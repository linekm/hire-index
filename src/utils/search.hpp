#ifndef SEARCH_HPP
#define SEARCH_HPP

#include <immintrin.h>
#include <cinttypes>

#include <typeinfo>
#include <utility>

#if defined(__AVX512F__)
#pragma message("Compile with AVX512F")
#elif defined(__AVX2__)
#pragma message("Compile with AVX2")
#elif defined(__SSE2__)
#pragma message("Compile with SSE2")
#endif

#define SHUF(i0, i1, i2, i3) (i0 + i1*4 + i2*16 + i3*64)


template<typename T>
void prefetch_array(const T* array, size_t n) {
    for (size_t i = 0; i + 8 < n; i += 8) {
        _mm_prefetch((char*)&array[i], _MM_HINT_T0);
    }
}

inline int linear_search_avx2_u64(const uint64_t *arr, int n, uint64_t key) {
    int i = 0;
    __m256i vkey = _mm256_set1_epi64x(key);
    __m256i cnt = _mm256_setzero_si256();
    for (; i + 8 < n; i += 8) {
        __m256i mask0 = _mm256_cmpgt_epi64(vkey, _mm256_loadu_si256((__m256i *)&arr[i+0]));
        __m256i mask1 = _mm256_cmpgt_epi64(vkey, _mm256_loadu_si256((__m256i *)&arr[i+4]));
        __m256i sum = _mm256_add_epi64(mask0, mask1);
        cnt = _mm256_sub_epi64(cnt, sum);
    }
    __m128i xcnt = _mm_add_epi64(_mm256_extracti128_si256(cnt, 1), _mm256_castsi256_si128(cnt));
    xcnt = _mm_add_epi64(xcnt, _mm_shuffle_epi32(xcnt, SHUF(2, 3, 0, 1)));
    int count = _mm_cvtsi128_si32(xcnt);

    // int64_t count = 0;
    // int64_t tmp[4];
    // _mm256_storeu_si256(reinterpret_cast<__m256i*>(tmp), cnt);

    // for (const long j : tmp) {
    //     count += j;
    // }

    for (; i < n; ++i) {
        count += ((arr[i] < key) & 1);
    }
    return count;
}

inline int linear_search_avx512_u64(const uint64_t *arr, int n, uint64_t key) {
    int i = 0;
    int count = 0;
    __m512i vkey = _mm512_set1_epi64(key);
    for (; i + 16 < n; i += 16) {
        __mmask8 mask0 = _mm512_cmpgt_epu64_mask(vkey, _mm512_loadu_si512((__m512i *)&arr[i]));
        __mmask8 mask1 = _mm512_cmpgt_epu64_mask(vkey, _mm512_loadu_si512((__m512i *)&arr[i+8]));
        count += _mm_popcnt_u32(mask0);
        count += _mm_popcnt_u32(mask1);
    }

    for (; i < n; ++i) {
        count += ((arr[i] < key) & 1);
    }
    return count;
}

inline void cmov4_internal(const uint64_t cond, uint32_t& dst,
                           const uint32_t& val) {
    asm volatile(
        "test %[mcond], %[mcond]\n\t"
        "cmovnz %[i2], %[i1]\n\t"
        : [i1] "=r"(dst)
        : [mcond] "r"(cond), "[i1]"(dst), [i2] "r"(val)
        :);
}

template<typename T>
int linear_search(const T* arr, int n, T key) {
    int lo = 0;
    int pos = 0;
    for (; lo < n && arr[lo] < key; ++lo) {
        ++pos;
    }
    return pos;
}


inline int avx2_u32_lower_bound(const uint32_t* arr, int n, uint32_t key) {
    int i = 0;
    __m256i targetVec = _mm256_set1_epi32(key);
    for (; i <= n - 8; i += 8) {
        __m256i chunk = _mm256_loadu_si256((__m256i*)&arr[i]);
        __m256i cmp = _mm256_cmpgt_epi32(targetVec, chunk);
        int mask = _mm256_movemask_epi8(cmp);
        if (mask != 0xFFFFFFFF) {
            for (int j = 0; j < 8; ++j) {
                if (arr[i + j] >= key) {
                    return i + j;
                }
            }
        }
    }
    for (; i < n; ++i) {
        if (arr[i] >= key) {
            return i;
        }
    }
    return n;
}

template<typename T>
int liner_scan_with_smid(const T* arr, int n, T key) {

#ifdef __AVX512F__
    if constexpr (sizeof(T) == 8) {
        return linear_search_avx512_u64(arr, n, key);
    } else if constexpr (sizeof(T) == 4) {
        return avx2_u32_lower_bound(arr, n, key);
    } else {
        return linear_search(arr, n, key);
    }
#elifdef __AVX2__
    if constexpr (sizeof(T) == 4) {
        return avx2_u32_lower_bound(arr, n, key);
    } else {
        return linear_search_avx2_u64(arr, n, key);
    }
#else
    return linear_search(arr, n, key);
#endif
}

template<typename T>
int array_lower_bound(const T* arr, int n, T key) {
    prefetch_array(arr, n);
    return liner_scan_with_smid(arr, n, key);
    // return linear_search(arr, n, key);
}

#endif //SEARCH_HPP
