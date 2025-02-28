
#ifndef ARRAY_HPP
#define ARRAY_HPP
#include <cstring>

template<typename T>
void shift_left_array(T* arr, size_t size, size_t shift) {
    if (shift == 0 || shift > size) return;
    memmove(arr, arr + shift, (size - shift) * sizeof(T));
}

template<typename T>
void shift_right_array(T* arr, size_t size, size_t shift) {
    if (shift == 0 || shift > size) return;
    memmove(arr + shift, arr, (size - shift) * sizeof(T));
}


#endif //ARRAY_HPP
