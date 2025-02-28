#pragma once

#include "../util.h"
#include <chrono>
#include <cmath>
#include <atomic>

class BaseSearch {
 public:
  template<typename Iterator, typename KeyType>
  static forceinline Iterator lower_bound(
    Iterator, Iterator,
		const KeyType&, Iterator,
    std::function<KeyType(Iterator)> = [](Iterator it)->KeyType{
      return static_cast<KeyType>(*it);
    }, 
    std::function<bool(const KeyType&, const KeyType&)> = [](const KeyType& key1, const KeyType& key2)->bool{
      return key1 < key2;
    });

  template<typename Iterator, typename KeyType>
  static forceinline Iterator upper_bound(
    Iterator, Iterator,
		const KeyType&, Iterator, 
    std::function<KeyType(Iterator)> = [](Iterator it)->KeyType{
      return static_cast<KeyType>(*it);
    },
    std::function<bool(const KeyType&, const KeyType&)> = [](const KeyType& key1, const KeyType& key2)->bool{
      return key1 < key2;
    });

  static std::string name();
};

template<int record>
class Search: public BaseSearch {
  public:
    static double searchAverageTime() { return 0; }
    static uint64_t searchTotalTime() { return 0; }
    static double searchBound() { return 0; }
    static void initSearch() {}
};

template<>
class Search<1>: public BaseSearch {
 public:
  static double searchAverageTime() { return (double)timing / search_num; }
  static uint64_t searchTotalTime() { return timing; }
  static double searchBound() { 
    return double(sum_search_bound) / research_num;
  }
  static void initSearch() { 
    timing = 0;
    search_num = 0;
    sum_search_bound = 0;
    research_num = 0;
  }

  static uint64_t timing;
  static size_t search_num;
  static uint64_t sum_search_bound;
  static size_t research_num;
};

template<>
class Search<2>: public BaseSearch {
 public:
  static double searchAverageTime() { return (double)timing / search_num; }
  static uint64_t searchTotalTime() { return timing; }
  static double searchBound() { return double(sum_search_bound) / research_num; }
  static void initSearch() { 
    timing = 0;
    search_num = 0;
    sum_search_bound = 0;
    research_num = 0;
  }

  static std::atomic<uint64_t> timing;
  static std::atomic<size_t> search_num;
  static std::atomic<uint64_t> sum_search_bound;
  static size_t research_num;
};