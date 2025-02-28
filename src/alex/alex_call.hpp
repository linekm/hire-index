//
// Created by xinyi on 24-12-1.
//

#ifndef ALEX_CALL_H
#define ALEX_CALL_H
#include "alex.h"

template<typename KeyType, typename ValueType>
alex::Alex<KeyType, ValueType>* alex_load(std::vector<std::pair<KeyType, ValueType>> &data) {
    size_t batch_size = data.size() * 0.25;
    size_t cnt = 0;
    auto* alex = new alex::Alex<KeyType,ValueType>();
    for (auto it : data) {
        alex->insert(it.first, it.second);
        ++cnt;
        if (cnt % batch_size == 0) {
            println("Insertion progress: {:.2f}", static_cast<double>(cnt) / data.size());
        }
    }
    return alex;
}

template<typename Key, typename Value>
std::vector<std::pair<Key, Value>> alex_range_search(alex::Alex<Key, Value> *alex, Key lower_bound, Key upper_bound) {
    auto itStart = alex->lower_bound(lower_bound);
    auto itEnd = alex->lower_bound(upper_bound);
    std::vector<std::pair<Key, Value>> searchResult;

    while (itStart != itEnd) {
        if (itStart.key() >= lower_bound && itStart.key() <= upper_bound) {
            searchResult.push_back(std::make_pair(itStart.key(), itStart.payload()));
        }
        ++itStart;
    }

    return searchResult;
}

template<typename Key, typename Value>
std::vector<std::pair<Key, Value>> alex_range_search(alex::Alex<Key, Value> *alex, Key lower_bound, int search_num) {
    auto itStart = alex->lower_bound(lower_bound);
    std::vector<std::pair<Key, Value>> searchResult;
    int res_cnt = 0;

    while (searchResult.size() < search_num && itStart != alex->end()) {
        searchResult.push_back(std::make_pair(itStart.key(), itStart.payload()));
        ++itStart;
    }
    return searchResult;
}

template<typename Key, typename Value>
size_t alex_index_size(alex::Alex<Key, Value> *alex) {
    return alex->model_size() + alex->data_size();
}

#endif //ALEX_CALL_H
