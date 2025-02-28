//
// Created by xinyi on 25-1-14.
//

#ifndef LIPP_CALL_H
#define LIPP_CALL_H

#include "lipp.h"

template<typename key_type, typename value_type>
class Lipp {
public:
    LIPP<key_type, value_type> index;
public:
    Lipp() {}
    ~Lipp() {}

    bool exist(const key_type& key) const {
        return index.exists(key);
    }

    value_type lookup(const key_type& key) {
        return index.at(key);
    }

    bool exists(const key_type& key) const {
        return index.exists(key);
    }

    void insert(const key_type& key, const value_type& value) {
        index.insert(key, value);
    }

    void bulk_load(const std::vector<std::pair<key_type, value_type>>& data) {
        index.bulk_load(data.data(), data.size());
        index.print_stats();
    }

    bool find(const key_type& key, value_type& value) const {
        return index.find(key, value);
    }

    std::vector<std::pair<key_type, value_type>> range_query(const key_type& start_key, size_t range_size) {
        auto it = index.lower_bound(start_key);
        std::vector<std::pair<key_type, value_type>> results;
        for (size_t i = 0; i < range_size && it != index.end(); ++i, ++it) {
            results.push_back({it->comp.data.key, it->comp.data.value});
        }
        return results;
    }

    uint64_t index_size() {
        return index.index_size();
    }

    void show() {
        index.show();
    }

    void erase(const key_type& key) {
        index.erase(key);
    }
};

#endif //LIPP_CALL_H
