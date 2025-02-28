//
// Created by xinyi on 25-1-13.
//

#ifndef PGM_CALL_DYNAMIC_HPP
#define PGM_CALL_DYNAMIC_HPP

#include <linear_model/pgm/pgm_index_dynamic.hpp>

using namespace pgm;

template<class key_type, class val_type>
DynamicPGMIndex<key_type, val_type> pgm_create_index(vector<pair<key_type, val_type>> &data) {
    auto sorted_data = data;
    sort(sorted_data.begin(), sorted_data.end());
    DynamicPGMIndex<key_type, val_type> pgm(sorted_data.begin(), sorted_data.end());
    return pgm;
}

template<class key_type, class val_type>
void pgm_update(DynamicPGMIndex<key_type, val_type>& pgm, pair<key_type, val_type> &kv) {
    pgm.insert_or_assign(kv.first, kv.second);
}

template<class key_type, class val_type>
void pgm_erase(DynamicPGMIndex<key_type, val_type>& pgm, key_type &key) {
    pgm.erase(key);
}

template<class key_type, class val_type>
vector<pair<key_type, val_type>> pgm_range_search(DynamicPGMIndex<key_type, val_type>& pgm, key_type &start_key, int range_size) {
    auto it = pgm.lower_bound(start_key);
    vector<pair<key_type, val_type>> result;
    for (int i = 0; i < range_size; ++i) {
        if (it == pgm.end())
            break;
        key_type key = it->first;
        val_type val = it->second;
        result.push_back(std::make_pair(key, val));
        ++it;
    }
    return result;
}

template<class key_type, class val_type>
val_type pgm_lookup(DynamicPGMIndex<key_type, val_type>& pgm, key_type &key) {
    auto it = pgm.lower_bound(key);
    if (it == pgm.end() || it->first != key)
        return 1;
    return it->second;
}

// Return the number of nodes in the PGM index with <inner_nodes_num, leaf_nodes_num>
template<class key_type, class val_type>
size_t pgm_leaves_num(DynamicPGMIndex<key_type, val_type>& pgm) {
    auto stat = pgm.get_stats();
    size_t num_of_leaves = 0;
    for (auto &level: stat) {
        num_of_leaves += level[0];
    }
    return num_of_leaves;
}

#endif //PGM_CALL_DYNAMIC_HPP
