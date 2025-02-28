#include <chrono>
#include <iostream>
#include <thread>

#include "opt_tree.hpp"

int main() {
    std::vector<uint64_t> test_seq;
    test_seq.push_back(1);
    test_seq.push_back(5);
    test_seq.push_back(6);
    test_seq.push_back(13);
    test_seq.push_back(15);
    test_seq.push_back(17);
    test_seq.push_back(30);
    test_seq.push_back(37);

    auto index = new OptBPlusTree<uint64_t, uint64_t>();

    for (auto key: test_seq) {
        pair<uint64_t, uint64_t> kv_pair = std::make_pair(key, 0);
        index->insert(kv_pair);
    }

}