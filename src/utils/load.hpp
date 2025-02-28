#ifndef DATA_LOADER
#define DATA_LOADER

#pragma once
#include <string>
#include <vector>
#include <fstream>
#include <cassert>
#include <sys/stat.h>  // POSIX头文件，用于mkdir
#include <sys/types.h> // POSIX头文件，用于mkdir
#include <errno.h>     // POSIX头文件，用于errno
#include <random>

#ifndef DATASET_RATIO
#define DATASET_RATIO 1
#endif

using namespace std;

void busy_wait_100ns() {
    auto start = std::chrono::high_resolution_clock::now();
    while (std::chrono::duration_cast<std::chrono::nanoseconds>(
           std::chrono::high_resolution_clock::now() - start).count() < 1000) {
           }
}

bool create_dir(const std::string& path) {
    size_t pos = 0;
    std::string current_path;

    while ((pos = path.find_first_of('/', pos)) != std::string::npos) {
        current_path = path.substr(0, pos);
        pos++;

        if (!current_path.empty() && current_path != "." && current_path != "..") {
            if (mkdir(current_path.c_str(), 0755) != 0 && errno != EEXIST) {
                return false;
            }
        }
    }

    if (mkdir(path.c_str(), 0755) != 0 && errno != EEXIST) {
        return false;
    }

    return true;
}

/*Loading SOSD datasets*/
template<typename K>
bool sosd_load_binary(string filename, vector<K> &v)
{
    ifstream ifs(filename, ios::in | ios::binary);
    assert(ifs);

    K size;
    ifs.read(reinterpret_cast<char*>(&size), sizeof(K));
    v.resize(size);
    ifs.read(reinterpret_cast<char*>(v.data()), size * sizeof(K));
    ifs.close();

    return ifs.good();
}

// Load preload data
vector<uint64_t> load_preload_data(string filename) {
    vector<uint64_t> data;
    ifstream ifs(filename);

    assert(ifs);

    uint64_t preload_data;

    while (ifs >> preload_data) {
        data.emplace_back(preload_data);
    }

    ifs.close();
    // assert(ifs.good());

    return data;
}

// Load operation data
vector<pair<uint32_t, uint64_t>> load_operation_data(string filename) {
    vector<pair<uint32_t, uint64_t>> data;
    ifstream ifs(filename);

    assert(ifs);

    uint32_t operation;
    uint64_t key;

    while (ifs >> operation >> key) {
        data.emplace_back(operation, key);
    }

    ifs.close();
    // assert(ifs.good());

    return data;
}

/*Dummy Workloads*/

#endif