#ifndef __PRINT_HELPER_HPP__
#define __PRINT_HELPER_HPP__

#pragma once
#include <fmt/core.h>
#include <fstream>
#include <iostream>
#include <map>
#include <vector>

#include "load.hpp"
#include "opt_tree.hpp"

using namespace std;

inline void print_compile_mode() {
#ifndef NODEBUG
#pragma message ("Build in debug mode")
  fmt::println("Run in debug mode");
#else
  fmt::println("Run in release mode");
#endif
  fmt::println("Fanout: {}", FANOUT);
  fmt::println("Error bound: {}", MAX_EPSILON);
  fmt::println("Min model size factor: {}", MODEL_NODE_MIN_SIZE_FACTOR);
  fmt::println("Max model size factor: {}", MODEL_NODE_MAX_SIZE_FACTOR);
  fmt::println("Dataset ratio: {}", DATASET_RATIO);
}

template <typename T>
void serialize_vec_to_csv(const std::vector<T>& data, const std::string& filename) {
  std::ofstream file(filename);

  if (!file.is_open()) {
    std::cerr << "Cannot open: " << filename << std::endl;
    return;
  }

  for (const auto& row : data) {
    file << row;
    file << " ";
    // file << "\n";
  }

  fmt::println("Saved to {}", filename);
  file.close();
}

template <typename T, typename P>
void serialize_vec_pair_to_csv(const std::vector<pair<P, T>>& data, const std::string& filename) {
  std::ofstream file(filename);

  if (!file.is_open()) {
    std::cerr << "Cannot open: " << filename << std::endl;
    return;
  }

  for (const auto& row : data) {
    file << row.first;
    file << ":";
    file << row.second;
    file << "\n";
  }

  fmt::println("Saved to {}", filename);
  file.close();
}

template <typename T, typename P, typename U>
void serialize_vec_tuple_to_csv(const std::vector<tuple<P, T, U>>& data, const std::string& filename) {
  std::ofstream file(filename);

  if (!file.is_open()) {
    std::cerr << "Cannot open: " << filename << std::endl;
    return;
  }

  for (const auto& row : data) {
    file << std::get<0>(row);
    file << ":";
    file << std::get<1>(row);
    file << ":";
    file << std::get<2>(row);
    file << "\n";
  }

  fmt::println("Saved to {}", filename);
  file.close();
}



#endif