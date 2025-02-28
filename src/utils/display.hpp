//
// Created by xinyi on 24-12-11.
//

#ifndef DISPLAY_H
#define DISPLAY_H

#include <iostream>

using namespace std;

void display_progress(const string& progress_name, const uint64_t current, const uint64_t total) {
    uint64_t progress = (current * 100) / total;

    std::string progressBar = "[";
    constexpr uint64_t barWidth = 50;
    const uint64_t pos = (current * barWidth) / total;
    for (uint64_t i = 0; i < barWidth; ++i) {
        if (i < pos) {
            progressBar += "=";
        } else if (i == pos) {
            progressBar += ">";
        } else {
            progressBar += " ";
        }
    }
    progressBar += "]";

    std::cout << "\r" << progress_name << ": " << progressBar << " " << progress << "%";
    std::cout.flush();
}

#endif //DISPLAY_H
