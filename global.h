//
// Created by thomas on 9/17/26.
//

#ifndef CACHEPROBLEM_GLOBAL_H
#define CACHEPROBLEM_GLOBAL_H

#pragma once
#include <vector>
#include <fstream>
#include <cmath>
#include <utility>

struct ProblemData {
    int nbr_videos = 0;
    int nbr_endpoints = 0;
    int nbr_requests = 0;
    int nbr_caches = 0;
    int cap_cache = 0;
    std::vector<int> videos;
    std::vector<int> endpoints;
    std::vector<std::vector<int>> network;
    std::vector<std::vector<int>> requests;
};

struct Solution {
    std::vector<std::vector<bool>> results;
    std::vector<int> used_capacity;
};

int eval_time_saved(const ProblemData& problemData, const std::vector<std::vector<bool>>& results);
Solution run_evol_algo(const ProblemData& problem);

inline void generate_json_output(const std::vector<std::pair<int, double>>& data) {
    if (data.empty()) return;

    std::ofstream out("fitness_plot.json");
    if (!out) return;

    out << "[\n";
    for (size_t i = 0; i < data.size(); ++i) {
        out << "  {\n";
        out << "    \"eval\": " << data[i].first << ",\n";
        out << "    \"fitness\": " << data[i].second << "\n";
        out << "  }";

        if (i < data.size() - 1) {
            out << ",\n";
        } else {
            out << "\n";
        }
    }
    out << "]\n";
    out.close();
}

#endif //CACHEPROBLEM_GLOBAL_H
