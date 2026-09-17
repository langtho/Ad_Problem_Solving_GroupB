//
// Created by thomas on 9/17/26.
//

#ifndef CACHEPROBLEM_GLOBAL_H
#define CACHEPROBLEM_GLOBAL_H

#pragma once
#include <vector>

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

#endif //CACHEPROBLEM_GLOBAL_H
