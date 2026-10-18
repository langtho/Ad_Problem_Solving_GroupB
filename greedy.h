#ifndef CACHEPROBLEM_GREEDY_H
#define CACHEPROBLEM_GREEDY_H

#pragma once
#include <string>
#include <vector>
#include "global.h"

enum class DecodingStrategy {
    STATIC_GREEDY,
    SOFT_SLACK_REPAIRED,
    DEMAND_CACHE_CENTRIC,
    FIXED_THRESHOLD_PROFILE,
    CO_EVOLVED_THRESHOLDS
};

struct DecodeParams {
    DecodingStrategy strategy = DecodingStrategy::STATIC_GREEDY;
    double tau_density = 0.0;
    double tau_reserve = 0.0;
    double soft_slack_ratio = 1.15;
};

struct DecodeState {
    std::vector<int> best_latency;
    std::vector<int> used_capacity;
    std::vector<char> placed;
    long long total_saved = 0;
    int start_idx = 0;
};

class GreedyDecoder {
public:
    explicit GreedyDecoder(const ProblemData& problem);

    const std::vector<int>& candidate_videos() const { return candidates; }
    const std::vector<double>& video_priorities() const { return priority; }
    std::vector<int> priority_order() const;

    long long decode(const std::vector<int>& order, Solution* solution = nullptr);
    long long decode(const std::vector<int>& order, Solution* solution, const DecodeParams& params, const DecodeState* start_state = nullptr, std::vector<DecodeState>* out_states = nullptr);
    const std::vector<char>& last_placed() const { return placed; }

private:
    const ProblemData& problem;
    long long total_requests = 0;

    std::vector<int> request_start;
    std::vector<int> request_endpoint;
    std::vector<int> request_count;
    std::vector<std::vector<std::pair<int, int>>> connections;

    std::vector<int> candidates;
    std::vector<double> priority;

    std::vector<int> best_latency;
    std::vector<int> used_capacity;
    std::vector<long long> cache_gain;
    std::vector<int> touched_caches;
    std::vector<char> placed;

    long long decode_static_greedy(const std::vector<int>& order, Solution* solution, const DecodeParams& params, const DecodeState* start_state, std::vector<DecodeState>* out_states);
    long long decode_soft_slack_repaired(const std::vector<int>& order, Solution* solution, const DecodeParams& params);
    long long decode_demand_cache_centric(const std::vector<int>& order, Solution* solution);
};

Solution run_greedy(const ProblemData& problem, double best_known_value, const std::string& instance_name);
Solution run_evol_algo_perm(const ProblemData& problem, double best_known_value, const std::string& instance_name);

#endif