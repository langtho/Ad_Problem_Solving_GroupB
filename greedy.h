#ifndef CACHEPROBLEM_GREEDY_H
#define CACHEPROBLEM_GREEDY_H

#pragma once
#include <string>
#include <vector>
#include "global.h"

enum class DecodingStrategy {
    STATIC_GREEDY,            // 0: Standard greedy decoder
    SOFT_SLACK_REPAIRED,      // 1: Soft capacity overflow + deterministic tail repair
    DEMAND_CACHE_CENTRIC,     // 2: Demand-weighted cache packing
    FIXED_THRESHOLD_PROFILE,  // 3: Static O(1) gate filters
    CO_EVOLVED_THRESHOLDS     // 4: Chromosome co-evolved threshold gates
};

struct DecodeParams {
    DecodingStrategy strategy = DecodingStrategy::STATIC_GREEDY;
    double tau_density = 0.0;  // Min gain/MB gate threshold
    double tau_reserve = 0.0;  // Reserved capacity fraction for top items
};
class GreedyDecoder {
public:
    explicit GreedyDecoder(const ProblemData& problem);

    // Videos that can save time: requested by an endpoint connected to a cache, and small enough for a cache
    const std::vector<int>& candidate_videos() const { return candidates; }

    // Potential time saved per MB of each video (0 for non-candidates)
    const std::vector<double>& video_priorities() const { return priority; }

    // Candidate videos sorted by decreasing priority
    std::vector<int> priority_order() const;

    long long decode(const std::vector<int>& order, Solution* solution = nullptr);

    // Block 5 Extended Decode Engine
    long long decode(const std::vector<int>& order, Solution* solution, const DecodeParams& params);

private:
    const ProblemData& problem;
    long long total_requests = 0;

    // Requests of video v: request_endpoint/request_count[request_start[v] .. request_start[v+1]-1]
    std::vector<int> request_start;
    std::vector<int> request_endpoint;
    std::vector<int> request_count;

    // Caches connected to each endpoint: (cache id, latency)
    std::vector<std::vector<std::pair<int, int>>> connections;

    std::vector<int> candidates;
    std::vector<double> priority;

    // Work buffers reused by decode()
    std::vector<int> best_latency;      // current best latency of each request
    std::vector<int> used_capacity;     // MB used in each cache
    std::vector<long long> cache_gain;  // time saved by adding the current video to each cache
    std::vector<int> touched_caches;

    long long decode_static_greedy(const std::vector<int>& order, Solution* solution, const DecodeParams& params);
    long long decode_soft_slack_repaired(const std::vector<int>& order, Solution* solution);
    long long decode_demand_cache_centric(const std::vector<int>& order, Solution* solution);
};

Solution run_greedy(const ProblemData& problem, double best_known_value, const std::string& instance_name);
Solution run_evol_algo_perm(const ProblemData& problem, double best_known_value, const std::string& instance_name);

#endif //CACHEPROBLEM_GREEDY_H
