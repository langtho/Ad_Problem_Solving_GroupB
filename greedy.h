#ifndef CACHEPROBLEM_GREEDY_H
#define CACHEPROBLEM_GREEDY_H

#pragma once
#include <string>
#include <vector>
#include "global.h"

// Greedy decoder: fills the caches by following an order of videos.
// The caches are filled in rounds: in each round, every video (in the given
// order) is added to the cache where it saves the most time, if it still saves
// time and fits. A video gets a second copy only in the next round, after every
// other video had a chance to get a place.
// Used alone (greedy method) and inside the permutation GA (option 2).
class GreedyDecoder {
public:
    explicit GreedyDecoder(const ProblemData& problem);

    // Videos that can save time: requested by an endpoint connected to a cache, and small enough for a cache
    const std::vector<int>& candidate_videos() const { return candidates; }

    // Potential time saved per MB of each video (0 for non-candidates)
    const std::vector<double>& video_priorities() const { return priority; }

    // Candidate videos sorted by decreasing priority
    std::vector<int> priority_order() const;

    // Fills the caches following `order` and returns the score (same formula as eval_time_saved).
    // If `solution` is given, the placement is written into it.
    long long decode(const std::vector<int>& order, Solution* solution = nullptr);

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
};

Solution run_greedy(const ProblemData& problem, double best_known_value, const std::string& instance_name);
Solution run_evol_algo_perm(const ProblemData& problem, double best_known_value, const std::string& instance_name);

#endif //CACHEPROBLEM_GREEDY_H
