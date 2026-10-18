#ifndef CACHEPROBLEM_GREEDY_H
#define CACHEPROBLEM_GREEDY_H

#pragma once
#include <string>
#include <vector>
#include "global.h"

enum class DecodingStrategy {
    STATIC_GREEDY = 0,
    SOFT_SLACK_REPAIRED = 1
};

struct DecodeParams {
    DecodingStrategy strategy = DecodingStrategy::STATIC_GREEDY;
    double soft_slack_ratio = 1.05; // 5% overflow allowance won Sweep 3 on dense topologies
};

class GreedyDecoder {
public:
    explicit GreedyDecoder(const ProblemData& problem);

    const std::vector<int>& candidate_videos() const { return candidates; }
    const std::vector<double>& video_priorities() const { return priority; }
    std::vector<int> priority_order() const;

    // Const + thread-local buffers allow lock-free parallel calls from OpenGA worker threads
    long long decode(const std::vector<int>& order, Solution* solution = nullptr, std::vector<char>* out_placed = nullptr) const;
    long long decode(const std::vector<int>& order, Solution* solution, const DecodeParams& params, std::vector<char>* out_placed = nullptr) const;

private:
    const ProblemData& problem;
    long long total_requests = 0;

    // CSR-style flat adjacency arrays for cache-friendly O(1) request lookups per video
    std::vector<int> request_start;
    std::vector<int> request_endpoint;
    std::vector<int> request_count;
    std::vector<std::vector<std::pair<int, int>>> connections;

    std::vector<int> candidates;
    std::vector<double> priority;

    long long decode_static_greedy(const std::vector<int>& order, Solution* solution, std::vector<char>* out_placed) const;
    long long decode_soft_slack_repaired(const std::vector<int>& order, Solution* solution, const DecodeParams& params, std::vector<char>* out_placed) const;
};

#endif // CACHEPROBLEM_GREEDY_H