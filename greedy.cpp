#include <algorithm>
#include <chrono>
#include <iostream>
#include "greedy.h"

using namespace std;

GreedyDecoder::GreedyDecoder(const ProblemData& problem) : problem(problem) {
    int V = problem.nbr_videos;
    int E = problem.nbr_endpoints;
    int C = problem.nbr_caches;

    // Caches connected to each endpoint (network[e][c] == 0 means no connection)
    connections.assign(E, {});
    for (int e = 0; e < E; ++e) {
        for (int c = 0; c < C; ++c) {
            if (problem.network[e][c] > 0) {
                connections[e].push_back({c, problem.network[e][c]});
            }
        }
    }

    // Requests grouped by video
    request_start.assign(V + 1, 0);
    for (int v = 0; v < V; ++v) {
        request_start[v] = request_endpoint.size();
        for (int e = 0; e < E; ++e) {
            int count = problem.requests[e][v];
            if (count > 0) {
                request_endpoint.push_back(e);
                request_count.push_back(count);
                total_requests += count;
            }
        }
    }
    request_start[V] = request_endpoint.size();

    // Priority of a video = time it could save if it was in the best cache of every endpoint, per MB
    priority.assign(V, 0.0);
    for (int v = 0; v < V; ++v) {
        if (problem.videos[v] > problem.cap_cache) continue;
        long long potential = 0;
        for (int r = request_start[v]; r < request_start[v + 1]; ++r) {
            int e = request_endpoint[r];
            int best = problem.endpoints[e];
            for (auto [c, latency] : connections[e]) {
                best = min(best, latency);
            }
            potential += (long long)request_count[r] * (problem.endpoints[e] - best);
        }
        if (potential > 0) {
            priority[v] = (double)potential / problem.videos[v];
            candidates.push_back(v);
        }
    }

    best_latency.resize(request_endpoint.size());
    used_capacity.resize(C);
    cache_gain.assign(C, 0);
}

vector<int> GreedyDecoder::priority_order() const {
    vector<int> order = candidates;
    stable_sort(order.begin(), order.end(), [this](int a, int b) { return priority[a] > priority[b]; });
    return order;
}

long long GreedyDecoder::decode(const vector<int>& order, Solution* solution) {
    for (size_t r = 0; r < best_latency.size(); ++r) {
        best_latency[r] = problem.endpoints[request_endpoint[r]]; // datacenter by default
    }
    fill(used_capacity.begin(), used_capacity.end(), 0);
    if (solution) {
        solution->results.assign(problem.nbr_caches, vector<bool>(problem.nbr_videos, false));
        solution->used_capacity.assign(problem.nbr_caches, 0);
    }

    // Filled in rounds: in each round, every video (in order) is added to at most
    // one more cache, its best one. A video is copied into several caches only
    // after the other videos had a chance to get a place.
    // A video that finds no useful cache is dropped: caches only fill up, so it never will.
    long long total_saved = 0;
    vector<int> active = order;
    vector<int> next_active;
    while (!active.empty()) {
        next_active.clear();
        for (int v : active) {
            int size = problem.videos[v];
            // Time saved by adding v to each cache that still has room for it
            for (int r = request_start[v]; r < request_start[v + 1]; ++r) {
                for (auto [c, latency] : connections[request_endpoint[r]]) {
                    if (latency < best_latency[r] && used_capacity[c] + size <= problem.cap_cache) {
                        if (cache_gain[c] == 0) touched_caches.push_back(c);
                        cache_gain[c] += (long long)request_count[r] * (best_latency[r] - latency);
                    }
                }
            }

            int best_cache = -1;
            long long best_gain = 0;
            for (int c : touched_caches) {
                if (cache_gain[c] > best_gain) {
                    best_gain = cache_gain[c];
                    best_cache = c;
                }
                cache_gain[c] = 0;
            }
            touched_caches.clear();
            if (best_cache < 0) continue; // no cache where v still saves time

            // Add v to the best cache and update the latency of its requests
            used_capacity[best_cache] += size;
            if (solution) {
                solution->results[best_cache][v] = true;
                solution->used_capacity[best_cache] += size;
            }
            for (int r = request_start[v]; r < request_start[v + 1]; ++r) {
                int latency = problem.network[request_endpoint[r]][best_cache];
                if (latency > 0 && latency < best_latency[r]) {
                    total_saved += (long long)request_count[r] * (best_latency[r] - latency);
                    best_latency[r] = latency;
                }
            }
            next_active.push_back(v);
        }
        swap(active, next_active);
    }

    return total_requests ? (total_saved * 1000) / total_requests : 0;
}

Solution run_greedy(const ProblemData& problem, double best_known_value, const string& instance_name) {
    auto start_time = chrono::high_resolution_clock::now();

    GreedyDecoder decoder(problem);
    Solution sol;
    long long score = decoder.decode(decoder.priority_order(), &sol);

    chrono::duration<double> elapsed = chrono::high_resolution_clock::now() - start_time;

    // A single evaluation: every target below the score is reached at evaluation 1
    vector<double> targets = generate_targets(best_known_value, 50);
    vector<pair<int, double>> history;
    for (double target : targets) {
        if (score >= target) history.push_back({1, target});
    }
    generate_json_output(instance_name, history, 1, 0, score, elapsed.count(), 1, "benchmark_results");

    cerr << "Greedy | score: " << score << " | time: " << elapsed.count() << "s\n";
    return sol;
}
