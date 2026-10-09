#include <algorithm>
#include <chrono>
#include <iostream>
#include "greedy.h"

using namespace std;

GreedyDecoder::GreedyDecoder(const ProblemData& problem) : problem(problem) {
    int V = problem.nbr_videos;
    int E = problem.nbr_endpoints;
    int C = problem.nbr_caches;

    connections.assign(E, {});
    for (int e = 0; e < E; ++e) {
        for (int c = 0; c < C; ++c) {
            if (problem.network[e][c] > 0) {
                connections[e].push_back({c, problem.network[e][c]});
            }
        }
    }

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
    DecodeParams default_params;
    return decode(order, solution, default_params);
}

long long GreedyDecoder::decode(const vector<int>& order, Solution* solution, const DecodeParams& params) {
    switch (params.strategy) {
        case DecodingStrategy::SOFT_SLACK_REPAIRED:
            return decode_soft_slack_repaired(order, solution);

        case DecodingStrategy::DEMAND_CACHE_CENTRIC:
            return decode_demand_cache_centric(order, solution);

        case DecodingStrategy::FIXED_THRESHOLD_PROFILE:
        case DecodingStrategy::CO_EVOLVED_THRESHOLDS:
        case DecodingStrategy::STATIC_GREEDY:
        default:
            return decode_static_greedy(order, solution, params);
    }
}

long long GreedyDecoder::decode_static_greedy(const vector<int>& order, Solution* solution, const DecodeParams& params) {
    for (size_t r = 0; r < best_latency.size(); ++r) {
        best_latency[r] = problem.endpoints[request_endpoint[r]];
    }
    fill(used_capacity.begin(), used_capacity.end(), 0);

    if (solution) {
        solution->sequence = order;
        solution->results.assign(problem.nbr_caches, vector<bool>(problem.nbr_videos, false));
        solution->used_capacity.assign(problem.nbr_caches, 0);
    }

    long long total_saved = 0;
    vector<int> active = order;
    vector<int> next_active;

    int total_active_vids = active.size();
    int top_reserve_idx = static_cast<int>(total_active_vids * (1.0 - params.tau_reserve));

    int vid_rank = 0;
    while (!active.empty()) {
        next_active.clear();
        for (int v : active) {
            int size = problem.videos[v];

            for (int r = request_start[v]; r < request_start[v + 1]; ++r) {
                for (auto [c, latency] : connections[request_endpoint[r]]) {
                    // Check reserve capacity constraint if threshold gate active
                    int eff_cap = (params.tau_reserve > 0.0 && vid_rank > top_reserve_idx) ?
                                  static_cast<int>(problem.cap_cache * (1.0 - params.tau_reserve)) : problem.cap_cache;

                    if (latency < best_latency[r] && used_capacity[c] + size <= eff_cap) {
                        if (cache_gain[c] == 0) touched_caches.push_back(c);
                        cache_gain[c] += (long long)request_count[r] * (best_latency[r] - latency);
                    }
                }
            }

            int best_cache = -1;
            long long best_gain = 0;
            for (int c : touched_caches) {
                double gain_per_mb = size > 0 ? static_cast<double>(cache_gain[c]) / size : 0.0;

                // Density gate filter
                if (cache_gain[c] > best_gain && gain_per_mb >= params.tau_density) {
                    best_gain = cache_gain[c];
                    best_cache = c;
                }
                cache_gain[c] = 0;
            }
            touched_caches.clear();
            if (best_cache < 0) continue;

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
            vid_rank++;
        }
        swap(active, next_active);
    }

    return total_requests ? (total_saved * 1000) / total_requests : 0;
}

long long GreedyDecoder::decode_soft_slack_repaired(const vector<int>& order, Solution* solution) {
    for (size_t r = 0; r < best_latency.size(); ++r) {
        best_latency[r] = problem.endpoints[request_endpoint[r]];
    }
    fill(used_capacity.begin(), used_capacity.end(), 0);

    vector<vector<bool>> local_results(problem.nbr_caches, vector<bool>(problem.nbr_videos, false));
    int soft_cap = static_cast<int>(problem.cap_cache * 1.15); // 15% Soft-Overflow

    vector<vector<pair<int, long long>>> placed_items(problem.nbr_caches);

    for (int v : order) {
        int size = problem.videos[v];
        for (int r = request_start[v]; r < request_start[v + 1]; ++r) {
            for (auto [c, latency] : connections[request_endpoint[r]]) {
                if (latency < best_latency[r] && used_capacity[c] + size <= soft_cap) {
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
        if (best_cache < 0) continue;

        used_capacity[best_cache] += size;
        placed_items[best_cache].push_back({v, best_gain});

        for (int r = request_start[v]; r < request_start[v + 1]; ++r) {
            int latency = problem.network[request_endpoint[r]][best_cache];
            if (latency > 0 && latency < best_latency[r]) {
                best_latency[r] = latency;
            }
        }
    }

    // Repair Pass: Entferne Items mit geringstem Gain/MB bis Hard Capacity eingehalten wird
    for (int c = 0; c < problem.nbr_caches; ++c) {
        while (used_capacity[c] > problem.cap_cache && !placed_items[c].empty()) {
            auto min_it = min_element(placed_items[c].begin(), placed_items[c].end(), [this](const auto& a, const auto& b) {
                double density_a = (double)a.second / problem.videos[a.first];
                double density_b = (double)b.second / problem.videos[b.first];
                return density_a < density_b;
            });

            used_capacity[c] -= problem.videos[min_it->first];
            placed_items[c].erase(min_it);
        }

        for (auto& item : placed_items[c]) {
            local_results[c][item.first] = true;
        }
    }

    if (solution) {
        solution->sequence = order;
        solution->results = local_results;
        solution->used_capacity = used_capacity;
    }

    return eval_time_saved(problem, local_results);
}

long long GreedyDecoder::decode_demand_cache_centric(const vector<int>& order, Solution* solution) {
    vector<vector<bool>> local_results(problem.nbr_caches, vector<bool>(problem.nbr_videos, false));

    vector<pair<long long, int>> cache_demand(problem.nbr_caches);
    for (int c = 0; c < problem.nbr_caches; ++c) {
        cache_demand[c] = {0, c};
    }

    for (int e = 0; e < problem.nbr_endpoints; ++e) {
        for (auto [c, lat] : connections[e]) {
            for (int v = 0; v < problem.nbr_videos; ++v) {
                cache_demand[c].first += problem.requests[e][v];
            }
        }
    }

    sort(cache_demand.rbegin(), cache_demand.rend());

    vector<int> cache_used(problem.nbr_caches, 0);

    for (auto [demand, c] : cache_demand) {
        for (int v : order) {
            int size = problem.videos[v];
            if (cache_used[c] + size <= problem.cap_cache) {
                cache_used[c] += size;
                local_results[c][v] = true;
            }
        }
    }

    if (solution) {
        solution->sequence = order;
        solution->results = local_results;
        solution->used_capacity = cache_used;
    }

    return eval_time_saved(problem, local_results);
}

Solution run_greedy(const ProblemData& problem, double best_known_value, const string& instance_name) {
    auto start_time = chrono::high_resolution_clock::now();

    GreedyDecoder decoder(problem);
    Solution sol;
    long long score = decoder.decode(decoder.priority_order(), &sol);

    chrono::duration<double> elapsed = chrono::high_resolution_clock::now() - start_time;

    vector<double> targets = generate_targets(best_known_value, 50);
    vector<pair<int, double>> history;
    for (double target : targets) {
        if (score >= target) history.push_back({1, target});
    }
    generate_json_output(instance_name, history, 1, 0, score, elapsed.count(), 1, "benchmark_results");

    //cerr << "Greedy | score: " << score << " | time: " << elapsed.count() << "s\n";
    return sol;
}