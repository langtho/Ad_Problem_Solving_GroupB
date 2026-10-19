#include "greedy.h"
#include <algorithm>

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

    // Flatten sparse 2D request matrix into contiguous CSR arrays indexed by video ID
    request_start.assign(V + 1, 0);
    for (int v = 0; v < V; ++v) {
        request_start[v] = static_cast<int>(request_endpoint.size());
        for (int e = 0; e < E; ++e) {
            int count = problem.requests[e][v];
            if (count > 0) {
                request_endpoint.push_back(e);
                request_count.push_back(count);
                total_requests += count;
            }
        }
    }
    request_start[V] = static_cast<int>(request_endpoint.size());

    // Pre-filter unplaceable videos and rank candidates by theoretical upper-bound gain per MB
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
            potential += static_cast<long long>(request_count[r]) * (problem.endpoints[e] - best);
        }
        if (potential > 0) {
            priority[v] = static_cast<double>(potential) / problem.videos[v];
            candidates.push_back(v);
        }
    }
}

vector<int> GreedyDecoder::priority_order() const {
    vector<int> order = candidates;
    stable_sort(order.begin(), order.end(), [this](int a, int b) { return priority[a] > priority[b]; });
    return order;
}

long long GreedyDecoder::decode(const vector<int>& order, Solution* solution, vector<char>* out_placed) const {
    DecodeParams default_params;
    return decode(order, solution, default_params, out_placed);
}

long long GreedyDecoder::decode(const vector<int>& order, Solution* solution, const DecodeParams& params, vector<char>* out_placed) const {
    if (params.strategy == DecodingStrategy::SOFT_SLACK_REPAIRED) {
        return decode_soft_slack_repaired(order, solution, params, out_placed);
    }
    return decode_static_greedy(order, solution, out_placed);
}

// Path B Decoder: Multi-placement per video to cover isolated clusters in sparse networks
long long GreedyDecoder::decode_static_greedy(const vector<int>& order, Solution* solution, vector<char>* out_placed) const {
    // Thread-local scratchpads avoid heap allocations per evaluation and prevent thread races
    static thread_local vector<int> best_latency;
    static thread_local vector<int> used_capacity;
    static thread_local vector<long long> cache_gain;
    static thread_local vector<int> touched_caches;

    best_latency.resize(request_endpoint.size());
    for (size_t r = 0; r < best_latency.size(); ++r) {
        best_latency[r] = problem.endpoints[request_endpoint[r]];
    }

    used_capacity.assign(problem.nbr_caches, 0);
    cache_gain.assign(problem.nbr_caches, 0);
    touched_caches.clear();

    if (out_placed) out_placed->assign(problem.nbr_videos, 0);

    if (solution) {
        solution->sequence = order;
        solution->results.assign(problem.nbr_caches, vector<bool>(problem.nbr_videos, false));
        solution->used_capacity.assign(problem.nbr_caches, 0);
    }

    long long total_saved = 0;

    for (int v : order) {
        int size = problem.videos[v];
        if (size <= 0 || size > problem.cap_cache) continue;

        // Repeatedly place video v into whichever cache yields the highest marginal latency reduction
        while (true) {
            int best_cache = -1;
            long long best_gain = 0;

            for (int r = request_start[v]; r < request_start[v + 1]; ++r) {
                for (auto [c, latency] : connections[request_endpoint[r]]) {
                    if (latency < best_latency[r] && used_capacity[c] + size <= problem.cap_cache) {
                        if (cache_gain[c] == 0) touched_caches.push_back(c);
                        cache_gain[c] += static_cast<long long>(request_count[r]) * (best_latency[r] - latency);
                    }
                }
            }

            for (int c : touched_caches) {
                if (cache_gain[c] > best_gain) {
                    best_gain = cache_gain[c];
                    best_cache = c;
                }
                cache_gain[c] = 0;
            }
            touched_caches.clear();
            if (best_cache < 0) break;

            used_capacity[best_cache] += size;
            if (out_placed) (*out_placed)[v] = 1;

            if (solution) {
                solution->results[best_cache][v] = true;
                solution->used_capacity[best_cache] += size;
            }

            for (int r = request_start[v]; r < request_start[v + 1]; ++r) {
                int latency = problem.network[request_endpoint[r]][best_cache];
                if (latency > 0 && latency < best_latency[r]) {
                    total_saved += static_cast<long long>(request_count[r]) * (best_latency[r] - latency);
                    best_latency[r] = latency;
                }
            }
        }
    }
    return total_requests ? (total_saved * 1000) / total_requests : 0;
}

// Path A Decoder: Single-placement + 5% soft capacity slack to prevent viral videos from monopolizing dense caches
// Path A Decoder: Multi-placement + 5% soft capacity slack to optimize knapsack density
long long GreedyDecoder::decode_soft_slack_repaired(const vector<int>& order, Solution* solution, const DecodeParams& params, vector<char>* out_placed) const {
    static thread_local vector<int> best_latency;
    static thread_local vector<int> used_capacity;
    static thread_local vector<long long> cache_gain;
    static thread_local vector<int> touched_caches;

    best_latency.resize(request_endpoint.size());
    for (size_t r = 0; r < best_latency.size(); ++r) {
        best_latency[r] = problem.endpoints[request_endpoint[r]];
    }

    used_capacity.assign(problem.nbr_caches, 0);
    cache_gain.assign(problem.nbr_caches, 0);
    touched_caches.clear();

    if (out_placed) out_placed->assign(problem.nbr_videos, 0);

    vector<vector<bool>> local_results(problem.nbr_caches, vector<bool>(problem.nbr_videos, false));
    int soft_cap = static_cast<int>(problem.cap_cache * params.soft_slack_ratio);

    vector<vector<pair<int, long long>>> placed_items(problem.nbr_caches);

    for (int v : order) {
        int size = problem.videos[v];
        if (size <= 0 || size > problem.cap_cache) continue;

        // UPGRADE: Added the while(true) loop to allow multi-placement in Soft Slack
        while (true) {
            int best_cache = -1;
            long long best_gain = 0;

            for (int r = request_start[v]; r < request_start[v + 1]; ++r) {
                for (auto [c, latency] : connections[request_endpoint[r]]) {
                    if (latency < best_latency[r] && used_capacity[c] + size <= soft_cap) {
                        if (cache_gain[c] == 0) touched_caches.push_back(c);
                        cache_gain[c] += static_cast<long long>(request_count[r]) * (best_latency[r] - latency);
                    }
                }
            }

            for (int c : touched_caches) {
                if (cache_gain[c] > best_gain) {
                    best_gain = cache_gain[c];
                    best_cache = c;
                }
                cache_gain[c] = 0;
            }
            touched_caches.clear();

            // If no profitable cache is found, break to move to the next video
            if (best_cache < 0) break;

            used_capacity[best_cache] += size;
            placed_items[best_cache].push_back({v, best_gain});

            // Update latency dynamically so the next loop iteration evaluates fresh margins
            for (int r = request_start[v]; r < request_start[v + 1]; ++r) {
                int latency = problem.network[request_endpoint[r]][best_cache];
                if (latency > 0 && latency < best_latency[r]) {
                    best_latency[r] = latency;
                }
            }
        }
    }

    // Feasibility repair: evict lowest gain-density items until hard cache capacity is satisfied
    for (int c = 0; c < problem.nbr_caches; ++c) {
        while (used_capacity[c] > problem.cap_cache && !placed_items[c].empty()) {
            auto min_it = min_element(placed_items[c].begin(), placed_items[c].end(), [this](const auto& a, const auto& b) {
                double density_a = static_cast<double>(a.second) / problem.videos[a.first];
                double density_b = static_cast<double>(b.second) / problem.videos[b.first];
                return density_a < density_b;
            });

            used_capacity[c] -= problem.videos[min_it->first];
            placed_items[c].erase(min_it);
        }

        for (const auto& item : placed_items[c]) {
            local_results[c][item.first] = true;
            if (out_placed) (*out_placed)[item.first] = 1;
        }
    }

    if (solution) {
        solution->sequence = order;
        solution->results = local_results;
        solution->used_capacity = used_capacity;
    }

    return eval_time_saved(problem, local_results);
}