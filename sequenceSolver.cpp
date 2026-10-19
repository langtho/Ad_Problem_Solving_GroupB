#include "Sequence_Solver.h"
#include "global.h"
#include <chrono>
#include <cmath>
#include <algorithm>
#include <set>
#include <mutex>
#include <stdexcept>

using namespace std;

Solution SequenceSolver::run(const ProblemData& problem, double, const string&) {
    return run_production(problem, 42, 230.0);
}

Solution SequenceSolver::run_production(
    const ProblemData& problem,
    unsigned int base_seed,
    double max_run_time_seconds
) {
    auto start_time = chrono::high_resolution_clock::now();

    GreedyDecoder decoder(problem);
    if (decoder.candidate_videos().empty()) {
        Solution empty_sol;
        empty_sol.results.assign(problem.nbr_caches, vector<bool>(problem.nbr_videos, false));
        empty_sol.used_capacity.assign(problem.nbr_caches, 0);
        return empty_sol;
    }

    DynamicHyperparameters dyn_params = calculate_dynamic_hyperparameters(problem, decoder);

    VideoOrder overall_best_order;
    double overall_best_fitness = -1.0;
    mutex pheno_mutex, state_mutex;
    set<size_t> generation_phenotypes;

    int restart_count = 0;

    // CONTINUOUS RESTART LOOP: Squeeze every drop of optimization out of the time limit
    while (chrono::duration<double>(chrono::high_resolution_clock::now() - start_time).count() < max_run_time_seconds - 3.0) {
        Perm_Engine ga;

        // Different seed per restart guarantees we explore new areas of the search space
        ga.set_seed(base_seed + restart_count);
        ga.multi_threading = true;
        ga.problem_mode = EA::GA_MODE::SOGA;
        ga.generation_max = 10000;
        ga.verbose = false;
        ga.tol_stall_best = 1e-6;
        ga.tol_stall_average = 1e-6;

        ga.population = dyn_params.population_size;
        ga.crossover_fraction = dyn_params.crossover_fraction;
        ga.mutation_rate = dyn_params.mutation_rate;
        ga.elite_count = dyn_params.elite_count;

        // Lower stall_max so we detect stagnation quickly and restart the search
        ga.best_stall_max = 150;
        ga.average_stall_max = 150;

        int ind_counter = 0;

        ga.init_genes = [&](VideoOrder& s, const function<double(void)>& rnd) {
            // Thread-safe Soft Stop: Fast return the baseline if time is out
            if (chrono::duration<double>(chrono::high_resolution_clock::now() - start_time).count() >= max_run_time_seconds) {
                s.videos = decoder.priority_order();
                return;
            }
            int idx;
            {
                lock_guard<mutex> lock(state_mutex);
                idx = ind_counter++;
            }

            // Inject the global best sequence into the new population to keep climbing
            if (idx == 0 && !overall_best_order.videos.empty()) {
                s = overall_best_order;
            } else {
                s = generate_initial_individual(dyn_params.init_strat, decoder, idx, ga.population, dyn_params.active_k, dyn_params.noise_scale, rnd);
            }
        };

        ga.eval_solution = [&](const VideoOrder& s, double& score) -> bool {
            // Thread-safe Soft Stop: Bypass heavy evaluation, return 0.0 so thread exits peacefully
            if (chrono::duration<double>(chrono::high_resolution_clock::now() - start_time).count() >= max_run_time_seconds) {
                score = 0.0;
                return true;
            }

            DecodeParams params;
            params.strategy = dyn_params.dec_strat;
            params.soft_slack_ratio = dyn_params.soft_slack_ratio;

            Solution sol;
            score = static_cast<double>(decoder.decode(s.videos, &sol, params));

            size_t p_hash = sol.used_capacity.size();
            for (int cap : sol.used_capacity) {
                p_hash ^= cap + 0x9e3779b9 + (p_hash << 6) + (p_hash >> 2);
            }

            {
                lock_guard<mutex> lock(state_mutex);
                if (score > overall_best_fitness) {
                    overall_best_fitness = score;
                    overall_best_order = s;
                }
            }

            {
                lock_guard<mutex> lock(pheno_mutex);
                if (generation_phenotypes.count(p_hash)) {
                    score *= 0.90; // Diversity penalty for clones
                } else {
                    generation_phenotypes.insert(p_hash);
                }
            }
            return true;
        };

        ga.calculate_SO_total_fitness = [](const Perm_Engine::thisChromosomeType& c) -> double {
            return -c.middle_costs;
        };

        ga.mutate = [&](const VideoOrder& s, const function<double(void)>& rnd, double) -> VideoOrder {
            vector<char> placed;
            DecodeParams params;
            params.strategy = dyn_params.dec_strat;
            params.soft_slack_ratio = dyn_params.soft_slack_ratio;

            // Run fast pre-decode using the *correct* path strategy so PROMOTE_UNPLACED is accurate
            decoder.decode(s.videos, nullptr, params, &placed);
            return execute_mutation(dyn_params.mut_strat, s, dyn_params.active_k, rnd, &placed);
        };

        ga.crossover = [&](const VideoOrder& p1, const VideoOrder& p2, const function<double(void)>& rnd) -> VideoOrder {
            return execute_crossover(dyn_params.cx_strat, p1, p2, problem.nbr_videos, dyn_params.active_k, rnd);
        };

        ga.SO_report_generation = [&](int, const Perm_Engine::thisGenerationType&, const VideoOrder&) {
            {
                lock_guard<mutex> lock(pheno_mutex);
                generation_phenotypes.clear();
            }
            if (chrono::duration<double>(chrono::high_resolution_clock::now() - start_time).count() >= max_run_time_seconds) {
                ga.user_request_stop = true;
            }
        };

        try {
            ga.solve();
        } catch (const runtime_error&) {}

        restart_count++;
    }

    // Safety fallback: if GA found nothing before timing out, use the raw priority order
    if (overall_best_order.videos.empty()) {
        overall_best_order.videos = decoder.priority_order();
    }

    Solution final_solution;
    final_solution.results.assign(problem.nbr_caches, vector<bool>(problem.nbr_videos, false));
    final_solution.used_capacity.assign(problem.nbr_caches, 0);

    DecodeParams final_params;
    final_params.strategy = dyn_params.dec_strat;
    final_params.soft_slack_ratio = dyn_params.soft_slack_ratio;
    final_solution.sequence = overall_best_order.videos;
    decoder.decode(final_solution.sequence, &final_solution, final_params);

    return final_solution;
}

// Hyper-Heuristic Router: Selects operators and Sweep 3 tuned parameters from network topology
DynamicHyperparameters SequenceSolver::calculate_dynamic_hyperparameters(
    const ProblemData& problem,
    const GreedyDecoder& decoder
) {
    DynamicHyperparameters params;
    int candidates_count = static_cast<int>(decoder.candidate_videos().size());

    long long possible_connections = static_cast<long long>(problem.nbr_endpoints) * problem.nbr_caches;
    long long actual_connections = 0;
    for (int e = 0; e < problem.nbr_endpoints; ++e) {
        for (int c = 0; c < problem.nbr_caches; ++c) {
            if (problem.network[e][c] > 0) actual_connections++;
        }
    }

    double network_density = possible_connections > 0 ? static_cast<double>(actual_connections) / possible_connections : 0.0;
    long long matrix_complexity = static_cast<long long>(problem.nbr_videos) * problem.nbr_endpoints * problem.nbr_caches;

    long long total_cache_cap = static_cast<long long>(problem.nbr_caches) * problem.cap_cache;
    long long total_video_size = 0;
    for (int s : problem.videos) total_video_size += s;
    double capacity_ratio = total_video_size > 0 ? static_cast<double>(total_cache_cap) / total_video_size : 0.0;

    if (network_density > 0.4) {
        params.crossover_fraction = 0.8;
        params.mutation_rate = 0.1;
    } else if (network_density > 0.1) {
        params.crossover_fraction = 0.65;
        params.mutation_rate = 0.2;
    } else {
        params.crossover_fraction = 0.4;
        params.mutation_rate = 0.35;
    }

    params.soft_slack_ratio = 1.05;

    if (matrix_complexity > 50000000) {
        params.population_size = 25;
    } else {
        params.population_size = max(30, min(80, candidates_count / 40));
    }

    if (matrix_complexity > 50000000 || network_density > 0.4) {
        params.init_strat    = InitStrategy::HYBRID_COMPOSITE;
        params.cx_strat      = CrossoverStrategy::ACTIVE_PREFIX_OX1;
        params.mut_strat     = MutationStrategy::PROMOTE_UNPLACED;
        params.ls_strat      = LocalSearchStrategy::NONE;
        // SWEEP 2 OVERRIDE: Unlock Multi-Placement universally
        params.dec_strat     = DecodingStrategy::STATIC_GREEDY;
        params.mutation_rate = 0.25;
        params.elite_count   = 2;
        params.noise_scale   = 0.75;
    } else {
        params.init_strat    = InitStrategy::DENSITY_REQUEST_SORT;
        params.cx_strat      = CrossoverStrategy::ACTIVE_SET_CROSSOVER;
        params.mut_strat     = MutationStrategy::PROMOTE_UNPLACED;
        params.ls_strat      = LocalSearchStrategy::NONE;
        params.dec_strat     = DecodingStrategy::STATIC_GREEDY;
        params.elite_count   = 4;
        params.noise_scale   = 0.50;
    }

    if (candidates_count == 0) {
        params.active_k = 0;
        return params;
    }

    double adaptive_k_ratio = min(0.40, max(0.05, capacity_ratio * 1.5));
    params.active_k = max(2, static_cast<int>(candidates_count * adaptive_k_ratio));

    return params;
}

VideoOrder SequenceSolver::init_exact_greedy(const GreedyDecoder& decoder) {
    VideoOrder order;
    order.videos = decoder.priority_order();
    return order;
}

VideoOrder SequenceSolver::init_pure_random(const GreedyDecoder& decoder, const function<double(void)>& rnd) {
    VideoOrder order;
    order.videos = decoder.priority_order();
    for (int i = static_cast<int>(order.videos.size()) - 1; i > 0; --i) {
        int j = static_cast<int>(rnd() * (i + 1)) % (i + 1);
        swap(order.videos[i], order.videos[j]);
    }
    return order;
}

// Log-normal multiplicative perturbation preserves relative ordering of extreme Zipfian outliers
VideoOrder SequenceSolver::init_noisy_mult(const GreedyDecoder& decoder, const function<double(void)>& rnd) {
    VideoOrder order;
    order.videos = decoder.priority_order();
    const vector<double>& priority = decoder.video_priorities();

    vector<pair<double, int>> keys;
    keys.reserve(order.videos.size());
    for (int v : order.videos) {
        keys.push_back({priority[v] * exp(2.0 * rnd() - 1.0), v});
    }
    sort(keys.begin(), keys.end(), [](const auto& a, const auto& b) { return a.first > b.first; });
    for (size_t i = 0; i < keys.size(); ++i) order.videos[i] = keys[i].second;
    return order;
}

// Box-Muller Gaussian additive perturbation shuffles medium-priority videos more aggressively
VideoOrder SequenceSolver::init_noisy_add(const GreedyDecoder& decoder, double noise_scale, const function<double(void)>& rnd) {
    VideoOrder order;
    order.videos = decoder.priority_order();
    const vector<double>& priority = decoder.video_priorities();

    double sum = 0.0, sq_sum = 0.0;
    int count = 0;
    for (int v : order.videos) {
        if (priority[v] > 0) {
            sum += priority[v];
            sq_sum += priority[v] * priority[v];
            count++;
        }
    }
    double mean = count > 0 ? sum / count : 1.0;
    double stddev = count > 1 ? sqrt(max(0.0, (sq_sum / count) - (mean * mean))) : 1.0;

    vector<pair<double, int>> keys;
    keys.reserve(order.videos.size());
    const double PI = acos(-1.0);

    for (int v : order.videos) {
        double u1 = max(1e-9, rnd());
        double u2 = rnd();
        double z0 = sqrt(-2.0 * log(u1)) * cos(2.0 * PI * u2);
        keys.push_back({priority[v] + (z0 * stddev * noise_scale), v});
    }
    sort(keys.begin(), keys.end(), [](const auto& a, const auto& b) { return a.first > b.first; });
    for (size_t i = 0; i < keys.size(); ++i) order.videos[i] = keys[i].second;
    return order;
}

VideoOrder SequenceSolver::init_density_sort(const GreedyDecoder& decoder, bool exact, const function<double(void)>& rnd) {
    VideoOrder order;
    order.videos = decoder.candidate_videos();
    const vector<double>& priority = decoder.video_priorities();

    vector<pair<double, int>> keys;
    keys.reserve(order.videos.size());
    for (int v : order.videos) {
        double val = exact ? priority[v] : priority[v] * exp(2.0 * rnd() - 1.0);
        keys.push_back({val, v});
    }
    sort(keys.begin(), keys.end(), [](const auto& a, const auto& b) { return a.first > b.first; });
    for (size_t i = 0; i < keys.size(); ++i) order.videos[i] = keys[i].second;
    return order;
}

VideoOrder SequenceSolver::init_active_front_shuffle(const GreedyDecoder& decoder, int active_k, const function<double(void)>& rnd) {
    VideoOrder order;
    order.videos = decoder.priority_order();
    int n = static_cast<int>(order.videos.size());
    if (n < 2) return order;

    int k = min(n, max(1, active_k));
    for (int i = k - 1; i > 0; --i) {
        int j = static_cast<int>(rnd() * (i + 1)) % (i + 1);
        swap(order.videos[i], order.videos[j]);
    }
    return order;
}

// Portfolio initialization for Path A: seeds 1 exact greedy + 4 structured diversity tiers
VideoOrder SequenceSolver::generate_initial_individual(
    InitStrategy strategy,
    const GreedyDecoder& decoder,
    int individual_index,
    int population_size,
    int active_prefix_k,
    double noise_scale,
    const function<double(void)>& rnd
) {
    switch (strategy) {
        case InitStrategy::HYBRID_COMPOSITE: {
            if (individual_index == 0) return init_exact_greedy(decoder);
            double ratio = static_cast<double>(individual_index) / population_size;
            if (ratio <= 0.50) return init_noisy_mult(decoder, rnd);
            if (ratio <= 0.75) return init_noisy_add(decoder, noise_scale, rnd);
            if (ratio <= 0.90) return init_active_front_shuffle(decoder, active_prefix_k, rnd);
            return init_pure_random(decoder, rnd);
        }
        case InitStrategy::DENSITY_REQUEST_SORT:
        default:
            return init_density_sort(decoder, individual_index == 0, rnd);
    }
}

// Path A Crossover: Order Crossover (OX1) restricted to the active prefix window [0, active_k)
VideoOrder SequenceSolver::active_prefix_ox1(
    const VideoOrder& p1,
    const VideoOrder& p2,
    int nbr_videos,
    int active_k,
    const function<double(void)>& rnd
) {
    int n = static_cast<int>(p1.videos.size());
    VideoOrder child;
    child.videos.assign(n, -1);
    if (n == 0) return child;

    int window = min(n, max(1, active_k));
    int a = static_cast<int>(rnd() * window) % window;
    int b = static_cast<int>(rnd() * window) % window;
    if (a > b) swap(a, b);

    vector<char> in_segment(nbr_videos, 0);
    for (int i = a; i <= b; ++i) {
        child.videos[i] = p1.videos[i];
        in_segment[p1.videos[i]] = 1;
    }

    int pos = 0;
    for (int v : p2.videos) {
        if (in_segment[v]) continue;
        if (pos == a) pos = b + 1;
        child.videos[pos++] = v;
    }
    return child;
}

// Path B Crossover: Preserves consensus active videos from both parents and samples disputed slots
VideoOrder SequenceSolver::active_set_crossover(
    const VideoOrder& p1,
    const VideoOrder& p2,
    int nbr_videos,
    int active_k,
    const function<double(void)>& rnd
) {
    int n = static_cast<int>(p1.videos.size());
    VideoOrder child;
    child.videos.reserve(n);
    if (n == 0) return child;

    int window = min(n, max(2, active_k));
    vector<char> in_p1(nbr_videos, 0), in_p2(nbr_videos, 0);
    for (int i = 0; i < window; ++i) {
        in_p1[p1.videos[i]] = 1;
        in_p2[p2.videos[i]] = 1;
    }

    vector<pair<int, int>> prefix;
    vector<pair<int, int>> pool;
    prefix.reserve(window);
    for (int i = 0; i < window; ++i) {
        if (in_p2[p1.videos[i]]) prefix.push_back({i, p1.videos[i]});
        else pool.push_back({i, p1.videos[i]});

        if (!in_p1[p2.videos[i]]) pool.push_back({i, p2.videos[i]});
    }

    int need = window - static_cast<int>(prefix.size());
    int pool_size = static_cast<int>(pool.size());
    for (int i = 0; i < need; ++i) {
        int remaining = pool_size - i;
        int j = i + static_cast<int>(rnd() * remaining) % remaining;
        swap(pool[i], pool[j]);
        prefix.push_back(pool[i]);
    }
    sort(prefix.begin(), prefix.end());

    vector<char> used(nbr_videos, 0);
    for (const auto& [pos, v] : prefix) {
        child.videos.push_back(v);
        used[v] = 1;
    }

    for (int v : p2.videos) {
        if (!used[v]) child.videos.push_back(v);
    }
    return child;
}

VideoOrder SequenceSolver::execute_crossover(
    CrossoverStrategy strategy,
    const VideoOrder& p1,
    const VideoOrder& p2,
    int nbr_videos,
    int active_k,
    const function<double(void)>& rnd
) {
    switch (strategy) {
        case CrossoverStrategy::ACTIVE_PREFIX_OX1:
            return active_prefix_ox1(p1, p2, nbr_videos, active_k, rnd);
        case CrossoverStrategy::ACTIVE_SET_CROSSOVER:
        default:
            return active_set_crossover(p1, p2, nbr_videos, active_k, rnd);
    }
}

VideoOrder SequenceSolver::swap_mutation(const VideoOrder& original, const function<double(void)>& rnd) {
    VideoOrder mutated = original;
    int n = static_cast<int>(mutated.videos.size());
    if (n < 2) return mutated;

    int i = static_cast<int>(rnd() * n) % n;
    int j = static_cast<int>(rnd() * n) % n;
    swap(mutated.videos[i], mutated.videos[j]);
    return mutated;
}

// Winning Mutation: Rotates a high-priority unplaced video ahead of an already-placed video
VideoOrder SequenceSolver::promote_unplaced(
    const VideoOrder& original,
    const vector<char>& placed,
    const function<double(void)>& rnd
) {
    VideoOrder mutated = original;
    int n = static_cast<int>(mutated.videos.size());
    if (n < 2) return mutated;

    vector<int> placed_pos, unplaced_pos;
    for (int i = 0; i < n; ++i) {
        if (placed[mutated.videos[i]]) placed_pos.push_back(i);
        else unplaced_pos.push_back(i);
    }
    if (placed_pos.empty() || unplaced_pos.empty()) return swap_mutation(original, rnd);

    int band = static_cast<int>(min(unplaced_pos.size(), placed_pos.size()));
    int from = unplaced_pos[static_cast<int>(rnd() * band) % band];

    int before = static_cast<int>(lower_bound(placed_pos.begin(), placed_pos.end(), from) - placed_pos.begin());
    int to = before > 0 ? placed_pos[static_cast<int>(rnd() * before) % before] : 0;

    rotate(mutated.videos.begin() + to, mutated.videos.begin() + from, mutated.videos.begin() + from + 1);
    return mutated;
}

VideoOrder SequenceSolver::execute_mutation(
    MutationStrategy,
    const VideoOrder& original,
    int,
    const function<double(void)>& rnd,
    const vector<char>* placed
) {
    if (!placed || placed->empty()) return swap_mutation(original, rnd);
    return promote_unplaced(original, *placed, rnd);
}