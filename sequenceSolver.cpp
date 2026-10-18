#include "Sequence_Solver.h"
#include "global.h"
#include <chrono>
#include <cmath>
#include <algorithm>
#include <set>
#include <stdexcept>

using namespace std;

Solution SequenceSolver::run(const ProblemData& problem, double best_known_value, const string& instance_name) {
    return run_benchmark(problem, best_known_value, instance_name);
}

Solution SequenceSolver::run_benchmark(
    const ProblemData& problem,
    double best_known_value,
    const string& instance_name,
    float pop_size_override,
    float cx_frac_override,
    float mut_rate_override,
    float k_mult_override,
    float slack_ratio_override,
    int elite_count_override,
    float noise_scale_override,
    int stall_max_override,
    unsigned int seed,
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

    DynamicHyperparameters dyn_params = calculate_dynamic_hyperparameters(
        problem, decoder, pop_size_override, cx_frac_override, mut_rate_override,
        k_mult_override, slack_ratio_override, elite_count_override, noise_scale_override, stall_max_override
    );

    Perm_Engine ga;
    ga.set_seed(seed);
    ga.multi_threading = false;
    ga.problem_mode = EA::GA_MODE::SOGA;
    ga.generation_max = 10000;
    ga.verbose = false;
    ga.tol_stall_best = 1e-6;
    ga.tol_stall_average = 1e-6;

    ga.population = dyn_params.population_size;
    ga.crossover_fraction = dyn_params.crossover_fraction;
    ga.mutation_rate = dyn_params.mutation_rate;
    ga.elite_count = dyn_params.elite_count;
    ga.best_stall_max = dyn_params.stall_max;
    ga.average_stall_max = dyn_params.stall_max;

    const int active_k = dyn_params.active_k;
    const int pop_size = ga.population;
    const double noise_scale = dyn_params.noise_scale;
    const double soft_slack = dyn_params.soft_slack_ratio;
    const InitStrategy init_strat = dyn_params.init_strat;
    const CrossoverStrategy cx_strat = dyn_params.cx_strat;
    const MutationStrategy mut_strat = dyn_params.mut_strat;
    const DecodingStrategy dec_strat = dyn_params.dec_strat;

    vector<double> targets = generate_targets(best_known_value, 50);
    set<size_t> generation_phenotypes;

    int eval_count = 0;
    int ind_counter = 0;
    int current_gen = 0;
    double best_fitness = -1.0;
    size_t current_target_idx = 0;
    vector<pair<int, double>> history;
    VideoOrder global_best_order;

    ga.init_genes = [&](VideoOrder& s, const function<double(void)>& rnd) {
        if (chrono::duration<double>(chrono::high_resolution_clock::now() - start_time).count() >= max_run_time_seconds) {
            throw runtime_error("TIMEOUT");
        }
        s = generate_initial_individual(init_strat, decoder, ind_counter++, pop_size, active_k, noise_scale, rnd);
    };

    ga.eval_solution = [&](const VideoOrder& s, double& score) -> bool {
        if (chrono::duration<double>(chrono::high_resolution_clock::now() - start_time).count() >= max_run_time_seconds) {
            throw runtime_error("TIMEOUT");
        }

        DecodeParams params;
        params.strategy = dec_strat;
        params.soft_slack_ratio = soft_slack;

        Solution sol;
        score = static_cast<double>(decoder.decode(s.videos, &sol, params));

        size_t p_hash = sol.used_capacity.size();
        for (int cap : sol.used_capacity) {
            p_hash ^= cap + 0x9e3779b9 + (p_hash << 6) + (p_hash >> 2);
        }
        if (generation_phenotypes.count(p_hash)) {
            score *= 0.90;
        } else {
            generation_phenotypes.insert(p_hash);
        }

        if (score > best_fitness) {
            best_fitness = score;
            global_best_order = s;
        }

        eval_count++;
        while (current_target_idx < targets.size() && best_fitness >= targets[current_target_idx]) {
            history.push_back({eval_count, targets[current_target_idx]});
            current_target_idx++;
        }
        return true;
    };

    ga.calculate_SO_total_fitness = [](const Perm_Engine::thisChromosomeType& c) -> double {
        return -c.middle_costs;
    };

    ga.mutate = [&](const VideoOrder& s, const function<double(void)>& rnd, double) -> VideoOrder {
        // Populate decoder.last_placed() via fast static greedy decode before promoting unplaced videos
        decoder.decode(s.videos);
        return execute_mutation(mut_strat, s, active_k, rnd, &decoder.last_placed());
    };

    ga.crossover = [&](const VideoOrder& p1, const VideoOrder& p2, const function<double(void)>& rnd) -> VideoOrder {
        return execute_crossover(cx_strat, p1, p2, problem.nbr_videos, active_k, rnd);
    };

    ga.SO_report_generation = [&](int gen, const Perm_Engine::thisGenerationType&, const VideoOrder&) {
        current_gen = gen;
        generation_phenotypes.clear();

        if (chrono::duration<double>(chrono::high_resolution_clock::now() - start_time).count() >= max_run_time_seconds) {
            ga.user_request_stop = true;
        }
    };

    try {
        ga.solve();
    } catch (const runtime_error&) {
    }

    chrono::duration<double> elapsed = chrono::high_resolution_clock::now() - start_time;

    Solution final_solution;
    final_solution.results.assign(problem.nbr_caches, vector<bool>(problem.nbr_videos, false));
    final_solution.used_capacity.assign(problem.nbr_caches, 0);

    VideoOrder best_final_order = global_best_order;
    const auto& chromosomes = ga.last_generation.chromosomes;
    if (best_final_order.videos.empty() && !chromosomes.empty() &&
        ga.last_generation.best_chromosome_index >= 0 &&
        ga.last_generation.best_chromosome_index < static_cast<int>(chromosomes.size())) {
        best_final_order = chromosomes[ga.last_generation.best_chromosome_index].genes;
    }

    if (!best_final_order.videos.empty()) {
        DecodeParams final_params;
        final_params.strategy = dec_strat;
        final_params.soft_slack_ratio = soft_slack;
        final_solution.sequence = best_final_order.videos;
        decoder.decode(final_solution.sequence, &final_solution, final_params);
    }

    generate_json_output(
        instance_name, history, eval_count, current_gen,
        best_fitness, elapsed.count(), pop_size, "benchmark_results"
    );

    return final_solution;
}

// region Initialisation Strategies

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
    for (size_t i = 0; i < keys.size(); ++i) {
        order.videos[i] = keys[i].second;
    }
    return order;
}

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
        // Box-Muller transform for Gaussian noise
        double u1 = max(1e-9, rnd());
        double u2 = rnd();
        double z0 = sqrt(-2.0 * log(u1)) * cos(2.0 * PI * u2);
        keys.push_back({priority[v] + (z0 * stddev * noise_scale), v});
    }

    sort(keys.begin(), keys.end(), [](const auto& a, const auto& b) { return a.first > b.first; });
    for (size_t i = 0; i < keys.size(); ++i) {
        order.videos[i] = keys[i].second;
    }
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
    for (size_t i = 0; i < keys.size(); ++i) {
        order.videos[i] = keys[i].second;
    }
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

// endregion

// region Crossover Strategies

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

    // Keep intersection of active prefixes; sample remaining slots from symmetric difference
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

// endregion

// region Mutation Strategies

VideoOrder SequenceSolver::swap_mutation(const VideoOrder& original, const function<double(void)>& rnd) {
    VideoOrder mutated = original;
    int n = static_cast<int>(mutated.videos.size());
    if (n < 2) return mutated;

    int i = static_cast<int>(rnd() * n) % n;
    int j = static_cast<int>(rnd() * n) % n;
    swap(mutated.videos[i], mutated.videos[j]);
    return mutated;
}

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

// endregion

// region Hyper-Heuristic Router & Dynamic Hyperparameters

DynamicHyperparameters SequenceSolver::calculate_dynamic_hyperparameters(
    const ProblemData& problem,
    const GreedyDecoder& decoder,
    float pop_size_override,
    float cx_frac_override,
    float mut_rate_override,
    float k_mult_override,
    float slack_ratio_override,
    int elite_count_override,
    float noise_scale_override,
    int stall_max_override
) {
    DynamicHyperparameters params;
    int candidates_count = static_cast<int>(decoder.candidate_videos().size());

    long long possible_connections = static_cast<long long>(problem.nbr_endpoints) * problem.nbr_caches;
    long long actual_connections = 0;
    for (int e = 0; e < problem.nbr_endpoints; ++e) {
        for (int c = 0; c < problem.nbr_caches; ++c) {
            if (problem.network[e][c] > 0) {
                actual_connections++;
            }
        }
    }

    double network_density = possible_connections > 0
        ? static_cast<double>(actual_connections) / possible_connections
        : 0.0;

    long long matrix_complexity = static_cast<long long>(problem.nbr_videos) * problem.nbr_endpoints * problem.nbr_caches;

    // Route topology to Path A (Large/Dense) or Path B (Small/Sparse)
    if (matrix_complexity > 50000000 || network_density > 0.4) {
        params.init_strat = InitStrategy::HYBRID_COMPOSITE;
        params.cx_strat   = CrossoverStrategy::ACTIVE_PREFIX_OX1;
        params.mut_strat  = MutationStrategy::PROMOTE_UNPLACED;
        params.ls_strat   = LocalSearchStrategy::NONE;
        params.dec_strat  = DecodingStrategy::SOFT_SLACK_REPAIRED;
    } else {
        params.init_strat = InitStrategy::DENSITY_REQUEST_SORT;
        params.cx_strat   = CrossoverStrategy::ACTIVE_SET_CROSSOVER;
        params.mut_strat  = MutationStrategy::PROMOTE_UNPLACED;
        params.ls_strat   = LocalSearchStrategy::NONE;
        params.dec_strat  = DecodingStrategy::STATIC_GREEDY;
    }

    if (candidates_count == 0) {
        params.population_size = 30;
        params.crossover_fraction = 0.5;
        params.mutation_rate = 0.2;
        params.active_k = 0;
        return params;
    }

    long long total_cache_cap = static_cast<long long>(problem.nbr_caches) * problem.cap_cache;
    long long total_video_size = 0;
    for (int s : problem.videos) {
        total_video_size += s;
    }

    double capacity_ratio = total_video_size > 0
        ? static_cast<double>(total_cache_cap) / total_video_size
        : 0.0;

    if (matrix_complexity > 50000000) {
        params.population_size = 20;
    } else {
        params.population_size = max(30, min(120, candidates_count / 50));
    }

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

    double k_mult = (k_mult_override > 0.0f) ? static_cast<double>(k_mult_override) : 1.5;
    double adaptive_k_ratio = min(0.40, max(0.05, capacity_ratio * k_mult));
    params.active_k = max(2, static_cast<int>(candidates_count * adaptive_k_ratio));

    // Set baseline defaults for new hyperparameters
    params.soft_slack_ratio = 1.15;
    params.elite_count = 2;
    params.noise_scale = 0.5;
    params.stall_max = 1000;

    // Apply Overrides if > 0 (or != -1)
    if (pop_size_override > 0.0f) params.population_size = static_cast<int>(pop_size_override);
    if (cx_frac_override > 0.0f)  params.crossover_fraction = static_cast<double>(cx_frac_override);
    if (mut_rate_override > 0.0f) params.mutation_rate = static_cast<double>(mut_rate_override);
    if (slack_ratio_override > 0.0f) params.soft_slack_ratio = static_cast<double>(slack_ratio_override);
    if (elite_count_override > -1) params.elite_count = elite_count_override;
    if (noise_scale_override > 0.0f) params.noise_scale = static_cast<double>(noise_scale_override);
    if (stall_max_override > -1) params.stall_max = stall_max_override;

    return params;
}

// endregion