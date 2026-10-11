#include "Sequence_Solver.h"
#include "global.h"
#include <chrono>
#include <cmath>
#include <iostream>
#include <algorithm>
#include <set>

using namespace std;

// Main execution loop for the permutation GA solver
Solution SequenceSolver::run(const ProblemData& problem, double best_known_value, const string& instance_name) {
    return run_benchmark(problem, best_known_value, instance_name, InitStrategy::BASELINE_EXACT_NOISY,CrossoverStrategy::BASELINE_OX1 ,MutationStrategy::BASELINE_SWAP, LocalSearchStrategy::NONE,DecodingStrategy::STATIC_GREEDY);
}

Solution SequenceSolver::run_benchmark(
    const ProblemData& problem,
    double best_known_value,
    const string& instance_name,
    InitStrategy init_strat,
    CrossoverStrategy cx_strat,
    MutationStrategy mut_strat,
    LocalSearchStrategy ls_strat,
    DecodingStrategy dec_strat,
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
    DynamicHyperparameters dyn_params = calculate_dynamic_hyperparameters(problem, decoder);

    Perm_Engine ga;
    ga.set_seed(seed);
    ga.multi_threading = false;
    ga.problem_mode = EA::GA_MODE::SOGA;
    ga.generation_max = 10000;
    ga.elite_count = 2;
    ga.verbose = false;

    ga.best_stall_max = 1000;
    ga.average_stall_max = 1000;
    ga.tol_stall_best = 1e-6;
    ga.tol_stall_average = 1e-6;

    ga.population = dyn_params.population_size;
    ga.crossover_fraction = dyn_params.crossover_fraction;
    ga.mutation_rate = dyn_params.mutation_rate;
    int active_k = dyn_params.active_k;


    vector<double> targets = generate_targets(best_known_value, 50);

    std::set<size_t> generation_phenotypes;

    int eval_count = 0;
    double best_fitness = -1.0;
    size_t current_target_idx = 0;
    int current_gen = 0;
    vector<pair<int, double>> history;

    int ind_counter = 0;
    int pop_size = ga.population;

    VideoOrder global_best_order;

    vector<VideoOrder> initial_pop_store;
    initial_pop_store.reserve(pop_size);

    double gen_0_sum = 0.0;
    double gen_0_best = -1.0;

    long long total_crossovers = 0;
    long long successful_crossovers = 0;
    long long neutral_crossovers = 0;

    long long total_mutations = 0;
    long long successful_mutations = 0;
    long long neutral_mutations = 0;

    long long total_ls_apps = 0;
    long long successful_ls_apps = 0;
    double total_ls_fitness_gain = 0.0;

    long long total_eval_time_us = 0;

    long long deleterious_crossovers = 0;
    double total_cx_deleterious_drop = 0.0;

    long long deleterious_mutations = 0;
    double total_mut_deleterious_drop = 0.0;

    long long total_ls_cap_hits = 0;

    ga.init_genes = [this, &decoder, &ind_counter, pop_size, active_k, init_strat, dec_strat, &initial_pop_store,&start_time,&max_run_time_seconds](VideoOrder& s, const function<double(void)>& rnd) {
        auto now = chrono::high_resolution_clock::now();
        if (chrono::duration<double>(now - start_time).count() >= max_run_time_seconds) {
            throw std::runtime_error("TIMEOUT");
        }
        s = generate_initial_individual(init_strat, decoder, ind_counter, pop_size, active_k, rnd);

        if (dec_strat == DecodingStrategy::CO_EVOLVED_THRESHOLDS) {
            s.thresholds = {rnd() * 0.3, rnd() * 0.2}; // Random init inside [0.0, 0.3] and [0.0, 0.2]
        }
        initial_pop_store.push_back(s);
        ind_counter++;
    };

    ga.eval_solution = [&](const VideoOrder& s, double& score) -> bool {
        auto t0 = chrono::high_resolution_clock::now();
        if (chrono::duration<double>(t0 - start_time).count() >= max_run_time_seconds) {
            throw std::runtime_error("TIMEOUT");
        }
        DecodeParams params;
        params.strategy = dec_strat;

        if (dec_strat == DecodingStrategy::FIXED_THRESHOLD_PROFILE) {
            params.tau_density = 0.15;
            params.tau_reserve = 0.10;
        } else if (dec_strat == DecodingStrategy::CO_EVOLVED_THRESHOLDS) {
            params.tau_density = s.thresholds[0];
            params.tau_reserve = s.thresholds[1];
        }

        Solution sol;
        score = static_cast<double>(decoder.decode(s.videos, &sol, params));

        size_t p_hash = sol.used_capacity.size();
        for (int cap : sol.used_capacity) {
            p_hash ^= cap + 0x9e3779b9 + (p_hash << 6) + (p_hash >> 2);
        }

        if (generation_phenotypes.count(p_hash)) {
            score = score * 0.90;
        } else {
            generation_phenotypes.insert(p_hash);
        }

        auto t1 = chrono::high_resolution_clock::now();
        total_eval_time_us += chrono::duration_cast<chrono::microseconds>(t1 - t0).count();

        if (eval_count < pop_size) {
            gen_0_sum += score;
            if (score > gen_0_best) gen_0_best = score;
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

    ga.mutate = [this, active_k, mut_strat, dec_strat, &decoder, &total_mutations, &successful_mutations, &neutral_mutations,&deleterious_mutations,&total_mut_deleterious_drop](const VideoOrder& s, const function<double(void)>& rnd, double shrink_scale) -> VideoOrder {
        DecodeParams params;
        params.strategy = dec_strat;
        if (dec_strat == DecodingStrategy::FIXED_THRESHOLD_PROFILE) {
            params.tau_density = 0.15; params.tau_reserve = 0.10;
        } else if (dec_strat == DecodingStrategy::CO_EVOLVED_THRESHOLDS) {
            params.tau_density = s.thresholds[0]; params.tau_reserve = s.thresholds[1];
        }
        // Decoded first: PROMOTE_UNPLACED reads which videos the decoder placed
        long long orig_score = decoder.decode(s.videos, nullptr, params);
        VideoOrder mutated = execute_mutation(mut_strat, s, active_k, rnd, &decoder.last_placed(),&decoder.video_priorities());

        // Co-evolve thresholds via Gaussian perturbation
        if (dec_strat == DecodingStrategy::CO_EVOLVED_THRESHOLDS) {
            mutated.thresholds = s.thresholds;
            mutated.thresholds[0] = max(0.0, min(0.5, mutated.thresholds[0] + (rnd() - 0.5) * 0.05));
            mutated.thresholds[1] = max(0.0, min(0.3, mutated.thresholds[1] + (rnd() - 0.5) * 0.05));
            params.tau_density = mutated.thresholds[0];
            params.tau_reserve = mutated.thresholds[1];
        }

        long long mut_score = decoder.decode(mutated.videos, nullptr, params);

        total_mutations++;
        if (mut_score > orig_score) {
            successful_mutations++;
        } else if (mut_score == orig_score) {
            neutral_mutations++;
        } else {
            deleterious_mutations++;
            total_mut_deleterious_drop += (orig_score - mut_score);
        }

        return mutated;
    };

    ga.crossover = [this, &problem, active_k, cx_strat, dec_strat, &decoder, &total_crossovers, &successful_crossovers, &neutral_crossovers,&deleterious_crossovers,&total_cx_deleterious_drop](const VideoOrder& p1, const VideoOrder& p2, const function<double(void)>& rnd) -> VideoOrder {

        // 1. Créer l'enfant d'abord
        VideoOrder child = execute_crossover(cx_strat, p1, p2, problem.nbr_videos, active_k, rnd);

        // 2. Mélanger les seuils pour l'enfant
        if (dec_strat == DecodingStrategy::CO_EVOLVED_THRESHOLDS) {
            double alpha = rnd();
            child.thresholds.resize(2);
            child.thresholds[0] = alpha * p1.thresholds[0] + (1.0 - alpha) * p2.thresholds[0];
            child.thresholds[1] = alpha * p1.thresholds[1] + (1.0 - alpha) * p2.thresholds[1];
        }

        // 3. Préparer les paramètres de décodage spécifiques à chaque individu
        DecodeParams params_p1, params_p2, params_child;
        params_p1.strategy = params_p2.strategy = params_child.strategy = dec_strat;

        if (dec_strat == DecodingStrategy::FIXED_THRESHOLD_PROFILE) {
            params_p1.tau_density = params_p2.tau_density = params_child.tau_density = 0.15;
            params_p1.tau_reserve = params_p2.tau_reserve = params_child.tau_reserve = 0.10;
        } else if (dec_strat == DecodingStrategy::CO_EVOLVED_THRESHOLDS) {
            params_p1.tau_density = p1.thresholds[0]; params_p1.tau_reserve = p1.thresholds[1];
            params_p2.tau_density = p2.thresholds[0]; params_p2.tau_reserve = p2.thresholds[1];
            params_child.tau_density = child.thresholds[0]; params_child.tau_reserve = child.thresholds[1];
        }

        // 4. Décoder avec les bons paramètres
        long long p1_score = decoder.decode(p1.videos, nullptr, params_p1);
        long long p2_score = decoder.decode(p2.videos, nullptr, params_p2);
        long long child_score = decoder.decode(child.videos, nullptr, params_child);
        long long max_parent_score = max(p1_score, p2_score);

        // 5. Mettre à jour les métriques
        total_crossovers++;
        if (child_score > max_parent_score) {
            successful_crossovers++;
        } else if (child_score == p1_score || child_score == p2_score) {
            neutral_crossovers++;
        } else {
            deleterious_crossovers++;
            total_cx_deleterious_drop += (max_parent_score - child_score);
        }

        return child;
    };

    ga.SO_report_generation = [this, ls_strat, active_k, &decoder, &eval_count, &total_ls_apps, &successful_ls_apps, &total_ls_fitness_gain, &current_gen, &dec_strat,&total_ls_cap_hits, &generation_phenotypes,&start_time,&max_run_time_seconds,&ga](
        int gen,
        const Perm_Engine::thisGenerationType& gen_obj,
        const VideoOrder& best
    ) {
        current_gen = gen;
        generation_phenotypes.clear();

        auto current_time = chrono::high_resolution_clock::now();
        chrono::duration<double> elapsed = current_time - start_time;
        if (elapsed.count() >= max_run_time_seconds) {
            ga.user_request_stop = true;
        }

        if (ls_strat != LocalSearchStrategy::NONE && !gen_obj.chromosomes.empty() && !ga.user_request_stop) {
            total_ls_apps++;
            long long extra_evals = 0;
            double gain = 0.0;
            int ls_k = min(active_k, 15);

            auto& non_const_gen = const_cast<Perm_Engine::thisGenerationType&>(gen_obj);
            int best_idx = (non_const_gen.best_chromosome_index >= 0 && non_const_gen.best_chromosome_index < static_cast<int>(non_const_gen.chromosomes.size())) ? non_const_gen.best_chromosome_index : 0;

            // Generate correct thresholds for Local Search
            DecodeParams params;
            params.strategy = dec_strat;
            if (dec_strat == DecodingStrategy::FIXED_THRESHOLD_PROFILE) {
                params.tau_density = 0.15; params.tau_reserve = 0.10;
            } else if (dec_strat == DecodingStrategy::CO_EVOLVED_THRESHOLDS && non_const_gen.chromosomes[best_idx].genes.thresholds.size() >= 2) {
                params.tau_density = non_const_gen.chromosomes[best_idx].genes.thresholds[0];
                params.tau_reserve = non_const_gen.chromosomes[best_idx].genes.thresholds[1];
            }

            bool hit_cap = false;
            VideoOrder refined = execute_local_search(
                ls_strat, non_const_gen.chromosomes[best_idx].genes, decoder, params, ls_k, extra_evals, gain, hit_cap
            );
            eval_count += extra_evals;
            if (hit_cap) total_ls_cap_hits++;

            if (gain > 0.0) {
                successful_ls_apps++;
                total_ls_fitness_gain += gain;
                non_const_gen.chromosomes[best_idx].genes = refined;
                non_const_gen.chromosomes[best_idx].middle_costs = -static_cast<double>(decoder.decode(refined.videos));
            }
        }
    };

    try {
        ga.solve();
    } catch (const std::runtime_error& e) {

    }

    chrono::duration<double> elapsed = chrono::high_resolution_clock::now() - start_time;

   // Calculate final sequence states
    double d0 = compute_population_diversity(initial_pop_store, active_k);
    double gen_0_mean = pop_size > 0 ? gen_0_sum / pop_size : 0.0;

    Solution final_solution;
    final_solution.results.assign(problem.nbr_caches, vector<bool>(problem.nbr_videos, false));
    final_solution.used_capacity.assign(problem.nbr_caches, 0);

    VideoOrder best_final_order = global_best_order;

    const auto& chromosomes = ga.last_generation.chromosomes;
    if (!chromosomes.empty() && ga.last_generation.best_chromosome_index >= 0 &&
        ga.last_generation.best_chromosome_index < static_cast<int>(chromosomes.size())) {
        best_final_order = chromosomes[ga.last_generation.best_chromosome_index].genes;
        }

    if (!best_final_order.videos.empty()) {
        DecodeParams final_params;
        final_params.strategy = dec_strat;
        if (dec_strat == DecodingStrategy::FIXED_THRESHOLD_PROFILE) {
            final_params.tau_density = 0.15;
            final_params.tau_reserve = 0.10;
        } else if (dec_strat == DecodingStrategy::CO_EVOLVED_THRESHOLDS && best_final_order.thresholds.size() >= 2) {
            final_params.tau_density = best_final_order.thresholds[0];
            final_params.tau_reserve = best_final_order.thresholds[1];
        }

        final_solution.sequence = best_final_order.videos;
        decoder.decode(final_solution.sequence, &final_solution, final_params);
    }

    export_benchmark_metrics(
        instance_name, history, eval_count, current_gen, best_fitness, elapsed.count(), pop_size,
        init_strat, cx_strat, mut_strat, ls_strat, dec_strat,
        active_k, gen_0_best, gen_0_mean, d0,
        total_crossovers, successful_crossovers, neutral_crossovers, deleterious_crossovers, total_cx_deleterious_drop,
        total_mutations, successful_mutations, neutral_mutations, deleterious_mutations, total_mut_deleterious_drop,
        total_ls_apps, successful_ls_apps, total_ls_cap_hits, total_ls_fitness_gain,
        total_eval_time_us,
        initial_pop_store, decoder, problem, final_solution
    );

    return final_solution;
}

// Generate prioritized genes, introducing noise if non-exact
VideoOrder SequenceSolver::init_genes(const GreedyDecoder& decoder, bool exact, const function<double(void)>& rnd) {
    VideoOrder order;
    order.videos = decoder.priority_order();

    if (exact) return order;

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


// region Initialisation approaches
// Exact Greedy Order
VideoOrder SequenceSolver::init_exact_greedy(const GreedyDecoder& decoder) {
    VideoOrder order;
    order.videos = decoder.priority_order();
    return order;
}

// Pure Uniform Random Order
VideoOrder SequenceSolver::init_pure_random(const GreedyDecoder& decoder, const function<double(void)>& rnd) {
    VideoOrder order;
    order.videos = decoder.priority_order();

    // Fisher-Yates Uniform Shuffle
    for (int i = static_cast<int>(order.videos.size()) - 1; i > 0; --i) {
        int j = static_cast<int>(rnd() * (i + 1)) % (i + 1);
        swap(order.videos[i], order.videos[j]);
    }
    return order;
}

// Multiplicative Log-Normal Noise
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

// Additive Gaussian Noise
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

    const double PI = std::acos(-1.0); // Standard portable Pi

    for (int v : order.videos) {
        double u1 = max(1e-9, rnd());
        double u2 = rnd();
        double z0 = sqrt(-2.0 * log(u1)) * cos(2.0 * PI * u2);

        double noisy_val = priority[v] + (z0 * stddev * noise_scale);
        keys.push_back({noisy_val, v});
    }

    sort(keys.begin(), keys.end(), [](const auto& a, const auto& b) { return a.first > b.first; });

    for (size_t i = 0; i < keys.size(); ++i) {
        order.videos[i] = keys[i].second;
    }
    return order;
}

// Raw Request Density Sort
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

// Active-Front Targeted Shuffle
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

// Initialization Dispatcher
VideoOrder SequenceSolver::generate_initial_individual(
    InitStrategy strategy,
    const GreedyDecoder& decoder,
    int individual_index,
    int population_size,
    int active_prefix_k,
    const function<double(void)>& rnd
) {
    switch (strategy) {
    case InitStrategy::PURE_RANDOM:
        return init_pure_random(decoder, rnd);

    case InitStrategy::NOISY_PRIORITY_ADD:
        if (individual_index == 0) return init_exact_greedy(decoder);
        return init_noisy_add(decoder, 0.5, rnd);

    case InitStrategy::DENSITY_REQUEST_SORT:
        return init_density_sort(decoder, individual_index == 0, rnd);

    case InitStrategy::ACTIVE_FRONT_SHUFFLE:
        if (individual_index == 0) return init_exact_greedy(decoder);
        return init_active_front_shuffle(decoder, active_prefix_k, rnd);

    case InitStrategy::HYBRID_COMPOSITE: {
        if (individual_index == 0) return init_exact_greedy(decoder);
        double ratio = static_cast<double>(individual_index) / population_size;
        if (ratio <= 0.50) return init_noisy_mult(decoder, rnd);
        if (ratio <= 0.75) return init_noisy_add(decoder, 0.5, rnd);
        if (ratio <= 0.90) return init_active_front_shuffle(decoder, active_prefix_k, rnd);
        return init_pure_random(decoder, rnd);
    }

    case InitStrategy::BASELINE_EXACT_NOISY:
    default:
        if (individual_index == 0) return init_exact_greedy(decoder);
        return init_noisy_mult(decoder, rnd);
    }
}

// endregion

// region Crossover Strategies

VideoOrder SequenceSolver::order_crossover(const VideoOrder& p1, const VideoOrder& p2, int nbr_videos, const function<double(void)>& rnd) {
    int n = p1.videos.size();
    VideoOrder child;
    child.videos.assign(n, -1);
    if (n == 0) return child;

    int a = static_cast<int>(rnd() * n) % n;
    int b = static_cast<int>(rnd() * n) % n;
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

VideoOrder SequenceSolver::active_prefix_ox1(const VideoOrder& p1, const VideoOrder& p2, int nbr_videos, int active_k, const function<double(void)>& rnd) {
    int n = p1.videos.size();
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

VideoOrder SequenceSolver::position_based_crossover(
    const VideoOrder& p1,
    const VideoOrder& p2,
    int nbr_videos,
    const function<double(void)>& rnd
) {
    int n = p1.videos.size();
    VideoOrder child;
    child.videos.assign(n, -1);
    if (n == 0) return child;

    // Determine tracking size based on total video count / max Video ID
    int tracking_size = max(nbr_videos, n);
    for (int v : p1.videos) if (v >= tracking_size) tracking_size = v + 1;
    for (int v : p2.videos) if (v >= tracking_size) tracking_size = v + 1;

    vector<char> selected(tracking_size, 0);
    for (int i = 0; i < n; ++i) {
        if (rnd() < 0.4) {
            child.videos[i] = p1.videos[i];
            selected[p1.videos[i]] = 1;
        }
    }

    int p2_idx = 0;
    for (int i = 0; i < n; ++i) {
        if (child.videos[i] == -1) {
            while (p2_idx < n && selected[p2.videos[p2_idx]]) {
                p2_idx++;
            }
            if (p2_idx < n) {
                child.videos[i] = p2.videos[p2_idx++];
            }
        }
    }
    return child;
}

VideoOrder SequenceSolver::partially_mapped_crossover(const VideoOrder& p1, const VideoOrder& p2, int nbr_videos, int active_k, const function<double(void)>& rnd) {
    int n = p1.videos.size();
    VideoOrder child = p1;
    if (n == 0) return child;

    int window = max(2, min(n, active_k));
    int a = static_cast<int>(rnd() * window) % window;
    int b = static_cast<int>(rnd() * window) % window;
    if (a > b) swap(a, b);

    // Indexed by video id, which can exceed the number of candidates
    vector<int> pos_in_child(nbr_videos, -1);
    for (int i = 0; i < n; ++i) pos_in_child[child.videos[i]] = i;

    for (int i = a; i <= b; ++i) {
        int val_p2 = p2.videos[i];
        if (child.videos[i] != val_p2) {
            int current_pos_of_p2 = pos_in_child[val_p2];
            int current_val_at_i = child.videos[i];

            swap(child.videos[i], child.videos[current_pos_of_p2]);
            pos_in_child[current_val_at_i] = current_pos_of_p2;
            pos_in_child[val_p2] = i;
        }
    }
    return child;
}

VideoOrder SequenceSolver::cycle_crossover(const VideoOrder& p1, const VideoOrder& p2, int nbr_videos) {
    int n = p1.videos.size();
    VideoOrder child;
    child.videos.assign(n, -1);
    if (n == 0) return child;

    // Indexed by video id, which can exceed the number of candidates
    vector<int> pos_p1(nbr_videos, -1);
    for (int i = 0; i < n; ++i) pos_p1[p1.videos[i]] = i;

    int start_idx = 0;
    int curr_idx = start_idx;
    while (child.videos[curr_idx] == -1) {
        child.videos[curr_idx] = p1.videos[curr_idx];
        int next_val = p2.videos[curr_idx];
        curr_idx = pos_p1[next_val];
    }

    for (int i = 0; i < n; ++i) {
        if (child.videos[i] == -1) {
            child.videos[i] = p2.videos[i];
        }
    }
    return child;
}

// Precedence Preservative Crossover restricted to the active prefix:
// each prefix slot takes the first unused video of a randomly chosen parent,
// so a video placed early in either parent stays early in the child.
VideoOrder SequenceSolver::active_prefix_ppx(const VideoOrder& p1, const VideoOrder& p2, int nbr_videos, int active_k, const function<double(void)>& rnd) {
    int n = p1.videos.size();
    VideoOrder child;
    child.videos.reserve(n);
    if (n == 0) return child;

    int window = min(n, max(1, active_k));
    vector<char> used(nbr_videos, 0);

    int i1 = 0, i2 = 0;
    for (int slot = 0; slot < window; ++slot) {
        bool from_p1 = rnd() < 0.5;
        const vector<int>& src = from_p1 ? p1.videos : p2.videos;
        int& idx = from_p1 ? i1 : i2;

        while (used[src[idx]]) ++idx;
        used[src[idx]] = 1;
        child.videos.push_back(src[idx]);
    }

    // Inactive tail: remaining videos in p2's order
    for (int v : p2.videos) {
        if (!used[v]) child.videos.push_back(v);
    }
    return child;
}

// Treats the active prefix as a set: videos active in both parents are always kept,
// the remaining slots are drawn from the videos active in only one parent.
VideoOrder SequenceSolver::active_set_crossover(const VideoOrder& p1, const VideoOrder& p2, int nbr_videos, int active_k, const function<double(void)>& rnd) {
    int n = p1.videos.size();
    VideoOrder child;
    child.videos.reserve(n);
    if (n == 0) return child;

    int window = min(n, max(2, active_k));

    vector<char> in_p1(nbr_videos, 0), in_p2(nbr_videos, 0);
    for (int i = 0; i < window; ++i) {
        in_p1[p1.videos[i]] = 1;
        in_p2[p2.videos[i]] = 1;
    }

    // (position in the source parent, video)
    vector<pair<int, int>> prefix;
    vector<pair<int, int>> pool;
    prefix.reserve(window);
    for (int i = 0; i < window; ++i) {
        if (in_p2[p1.videos[i]]) prefix.push_back({i, p1.videos[i]});
        else pool.push_back({i, p1.videos[i]});

        if (!in_p1[p2.videos[i]]) pool.push_back({i, p2.videos[i]});
    }

    // Partial Fisher-Yates over the videos the parents disagree on
    int need = window - static_cast<int>(prefix.size());
    int pool_size = pool.size();
    for (int i = 0; i < need; ++i) {
        int remaining = pool_size - i;
        int j = i + static_cast<int>(rnd() * remaining) % remaining;
        swap(pool[i], pool[j]);
        prefix.push_back(pool[i]);
    }

    // Each video keeps the rank it had in its source parent
    sort(prefix.begin(), prefix.end());

    vector<char> used(nbr_videos, 0);
    for (const auto& [pos, v] : prefix) {
        child.videos.push_back(v);
        used[v] = 1;
    }

    // Inactive tail: remaining videos in p2's order
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

        case CrossoverStrategy::POSITION_BASED_POS:
            return position_based_crossover(p1, p2,nbr_videos, rnd);

        case CrossoverStrategy::PARTIALLY_MAPPED_PMX:
            return partially_mapped_crossover(p1, p2, nbr_videos, active_k, rnd);

        case CrossoverStrategy::CYCLE_CROSSOVER_CX:
            return cycle_crossover(p1, p2, nbr_videos);

        case CrossoverStrategy::ACTIVE_PREFIX_PPX:
            return active_prefix_ppx(p1, p2, nbr_videos, active_k, rnd);

        case CrossoverStrategy::ACTIVE_SET_CROSSOVER:
            return active_set_crossover(p1, p2, nbr_videos, active_k, rnd);

        case CrossoverStrategy::BASELINE_OX1:
        default:
            return order_crossover(p1, p2, nbr_videos, rnd);
    }
}

// endregion

// region Mutation Strategies

VideoOrder SequenceSolver::swap_mutation(const VideoOrder& original, const function<double(void)>& rnd) {
    VideoOrder mutated = original;
    int n = mutated.videos.size();
    if (n < 2) return mutated;

    int i = static_cast<int>(rnd() * n) % n;
    int j = static_cast<int>(rnd() * n) % n;
    swap(mutated.videos[i], mutated.videos[j]);

    return mutated;
}

VideoOrder SequenceSolver::active_prefix_swap(const VideoOrder& original, int active_k, const function<double(void)>& rnd) {
    VideoOrder mutated = original;
    int n = mutated.videos.size();
    if (n < 2) return mutated;

    int window = min(n, max(1, active_k));
    int i = static_cast<int>(rnd() * window) % window;
    int j = static_cast<int>(rnd() * window) % window;
    swap(mutated.videos[i], mutated.videos[j]);

    return mutated;
}

VideoOrder SequenceSolver::active_prefix_inversion(const VideoOrder& original, int active_k, const function<double(void)>& rnd) {
    VideoOrder mutated = original;
    int n = mutated.videos.size();
    if (n < 2) return mutated;

    int window = min(n, max(1, active_k));
    int a = static_cast<int>(rnd() * window) % window;
    int b = static_cast<int>(rnd() * window) % window;
    if (a > b) swap(a, b);

    reverse(mutated.videos.begin() + a, mutated.videos.begin() + b + 1);
    return mutated;
}

VideoOrder SequenceSolver::active_prefix_insertion(const VideoOrder& original, int active_k, const function<double(void)>& rnd) {
    VideoOrder mutated = original;
    int n = mutated.videos.size();
    if (n < 2) return mutated;

    int window = min(n, max(1, active_k));
    int from = static_cast<int>(rnd() * window) % window;
    int to = static_cast<int>(rnd() * window) % window;

    if (from == to) return mutated;

    int val = mutated.videos[from];
    mutated.videos.erase(mutated.videos.begin() + from);
    mutated.videos.insert(mutated.videos.begin() + to, val);

    return mutated;
}

VideoOrder SequenceSolver::boundary_swap(const VideoOrder& original, int active_k, const function<double(void)>& rnd) {
    VideoOrder mutated = original;
    int n = mutated.videos.size();
    if (n < 2) return mutated;

    int active_limit = max(1, min(n - 1, active_k));
    int i = static_cast<int>(rnd() * active_limit) % active_limit; // Inside active prefix

    int boundary_span = max(1, min(active_k, n - active_limit));
    int j = active_limit + (static_cast<int>(rnd() * boundary_span) % boundary_span); // In boundary tail

    if (j < n) {
        swap(mutated.videos[i], mutated.videos[j]);
    }
    return mutated;
}

// Moves a video the decoder could not place in front of a video it did place.
// Uses the real placed/unplaced split instead of the active_k estimate.
VideoOrder SequenceSolver::promote_unplaced(const VideoOrder& original, const vector<char>& placed, const function<double(void)>& rnd) {
    VideoOrder mutated = original;
    int n = mutated.videos.size();
    if (n < 2) return mutated;

    vector<int> placed_pos, unplaced_pos;
    for (int i = 0; i < n; ++i) {
        if (placed[mutated.videos[i]]) placed_pos.push_back(i);
        else unplaced_pos.push_back(i);
    }
    if (placed_pos.empty() || unplaced_pos.empty()) return swap_mutation(original, rnd);

    // Only the first unplaced videos are candidates, as many as there are placed ones
    int band = min(unplaced_pos.size(), placed_pos.size());
    int from = unplaced_pos[static_cast<int>(rnd() * band) % band];

    // Target: the position of a random placed video located before it
    int before = lower_bound(placed_pos.begin(), placed_pos.end(), from) - placed_pos.begin();
    int to = before > 0 ? placed_pos[static_cast<int>(rnd() * before) % before] : 0;

    rotate(mutated.videos.begin() + to, mutated.videos.begin() + from, mutated.videos.begin() + from + 1);
    return mutated;
}

VideoOrder SequenceSolver::graph_aware_bottleneck_repair(const VideoOrder& original, const vector<char>& placed, const std::vector<double>& priorities, int active_k, const function<double(void)>& rnd) {
    VideoOrder mutated = original;
    int n = mutated.videos.size();
    if (n < 2) return mutated;

    int best_unplaced_vid = -1;
    double max_priority = -1.0;
    int original_pos = -1;

    for (int i = 0; i < n; ++i) {
        int v = mutated.videos[i];

        // priorities is now safely accessed as a reference
        if (!placed[v] && priorities[v] > max_priority) {
            max_priority = priorities[v];
            best_unplaced_vid = v;
            original_pos = i;
        }
    }

    if (best_unplaced_vid == -1) return swap_mutation(original, rnd);

    mutated.videos.erase(mutated.videos.begin() + original_pos);

    int insertion_window = max(1, active_k / 10);
    int target_pos = static_cast<int>(rnd() * insertion_window) % insertion_window;

    mutated.videos.insert(mutated.videos.begin() + target_pos, best_unplaced_vid);

    return mutated;
}

VideoOrder SequenceSolver::execute_mutation(
    MutationStrategy strategy,
    const VideoOrder& original,
    int active_k,
    const function<double(void)>& rnd,
    const vector<char>* placed,
    const std::vector<double>* priorities
) {
    switch (strategy) {
        case MutationStrategy::PROMOTE_UNPLACED:
            if (!placed) return boundary_swap(original, active_k, rnd);
            return promote_unplaced(original, *placed, rnd);

        case MutationStrategy::ACTIVE_PREFIX_SWAP:
            return active_prefix_swap(original, active_k, rnd);

        case MutationStrategy::ACTIVE_PREFIX_INVERSION:
            return active_prefix_inversion(original, active_k, rnd);

        case MutationStrategy::ACTIVE_PREFIX_INSERTION:
            return active_prefix_insertion(original, active_k, rnd);

        case MutationStrategy::BOUNDARY_SWAP:
            return boundary_swap(original, active_k, rnd);

        case MutationStrategy::GRAPH_AWARE_BOTTLENECK_REPAIR:
        // If we are missing either array, fall back safely
            if (!placed || !priorities) return boundary_swap(original, active_k, rnd);
            return graph_aware_bottleneck_repair(original, *placed, *priorities, active_k, rnd);

        case MutationStrategy::BASELINE_SWAP:
        default:
            return swap_mutation(original, rnd);
    }
}

// endregion

// region Local Search

VideoOrder SequenceSolver::ls_first_improvement_swap(VideoOrder& input, GreedyDecoder& decoder, const DecodeParams& params, int active_k, long long& evals, double& gain, bool& hit_cap) {
    VideoOrder best_order = input;
    int n = best_order.videos.size();
    int k = min(n, active_k);

    std::vector<DecodeState> states;
    states.reserve(k + 1);
    long long base_score = decoder.decode(best_order.videos, nullptr, params, nullptr, &states);

    evals = 0;
    gain = 0.0;
    const int max_evals = 300; // Hard safety cap per generation
    hit_cap=false;

    for (int i = 0; i < k && evals < max_evals; ++i) {
        for (int j = i + 1; j < k && evals < max_evals; ++j) {
            swap(best_order.videos[i], best_order.videos[j]);
            long long new_score = decoder.decode(best_order.videos, nullptr, params, &states[i], nullptr);
            evals++;

            if (new_score > base_score) {
                gain = static_cast<double>(new_score - base_score);
                return best_order; // First improvement accepted immediately
            }
            swap(best_order.videos[i], best_order.videos[j]); // Revert
        }
    }
    if (evals >= max_evals) hit_cap = true;
    return best_order;
}

VideoOrder SequenceSolver::ls_best_improvement_insert(VideoOrder& input, GreedyDecoder& decoder, const DecodeParams& params, int active_k, long long& evals, double& gain, bool& hit_cap) {
    VideoOrder best_order = input;

    int n = best_order.videos.size();
    int k = min(n, active_k);

    std::vector<DecodeState> states;
    states.reserve(k + 1);
    long long base_score = decoder.decode(best_order.videos, nullptr, params, nullptr, &states);
    long long max_score = base_score;

    evals = 0;
    gain = 0.0;
    int best_from = -1, best_to = -1;
    const int max_evals = 300;
    hit_cap=false;

    for (int from = 0; from < k && evals < max_evals; ++from) {
        VideoOrder temp = input;
        int val = temp.videos[from];
        temp.videos.erase(temp.videos.begin() + from);

        for (int to = 0; to < k && evals < max_evals; ++to) {
            if (from == to) continue;
            temp.videos.insert(temp.videos.begin() + to, val);

            // FIX: Evaluate temp.videos, and use the minimum of from/to for the state checkpoint
            int earliest_change = min(from, to);
            long long current_score = decoder.decode(temp.videos, nullptr, params, &states[earliest_change], nullptr);
            evals++;

            if (current_score > max_score) {
                max_score = current_score;
                best_from = from;
                best_to = to;
            }
            temp.videos.erase(temp.videos.begin() + to);
        }
    }

    if (best_from != -1) {
        int val = best_order.videos[best_from];
        best_order.videos.erase(best_order.videos.begin() + best_from);
        best_order.videos.insert(best_order.videos.begin() + best_to, val);
        gain = static_cast<double>(max_score - base_score);
    }
    if (evals >= max_evals) hit_cap = true;
    return best_order;
}

VideoOrder SequenceSolver::ls_active_window_2opt(VideoOrder& input, GreedyDecoder& decoder, const DecodeParams& params, int active_k, long long& evals, double& gain, bool& hit_cap) {
    VideoOrder best_order = input;
    int n = best_order.videos.size();
    int k = min(n, active_k);

    std::vector<DecodeState> states;
    states.reserve(k + 1);
    long long base_score = decoder.decode(best_order.videos, nullptr, params, nullptr, &states);

    evals = 0;
    gain = 0.0;
    const int max_evals = 300;
    hit_cap=false;

    for (int i = 0; i < k && evals < max_evals; ++i) {
        for (int j = i + 1; j < k && evals < max_evals; ++j) {
            reverse(best_order.videos.begin() + i, best_order.videos.begin() + j + 1);
            long long new_score =  decoder.decode(best_order.videos, nullptr, params, &states[i], nullptr);
            evals++;

            if (new_score > base_score) {
                gain = static_cast<double>(new_score - base_score);
                return best_order;
            }
            reverse(best_order.videos.begin() + i, best_order.videos.begin() + j + 1); // Revert
        }
    }
    if (evals >= max_evals) hit_cap = true;
    return best_order;
}

VideoOrder SequenceSolver::ls_request_frequency_hill_climb(VideoOrder& input, GreedyDecoder& decoder, const DecodeParams& params, int active_k, long long& evals, double& gain, bool& hit_cap) {
    VideoOrder best_order = input;
    int n = best_order.videos.size();
    int k = min(n, active_k);

    std::vector<DecodeState> states;
    states.reserve(k + 1);
    long long base_score = decoder.decode(best_order.videos, nullptr, params, nullptr, &states);
    const vector<double>& priorities = decoder.video_priorities();

    evals = 0;
    gain = 0.0;
    const int max_evals = 300;
    hit_cap = false;

    for (int i = 1; i < k && evals < max_evals; ++i) {
        int curr_video = best_order.videos[i];
        int prev_video = best_order.videos[i - 1];

        if (priorities[curr_video] > priorities[prev_video]) {
            swap(best_order.videos[i], best_order.videos[i - 1]);

            long long new_score = decoder.decode(best_order.videos, nullptr, params, &states[i - 1], nullptr);
            evals++;

            if (new_score > base_score) {
                gain = static_cast<double>(new_score - base_score);
                return best_order;
            }
            swap(best_order.videos[i], best_order.videos[i - 1]);
        }
    }
    if (evals >= max_evals) hit_cap = true;
    return best_order;
}

VideoOrder SequenceSolver::execute_local_search(
    LocalSearchStrategy strategy,
    VideoOrder& input,
    GreedyDecoder& decoder,
    const DecodeParams& params,
    int active_k,
    long long& evals_used,
    double& fitness_gain,
    bool& hit_cap
) {
    switch (strategy) {
        case LocalSearchStrategy::FIRST_IMPROVEMENT_SWAP:
            return ls_first_improvement_swap(input, decoder, params, active_k, evals_used, fitness_gain, hit_cap);

        case LocalSearchStrategy::BEST_IMPROVEMENT_ACTIVE_INSERT:
            return ls_best_improvement_insert(input, decoder, params, active_k, evals_used, fitness_gain, hit_cap);

        case LocalSearchStrategy::ACTIVE_WINDOW_2OPT_SCAN:
            return ls_active_window_2opt(input, decoder, params, active_k, evals_used, fitness_gain, hit_cap);

        case LocalSearchStrategy::REQUEST_FREQUENCY_GREEDY_HILL_CLIMB:
            return ls_request_frequency_hill_climb(input, decoder, params, active_k, evals_used, fitness_gain, hit_cap);

        case LocalSearchStrategy::NONE:
        default:
            evals_used = 0;
            fitness_gain = 0.0;
            hit_cap = false;
            return input;
    }
}
// endregion

// region metrics
// Diversity Calculation Across Active Prefix
double SequenceSolver::compute_population_diversity(const vector<VideoOrder>& pop, int prefix_k) {
    if (pop.size() < 2) return 0.0;
    size_t total_videos = pop[0].videos.size();
    if (total_videos == 0) return 0.0;

    int max_vid = 0;
    for (const auto& ind : pop) {
        for (int v : ind.videos) {
            if (v > max_vid) max_vid = v;
        }
    }

    size_t k = min(total_videos, static_cast<size_t>(max(2, prefix_k)));
    double total_distance = 0.0;
    int comparisons = 0;

    for (size_t i = 0; i < pop.size(); ++i) {
        vector<int> pos_a(max_vid + 1, -1);
        for (size_t idx = 0; idx < total_videos; ++idx) {
            pos_a[pop[i].videos[idx]] = static_cast<int>(idx);
        }

        for (size_t j = i + 1; j < pop.size(); ++j) {
            double pair_dist = 0.0;
            for (size_t idx = 0; idx < k; ++idx) {
                int vid = pop[j].videos[idx];
                if (vid <= max_vid && pos_a[vid] != -1) {
                    pair_dist += abs(static_cast<int>(idx) - pos_a[vid]);
                }
            }
            total_distance += pair_dist / k;
            comparisons++;
        }
    }
    return comparisons > 0 ? total_distance / comparisons : 0.0;
}

double SequenceSolver::compute_cache_utilization(const ProblemData& problem, const Solution& sol) {
    long long total_used = 0;
    for (int cap : sol.used_capacity) {
        total_used += cap;
    }
    long long total_cap = static_cast<long long>(problem.nbr_caches) * problem.cap_cache;
    return total_cap > 0 ? static_cast<double>(total_used) / total_cap : 0.0;
}

void SequenceSolver::export_benchmark_metrics(
    const string& instance_name, const vector<pair<int, double>>& history,
    int eval_count, int current_gen, double best_fitness, double elapsed_seconds, int pop_size,
    InitStrategy init_strat, CrossoverStrategy cx_strat, MutationStrategy mut_strat, LocalSearchStrategy ls_strat, DecodingStrategy dec_strat,
    int active_k, double gen_0_best, double gen_0_mean, double d0,
    long long t_cx, long long s_cx, long long n_cx, long long d_cx, double drop_cx,
    long long t_mut, long long s_mut, long long n_mut, long long d_mut, double drop_mut,
    long long t_ls, long long s_ls, long long cap_ls, double gain_ls,
    long long eval_time_us,
    const vector<VideoOrder>& initial_pop, GreedyDecoder& decoder, const ProblemData& problem, const Solution& final_solution
) {
    // 1. Calculate Standard Rates
    double cx_success_rate = t_cx > 0 ? (double)s_cx / t_cx : 0.0;
    double cx_neutral_rate = t_cx > 0 ? (double)n_cx / t_cx : 0.0;
    double cx_del_avg = d_cx > 0 ? drop_cx / d_cx : 0.0;

    double mut_success_rate = t_mut > 0 ? (double)s_mut / t_mut : 0.0;
    double mut_neutral_rate = t_mut > 0 ? (double)n_mut / t_mut : 0.0;
    double mut_del_avg = d_mut > 0 ? drop_mut / d_mut : 0.0;

    double ls_success_rate = t_ls > 0 ? (double)s_ls / t_ls : 0.0;
    double ls_avg_delta = s_ls > 0 ? gain_ls / s_ls : 0.0;
    double ls_cap_rate = t_ls > 0 ? (double)cap_ls / t_ls : 0.0;

    double avg_eval_us = eval_count > 0 ? (double)eval_time_us / eval_count : 0.0;
    double cache_utilization = compute_cache_utilization(problem, final_solution);

    // 2. Calculate Phenotypic Diversity (with correct decode parameters)
    set<std::string> unique_phenotypes;
    for (const auto& ind : initial_pop) {
        Solution temp_sol;
        DecodeParams params;
        params.strategy = dec_strat;
        if (dec_strat == DecodingStrategy::FIXED_THRESHOLD_PROFILE) {
            params.tau_density = 0.15;
            params.tau_reserve = 0.10;
        } else if (dec_strat == DecodingStrategy::CO_EVOLVED_THRESHOLDS && ind.thresholds.size() >= 2) {
            params.tau_density = ind.thresholds[0];
            params.tau_reserve = ind.thresholds[1];
        }

        decoder.decode(ind.videos, &temp_sol, params);
        std::string hash_str;
        for (int cap : temp_sol.used_capacity) hash_str += std::to_string(cap) + "-";
        unique_phenotypes.insert(hash_str);
    }
    int phenotypic_diversity_d0 = unique_phenotypes.size();

    // 3. Calculate Residual Slack
    double residual_slack_mb = 0.0;
    long saturated_caches = 0;
    for (int cap : final_solution.used_capacity) {
        if (cap > 0) {
            residual_slack_mb += (problem.cap_cache - cap);
            saturated_caches++;
        }
    }
    residual_slack_mb = saturated_caches > 0 ? residual_slack_mb / saturated_caches : 0.0;

    // 4. Output
    generate_json_output(
        instance_name, history, eval_count, current_gen, best_fitness, elapsed_seconds, pop_size,
        "benchmark_results", static_cast<int>(init_strat), active_k, gen_0_best, gen_0_mean, d0,
        static_cast<int>(cx_strat), cx_success_rate, cx_neutral_rate,
        static_cast<int>(mut_strat), mut_success_rate, mut_neutral_rate,
        static_cast<int>(ls_strat), ls_success_rate, ls_avg_delta,
        static_cast<int>(dec_strat), avg_eval_us, cache_utilization,
        cx_del_avg, mut_del_avg, ls_cap_rate, residual_slack_mb, phenotypic_diversity_d0
    );
}
//endregion

DynamicHyperparameters SequenceSolver::calculate_dynamic_hyperparameters(const ProblemData& problem, const GreedyDecoder& decoder) {
    DynamicHyperparameters params;
    int candidates_count = decoder.candidate_videos().size();

    if (candidates_count == 0) {
        params.population_size = 30;
        params.crossover_fraction = 0.5;
        params.mutation_rate = 0.2;
        params.active_k = 0;
        return params;
    }

    long long possible_connections = static_cast<long long>(problem.nbr_endpoints) * problem.nbr_caches;
    long long actual_connections = 0;

    for (int e = 0; e < problem.nbr_endpoints; ++e) {
        for (int c = 0; c < problem.nbr_caches; ++c) {
            if (problem.network[e][c] > 0) {
                actual_connections++;
            }
        }
    }

    double network_density = possible_connections > 0 ? static_cast<double>(actual_connections) / possible_connections : 0.0;

    long long total_cache_cap = static_cast<long long>(problem.nbr_caches) * problem.cap_cache;
    long long total_video_size = 0;

    for (int s : problem.videos) {
        total_video_size += s;
    }

    double capacity_ratio = total_video_size > 0 ? static_cast<double>(total_cache_cap) / total_video_size : 0.0;

    candidates_count = decoder.candidate_videos().size();

    long long matrix_complexity = static_cast<long long>(problem.nbr_videos) * problem.nbr_endpoints * problem.nbr_caches;
    if (matrix_complexity > 50000000) {
        params.population_size = 20;
    } else {
        params.population_size = std::max(30, std::min(120, candidates_count / 50));
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

    double adaptive_k_ratio = std::min(0.40, std::max(0.05, capacity_ratio * 1.5));
    params.active_k = std::max(2, static_cast<int>(candidates_count * adaptive_k_ratio));

    return params;
}