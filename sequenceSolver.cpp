#include "Sequence_Solver.h"
#include "global.h"
#include <chrono>
#include <cmath>
#include <iostream>
#include <algorithm>

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
    DecodingStrategy dec_strat
) {
    auto start_time = chrono::high_resolution_clock::now();

    GreedyDecoder decoder(problem);
    Perm_Engine ga;

    ga.set_seed(42);
    ga.multi_threading = false;
    ga.problem_mode = EA::GA_MODE::SOGA;
    ga.population = 30;
    ga.generation_max = 50;
    ga.crossover_fraction = 0.7;
    ga.mutation_rate = 0.2;
    ga.elite_count = 2;
    ga.verbose = false;

    vector<double> targets = generate_targets(best_known_value, 50);

    int eval_count = 0;
    double best_fitness = -1.0;
    size_t current_target_idx = 0;
    int current_gen = 0;
    vector<pair<int, double>> history;

    int active_k = max(2, static_cast<int>(decoder.candidate_videos().size()) / 4);
    int ind_counter = 0;
    int pop_size = ga.population;

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

    ga.init_genes = [this, &decoder, &ind_counter, pop_size, active_k, init_strat, dec_strat, &initial_pop_store](VideoOrder& s, const function<double(void)>& rnd) {
        s = generate_initial_individual(init_strat, decoder, ind_counter, pop_size, active_k, rnd);

        if (dec_strat == DecodingStrategy::CO_EVOLVED_THRESHOLDS) {
            s.thresholds = {rnd() * 0.3, rnd() * 0.2}; // Random init inside [0.0, 0.3] and [0.0, 0.2]
        }
        initial_pop_store.push_back(s);
        ind_counter++;
    };

    ga.eval_solution = [&](const VideoOrder& s, double& score) -> bool {
        auto t0 = chrono::high_resolution_clock::now();

        DecodeParams params;
        params.strategy = dec_strat;

        if (dec_strat == DecodingStrategy::FIXED_THRESHOLD_PROFILE) {
            params.tau_density = 0.15;
            params.tau_reserve = 0.10;
        } else if (dec_strat == DecodingStrategy::CO_EVOLVED_THRESHOLDS) {
            params.tau_density = s.thresholds[0];
            params.tau_reserve = s.thresholds[1];
        }

        score = static_cast<double>(decoder.decode(s.videos, nullptr, params));

        auto t1 = chrono::high_resolution_clock::now();
        total_eval_time_us += chrono::duration_cast<chrono::microseconds>(t1 - t0).count();

        if (eval_count < pop_size) {
            gen_0_sum += score;
            if (score > gen_0_best) gen_0_best = score;
        }

        if (score > best_fitness) {
            best_fitness = score;
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

    ga.mutate = [this, active_k, mut_strat, dec_strat, &decoder, &total_mutations, &successful_mutations, &neutral_mutations](const VideoOrder& s, const function<double(void)>& rnd, double shrink_scale) -> VideoOrder {
        VideoOrder mutated = execute_mutation(mut_strat, s, active_k, rnd);

        // Co-evolve thresholds via Gaussian perturbation
        if (dec_strat == DecodingStrategy::CO_EVOLVED_THRESHOLDS) {
            mutated.thresholds = s.thresholds;
            mutated.thresholds[0] = max(0.0, min(0.5, mutated.thresholds[0] + (rnd() - 0.5) * 0.05));
            mutated.thresholds[1] = max(0.0, min(0.3, mutated.thresholds[1] + (rnd() - 0.5) * 0.05));
        }

        long long orig_score = decoder.decode(s.videos);
        long long mut_score = decoder.decode(mutated.videos);

        total_mutations++;
        if (mut_score > orig_score) {
            successful_mutations++;
        } else if (mut_score == orig_score) {
            neutral_mutations++;
        }

        return mutated;
    };

    ga.crossover = [this, &problem, active_k, cx_strat, dec_strat, &decoder, &total_crossovers, &successful_crossovers, &neutral_crossovers](const VideoOrder& p1, const VideoOrder& p2, const function<double(void)>& rnd) -> VideoOrder {
        VideoOrder child = execute_crossover(cx_strat, p1, p2, problem.nbr_videos, active_k, rnd);

        // Blend crossover on thresholds
        if (dec_strat == DecodingStrategy::CO_EVOLVED_THRESHOLDS) {
            double alpha = rnd();
            child.thresholds.resize(2);
            child.thresholds[0] = alpha * p1.thresholds[0] + (1.0 - alpha) * p2.thresholds[0];
            child.thresholds[1] = alpha * p1.thresholds[1] + (1.0 - alpha) * p2.thresholds[1];
        }

        long long p1_score = decoder.decode(p1.videos);
        long long p2_score = decoder.decode(p2.videos);
        long long child_score = decoder.decode(child.videos);
        long long max_parent_score = max(p1_score, p2_score);

        total_crossovers++;
        if (child_score > max_parent_score) {
            successful_crossovers++;
        } else if (child_score == p1_score || child_score == p2_score) {
            neutral_crossovers++;
        }

        return child;
    };

    double max_run_time_seconds = 180.0;

    ga.SO_report_generation = [this, ls_strat, active_k, &decoder, &eval_count, &total_ls_apps, &successful_ls_apps, &total_ls_fitness_gain, &current_gen, start_time, max_run_time_seconds, &ga](
        int gen,
        const Perm_Engine::thisGenerationType& gen_obj,
        const VideoOrder& best
    ) {
        current_gen = gen;
        /*
        auto current_time = chrono::high_resolution_clock::now();
        chrono::duration<double> elapsed = current_time - start_time;
        if (elapsed.count() >= max_run_time_seconds) {
            ga.user_request_stop = true; // OpenGA stops evolution and moves directly to JSON output
        }*/

        if (ls_strat != LocalSearchStrategy::NONE && !gen_obj.chromosomes.empty()) {
            total_ls_apps++;
            long long extra_evals = 0;
            double gain = 0.0;

            int ls_k = min(active_k, 15); // Restrict search depth on large instances

            auto& non_const_gen = const_cast<Perm_Engine::thisGenerationType&>(gen_obj);

            int best_idx = 0;
            if (non_const_gen.best_chromosome_index >= 0 &&
                non_const_gen.best_chromosome_index < static_cast<int>(non_const_gen.chromosomes.size())) {
                best_idx = non_const_gen.best_chromosome_index;
                }

            VideoOrder refined = execute_local_search(
                ls_strat,
                non_const_gen.chromosomes[best_idx].genes,
                decoder,
                ls_k,
                extra_evals,
                gain
            );
            eval_count += extra_evals;

            if (gain > 0.0) {
                successful_ls_apps++;
                total_ls_fitness_gain += gain;
                non_const_gen.chromosomes[best_idx].genes = refined;
                non_const_gen.chromosomes[best_idx].middle_costs = -static_cast<double>(decoder.decode(refined.videos));
            }
        }
    };

    ga.solve();

    chrono::duration<double> elapsed = chrono::high_resolution_clock::now() - start_time;

    double d0 = compute_population_diversity(initial_pop_store, active_k);
    double gen_0_mean = pop_size > 0 ? gen_0_sum / pop_size : 0.0;

    double cx_success_rate = total_crossovers > 0 ? (double)successful_crossovers / total_crossovers : 0.0;
    double cx_neutral_rate = total_crossovers > 0 ? (double)neutral_crossovers / total_crossovers : 0.0;

    double mut_success_rate = total_mutations > 0 ? (double)successful_mutations / total_mutations : 0.0;
    double mut_neutral_rate = total_mutations > 0 ? (double)neutral_mutations / total_mutations : 0.0;

    double ls_success_rate = total_ls_apps > 0 ? (double)successful_ls_apps / total_ls_apps : 0.0;
    double ls_avg_delta = successful_ls_apps > 0 ? total_ls_fitness_gain / successful_ls_apps : 0.0;

    double avg_eval_us = eval_count > 0 ? static_cast<double>(total_eval_time_us) / eval_count : 0.0;

    // Pre-allocate final solution buffer space cleanly
    Solution final_solution;
    final_solution.results.assign(problem.nbr_caches, vector<bool>(problem.nbr_videos, false));
    final_solution.used_capacity.assign(problem.nbr_caches, 0);

    const auto& chromosomes = ga.last_generation.chromosomes;
    if (!chromosomes.empty()) {
        int best_idx = 0;
        if (ga.last_generation.best_chromosome_index >= 0 &&
            ga.last_generation.best_chromosome_index < static_cast<int>(chromosomes.size())) {
            best_idx = ga.last_generation.best_chromosome_index;
        }

        DecodeParams final_params;
        final_params.strategy = dec_strat;
        if (dec_strat == DecodingStrategy::FIXED_THRESHOLD_PROFILE) {
            final_params.tau_density = 0.15;
            final_params.tau_reserve = 0.10;
        } else if (dec_strat == DecodingStrategy::CO_EVOLVED_THRESHOLDS) {
            final_params.tau_density = chromosomes[best_idx].genes.thresholds[0];
            final_params.tau_reserve = chromosomes[best_idx].genes.thresholds[1];
        }

        final_solution.sequence = chromosomes[best_idx].genes.videos;
        decoder.decode(final_solution.sequence, &final_solution, final_params);
    }

    double cache_utilization = compute_cache_utilization(problem, final_solution);

    generate_json_output(
        instance_name,
        history,
        eval_count,
        current_gen,
        best_fitness,
        elapsed.count(),
        ga.population,
        "benchmark_results",
        static_cast<int>(init_strat),
        active_k,
        gen_0_best,
        gen_0_mean,
        d0,
        static_cast<int>(cx_strat),
        cx_success_rate,
        cx_neutral_rate,
        static_cast<int>(mut_strat),
        mut_success_rate,
        mut_neutral_rate,
        static_cast<int>(ls_strat),
        ls_success_rate,
        ls_avg_delta,
        static_cast<int>(dec_strat),
        avg_eval_us,
        cache_utilization
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
    order.videos = decoder.candidate_videos();

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

    int k = max(2, min(n, active_k));

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

    int window = max(2, min(n, active_k));
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

VideoOrder SequenceSolver::partially_mapped_crossover(const VideoOrder& p1, const VideoOrder& p2, int active_k, const function<double(void)>& rnd) {
    int n = p1.videos.size();
    VideoOrder child = p1;
    if (n == 0) return child;

    int window = max(2, min(n, active_k));
    int a = static_cast<int>(rnd() * window) % window;
    int b = static_cast<int>(rnd() * window) % window;
    if (a > b) swap(a, b);

    vector<int> pos_in_child(n);
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

VideoOrder SequenceSolver::cycle_crossover(const VideoOrder& p1, const VideoOrder& p2) {
    int n = p1.videos.size();
    VideoOrder child;
    child.videos.assign(n, -1);
    if (n == 0) return child;

    vector<int> pos_p1(n);
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
            return partially_mapped_crossover(p1, p2, active_k, rnd);

        case CrossoverStrategy::CYCLE_CROSSOVER_CX:
            return cycle_crossover(p1, p2);

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

    int window = max(2, min(n, active_k));
    int i = static_cast<int>(rnd() * window) % window;
    int j = static_cast<int>(rnd() * window) % window;
    swap(mutated.videos[i], mutated.videos[j]);

    return mutated;
}

VideoOrder SequenceSolver::active_prefix_inversion(const VideoOrder& original, int active_k, const function<double(void)>& rnd) {
    VideoOrder mutated = original;
    int n = mutated.videos.size();
    if (n < 2) return mutated;

    int window = max(2, min(n, active_k));
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

    int window = max(2, min(n, active_k));
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

VideoOrder SequenceSolver::execute_mutation(
    MutationStrategy strategy,
    const VideoOrder& original,
    int active_k,
    const function<double(void)>& rnd
) {
    switch (strategy) {
        case MutationStrategy::ACTIVE_PREFIX_SWAP:
            return active_prefix_swap(original, active_k, rnd);

        case MutationStrategy::ACTIVE_PREFIX_INVERSION:
            return active_prefix_inversion(original, active_k, rnd);

        case MutationStrategy::ACTIVE_PREFIX_INSERTION:
            return active_prefix_insertion(original, active_k, rnd);

        case MutationStrategy::BOUNDARY_SWAP:
            return boundary_swap(original, active_k, rnd);

        case MutationStrategy::BASELINE_SWAP:
        default:
            return swap_mutation(original, rnd);
    }
}

// endregion

// region Local Search

VideoOrder SequenceSolver::ls_first_improvement_swap(VideoOrder& input, GreedyDecoder& decoder, int active_k, long long& evals, double& gain) {
    VideoOrder best_order = input;
    long long base_score = decoder.decode(best_order.videos);
    int n = best_order.videos.size();
    int k = min(n, active_k);

    evals = 0;
    gain = 0.0;
    const int max_evals = 300; // Hard safety cap per generation

    for (int i = 0; i < k && evals < max_evals; ++i) {
        for (int j = i + 1; j < k && evals < max_evals; ++j) {
            swap(best_order.videos[i], best_order.videos[j]);
            long long new_score = decoder.decode(best_order.videos);
            evals++;

            if (new_score > base_score) {
                gain = static_cast<double>(new_score - base_score);
                return best_order; // First improvement accepted immediately
            }
            swap(best_order.videos[i], best_order.videos[j]); // Revert
        }
    }
    return best_order;
}

VideoOrder SequenceSolver::ls_best_improvement_insert(VideoOrder& input, GreedyDecoder& decoder, int active_k, long long& evals, double& gain) {
    VideoOrder best_order = input;
    long long base_score = decoder.decode(best_order.videos);
    long long max_score = base_score;
    int n = best_order.videos.size();
    int k = min(n, active_k);

    evals = 0;
    gain = 0.0;
    int best_from = -1, best_to = -1;
    const int max_evals = 300;

    for (int from = 0; from < k && evals < max_evals; ++from) {
        VideoOrder temp = input;
        int val = temp.videos[from];
        temp.videos.erase(temp.videos.begin() + from);

        for (int to = 0; to < k && evals < max_evals; ++to) {
            if (from == to) continue;
            temp.videos.insert(temp.videos.begin() + to, val);
            long long current_score = decoder.decode(temp.videos);
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
    return best_order;
}

VideoOrder SequenceSolver::ls_active_window_2opt(VideoOrder& input, GreedyDecoder& decoder, int active_k, long long& evals, double& gain) {
    VideoOrder best_order = input;
    long long base_score = decoder.decode(best_order.videos);
    int n = best_order.videos.size();
    int k = min(n, active_k);

    evals = 0;
    gain = 0.0;
    const int max_evals = 300;

    for (int i = 0; i < k && evals < max_evals; ++i) {
        for (int j = i + 1; j < k && evals < max_evals; ++j) {
            reverse(best_order.videos.begin() + i, best_order.videos.begin() + j + 1);
            long long new_score = decoder.decode(best_order.videos);
            evals++;

            if (new_score > base_score) {
                gain = static_cast<double>(new_score - base_score);
                return best_order;
            }
            reverse(best_order.videos.begin() + i, best_order.videos.begin() + j + 1); // Revert
        }
    }
    return best_order;
}

VideoOrder SequenceSolver::ls_request_frequency_hill_climb(VideoOrder& input, GreedyDecoder& decoder, int active_k, long long& evals, double& gain) {
    VideoOrder best_order = input;
    long long base_score = decoder.decode(best_order.videos);
    const vector<double>& priorities = decoder.video_priorities();

    evals = 0;
    gain = 0.0;
    int k = min(static_cast<int>(best_order.videos.size()), active_k);
    const int max_evals = 300;

    for (int i = 1; i < k && evals < max_evals; ++i) {
        int curr_video = best_order.videos[i];
        int prev_video = best_order.videos[i - 1];

        if (priorities[curr_video] > priorities[prev_video]) {
            swap(best_order.videos[i], best_order.videos[i - 1]);
            long long new_score = decoder.decode(best_order.videos);
            evals++;

            if (new_score > base_score) {
                gain = static_cast<double>(new_score - base_score);
                return best_order;
            }
            swap(best_order.videos[i], best_order.videos[i - 1]); // Revert
        }
    }
    return best_order;
}

VideoOrder SequenceSolver::execute_local_search(
    LocalSearchStrategy strategy,
    VideoOrder& input,
    GreedyDecoder& decoder,
    int active_k,
    long long& evals_used,
    double& fitness_gain
) {
    switch (strategy) {
        case LocalSearchStrategy::FIRST_IMPROVEMENT_SWAP:
            return ls_first_improvement_swap(input, decoder, active_k, evals_used, fitness_gain);

        case LocalSearchStrategy::BEST_IMPROVEMENT_ACTIVE_INSERT:
            return ls_best_improvement_insert(input, decoder, active_k, evals_used, fitness_gain);

        case LocalSearchStrategy::ACTIVE_WINDOW_2OPT_SCAN:
            return ls_active_window_2opt(input, decoder, active_k, evals_used, fitness_gain);

        case LocalSearchStrategy::REQUEST_FREQUENCY_GREEDY_HILL_CLIMB:
            return ls_request_frequency_hill_climb(input, decoder, active_k, evals_used, fitness_gain);

        case LocalSearchStrategy::NONE:
        default:
            evals_used = 0;
            fitness_gain = 0.0;
            return input;
    }
}
// endregion

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

