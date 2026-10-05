// Option 2: genetic algorithm with a permutation encoding.
// A chromosome is an order of the candidate videos. It is decoded by the
// greedy decoder (greedy.cpp), which fills the caches following this order,
// so every chromosome is a valid solution.
// Same GA settings as run_evol_algo (option 1) so that both can be compared.

#include <chrono>
#include <cmath>
#include <iostream>
#include <vector>
#include "openGA.hpp"
#include "global.h"
#include "greedy.h"

using namespace std;

namespace {

struct VideoOrder {
    vector<int> videos;
};

using Perm_Engine = EA::Genetic<VideoOrder, double>;

// First chromosome = greedy priority order; the others = the same order with noise on the priorities
VideoOrder init_order(const GreedyDecoder& decoder, bool exact, const function<double(void)>& rnd) {
    VideoOrder order;
    order.videos = decoder.priority_order();
    if (exact) return order;

    const vector<double>& priority = decoder.video_priorities();
    vector<pair<double, int>> keys;
    keys.reserve(order.videos.size());
    for (int v : order.videos) {
        keys.push_back({priority[v] * exp(2.0 * rnd() - 1.0), v}); // priority x [1/e, e]
    }
    sort(keys.begin(), keys.end(), [](const auto& a, const auto& b) { return a.first > b.first; });
    for (size_t i = 0; i < keys.size(); ++i) {
        order.videos[i] = keys[i].second;
    }
    return order;
}

// Swap mutation: exchange the positions of two videos
VideoOrder swap_mutation(const VideoOrder& original, const function<double(void)>& rnd) {
    VideoOrder mutated = original;
    int n = mutated.videos.size();
    if (n < 2) return mutated;
    int i = static_cast<int>(rnd() * n) % n;
    int j = static_cast<int>(rnd() * n) % n;
    swap(mutated.videos[i], mutated.videos[j]);
    return mutated;
}

// Order crossover (OX): the child keeps a segment of parent 1, the other
// videos are taken in the order of parent 2. Every video appears exactly once.
VideoOrder order_crossover(const VideoOrder& p1, const VideoOrder& p2, int nbr_videos, const function<double(void)>& rnd) {
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
        if (pos == a) pos = b + 1; // skip the segment
        child.videos[pos++] = v;
    }
    return child;
}

}

Solution run_evol_algo_perm(const ProblemData& problem, double best_known_value, const string& instance_name) {
    auto start_time = chrono::high_resolution_clock::now();

    GreedyDecoder decoder(problem);
    Perm_Engine ga;

    ga.set_seed(42);
    ga.multi_threading = false; // the decoder uses shared work buffers

    ga.problem_mode = EA::GA_MODE::SOGA;
    ga.population = 30;
    ga.generation_max = 50;

    ga.crossover_fraction = 0.7;
    ga.mutation_rate = 0.2;

    ga.elite_count = 2;
    ga.verbose = false;

    vector<double> targets = generate_targets(best_known_value, 50);

    // Eval tracker (same as run_evol_algo)
    int eval_count = 0;
    double best_fitness = -1.0;
    size_t current_target_idx = 0;
    int current_gen = 0;
    vector<pair<int, double>> history;

    bool first_chromosome = true;
    ga.init_genes = [&decoder, &first_chromosome](VideoOrder& s, const function<double(void)>& rnd) {
        s = init_order(decoder, first_chromosome, rnd);
        first_chromosome = false;
    };

    ga.eval_solution = [&](const VideoOrder& s, double& score) -> bool {
        score = static_cast<double>(decoder.decode(s.videos));

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

    ga.mutate = [](const VideoOrder& s, const function<double(void)>& rnd, double shrink_scale) -> VideoOrder {
        return swap_mutation(s, rnd);
    };

    ga.crossover = [&problem](const VideoOrder& p1, const VideoOrder& p2, const function<double(void)>& rnd) -> VideoOrder {
        return order_crossover(p1, p2, problem.nbr_videos, rnd);
    };

    ga.SO_report_generation = [&current_gen](int gen, const Perm_Engine::thisGenerationType& gen_obj, const VideoOrder& best) {
        current_gen = gen;
        cout << "Generation " << gen
             << " | Best saved: " << -gen_obj.best_total_cost
             << " | Avg saved: " << -gen_obj.average_cost
             << "\n";
    };

    ga.solve();

    chrono::duration<double> elapsed = chrono::high_resolution_clock::now() - start_time;
    generate_json_output(instance_name, history, eval_count, current_gen, best_fitness, elapsed.count(), ga.population, "benchmark_results");

    Solution sol;
    decoder.decode(ga.last_generation.chromosomes[ga.last_generation.best_chromosome_index].genes.videos, &sol);
    return sol;
}
