#include <iostream>
#include <vector>
#include <functional>
#include "openGA.hpp"

using namespace std;

struct ProblemData {
    int nbr_videos = 0;
    int nbr_endpoints = 0;
    int nbr_requests = 0;
    int nbr_caches = 0;
    int cap_cache = 0;
    std::vector<int> videos;
    std::vector<int> endpoints;
    std::vector<std::vector<int>> network;
    std::vector<std::vector<int>> requests;
};

struct Solution {
    vector<vector<bool>> results;
    std::vector<int> used_capacity;
};

using GA_Engine = EA::Genetic<Solution, double>;

// Forward declarations
void init_genes(Solution& s, const ProblemData& problem, const std::function<double(void)>& rnd);
bool can_add_video(const Solution& s, const ProblemData& problem, int cache_id, int video_id);
void add_video(Solution& s, const ProblemData& problem, int cache_id, int video_id);
void remove_video(Solution& s, const ProblemData& problem, int cache_id, int video_id);
Solution mutate_solution(const Solution& original, const ProblemData& problem, const std::function<double(void)>& rnd);
void show_generation_summary(int generation_number, const GA_Engine::thisGenerationType& generation, const Solution& best_solution);

int eval_time_saved(const ProblemData& problemData, const vector<vector<bool>>& results);

Solution run_evol_algo(const ProblemData& problem) {
    GA_Engine ga;

    ga.problem_mode = EA::GA_MODE::SOGA;
    ga.population = 30;
    ga.generation_max = 50;

    ga.crossover_fraction = 0.7;
    ga.mutation_rate = 0.2;

    ga.elite_count = 2;
    ga.verbose = false;

    ga.init_genes = [&problem](Solution& s, const std::function<double(void)>& rnd) {
        init_genes(s, problem, rnd);
    };

    ga.eval_solution = [&problem](const Solution& s, double& score) -> bool {
        score = static_cast<double>(eval_time_saved(problem, s.results));
        return true;
    };

    ga.calculate_SO_total_fitness = [](const GA_Engine::thisChromosomeType& c) -> double {
        return -c.middle_costs;
    };

    ga.mutate = [&problem](const Solution& s, const std::function<double(void)>& rnd, double shrink_scale) -> Solution {
        return mutate_solution(s, problem, rnd);
    };

    ga.crossover = [](const Solution& p1, const Solution& p2, const std::function<double(void)>& rnd) -> Solution {
        return p1;
    };

    ga.SO_report_generation = [](int gen, const GA_Engine::thisGenerationType& gen_obj, const Solution& best_sol) {
        show_generation_summary(gen, gen_obj, best_sol);
    };

    ga.solve();

    return ga.last_generation.chromosomes[ga.last_generation.best_chromosome_index].genes;
}

void init_genes(Solution& s, const ProblemData& problem, const std::function<double(void)>& rnd) {
    s.results.assign(problem.nbr_caches, std::vector<bool>(problem.nbr_videos, false));
    s.used_capacity.assign(problem.nbr_caches, 0);
    for (int c = 0; c < problem.nbr_caches; ++c) {
        int start_idx = static_cast<int>(rnd() * problem.nbr_videos) % problem.nbr_videos;
        for (int step = 0; step < problem.nbr_videos; ++step) {
            int v = (start_idx + step) % problem.nbr_videos;
            if (can_add_video(s, problem, c, v) && rnd() < 0.25) {
                add_video(s, problem, c, v);
            }
        }
    }
}

// Constraints checker
bool can_add_video(const Solution& s, const ProblemData& problem, int cache_id, int video_id) {
    if (s.results[cache_id][video_id]) {
        return true;
    }
    return (s.used_capacity[cache_id] + problem.videos[video_id]) <= problem.cap_cache;
}

// Solution updaters
void add_video(Solution& s, const ProblemData& problem, int cache_id, int video_id) {
    if (!s.results[cache_id][video_id]) {
        s.results[cache_id][video_id] = true;
        s.used_capacity[cache_id] += problem.videos[video_id];
    }
}

void remove_video(Solution& s, const ProblemData& problem, int cache_id, int video_id) {
    if (s.results[cache_id][video_id]) {
        s.results[cache_id][video_id] = false;
        s.used_capacity[cache_id] -= problem.videos[video_id];
    }
}

// Mutation operator
Solution mutate_solution(const Solution& original, const ProblemData& problem, const std::function<double(void)>& rnd) {
    Solution mutated = original;
    int cache_idx = static_cast<int>(rnd() * problem.nbr_caches) % problem.nbr_caches;
    int video_idx = static_cast<int>(rnd() * problem.nbr_videos) % problem.nbr_videos;

    if (mutated.results[cache_idx][video_idx]) {
        remove_video(mutated, problem, cache_idx, video_idx);
    } else if (can_add_video(mutated, problem, cache_idx, video_idx)) {
        add_video(mutated, problem, cache_idx, video_idx);
    }
    return mutated;
}

void show_generation_summary(int generation_number,
                             const GA_Engine::thisGenerationType& generation,
                             const Solution& best_solution) {
    double best_time_saved = -generation.best_total_cost;
    double avg_time_saved = -generation.average_cost;

    std::cout << "Generation " << generation_number
              << " | Best saved: " << best_time_saved
              << " | Avg saved: " << avg_time_saved
              << "\n";
}