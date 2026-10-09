//
// Created by thomas on 9/17/26.
//

#ifndef CACHEPROBLEM_GLOBAL_H
#define CACHEPROBLEM_GLOBAL_H

#pragma once
#include <vector>
#include <fstream>
#include <cmath>
#include <filesystem>
#include <utility>

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
    std::vector<int> sequence;
    std::vector<std::vector<bool>> results;
    std::vector<int> used_capacity;
};

int eval_time_saved(const ProblemData& problemData, const std::vector<std::vector<bool>>& results);
Solution run_evol_algo(const ProblemData& problem, double best_known_value, const std::string& instance_name);
Solution run_evol_algo_newEncoding(const ProblemData& problem, double best_known_value, const std::string& instance_name);





inline vector<double> generate_targets(const double current_best_value, const int number_targets)
{
    std::vector<double> targets;
    targets.reserve(number_targets);

    double min_val = std::max(1.0, current_best_value * 0.01);
    double max_val = current_best_value;

    for (int i = 0; i < number_targets; ++i) {
        double t = static_cast<double>(i) / (number_targets - 1);
        double target = min_val * std::pow(max_val / min_val, t);
        targets.push_back(target);
    }

    return targets;
}


inline void generate_json_output(
    string instance_name,
    const std::vector<std::pair<int, double>>& data,
    int total_evals,
    int total_gens,
    double best_achieved,
    double total_seconds,
    int pop_size,
    const std::string& output_dir = ".",
    int init_strategy_id = 0,
    int active_k = 0,
    double gen_0_best = 0.0,
    double gen_0_mean = 0.0,
    double d0 = 0.0,
    int crossover_strategy_id = 0,
    double cx_success_rate = 0.0,
    double cx_neutral_rate = 0.0,
    int mutation_strategy_id = 0,
    double mut_success_rate = 0.0,
    double mut_neutral_rate = 0.0,
    int local_search_strategy_id = 0,
    double ls_success_rate = 0.0,
    double ls_avg_delta = 0.0,
    int decoding_strategy_id = 0,
    double avg_eval_us = 0.0,
    double cache_utilization = 0.0
) {
    std::error_code ec;
    std::filesystem::create_directories(output_dir, ec);

    std::string filename = output_dir + "/fitness_plot_" + instance_name + ".json";

    std::ofstream out(filename);
    if (!out) {
        return;
    }

    out << "{\n";
    out << "  \"instance\": \"" << instance_name << "\",\n";
    out << "  \"init_strategy_id\": " << init_strategy_id << ",\n";
    out << "  \"crossover_strategy_id\": " << crossover_strategy_id << ",\n";
    out << "  \"mutation_strategy_id\": " << mutation_strategy_id << ",\n";
    out << "  \"local_search_strategy_id\": " << local_search_strategy_id << ",\n";
    out << "  \"decoding_strategy_id\": " << decoding_strategy_id << ",\n";
    out << "  \"active_prefix_k\": " << active_k << ",\n";
    out << "  \"gen_0_best_score\": " << gen_0_best << ",\n";
    out << "  \"gen_0_mean_score\": " << gen_0_mean << ",\n";
    out << "  \"gen_0_diversity_d0\": " << d0 << ",\n";
    out << "  \"crossover_success_rate\": " << cx_success_rate << ",\n";
    out << "  \"crossover_neutral_rate\": " << cx_neutral_rate << ",\n";
    out << "  \"mutation_success_rate\": " << mut_success_rate << ",\n";
    out << "  \"mutation_neutral_rate\": " << mut_neutral_rate << ",\n";
    out << "  \"local_search_success_rate\": " << ls_success_rate << ",\n";
    out << "  \"local_search_avg_delta\": " << ls_avg_delta << ",\n";
    out << "  \"avg_eval_microseconds\": " << avg_eval_us << ",\n";
    out << "  \"cache_utilization_ratio\": " << cache_utilization << ",\n";
    out << "  \"total_evaluations\": " << total_evals << ",\n";
    out << "  \"total_generations\": " << total_gens << ",\n";
    out << "  \"population_size\": " << pop_size << ",\n";
    out << "  \"best_fitness_achieved\": " << best_achieved << ",\n";
    out << "  \"total_time_seconds\": " << total_seconds << ",\n";
    out << "  \"targets_hit\": [\n";

    for (size_t i = 0; i < data.size(); ++i) {
        out << "    {\n";
        out << "      \"eval\": " << data[i].first << ",\n";
        out << "      \"fitness\": " << data[i].second << "\n";
        out << "    }";

        if (i < data.size() - 1) {
            out << ",\n";
        } else {
            out << "\n";
        }
    }
    out << "  ]\n";
    out << "}\n";
    out.close();
}

#endif //CACHEPROBLEM_GLOBAL_H
