//
// Created by thomas on 9/17/26.
//

#ifndef CACHEPROBLEM_GLOBAL_H
#define CACHEPROBLEM_GLOBAL_H

#pragma once
#include <vector>
#include <fstream>
#include <cmath>
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
    std::vector<std::vector<bool>> results;
    std::vector<int> used_capacity;
};



int eval_time_saved(const ProblemData& problemData, const std::vector<std::vector<bool>>& results);
Solution run_evol_algo(const ProblemData& problem, double best_known_value, const std::string& instance_name);

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

inline void generate_json_output(string instance_name, const std::vector<std::pair<int, double>>& data, int total_evals, int total_gens, double best_achieved, double total_seconds, int pop_size,const std::string& output_dir = ".") {

    std::string filename = output_dir + "/fitness_plot_" + instance_name + ".json";


    std::ofstream out(filename);
    if (!out) return;

    out << "{\n";
    out << "  \"instance\": \"" << instance_name << "\",\n";
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
