//
// Created by thomas on 9/30/26.
//

#ifndef CACHEPROBLEM_SEQUENCE_SOLVER_H
#define CACHEPROBLEM_SEQUENCE_SOLVER_H

#include "Solver_Interface.h"
#include "openGA.hpp"
#include <functional>

struct VideoCachePreference {
    int cache_id;
    int potential_score;
};

class SequenceSolver : public ISolver {
public:
    Solution run(const ProblemData& problem, double best_known_value, const std::string& instance_name) override;

private:
    std::vector<std::vector<VideoCachePreference>> prefs;
    std::vector<std::vector<VideoCachePreference>> precompute_preferences(const ProblemData& problem);
    void init_genes(Solution& s, const ProblemData& problem, const std::function<double(void)>& rnd);
    Solution mutate_solution(const Solution& original, const ProblemData& problem, const std::function<double(void)>& rnd);
    Solution crossover_solution(const Solution& p1, const Solution& p2, const ProblemData& problem, const std::function<double(void)>& rnd);
    void calculate_result(Solution& s, const ProblemData& problem);
    void show_generation_summary(int generation_number, const EA::Genetic<Solution, double>::thisGenerationType& generation, const Solution& best_solution);
};

#endif //CACHEPROBLEM_SEQUENCE_SOLVER_H
