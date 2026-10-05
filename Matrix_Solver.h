//
// Created by thomas on 9/30/26.
//

#ifndef MATRIXSOLVER_H
#define MATRIXSOLVER_H

#include "Solver_Interface.h"
#include "openGA.hpp"

class MatrixSolver : public ISolver {
public:
    Solution run(const ProblemData& problem, double best_known_value, const std::string& instance_name) override;

private:
    void init_genes(Solution& s, const ProblemData& problem, const std::function<double(void)>& rnd);
    bool can_add_video(const Solution& s, const ProblemData& problem, int cache_id, int video_id);
    void add_video(Solution& s, const ProblemData& problem, int cache_id, int video_id);
    void remove_video(Solution& s, const ProblemData& problem, int cache_id, int video_id);
    Solution mutate_solution(const Solution& original, const ProblemData& problem, const std::function<double(void)>& rnd);
    Solution crossover_solution(const Solution& p1, const Solution& p2, const ProblemData& problem, const std::function<double(void)>& rnd);
};

#endif
