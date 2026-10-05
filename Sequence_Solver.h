//
// Created by thomas on 9/30/26.
//

#ifndef CACHEPROBLEM_SEQUENCE_SOLVER_H
#define CACHEPROBLEM_SEQUENCE_SOLVER_H

#include "Solver_Interface.h"
#include "openGA.hpp"
#include "greedy.h"
#include <functional>
#include <vector>
#include <string>

// Encodes the priority sequence of candidate videos
struct VideoOrder {
    std::vector<int> videos;
};

using Perm_Engine = EA::Genetic<VideoOrder, double>;

class SequenceSolver : public ISolver {
public:
    Solution run(const ProblemData& problem, double best_known_value, const std::string& instance_name) override;

private:
    VideoOrder init_genes(const GreedyDecoder& decoder, bool exact, const std::function<double(void)>& rnd);
    VideoOrder swap_mutation(const VideoOrder& original, const std::function<double(void)>& rnd);
    VideoOrder order_crossover(const VideoOrder& p1, const VideoOrder& p2, int nbr_videos, const std::function<double(void)>& rnd);
};

#endif //CACHEPROBLEM_SEQUENCE_SOLVER_H
