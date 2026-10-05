//
// Created by thomas on 9/30/26.
//

#ifndef CACHEPROBLEM_SOLVER_INTERFACE_H
#define CACHEPROBLEM_SOLVER_INTERFACE_H

#include "global.h"
#include <string>

class ISolver {
public:
    virtual ~ISolver() = default;
    virtual Solution run(const ProblemData& problem, double best_known_value, const std::string& instance_name) = 0;
};

#endif //CACHEPROBLEM_SOLVER_INTERFACE_H
