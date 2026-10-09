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
    std::vector<double> thresholds = {0.0, 0.0};
};

enum class InitStrategy {
    BASELINE_EXACT_NOISY,
    PURE_RANDOM,
    NOISY_PRIORITY_ADD,
    DENSITY_REQUEST_SORT,
    ACTIVE_FRONT_SHUFFLE,
    HYBRID_COMPOSITE
};

enum class CrossoverStrategy {
    BASELINE_OX1,
    ACTIVE_PREFIX_OX1,
    POSITION_BASED_POS,
    PARTIALLY_MAPPED_PMX,
    CYCLE_CROSSOVER_CX
};

enum class MutationStrategy {
    BASELINE_SWAP,
    ACTIVE_PREFIX_SWAP,
    ACTIVE_PREFIX_INVERSION,
    ACTIVE_PREFIX_INSERTION,
    BOUNDARY_SWAP
};

enum class LocalSearchStrategy {
    NONE,
    FIRST_IMPROVEMENT_SWAP,
    BEST_IMPROVEMENT_ACTIVE_INSERT,
    ACTIVE_WINDOW_2OPT_SCAN,
    REQUEST_FREQUENCY_GREEDY_HILL_CLIMB
};


using Perm_Engine = EA::Genetic<VideoOrder, double>;

class SequenceSolver : public ISolver {
public:
    Solution run(const ProblemData& problem, double best_known_value, const std::string& instance_name) override;

    Solution run_benchmark( const ProblemData& problem, double best_known_value, const std::string& instance_name, InitStrategy init_strat = InitStrategy::BASELINE_EXACT_NOISY,  CrossoverStrategy = CrossoverStrategy::BASELINE_OX1,MutationStrategy mut_strat = MutationStrategy::BASELINE_SWAP,LocalSearchStrategy ls_strat = LocalSearchStrategy::NONE,DecodingStrategy dec_strat = DecodingStrategy::STATIC_GREEDY);

    VideoOrder generate_initial_individual(
        InitStrategy strategy,
        const GreedyDecoder& decoder,
        int individual_index,
        int population_size,
        int active_prefix_k,
        const std::function<double(void)>& rnd
    );



    VideoOrder execute_crossover(
        CrossoverStrategy strategy,
        const VideoOrder& p1,
        const VideoOrder& p2,
        int nbr_videos,
        int active_k,
        const std::function<double(void)>& rnd
    );

    VideoOrder execute_mutation(
        MutationStrategy strategy,
        const VideoOrder& original,
        int active_k,
        const std::function<double(void)>& rnd
    );

    VideoOrder execute_local_search(
        LocalSearchStrategy strategy,
        VideoOrder& input,
         GreedyDecoder& decoder,
        int active_k,
        long long& evals_used,
        double& fitness_gain
    );


private:
    VideoOrder init_genes(const GreedyDecoder& decoder, bool exact, const std::function<double(void)>& rnd);

    // Initialisation approaches
    VideoOrder init_exact_greedy(const GreedyDecoder& decoder);
    VideoOrder init_pure_random(const GreedyDecoder& decoder, const std::function<double(void)>& rnd);
    VideoOrder init_noisy_mult(const GreedyDecoder& decoder, const std::function<double(void)>& rnd);
    VideoOrder init_noisy_add(const GreedyDecoder& decoder, double noise_scale, const std::function<double(void)>& rnd);
    VideoOrder init_density_sort(const GreedyDecoder& decoder, bool exact, const std::function<double(void)>& rnd);
    VideoOrder init_active_front_shuffle(const GreedyDecoder& decoder, int active_k, const std::function<double(void)>& rnd);


    VideoOrder order_crossover(const VideoOrder& p1, const VideoOrder& p2, int nbr_videos, const std::function<double(void)>& rnd);
    VideoOrder active_prefix_ox1(const VideoOrder& p1, const VideoOrder& p2, int nbr_videos, int active_k, const std::function<double(void)>& rnd);
    VideoOrder position_based_crossover(const VideoOrder& p1, const VideoOrder& p2,int nbr_videos, const std::function<double(void)>& rnd);
    VideoOrder partially_mapped_crossover(const VideoOrder& p1, const VideoOrder& p2, int active_k, const std::function<double(void)>& rnd);
    VideoOrder cycle_crossover(const VideoOrder& p1, const VideoOrder& p2);


    VideoOrder swap_mutation(const VideoOrder& original, const std::function<double(void)>& rnd);
    VideoOrder active_prefix_swap(const VideoOrder& original, int active_k, const std::function<double(void)>& rnd);
    VideoOrder active_prefix_inversion(const VideoOrder& original, int active_k, const std::function<double(void)>& rnd);
    VideoOrder active_prefix_insertion(const VideoOrder& original, int active_k, const std::function<double(void)>& rnd);
    VideoOrder boundary_swap(const VideoOrder& original, int active_k, const std::function<double(void)>& rnd);

    VideoOrder ls_first_improvement_swap( VideoOrder& input,  GreedyDecoder& decoder, int active_k, long long& evals, double& gain);
    VideoOrder ls_best_improvement_insert( VideoOrder& input,  GreedyDecoder& decoder, int active_k, long long& evals, double& gain);
    VideoOrder ls_active_window_2opt( VideoOrder& input,  GreedyDecoder& decoder, int active_k, long long& evals, double& gain);
    VideoOrder ls_request_frequency_hill_climb( VideoOrder& input,  GreedyDecoder& decoder, int active_k, long long& evals, double& gain);

    //Benchmark Functions
    double compute_population_diversity(const std::vector<VideoOrder>& pop, int prefix_k);
    double compute_cache_utilization(const ProblemData& problem, const Solution& sol);

};

#endif //CACHEPROBLEM_SEQUENCE_SOLVER_H
