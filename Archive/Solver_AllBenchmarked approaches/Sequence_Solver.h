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
    CYCLE_CROSSOVER_CX,
    ACTIVE_PREFIX_PPX,
    ACTIVE_SET_CROSSOVER
};

enum class MutationStrategy {
    BASELINE_SWAP,
    ACTIVE_PREFIX_SWAP,
    ACTIVE_PREFIX_INVERSION,
    ACTIVE_PREFIX_INSERTION,
    BOUNDARY_SWAP,
    PROMOTE_UNPLACED,
    GRAPH_AWARE_BOTTLENECK_REPAIR
};

enum class LocalSearchStrategy {
    NONE,
    FIRST_IMPROVEMENT_SWAP,
    BEST_IMPROVEMENT_ACTIVE_INSERT,
    ACTIVE_WINDOW_2OPT_SCAN,
    REQUEST_FREQUENCY_GREEDY_HILL_CLIMB
};

struct DynamicHyperparameters {
    int population_size;
    double crossover_fraction;
    double mutation_rate;
    int active_k;
};


using Perm_Engine = EA::Genetic<VideoOrder, double>;

class SequenceSolver : public ISolver {
public:
    Solution run(const ProblemData& problem, double best_known_value, const std::string& instance_name) override;

    Solution run_benchmark( const ProblemData& problem, double best_known_value, const std::string& instance_name, InitStrategy init_strat = InitStrategy::BASELINE_EXACT_NOISY,  CrossoverStrategy = CrossoverStrategy::BASELINE_OX1,MutationStrategy mut_strat = MutationStrategy::BASELINE_SWAP,LocalSearchStrategy ls_strat = LocalSearchStrategy::NONE,DecodingStrategy dec_strat = DecodingStrategy::STATIC_GREEDY, unsigned int seed = 42, double max_run_time_seconds=230.0);

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
        const std::function<double(void)>& rnd,
        const std::vector<char>* placed = nullptr,
        const std::vector<double>* priorities = nullptr

    );

    VideoOrder execute_local_search(
        LocalSearchStrategy strategy,
        VideoOrder& input,
        GreedyDecoder& decoder,
        const DecodeParams& params,
        int active_k,
        long long& evals_used,
        double& fitness_gain,
        bool& hit_cap
    );


private:
    DynamicHyperparameters calculate_dynamic_hyperparameters(const ProblemData& problem, const GreedyDecoder& decoder);

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
    VideoOrder partially_mapped_crossover(const VideoOrder& p1, const VideoOrder& p2, int nbr_videos, int active_k, const std::function<double(void)>& rnd);
    VideoOrder cycle_crossover(const VideoOrder& p1, const VideoOrder& p2, int nbr_videos);
    VideoOrder active_prefix_ppx(const VideoOrder& p1, const VideoOrder& p2, int nbr_videos, int active_k, const std::function<double(void)>& rnd);
    VideoOrder active_set_crossover(const VideoOrder& p1, const VideoOrder& p2, int nbr_videos, int active_k, const std::function<double(void)>& rnd);


    VideoOrder swap_mutation(const VideoOrder& original, const std::function<double(void)>& rnd);
    VideoOrder active_prefix_swap(const VideoOrder& original, int active_k, const std::function<double(void)>& rnd);
    VideoOrder active_prefix_inversion(const VideoOrder& original, int active_k, const std::function<double(void)>& rnd);
    VideoOrder active_prefix_insertion(const VideoOrder& original, int active_k, const std::function<double(void)>& rnd);
    VideoOrder boundary_swap(const VideoOrder& original, int active_k, const std::function<double(void)>& rnd);
    VideoOrder promote_unplaced(const VideoOrder& original, const std::vector<char>& placed, const std::function<double(void)>& rnd);
    VideoOrder graph_aware_bottleneck_repair(const VideoOrder& original, const std::vector<char>& placed, const std::vector<double>& priorities, int active_k, const std::function<double(void)>& rnd);

    VideoOrder ls_first_improvement_swap( VideoOrder& input,  GreedyDecoder& decoder, const DecodeParams& params, int active_k, long long& evals, double& gain, bool& hit_cap);
    VideoOrder ls_best_improvement_insert( VideoOrder& input,  GreedyDecoder& decoder, const DecodeParams& params ,int active_k, long long& evals, double& gain, bool& hit_cap);
    VideoOrder ls_active_window_2opt( VideoOrder& input,  GreedyDecoder& decoder, const DecodeParams& params, int active_k, long long& evals, double& gain, bool& hit_cap);
    VideoOrder ls_request_frequency_hill_climb( VideoOrder& input,  GreedyDecoder& decoder, const DecodeParams& params, int active_k, long long& evals, double& gain, bool& hit_cap);

    //Benchmark Functions
    double compute_population_diversity(const std::vector<VideoOrder>& pop, int prefix_k);
    double compute_cache_utilization(const ProblemData& problem, const Solution& sol);

    void export_benchmark_metrics(
    const std::string& instance_name, const std::vector<std::pair<int, double>>& history,
    int eval_count, int current_gen, double best_fitness, double elapsed_seconds, int pop_size,
    InitStrategy init_strat, CrossoverStrategy cx_strat, MutationStrategy mut_strat, LocalSearchStrategy ls_strat, DecodingStrategy dec_strat,
    int active_k, double gen_0_best, double gen_0_mean, double d0,
    int best_found_gen, double d_final, int phenotypic_diversity_final,
    long long t_cx, long long s_cx, long long n_cx, long long d_cx, double drop_cx,
    long long t_mut, long long s_mut, long long n_mut, long long d_mut, double drop_mut,
    long long t_ls, long long s_ls, long long cap_ls, double gain_ls,
    long long eval_time_us,
    const std::vector<VideoOrder>& initial_pop, GreedyDecoder& decoder, const ProblemData& problem, const Solution& final_solution
);
};

#endif //CACHEPROBLEM_SEQUENCE_SOLVER_H
