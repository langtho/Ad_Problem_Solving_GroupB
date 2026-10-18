#ifndef CACHEPROBLEM_SEQUENCE_SOLVER_H
#define CACHEPROBLEM_SEQUENCE_SOLVER_H

#include "Solver_Interface.h"
#include "openGA.hpp"
#include "greedy.h"
#include <functional>
#include <vector>
#include <string>

struct VideoOrder {
    std::vector<int> videos;
};

enum class InitStrategy {
    DENSITY_REQUEST_SORT = 3,
    HYBRID_COMPOSITE = 5
};

enum class CrossoverStrategy {
    ACTIVE_PREFIX_OX1 = 1,
    ACTIVE_SET_CROSSOVER = 6
};

enum class MutationStrategy {
    PROMOTE_UNPLACED = 5
};

enum class LocalSearchStrategy {
    NONE = 0
};

struct DynamicHyperparameters {
    int population_size;
    double crossover_fraction;
    double mutation_rate;
    int active_k;
    double soft_slack_ratio;
    int elite_count;
    double noise_scale;
    int stall_max;
    InitStrategy init_strat;
    CrossoverStrategy cx_strat;
    MutationStrategy mut_strat;
    LocalSearchStrategy ls_strat;
    DecodingStrategy dec_strat;
};

using Perm_Engine = EA::Genetic<VideoOrder, double>;

class SequenceSolver : public ISolver {
public:
    Solution run(const ProblemData& problem, double best_known_value, const std::string& instance_name) override;

    Solution run_benchmark(
        const ProblemData& problem,
        double best_known_value,
        const std::string& instance_name,
        float pop_size_override = -1.0f,
        float cx_frac_override = -1.0f,
        float mut_rate_override = -1.0f,
        float k_mult_override = -1.0f,
        float slack_ratio_override = -1.0f,
        int elite_count_override = -1,
        float noise_scale_override = -1.0f,
        int stall_max_override = -1,
        unsigned int seed = 42,
        double max_run_time_seconds = 230.0
    );

    VideoOrder generate_initial_individual(
        InitStrategy strategy,
        const GreedyDecoder& decoder,
        int individual_index,
        int population_size,
        int active_prefix_k,
        double noise_scale,
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
        const std::vector<char>* placed = nullptr
    );

private:
    DynamicHyperparameters calculate_dynamic_hyperparameters(
        const ProblemData& problem,
        const GreedyDecoder& decoder,
        float pop_size_override,
        float cx_frac_override,
        float mut_rate_override,
        float k_mult_override,
        float slack_ratio_override,
        int elite_count_override,
        float noise_scale_override,
        int stall_max_override
    );

    VideoOrder init_exact_greedy(const GreedyDecoder& decoder);
    VideoOrder init_pure_random(const GreedyDecoder& decoder, const std::function<double(void)>& rnd);
    VideoOrder init_noisy_mult(const GreedyDecoder& decoder, const std::function<double(void)>& rnd);
    VideoOrder init_noisy_add(const GreedyDecoder& decoder, double noise_scale, const std::function<double(void)>& rnd);
    VideoOrder init_density_sort(const GreedyDecoder& decoder, bool exact, const std::function<double(void)>& rnd);
    VideoOrder init_active_front_shuffle(const GreedyDecoder& decoder, int active_k, const std::function<double(void)>& rnd);

    VideoOrder active_prefix_ox1(const VideoOrder& p1, const VideoOrder& p2, int nbr_videos, int active_k, const std::function<double(void)>& rnd);
    VideoOrder active_set_crossover(const VideoOrder& p1, const VideoOrder& p2, int nbr_videos, int active_k, const std::function<double(void)>& rnd);

    VideoOrder swap_mutation(const VideoOrder& original, const std::function<double(void)>& rnd);
    VideoOrder promote_unplaced(const VideoOrder& original, const std::vector<char>& placed, const std::function<double(void)>& rnd);
};

#endif