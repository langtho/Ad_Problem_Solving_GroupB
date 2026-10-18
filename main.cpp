#ifndef MAIN
#define MAIN

#include <iostream>
#include <string>
#include <sstream>
#include <vector>
#include "Solver_Interface.h"
#include "Sequence_Solver.h"
#include "global.h"

using namespace std;

bool constraint_checking(int& cache, int& video, ProblemData problem_data, vector<vector<bool>>& results);
void add_video_to_cache(int& cache, int& video, vector<vector<bool>>& results);
ProblemData read_problem_data();

int main(int argc, char* argv[])
{
    double best_known_value = 0.0;
    if (argc >= 2 && argv[1] != nullptr) {
        best_known_value = stod(argv[1]);
    }

    string instance_name = "default_instance";
    if (argc >= 3 && argv[2] != nullptr) {
        instance_name = argv[2];
    }

    // Inside main()
    float pop_size_override = -1.0f;
    float cx_frac_override  = -1.0f;
    float mut_rate_override = -1.0f;
    float k_mult_override   = -1.0f;
    float slack_ratio_override = -1.0f;
    int elite_count_override = -1;
    float noise_scale_override = -1.0f;
    int stall_max_override = -1;

    if (argc >= 4 && argv[3] != nullptr) pop_size_override = stof(argv[3]);
    if (argc >= 5 && argv[4] != nullptr) cx_frac_override  = stof(argv[4]);
    if (argc >= 6 && argv[5] != nullptr) mut_rate_override = stof(argv[5]);
    if (argc >= 7 && argv[6] != nullptr) k_mult_override   = stof(argv[6]);
    if (argc >= 8 && argv[7] != nullptr) slack_ratio_override = stof(argv[7]);
    if (argc >= 9 && argv[8] != nullptr) elite_count_override = stoi(argv[8]);
    if (argc >= 10 && argv[9] != nullptr) noise_scale_override = stof(argv[9]);
    if (argc >= 11 && argv[10] != nullptr) stall_max_override = stoi(argv[10]);

    unsigned int seed = 42;
    if (argc >= 12 && argv[11] != nullptr) seed = static_cast<unsigned int>(stoul(argv[11]));

    double max_time_seconds = 230.0;
    if (argc >= 13 && argv[12] != nullptr) max_time_seconds = stod(argv[12]);

    ProblemData problemData = read_problem_data();

    ostringstream rn;
    rn << instance_name
       << "_pop"  << pop_size_override
       << "_cx"   << cx_frac_override
       << "_mut"  << mut_rate_override
       << "_k"    << k_mult_override
       << "_slk"  << slack_ratio_override
       << "_eli"  << elite_count_override
       << "_nsc"  << noise_scale_override
       << "_stl"  << stall_max_override
       << "_seed" << seed;

    SequenceSolver solver;
    Solution sol = solver.run_benchmark(
        problemData,
        best_known_value,
        rn.str(),
        pop_size_override,
        cx_frac_override,
        mut_rate_override,
        k_mult_override,
        slack_ratio_override,
        elite_count_override,
        noise_scale_override,
        stall_max_override,
        seed,
        max_time_seconds
    );

    cout << problemData.nbr_caches << "\n";
    for (int i = 0; i < problemData.nbr_caches; i++) {
        cout << i << " ";
        for (int y = 0; y < problemData.nbr_videos; y++) {
            if (sol.results[i][y]) {
                cout << y << " ";
            }
        }
        cout << "\n";
    }
    return 0;
}

ProblemData read_problem_data()
{
    int line = 0;
    ProblemData problemData;
    int endpoint_counter = 0;

    for (string s; getline(cin, s);) {
        istringstream stream(s);
        if (line == 0) {
            stream >> problemData.nbr_videos;
            problemData.videos.resize(problemData.nbr_videos);
            stream >> problemData.nbr_endpoints;
            problemData.endpoints.resize(problemData.nbr_endpoints);
            stream >> problemData.nbr_requests;
            stream >> problemData.nbr_caches;
            stream >> problemData.cap_cache;

            problemData.network.resize(problemData.nbr_endpoints, vector<int>(problemData.nbr_caches, 0));
            problemData.requests.resize(problemData.nbr_endpoints, vector<int>(problemData.nbr_videos, 0));
        } else if (line == 1) {
            for (int i = 0; i < problemData.nbr_videos; i++) {
                stream >> problemData.videos[i];
            }
        } else if (line > 1 && endpoint_counter != problemData.nbr_endpoints) {
            stream >> problemData.endpoints[endpoint_counter];
            int nb_caches_temp;
            stream >> nb_caches_temp;

            for (int i = 0; i < nb_caches_temp; i++) {
                getline(cin, s);
                istringstream sub_stream(s);
                line++;

                int curr_cache;
                sub_stream >> curr_cache;
                sub_stream >> problemData.network[endpoint_counter][curr_cache];
            }
            endpoint_counter++;
        } else {
            int curr_endpoint, curr_video;
            stream >> curr_video;
            stream >> curr_endpoint;
            stream >> problemData.requests[curr_endpoint][curr_video];
        }
        line++;
    }

    return problemData;
}

int eval_time_saved(const ProblemData& problemData, const vector<vector<bool>>& results)
{
    long long total_saved = 0;
    long long total_requests = 0;

    for (int e = 0; e < problemData.nbr_endpoints; ++e) {
        for (int v = 0; v < problemData.nbr_videos; ++v) {
            int req_count = problemData.requests[e][v];
            if (req_count <= 0) continue;

            total_requests += req_count;
            int best_latency = problemData.endpoints[e];

            for (int c = 0; c < problemData.nbr_caches; ++c) {
                if (problemData.network[e][c] > 0 && results[c][v]) {
                    best_latency = min(best_latency, problemData.network[e][c]);
                }
            }

            total_saved += static_cast<long long>(problemData.endpoints[e] - best_latency) * req_count;
        }
    }

    return total_requests > 0 ? static_cast<int>((total_saved * 1000) / total_requests) : 0;
}

bool constraint_checking(int& cache, int& video, ProblemData problem_data, vector<vector<bool>>& results)
{
    int cache_size = 0;
    for (int i = 0; i < problem_data.nbr_videos; i++) {
        if (results[cache][i] || i == video) {
            cache_size += problem_data.videos[i];
        }
    }
    return cache_size <= problem_data.cap_cache;
}

void add_video_to_cache(int& cache, int& video, vector<vector<bool>>& results)
{
    results[cache][video] = true;
}

#endif