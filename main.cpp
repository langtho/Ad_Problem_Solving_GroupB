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

ProblemData read_problem_data();

int main(int argc, char* argv[])
{
    // Production CLI: ./solver [seed] [max_time_seconds] < instance.in > instance.out
    unsigned int seed = 42;
    if (argc >= 2 && argv[1] != nullptr) seed = static_cast<unsigned int>(stoul(argv[1]));

    double max_time_seconds = 230.0;
    if (argc >= 3 && argv[2] != nullptr) max_time_seconds = stod(argv[2]);

    ProblemData problemData = read_problem_data();

    SequenceSolver solver;
    Solution sol = solver.run_production(problemData, seed, max_time_seconds);

    // Standard Google Hash Code submission format
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

#endif