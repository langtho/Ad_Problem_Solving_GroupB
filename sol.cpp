#ifndef MAIN
#define MAIN

#include <iostream>
#include <string>
#include <sstream>
#include <vector>
#include <cstdlib>
#include "global.h"
#include "greedy.h"


using namespace std;
bool constraint_checking(int& cache, int& video, int& nbr_videos, int& cap_cache, vector<int>& videos, vector<vector<bool>>& results);
void add_video_to_cache(int& cache, int& video, vector<vector<bool>>& results);

// Usage: solver [best_known_value] [instance_name] [method] < instance.in
//   method = ga (default, matrix encoding) | greedy | ga_perm (permutation encoding + greedy decoder)
int main(int argc, char* argv[]){

    double best_known_value = (argc > 1) ? atof(argv[1]) : 0;
    string instance_name = (argc > 2) ? argv[2] : "instance";
    string method = (argc > 3) ? argv[3] : "ga";

    int line=0;
    ProblemData problemData;
    vector<vector<bool>> results;

    int endpoint_counter=0;
    
    //Entree des données
    for (string s; getline(cin, s);) {
            istringstream stream(s);
            if(line==0){
               stream >> problemData.nbr_videos ;
                problemData.videos.resize(problemData.nbr_videos);
                stream >> problemData.nbr_endpoints;
                problemData.endpoints.resize(problemData.nbr_endpoints);
                stream >> problemData.nbr_requests;
                stream >> problemData.nbr_caches;
                stream >> problemData.cap_cache;

                problemData.network.resize(problemData.nbr_endpoints, vector<int>(problemData.nbr_caches, 0));
                problemData.requests.resize(problemData.nbr_endpoints, vector<int>(problemData.nbr_videos, 0));
                results.resize(problemData.nbr_caches, vector<bool>(problemData.nbr_videos, false));
            }else if(line==1){
                for(int i =0; i<problemData.nbr_videos; i++)
                {
                    stream >> problemData.videos[i];
                }
            }else if(line>1 && endpoint_counter != problemData.nbr_endpoints){
                stream>> problemData.endpoints[endpoint_counter];
                int nb_caches_temp;
                stream>> nb_caches_temp;

                for(int i=0; i<nb_caches_temp;i++){
                    getline(cin, s);
                    istringstream stream(s);
                    line++;

                    int curr_cache;
                    stream>> curr_cache;
                    stream >> problemData.network[endpoint_counter][curr_cache];

                }
                endpoint_counter++;
            }else{
                int curr_endpoint,curr_video,curr_count;
                stream >> curr_video;
                stream >> curr_endpoint;
                stream >> curr_count;
                // the same (video, endpoint) pair can appear on several lines: the counts add up
                problemData.requests[curr_endpoint][curr_video] += curr_count;
            }

            line++;
        }
    
        Solution sol;
        if (method == "greedy") {
            sol = run_greedy(problemData, best_known_value, instance_name + "_greedy");
        } else if (method == "ga_perm") {
            sol = run_evol_algo_perm(problemData, best_known_value, instance_name + "_ga_perm");
        } else {
            sol = run_evol_algo(problemData, best_known_value, instance_name);
        }
       
        //Sortie des donnes
        cout<<problemData.nbr_caches<<endl;
        for(int i=0; i<problemData.nbr_caches;i++){
            cout<<i<<" ";
            for(int y=0; y<problemData.nbr_videos; y++){
                if(sol.results[i][y]==true){
                    cout<<y<<" ";
                }
            }
            cout<<endl;
        }
}



int eval_time_saved(const ProblemData& problemData,const vector<vector<bool>>& results){
    long long total_saved = 0;
    long long total_requests = 0;

    for (int e = 0; e < problemData.nbr_endpoints; ++e) {
        for (int v = 0; v < problemData.nbr_videos; ++v) {
            int req_count = problemData.requests[e][v];
            if (req_count <= 0) continue;

            total_requests += req_count;

            int best_latency = problemData.endpoints[e]; //par défaut datacenter

            for (int c = 0; c < problemData.nbr_caches; ++c) {
                bool cache_connected = (problemData.network[e][c] > 0); // 0 = pas de lien
                bool video_in_cache = results[c][v];
                if (cache_connected && video_in_cache) {
                    best_latency = min(best_latency, problemData.network[e][c]);
                }
            }

            int saved = problemData.endpoints[e] - best_latency; //latency serv->endpoint - latency cache->endpoint
            total_saved += saved*req_count;
        }

    }

    return (total_saved*1000)/total_requests;
}

bool constraint_checking(int& cache, int& video, ProblemData problem_data,vector<vector<bool>>& results){
    int cache_size=0;
    for(int i=0; i<problem_data.nbr_videos; i++){
        if(results[cache][i]==true || i==video){
            cache_size+=problem_data.videos[i];
        }
    }
    //cout<<"Cache: "<<cache_size<<" <=  "<<cap_cache<<endl;
    return cache_size <= problem_data.cap_cache;
}

void add_video_to_cache(int& cache, int& video, vector<vector<bool>>& results){
    results[cache][video]=true;
}

#endif
