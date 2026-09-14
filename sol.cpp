#include <iostream>
#include <string>
#include <sstream>
#include <vector>
#include <openGA.hpp>

using namespace std;
bool constraint_checking(int& cache, int& video, int& nbr_videos, int& cap_cache, vector<int>& videos, vector<vector<bool>>& results);
void add_video_to_cache(int& cache, int& video, vector<vector<bool>>& results);
int eval_time_saved(int& nbr_videos, int& nbr_caches, int& nbr_endpoints, int& nbr_requests, const vector<int>& endpoints, vector<int>& videos, vector<vector<int>>& network, vector<vector<int>>& requests, vector<vector<bool>>& results);


int main(){



    int line=0;
    vector<int> videos;
    vector<int> endpoints;
    vector<int> caches;
    int nbr_videos, nbr_endpoints, nbr_requests, nbr_caches ,cap_cache;


    vector<vector<int>> network;
    vector<vector<int>> requests;
    vector<vector<bool>> results;


    int endpoint_counter=0;
    
    //Entree des données
    for (string s; getline(cin, s);) {
            istringstream stream(s);
            if(line==0){
               stream >> nbr_videos ;
                videos.resize(nbr_videos);
                stream >> nbr_endpoints;
                endpoints.resize(nbr_endpoints);
                stream >> nbr_requests;
                stream >> nbr_caches;
                caches.resize(nbr_caches);
                stream >> cap_cache;

                network.resize(nbr_endpoints, vector<int>(nbr_caches, 0));
                requests.resize(nbr_endpoints, vector<int>(nbr_videos, 0));
                results.resize(nbr_caches, vector<bool>(nbr_videos, false));
            }else if(line==1){
                for(int i =0; i<nbr_videos; i++)
                {
                    stream >> videos[i];
                }
            }else if(line>1 && endpoint_counter != nbr_endpoints){
                stream>> endpoints[endpoint_counter];
                int nb_caches_temp;
                stream>> nb_caches_temp;

                for(int i=0; i<nb_caches_temp;i++){
                    getline(cin, s);
                    istringstream stream(s);
                    line++;

                    int curr_cache;
                    stream>> curr_cache;
                    stream >> network[endpoint_counter][curr_cache];

                }
                endpoint_counter++;
            }else{
                int curr_endpoint,curr_video;
                stream >> curr_video;
                stream >> curr_endpoint;
                stream >> requests[curr_endpoint][curr_video];
            }

            line++;
        }
    
        //Solution valide
        for(int i=0; i<nbr_endpoints; i++){
            for (int y = 0; y < nbr_videos; y++)
            {
                if(requests[i][y]>0){
                    //cout<<"Endpoint "<<i<<" has requested video "<<y<<" "<<requests[i][y]<<" times"<<endl;
                    bool already_in_cache=false;
                    for (int c = 0; c < nbr_caches && !already_in_cache ; c++)
                    {
                        //cout<<"Checking cache "<<c<<" for video "<<y<<"Network: "<<network[i][c]<<endl;
                        if(network[i][c]>0 && constraint_checking(c,y,nbr_videos,cap_cache,videos,results)){
                            //cout<<"Adding video "<<y<<" to cache "<<c<<endl;
                            add_video_to_cache(c,y,results);
                            already_in_cache=true;
                        }
                    }
                }
            }  
        }
       
        //Sortie des donnes
        cout<<nbr_caches<<endl;
        for(int i=0; i<nbr_caches;i++){
            cout<<i<<" ";
            for(int y=0; y<nbr_videos; y++){
                if(results[i][y]==true){
                    cout<<y<<" ";
                }
            }
            cout<<endl;
        }
        
        cout<<"Solution:"<< eval_time_saved(nbr_videos,nbr_caches,nbr_endpoints,nbr_requests,endpoints,videos,network,requests,results)<<endl;
}



int eval_time_saved(int& nbr_videos, int& nbr_caches, int& nbr_endpoints, int& nbr_requests, const vector<int>& endpoints, vector<int>& videos, vector<vector<int>>& network, vector<vector<int>>& requests, vector<vector<bool>>& results){
    long long total_saved = 0;
    long long total_requests = 0;

    for (int e = 0; e < nbr_endpoints; ++e) {
        for (int v = 0; v < nbr_videos; ++v) {
            int req_count = requests[e][v];
            if (req_count <= 0) continue;

            total_requests += req_count;

            int best_latency = endpoints[e]; //par défaut datacenter

            for (int c = 0; c < nbr_caches; ++c) {
                bool cache_connected = (network[e][c] > 0); // 0 = pas de lien
                bool video_in_cache = results[c][v];
                if (cache_connected && video_in_cache) {
                    best_latency = min(best_latency, network[e][c]);
                }
            }

            int saved = endpoints[e] - best_latency; //latency serv->endpoint - latency cache->endpoint
            total_saved += saved*req_count;
        }

    }

    return (total_saved*1000)/total_requests;
}

bool constraint_checking(int& cache, int& video, int& nbr_videos, int& cap_cache, vector<int>& videos, vector<vector<bool>>& results){
    int cache_size=0;
    for(int i=0; i<nbr_videos; i++){
        if(results[cache][i]==true || i==video){
            cache_size+=videos[i];
        }
    }
    //cout<<"Cache: "<<cache_size<<" <=  "<<cap_cache<<endl;
    return cache_size <= cap_cache;
}

void add_video_to_cache(int& cache, int& video, vector<vector<bool>>& results){
    results[cache][video]=true;
}