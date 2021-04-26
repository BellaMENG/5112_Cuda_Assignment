/*
 * Name: MENG Zihan
 * Student id: 20412027
 * ITSC email: zmengaa@connect.ust.hk
 *
 * COMPILE: nvcc -std=c++11 clustering_cuda_skeleton.cu clustering_impl.cpp main.cpp -o cuda
 * RUN:     ./cuda <path> <epsilon> <mu> <num_blocks_per_grid> <num_threads_per_block>
 */

#include <iostream>
#include "clustering.h"

using namespace std;
// Define variables or functions here

__device__
int get_num_com_nbrs(int *nbrs, int left_start, int left_end, int right_start, int right_end) {
    int left_pos = left_start, right_pos = right_start, num_com_nbrs = 0;

    while (left_pos < left_end && right_pos < right_end) {
        if (nbrs[left_pos] == nbrs[right_pos]) {
            num_com_nbrs++;
            left_pos++;
            right_pos++;
        } else if (nbrs[left_pos] < nbrs[right_pos]) {
            left_pos++;
        } else {
            right_pos++;
        }
    }
    return num_com_nbrs;
}

// findPivots<<<blocks, threads>>>(num_vs, num_es, epsilon, mu, d_nbr_offs, d_nbrs, d_pivots, d_num_sim_nbrs, d_sim_nbrs);
__global__
void findPivots(int start, int end, int num_blocks_per_grid, int num_threads_per_block, int num_vs, int num_es, float ep, int mu, int* d_nbr_offs, int* d_nbrs, bool* d_pivots, int* d_num_sim_nbrs, int* d_sim_nbrs) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    int nthread = blockDim.x * gridDim.x;
    
    for (int i = tid; i < num_vs; i += nthread) {
        if (i < start || i >= end)
            continue;
        int left_start = d_nbr_offs[i];
        int left_end = d_nbr_offs[i + 1];
        int left_size = left_end - left_start;

//        d_sim_nbrs[i] = new int[left_size];
//        cudaMalloc(&d_sim_nbrs[i], left_size*sizeof(int));
//        d_sim_nbrs[i] = (int*)malloc(left_size * sizeof(int));
        
        // loop over all neighbors of i
        for (int j = left_start; j < left_end; j++) {
            int nbr_id = d_nbrs[j];
            
            int right_start = d_nbr_offs[nbr_id];
            int right_end = d_nbr_offs[nbr_id + 1];
            int right_size = right_end - right_start;
            
            // compute the similarity
            int num_com_nbrs = get_num_com_nbrs(d_nbrs, left_start, left_end, right_start, right_end);

            float sim = (num_com_nbrs + 2) / std::sqrt((left_size + 1.0) * (right_size + 1.0));
            
            if (sim > ep) {
                d_sim_nbrs[left_start + d_num_sim_nbrs[i]] = nbr_id;
                d_num_sim_nbrs[i]++;
            }
        }
        if (d_num_sim_nbrs[i] > mu) {
            d_pivots[i] = true;
        }
    }
}

void expansion(int cur_id, int num_clusters, int *nbr_offs, int *num_sim_nbrs, int *sim_nbrs,
               bool *visited, bool *pivots, int *cluster_result) {
    
    for (int i = 0; i < num_sim_nbrs[cur_id]; i++) {
        int nbr_id = sim_nbrs[nbr_offs[cur_id] + i];
        if ((pivots[nbr_id])&&(!visited[nbr_id])){
            visited[nbr_id] = true;
            cluster_result[nbr_id] = num_clusters;
            expansion(nbr_id, num_clusters, nbr_offs, num_sim_nbrs, sim_nbrs, visited, pivots,
                        cluster_result);
        }
    }
}

void cuda_scan(int num_vs, int num_es, int *nbr_offs, int *nbrs,
        float epsilon, int mu, int num_blocks_per_grid, int num_threads_per_block,
        int &num_clusters, int *cluster_result) {

    // Fill in the cuda_scan function here
    
    // two steps: 1. find the pivots; 2. cluster and label the results
    int* d_nbr_offs, *d_nbrs;
    
    // declare all of the variabled that will be used
    bool* h_pivots, *d_pivots;
    int* h_num_sim_nbrs, *d_num_sim_nbrs;
    int* h_sim_nbrs, *d_sim_nbrs;
    
    // malloc for host and device variables
    size_t size_offs = (num_vs+1) * sizeof(int);
    size_t size_nbrs = (num_es+1) * sizeof(int);
    
    size_t size_pivots = num_vs * sizeof(bool);
    size_t size_num = num_vs * sizeof(int);
//    size_t size_sim = num_vs * sizeof(int*);
    
    cudaMalloc(&d_nbr_offs, size_offs);
    cudaMalloc(&d_nbrs, size_nbrs);
    
    
    h_pivots = (bool*)calloc(num_vs, sizeof(bool));
    cudaMalloc(&d_pivots, size_pivots);
    cudaMemset(d_pivots, 0, size_pivots);
    
    h_num_sim_nbrs = (int*)calloc(num_vs, sizeof(int));
    cudaMalloc(&d_num_sim_nbrs, size_num);
    cudaMemset(d_num_sim_nbrs, 0, size_num);
    
    h_sim_nbrs = (int*)malloc(size_nbrs);
    cudaMalloc(&d_sim_nbrs, size_nbrs);
    
    // copy the parameters to the device
    cudaMemcpy(d_nbr_offs, nbr_offs, size_offs, cudaMemcpyHostToDevice);
    cudaMemcpy(d_nbrs, nbrs, size_nbrs, cudaMemcpyHostToDevice);
    
    dim3 blocks(num_blocks_per_grid);
    dim3 threads(num_threads_per_block);

    int nthread = num_blocks_per_grid * num_threads_per_block;
    int start, end;
    // stage 1: find the pivot nodes
    for (int i = 0; i < num_vs/(nthread*100); ++i) {
        start = i*(nthread*100);
        end = (i + 1)*(nthread*100);
        if (end > num_vs)
            end = num_vs;
        findPivots<<<blocks, threads>>>(start, end, num_blocks_per_grid, num_threads_per_block, num_vs, num_es, epsilon, mu, d_nbr_offs, d_nbrs, d_pivots, d_num_sim_nbrs, d_sim_nbrs);
    }
    // copy the pivots results back from the device
    /*
        // for debug
        if (num_vs <= 50) {
            for (int i = 0; i < num_vs; ++i) {
                std::cout << pivots[i] << " ";
            }
            cout << endl;
            for (int i = 0; i < num_vs; ++i) {
                std::cout << "node " << i << ": ";
                for (int j = 0; j < num_sim_nbrs[i]; ++j) {
                    std::cout << sim_nbrs[i][j] << " ";
                }
                cout << endl;
            }

        }
    */

    cudaMemcpy(h_pivots, d_pivots, size_pivots, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_num_sim_nbrs, d_num_sim_nbrs, size_num, cudaMemcpyDeviceToHost);
    
    cudaMemcpy(h_sim_nbrs, d_sim_nbrs, size_nbrs, cudaMemcpyDeviceToHost);
    

    // stage 2: cluster. sequential version
    bool *visited = new bool[num_vs]();
    
    for (int i = 0; i < num_vs; i++) {
        if (!h_pivots[i] || visited[i]) continue;

        visited[i] = true;
        cluster_result[i] = i;
        expansion(i, i, nbr_offs, h_num_sim_nbrs, h_sim_nbrs, visited, h_pivots, cluster_result);

        num_clusters++;
    }
    
    // free mem allocation
    cudaFree(d_nbr_offs);
    cudaFree(d_nbrs);
    free(h_pivots);
    cudaFree(d_pivots);
    free(h_num_sim_nbrs);
    cudaFree(d_num_sim_nbrs);
    free(h_sim_nbrs);
    cudaFree(d_sim_nbrs);
}
