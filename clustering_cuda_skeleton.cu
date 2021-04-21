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

// Define variables or functions here

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


__global__
void findPivots(int num_vs, int num_es, float ep, int mu, int* d_nbr_offs, int* d_nbrs, int* d_pivots, int* d_num_sim_nbrs, int** d_sim_nbrs) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    int element_skip = blockDim.x * gridDim.x;
    for (int i = tid; i < num_vs; i += element_skip) {
        int left_start = d_nbrs[d_nbr_offs[i]];
        int left_end = d_nbrs[d_nbr_offs[i + 1]];
        int left_size = left_end - left_start;
        
        d_sim_nbrs[i] = new int[left_size];
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
                d_sim_nbrs[i][d_num_sim_nbrs[i]] = nbr_id;
                d_num_sim_nbrs[i]++;
            }
        }
        if (d_num_sim_nbrs[i] > mu) {
            d_pivots[i] = true;
        }
    }
}

void cuda_scan(int num_vs, int num_es, int *nbr_offs, int *nbrs,
        float epsilon, int mu, int num_blocks_per_grid, int num_threads_per_block,
        int &num_clusters, int *cluster_result) {

    // Fill in the cuda_scan function here
    
    // two steps: 1. find the pivots; 2. cluster and label the results
    int* d_nbr_offs, d_nbrs;
    int* d_cluster_result;
    
    // declare all of the variabled that will be used
    bool* h_pivots, d_pivots;
    int* h_num_sim_nbrs, d_num_sim_nbrs;
    int** h_sim_nbrs, d_sim_nbrs;
    
    // malloc for host and device variables
    size_t size_offs = (num_vs+1) * sizeof(int);
    size_t size_nbrs = (num_es+1) * sizeof(int);
    size_t size_results = num_vs * sizeof(int);
    
    size_t size_pivots = num_vs * sizeof(bool);
    size_t size_num = num_vs * sizeof(int);
    size_t size_sim = num_vs * sizeof(int*);
    
    cudaMalloc(&d_nbr_offs, size_offs);
    cudaMalloc(&d_nbrs, size_nbrs);
    
    cudaMalloc(&d_cluster_result, size_results);
    
    h_pivots = (bool*)calloc(num_vs, sizeof(bool));
    cudaMalloc(&d_pivots, size_pivots);
    cudaMemset(d_pivots, 0, size_pivots);
    
    h_num_sim_nbrs = (int*)calloc(num_vs, sizeof(int));
    cudaMalloc(&d_num_sim_nbrs, size_num);
    cudaMemset(d_num_sim_nbrs, 0, size_num);
    
    h_sim_nbrs = (int**)malloc(size_sim);
    cudaMalloc(&d_sim_nbrs, size_sim);
    
    dim3 blocks(num_blocks_per_grid);
    dim3 threads(num_threads_per_block);

    findPivots<<<blocks, threads>>>(num_vs, num_es, epsilon, mu, d_nbr_offs, d_nbrs, d_pivots, d_num_sim_nbrs, d_sim_nbrs);
    // copy parameters from host to device
    cudaMemcpy(d_nbr_offs, nbr_offs, size_offs, cudaMemcpyHostToDevice);
    cudaMemcpy(d_nbrs, nbrs, size_nbrs, cudaMemcpyHostToDevice);
}
