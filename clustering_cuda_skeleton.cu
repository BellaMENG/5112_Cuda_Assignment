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

__global__
void findPivots(int num_vs, int num_es) {
    
}

void cuda_scan(int num_vs, int num_es, int *nbr_offs, int *nbrs,
        float epsilon, int mu, int num_blocks_per_grid, int num_threads_per_block,
        int &num_clusters, int *cluster_result) {

    // Fill in the cuda_scan function here
    dim3 blocks(num_blocks_per_grid);
    dim3 threads(num_threads_per_block);
    
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
    
    // copy parameters from host to device
    cudaMemcpy(d_nbr_offs, nbr_offs, size_offs, cudaMemcpyHostToDevice);
    cudaMemcpy(d_nbrs, nbrs, size_nbrs, cudaMemcpyHostToDevice);
}
