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
void findPivots() {
    
}

void cuda_scan(int num_vs, int num_es, int *nbr_offs, int *nbrs,
        float epsilon, int mu, int num_blocks_per_grid, int num_threads_per_block,
        int &num_clusters, int *cluster_result) {

    // Fill in the cuda_scan function here
    dim3 blocks(num_blocks_per_grid);
    dim3 threads(num_threads_per_block);
    
    // two steps: 1. find the pivots; 2. cluster and label the results
    
}
