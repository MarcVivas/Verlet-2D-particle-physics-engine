
#include <cuda_runtime.h>
#include "UpdatePositions.h"
#include "helper_math.h" 
#include <iostream>
// The CUDA kernel definition
__global__ void updatePos(float4* positions, float4* previousPositions, float4* velocities, float4 force, float delta_time, int total_particles) {

    // This is the CUDA kernel code that updates the particle positions.

    // Get the global thread index
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    // Check if the thread index is within the valid range of particles
    if (tid < total_particles) {
        float4 pos = positions[tid];
        float4 prev = previousPositions[tid];
        
      
        previousPositions[tid] = pos;

        //// Compute velocity
        float4 vel = (pos - prev) + force * delta_time * delta_time;
        velocities[tid] = vel;
        
        // Update position
        positions[tid] = make_float4(pos.x + vel.x, pos.y + vel.y  , 0.f, 0.f); 
    }
}


__host__ void updatePositions(float4* positions, float4* previousPositions, float4* velocities, float4 force, float delta_time, int total_particles, int numBlocks, int threadsPerBlock){
    updatePos<<<numBlocks, threadsPerBlock>>>(positions, previousPositions, velocities, force, delta_time, total_particles);
    cudaDeviceSynchronize();
}

