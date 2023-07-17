
#include <cuda_runtime.h>
#include "UpdatePositions.h"

// The CUDA kernel definition
//__global__ void updatePos(float4* positions, float4* accelerations, float4* previousPositions, float4* velocities,  float4* forces, float delta_time, int total_particles) {
__global__ void updatePos(float4* positions, float4* previousPositions, float4* velocities, float4 force, float delta_time, int total_particles) {

    // This is the CUDA kernel code that updates the particle positions.
    // You can access and modify the particle data here.

    // Get the global thread index
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    // Check if the thread index is within the valid range of particles
    if (tid < total_particles) {
        positions[tid].x += 10.f;
        float4 pos = positions[tid];
        float4 prev = previousPositions[tid];

        // Compute velocity
        float4 vel = make_float4(pos.x - prev.x, pos.y - prev.y, 0.f, 0.f);
        velocities[tid] = vel;

        // Save the current position
        previousPositions[tid] = pos;

        // Update acceleration
        // F = MA;
        // A = F/M
        
        // Update position
        positions[tid] = make_float4(pos.x + vel.x, pos.y + vel.y + force.y * delta_time, 0.f, 0.f); 
    }
}


void updatePositions(float4* positions, float4* previousPositions, float4* velocities, float4 force, float delta_time, int total_particles, int numBlocks, int threadsPerBlock){
    updatePos<<<numBlocks, threadsPerBlock>>>(positions, previousPositions, velocities, force, delta_time, total_particles);
    cudaDeviceSynchronize();
}

