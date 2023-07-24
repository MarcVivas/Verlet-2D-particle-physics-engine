
#include <cuda_runtime.h>
#include "helper_math.h" 
#include "AddParticle.h"



__global__ void add(float4* positions, float4* masses, float4*accelerations, float4*velocities, float4*forces, float4*previousPositions, float4 pos, float4 acc, float4 vel, float4 f, int total_particles) {
    // Get the global thread index
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    // Check if the thread index is within the valid range of particles
    if (tid == 0) {
        int i = total_particles - 1;
        positions[i] = pos;
        accelerations[i] = acc;
        velocities[i] = vel;
        forces[i] = f;
        previousPositions[i] = pos;
     }
}

__host__ void addNewParticle(float4* positions, float4* masses, float4*accelerations, float4*velocities, float4*forces, float4*previousPositions, float4 pos, float4 acc, float4 vel, float4 f, int total_particles){
    add<<<1, 1>>>(positions, masses, accelerations, velocities, forces, previousPositions, pos, acc, vel, f, total_particles);
    cudaDeviceSynchronize();
}

