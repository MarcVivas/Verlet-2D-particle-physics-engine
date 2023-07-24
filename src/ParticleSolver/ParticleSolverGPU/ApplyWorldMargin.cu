
#include <cuda_runtime.h>
#include "helper_math.h" 
#include "ApplyWorldMargin.h"
#include <iostream>


__global__ void applyMargin(float4* positions, float4* previousPositions, float4* masses, float4* velocities, float4 worldCenter, float worldRadius, int total_particles) {


    // Get the global thread index
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    // Check if the thread index is within the valid range of particles
    if (tid < total_particles) {
        float particleRadius = masses[tid].x;

        float4 pos = positions[tid];

        float4 vec_particle_world_center = pos-worldCenter;

        float distanceFromCenter = dot(vec_particle_world_center, vec_particle_world_center);

        float maxRadius =  worldRadius - particleRadius;

        if(distanceFromCenter > maxRadius * maxRadius){
            // Move the particle back to the nearest point on the circle's edge
            positions[tid] = worldCenter + maxRadius * normalize(vec_particle_world_center);
        }

    }
}

__host__ void applyWorldCircleMargin(float4* positions, float4* previousPositions, float4* masses, float4* velocities, float2 worldCenter, float worldRadius, int numBlocks, int threadsPerBlock, int total_particles){
    applyMargin<<<numBlocks, threadsPerBlock>>>(positions, previousPositions, masses, velocities, make_float4(worldCenter.x, worldCenter.y, 0.f, 0.f), worldRadius, total_particles);
    cudaDeviceSynchronize();
}

