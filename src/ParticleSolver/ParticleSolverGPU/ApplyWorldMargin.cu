
#include <cuda_runtime.h>
#include "helper_math.h" 
#include "ApplyWorldMargin.h"



__global__ void applyMargin(float4* positions, float4* masses, float4 worldCenter, float worldRadius, int total_particles) {


    // Get the global thread index
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    // Check if the thread index is within the valid range of particles
    if (tid < total_particles) {
        float particleRadius = masses[tid].x;

        float4 pos = positions[tid];

        float4 vec_particle_world_center = pos-worldCenter;

        float distanceFromCenter = length(vec_particle_world_center);

        float maxRadius =  worldRadius - particleRadius;

        if(distanceFromCenter > maxRadius){
            float4 normalizedVec = normalize(vec_particle_world_center);
            positions[tid] -= normalizedVec * (distanceFromCenter - worldRadius + particleRadius);
        }
    }
}

void applyWorldCircleMargin(float4* positions, float4* masses, float2 worldCenter, float worldRadius, int numBlocks, int threadsPerBlock, int total_particles){
    applyMargin<<<numBlocks, threadsPerBlock>>>(positions, masses, make_float4(worldCenter.x, worldCenter.y, 0.f, 0.f), worldRadius, total_particles);
    cudaDeviceSynchronize();
}

