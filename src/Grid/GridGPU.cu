#include "GridGPU.h"
#include <iostream>
#include "helper_math.h" 


__host__ GridGPU::GridGPU(float2 worldDim, unsigned int bucketCap, unsigned int bucketsPerDim) {
    this->bucketsPerDimension = bucketsPerDim;
    this->totalBuckets = bucketsPerDimension * bucketsPerDimension;
    this->worldDimensions = worldDim;
    this->bucketCapacity = bucketCap;
    this->bucketSize = worldDimensions.x / (float) bucketsPerDimension;

    this->bucketsParticleIds = new int[totalBuckets * bucketCapacity];
    cudaMalloc(&devBucketsParticleIds, sizeof(int) * totalBuckets * bucketCapacity);
    
    this->bucketsNumParticles = new int[totalBuckets];
    memset(bucketsNumParticles, 0, sizeof(int) * totalBuckets);
    cudaMalloc(&devBucketsNumParticles, sizeof(int) * totalBuckets);
    cudaMemcpy(devBucketsNumParticles, bucketsNumParticles, sizeof(int) * totalBuckets, cudaMemcpyHostToDevice);

    minDimensions = make_int2(0, 0);
    maxDimensions = make_int2(bucketsPerDimension - 1, bucketsPerDimension - 1);

    
}

__global__ void resetGrid(int* bucketsNumParticles, int totalBuckets) {

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid < totalBuckets) {
        bucketsNumParticles[tid] = 0;
    }
}

void GridGPU::reset(double blockSize) {
    int totalBlocks = ceil((double) totalBuckets / blockSize);
    resetGrid << <totalBlocks, blockSize >> > (devBucketsNumParticles, totalBuckets);
    cudaDeviceSynchronize();
}




__host__ GridGPU::~GridGPU(){
    delete[] this->bucketsNumParticles;
    delete[] this->bucketsParticleIds;
    cudaFree(devBucketsNumParticles);
    cudaFree(devBucketsParticleIds);
}



__device__ int getBucketIdByPosition(float4 particlePos, int2 minDimensions, int2 maxDimensions, float bucketSize, int bucketsPerDimension) {
    const int2 bucketCoords = clamp(
        make_int2(particlePos.x / bucketSize, particlePos.y / bucketSize),
        minDimensions,
        maxDimensions
    );
    return bucketCoords.x + bucketCoords.y * bucketsPerDimension;
}


__global__ void buildGrid(int totalParticles, float4 *positions, int* bucketsNumParticles, int* bucketsParticlesIds, int2 minDim, int2 maxDim, float bucketSize, int bucketsPerDimension, int bucketCapacity) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < totalParticles) {
        int bucketId = getBucketIdByPosition(positions[tid], minDim, maxDim, bucketSize, bucketsPerDimension);
    
        // Add the particle id to the bucket
        int offset = atomicAdd(&bucketsNumParticles[bucketId], 1);

        bucketsParticlesIds[offset + bucketCapacity * bucketId] = tid;
    }
}


__host__ void GridGPU::build(double blockSize, int totalParticles, float4* devPositions) {

    int totalBlocks = ceil((double)totalParticles / blockSize);
    buildGrid << <totalBlocks, blockSize >> > (
        totalParticles, 
        devPositions,
        devBucketsNumParticles, 
        devBucketsParticleIds,
        minDimensions, 
        maxDimensions, 
        bucketSize, 
        bucketsPerDimension,
        bucketCapacity
        );
    cudaDeviceSynchronize();

}

__device__ int getOtherBucketId(int bucketId, int otherBucketOffset, int rowLength) {
    int offsets[] = { -(rowLength + 1), -rowLength, -(rowLength - 1), -1, 1, rowLength - 1, rowLength, rowLength + 1, 0 };
    return bucketId + offsets[otherBucketOffset];

}

#define PERCENT 0.2900f // usually 0.2 to 0.8
#define SLOP 0.04f // usually 0.01 to 0.1 

__device__ void solveCollision(float4 particle_1_radius, float4 particle_2_radius, float4 particle_1_pos, float4 particle_2_pos, float4* positions, int particleId, int otherParticleId, float4 * vels) {

    float4 vec_i_j = particle_1_pos - particle_2_pos;

    // Direction of collision
    float4 collisionVectorDirection = normalize(vec_i_j);
    
    // Distance between both particles
    float current_distance = length(vec_i_j);
   

    float inv_mass1 = 1.f / (particle_1_radius.x * particle_1_radius.x);
    float inv_mass2 = 1.f / (particle_2_radius.x * particle_2_radius.x);

    float penetration = (particle_1_radius.x + particle_2_radius.x) - current_distance;

    float4 correction = (fmax(penetration - SLOP, 0.f) / (inv_mass1 + inv_mass2) ) * PERCENT * collisionVectorDirection;

    atomicAdd(&positions[particleId].x, (inv_mass1) * correction.x);
    atomicAdd(&positions[particleId].y, (inv_mass1) * correction.y);

    atomicAdd(&positions[otherParticleId].x, -(inv_mass2) * correction.x);
    atomicAdd(&positions[otherParticleId].y, -(inv_mass2) * correction.y);
}

__device__ bool areColliding(float4 particle_1_radius, float4 particle_2_radius, float4 particle_1_pos, float4 particle_2_pos) {
    float4 vec_i_j = particle_2_pos - particle_1_pos;
    return powf(particle_1_radius.x + particle_2_radius.x, 2.f) > dot(vec_i_j, vec_i_j);
}


__global__ void solveGridCollisions(float4* positions, float4* radius, float4* velocities,int* bucketsNumParticles, int* bucketsParticlesIds, int totalParticles, int totalBuckets, int bucketCapacity, int2 minDimensions, int2 maxDimensions, float bucketSize, int bucketsPerDimension) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < totalParticles) {
        float4 pos = positions[tid];
        float4 rad = radius[tid];
        int bucketId = getBucketIdByPosition(pos, minDimensions, maxDimensions, bucketSize, bucketsPerDimension);
#pragma unroll
        for (int i = 0; i < 9; i++) {
            int otherBucketId = bucketId < totalBuckets ? getOtherBucketId(bucketId, i, bucketsPerDimension) : totalBuckets+4;
            if (otherBucketId < totalBuckets) {
                int otherBucketTotalParticles = bucketsNumParticles[otherBucketId];
                for (int j = 0; j < otherBucketTotalParticles; j++) {
                    int otherParticleId = bucketsParticlesIds[j + otherBucketId * bucketCapacity];
                    if (tid < otherParticleId && otherParticleId < totalParticles) {
                        float4 particle_2_radius = radius[otherParticleId];
                        float4 particle_2_pos = positions[otherParticleId];
                        if (areColliding(rad, particle_2_radius, pos, particle_2_pos)) {
                            solveCollision(rad, particle_2_radius, pos, particle_2_pos, positions, tid, otherParticleId, velocities);
                        }
                    }
                }
            }
        }
    }
}


__host__ void GridGPU::collisions(double blockSize, float4* devPositions, float4* devRadius, int totalParticles, float4* devVelocities) {
    
    // Bucket collisions
    int totalBlocks = ceil((double)totalParticles / blockSize);
    solveGridCollisions << <totalBlocks, blockSize >> > (
        devPositions,
        devRadius,
        devVelocities,
        devBucketsNumParticles,
        devBucketsParticleIds,
        totalParticles,
        totalBuckets,
        bucketCapacity,
        minDimensions,
        maxDimensions,
        bucketSize,
        bucketsPerDimension
        );
    cudaDeviceSynchronize();
}