#include <cuda_runtime.h>


#ifndef N_BODY_GRIDGPU_H
#define N_BODY_GRIDGPU_H


class GridGPU{
public:
    __host__ GridGPU(float2 worldDimensions, unsigned int bucketCapacity, unsigned int bucketsPerDimension);
    ~GridGPU();
    float2 worldDimensions;
    float bucketSize;
    int bucketsPerDimension;
    int2 maxDimensions;
    int2 minDimensions;
    int bucketCapacity;
    int totalBuckets;
    int* bucketsParticleIds;
    int* bucketsNumParticles;
    int* devBucketsParticleIds;
    int* devBucketsNumParticles;
    int blockSize;
    __host__ void reset(double blockSize);
    __host__ void build(double blockSize, int totalParticles, float4 * devPositions);
    __host__ void collisions(double blockSize, float4* devPositions, float4* devRadius, int totalParticles, float4* devVelocities);


};


#endif //N_BODY_GRIDGPU_H
