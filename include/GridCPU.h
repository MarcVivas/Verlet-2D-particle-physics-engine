#include <omp.h>
#include "Grid.h"

#ifndef N_BODY_GRIDCPU_H
#define N_BODY_GRIDCPU_H


class GridCPU: public Grid{
public:
    omp_lock_t* bucketLocks;
    GridCPU(glm::vec3 worldDimensions, unsigned int bucketCapacity, unsigned int bucketsPerDimension);
    void updateGrid(ParticleSystem *particleSystem, int i) override;
    ~GridCPU() override;
    void resetGrid() override;
    Bucket* getBucketByPosition(glm::vec4 particlePos) override;
    Bucket* getBucketById(unsigned int bucketId) override;
    unsigned int getTotalBuckets() override;
    int getRowLength();
};


#endif //N_BODY_GRIDCPU_H
