
#include "ParticleSolver.h"
#include "GridGPU.h"
#ifndef N_BODY_PARTICLESOLVERGPU_H
#define N_BODY_PARTICLESOLVERGPU_H


class ParticleSolverGPU: public ParticleSolver {
public:
    ParticleSolverGPU(double block_size, float stepSize, glm::vec3 worldDimensions, GridGPU* grid);
    ~ParticleSolverGPU();
    bool usesGPU() override;
    void updateParticlePositions(ParticleSystem *particles) override;
    void updateParticlesPositions(ParticleSystem* particles, float4** cudaData) override;

protected:
    double blockSize;
    float stepSize;
    float4 force;
    float2 worldDim;
    float2 worldCenter;
    GridGPU *grid;
    float worldRadius;
   
};


#endif //N_BODY_PARTICLESOLVERGPU_H
