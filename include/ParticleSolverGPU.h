
#include "ParticleSolver.h"

#ifndef N_BODY_PARTICLESOLVERGPU_H
#define N_BODY_PARTICLESOLVERGPU_H


class ParticleSolverGPU: public ParticleSolver {
public:
    ParticleSolverGPU(double block_size, float stepSize);
    ~ParticleSolverGPU();
    bool usesGPU() override;
    void updateParticlePositions(ParticleSystem *particles) override;
    void updateParticlesPositions(ParticleSystem* particles, float4** cudaData) override;
protected:
    double blockSize;
    float stepSize;
    float4 force;
   
};


#endif //N_BODY_PARTICLESOLVERGPU_H
