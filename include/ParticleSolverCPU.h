
#include "ParticleSolver.h"
#include "GridCPU.h"

#ifndef N_BODY_PARTICLESOLVERCPUSEQUENTIAL_H
#define N_BODY_PARTICLESOLVERCPUSEQUENTIAL_H


class ParticleSolverCPU: public ParticleSolver  {
public:
    ParticleSolverCPU(GridCPU *grid, float timeStep, glm::vec3 worldDim);
    void updateParticlePositions(ParticleSystem *particles) override;
    bool usesGPU() override;
protected:
    float G;
    GridCPU *gridCpu;
    float timeStep;
    void computeForce(ParticleSystem *particles, const unsigned int particleId);
    void solveBucketCollisions(ParticleSystem *particles, const unsigned int bucketId);
    bool areColliding(ParticleSystem *particles, const unsigned int i, const unsigned int j);
    void solveCollision(ParticleSystem *particles, const unsigned int i, const unsigned int j);
    void applyWorldMargin(ParticleSystem *particles, unsigned int particleId);
    glm::vec3 worldDimensions;
    glm::vec4 worldCenter;
    float worldRadius;
    void solveParticleCollisions(ParticleSystem *particles, unsigned int particleId, unsigned int bucketId);

};


#endif //N_BODY_PARTICLESOLVERCPUSEQUENTIAL_H
