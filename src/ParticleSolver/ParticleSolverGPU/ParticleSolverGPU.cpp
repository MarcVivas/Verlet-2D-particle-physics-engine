
#include "ParticleSolverGPU.h"
#include "UpdatePositions.h"
#include "ApplyWorldMargin.h"

ParticleSolverGPU::ParticleSolverGPU(double block_size, float step_size, glm::vec3 worldDimensions) {
    this->blockSize = block_size;
    this->stepSize = step_size*step_size;
    this->force = make_float4(0.f, -7000.2f * 7000.2f, 0.f, 0.f);
    this->worldDim = make_float2(worldDim.x, worldDim.y);
    this->worldCenter = make_float2(worldDim.x /2.f, worldDim.y/2.f);
    this->worldRadius = worldDim.x - 0.13 * worldDim.x;
}

void ParticleSolverGPU::updateParticlesPositions(ParticleSystem* particles, float4** cudaData) {
    int numBlocks = ceil(particles->size() / blockSize);
    
    float4* positions = cudaData[0];
    float4* previousPositions = cudaData[5];
    float4* velocities = cudaData[1];
    float4* masses = cudaData[3];

    applyWorldCircleMargin(positions, masses, worldCenter, worldRadius, numBlocks, blockSize, particles->size());
    updatePositions(positions, previousPositions, velocities, force, stepSize, particles->size(), numBlocks, blockSize);
}


void ParticleSolverGPU::updateParticlePositions(ParticleSystem *particles){}


ParticleSolverGPU::~ParticleSolverGPU() noexcept {

}

bool ParticleSolverGPU::usesGPU() {return true;}

