
#include "ParticleSolverGPU.h"
#include "GridGPU.h"
#include "UpdatePositions.h"
#include "ApplyWorldMargin.h"
#include <iostream>

ParticleSolverGPU::ParticleSolverGPU(double block_size, float step_size, glm::vec3 worldDimensions, GridGPU *gridGpu) {
    this->blockSize = block_size;
    this->stepSize = step_size;
    this->force = make_float4(0.f, -9.8f, 0.f, 0.f);
    this->worldDim = make_float2(worldDimensions.x, worldDimensions.y);
    this->worldCenter = make_float2(worldDimensions.x /2.f, worldDimensions.y/2.f);
    this->worldRadius = worldDimensions.x - 0.13 * worldDimensions.x;
    this->grid = gridGpu;
}

void ParticleSolverGPU::updateParticlesPositions(ParticleSystem* particles, float4** cudaData) {
    int numBlocks = ceil(particles->size() / blockSize);
    
    float4* positions = cudaData[0];
    float4* previousPositions = cudaData[5];
    float4* velocities = cudaData[1];
    float4* masses = cudaData[3];
    
    grid->reset(blockSize);

    applyWorldCircleMargin(positions, previousPositions, masses, velocities, worldCenter, worldRadius, numBlocks, blockSize, particles->size());

    grid->build(blockSize, particles->size(), positions);

    grid->collisions(blockSize, positions, masses, particles->size(), velocities);

    updatePositions(positions, previousPositions, velocities, force, stepSize, particles->size(), numBlocks, blockSize);
    
}


void ParticleSolverGPU::updateParticlePositions(ParticleSystem *particles){}


ParticleSolverGPU::~ParticleSolverGPU() noexcept {
    delete this->grid;
}

bool ParticleSolverGPU::usesGPU() {return true;}

