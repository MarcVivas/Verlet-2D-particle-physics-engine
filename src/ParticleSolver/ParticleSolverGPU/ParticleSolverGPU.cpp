
#include "ParticleSolverGPU.h"
#include "UpdatePositions.h"


ParticleSolverGPU::ParticleSolverGPU(double block_size, float step_size) {
    this->blockSize = block_size;
    this->stepSize = step_size*step_size;
    this->force = make_float4(0.f, -7000.2f * 7000.2f, 0.f, 0.f);
}

void ParticleSolverGPU::updateParticlesPositions(ParticleSystem* particles, float4** cudaData) {
    int numBlocks = ceil(particles->size() / blockSize);
    
    float4* positions = cudaData[0];
    float4* previousPositions = cudaData[5];
    float4* velocities = cudaData[1];
    updatePositions(positions, previousPositions, velocities, force, numBlocks, blockSize, stepSize, particles->size());
}


void ParticleSolverGPU::updateParticlePositions(ParticleSystem *particles){}


ParticleSolverGPU::~ParticleSolverGPU() noexcept {

}

bool ParticleSolverGPU::usesGPU() {return true;}

