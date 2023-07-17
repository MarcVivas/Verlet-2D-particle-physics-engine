
#include "ParticleSolverGPU.h"
#include <cuda_runtime.h>


ParticleSolverGPU::ParticleSolverGPU(double block_size, float step_size) {
    this->blockSize = block_size;
    this->stepSize = step_size;
    
}

void ParticleSolverGPU::updateParticlePositions(ParticleSystem *particles){
    int numBlocks = ceil(particles->size() / blockSize);

    //updatePositions<<<numBlocks,blockSize>>>(particles->getPositions(), particles->getAccelerations(), particles->getPreviousPositions(), particles->getVelocities(), particles->getForces(), stepSize * stepSize, particles->size());
    updatePositions<<<numBlocks,blockSize>>>(stepSize * stepSize, particles->size());

    cudaDeviceSynchronize();

}


ParticleSolverGPU::~ParticleSolverGPU() noexcept {

}

bool ParticleSolverGPU::usesGPU() {return true;}

// The CUDA kernel definition
//__global__ void updatePositions(glm::vec4* positions, glm::vec4* accelerations, glm::vec4* previousPositions, glm::vec4* velocities,  glm::vec4* forces, float delta_time, int total_particles) {
__global__ void updatePositions(float delta_time, int total_particles) {

    // This is the CUDA kernel code that updates the particle positions.
    // You can access and modify the particle data here.

    // Get the global thread index
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    // Check if the thread index is within the valid range of particles
    if (tid < total_particles) {
        //// Compute velocity
        //velocities[tid] = positions[tid] - previousPositions[tid];

        //// Save the current position
        //previousPositions[tid] = positions[tid];

        //// Update acceleration
        //// F = MA;
        //// A = F/M
        //accelerations[tid] = forces[tid];

        //// Update position
        //positions[tid] += velocities[tid] + accelerations[tid] * delta_time;
    }
}