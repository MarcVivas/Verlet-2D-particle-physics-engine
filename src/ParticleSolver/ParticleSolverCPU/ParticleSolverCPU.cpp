
#include "ParticleSolverCPU.h"
#include <glm/gtx/norm.hpp>
#include <iostream>

ParticleSolverCPU::ParticleSolverCPU(GridCPU *grid, float stepSize, glm::vec3 worldDim): ParticleSolver() {
    this->timeStep = stepSize;
    this->G = -7000.2f*7000.2f;
    this->worldDimensions = worldDim;
    this->worldCenter = glm::vec4(worldDim, 0.f) / 2.f;
    this->worldRadius = worldDimensions.x - 0.13 * worldDimensions.x;
    this->gridCpu = grid;
}

void ParticleSolverCPU::updateParticlePositions(ParticleSystem *particles){
    int subSteps = 1;

    for(int k = 0; k < subSteps; k++){
        gridCpu->resetGrid();

        #pragma omp parallel for schedule(static)
        for(int i = 0; i<particles->size(); i++){
            gridCpu->updateGrid(particles, i);
            computeForce(particles, i);
            applyWorldMargin(particles, i);
        }

        #pragma omp parallel for schedule(dynamic)
        for (int subGrid = 0; subGrid < gridCpu->getRowLength(); subGrid += 3) {
            bool enough = false;
            for (auto i = 0; i < 3; i++) {
                #pragma omp task
                {
                    for (auto j = 0; j < gridCpu->getRowLength(); j++) {
                        auto bucketId = subGrid + i + gridCpu->getRowLength() * j;
                        if (bucketId >= gridCpu->getTotalBuckets()) {
                            enough = true;
                            break;
                        }
                        solveBucketCollisions(particles, bucketId);
                    }
                }
                #pragma omp taskwait
                if (enough) break;
            }
        }

        #pragma omp parallel for schedule(static)
        for(int i = 0; i < particles->size(); i++){
            particles->updateParticlePosition(i, this->timeStep / (float)subSteps);
        }
    }


}

void ParticleSolverCPU::applyWorldMargin(ParticleSystem *particles, unsigned int particleId) {
    float particleRadius = particles->getMasses()[particleId].x;

    glm::vec4 vec_particle_world_center = particles->getPositions()[particleId] - worldCenter;

    float distanceFromCenter = glm::length(vec_particle_world_center);

    float maxRadius =  worldRadius - particleRadius;

    if(distanceFromCenter > maxRadius){
        glm::vec4 normalizedVec = glm::normalize(vec_particle_world_center);
        particles->getPositions()[particleId] -= normalizedVec * (distanceFromCenter - worldRadius + particleRadius);
    }
}

void
ParticleSolverCPU::computeForce(ParticleSystem *particles, const unsigned int particleId) {
    particles->getForces()[particleId] = glm::vec4 (0.f, G, 0.0f, 0.0f);
}

void
ParticleSolverCPU::solveCollision(ParticleSystem *particles, const unsigned int i, const unsigned int j) {
    glm::vec4 particle_1_pos = particles->getPositions()[i];
    glm::vec4 particle_2_pos = particles->getPositions()[j];
    float particle_1_radius = particles->getMasses()[i].x;
    float particle_2_radius = particles->getMasses()[j].x;
    glm::vec4 vec_i_j = particle_1_pos - particle_2_pos;


    // Direction of collision
    glm::vec4 collisionVectorDirection = glm::normalize(vec_i_j);

    // Distance between both particles
    float current_distance = glm::length(vec_i_j);

    // Minimum distance when there's no collision
    float desired_distance =  particle_1_radius + particle_2_radius;

    // Distance to move both particles
    float distance_to_move = (desired_distance - current_distance) * 0.5f;

    particles->getPositions()[i] = particle_1_pos + (collisionVectorDirection * distance_to_move);
    particles->getPositions()[j] = particle_2_pos - (collisionVectorDirection * distance_to_move);
}

void ParticleSolverCPU::solveBucketCollisions(ParticleSystem *particles, unsigned int bucketId) {
    auto bucket = gridCpu->getBucketById(bucketId);
    glm::uvec4* bucketParticles = bucket->getParticles();
    for(auto i = 0; i < bucket->getNumParticles(); i++){
        auto particleId = bucketParticles[i].x;
        solveParticleCollisions(particles, particleId, bucketId);
    }
}

void ParticleSolverCPU::solveParticleCollisions(ParticleSystem *particles, unsigned int particleId,
                                                          unsigned int bucketId) {
    // Get the bucket where the particle is located
    Bucket* bucket = this->gridCpu->getBucketById(bucketId);

    // Compute collisions with other buckets
    Bucket *otherBucket = nullptr;
    int rowLength = gridCpu->getRowLength();
    int currentBucketId = bucket->getBucketId();

    int offsets[] = {-(rowLength+1), -rowLength, -(rowLength-1), -1, 1, rowLength-1, rowLength, rowLength+1};

    for(int j = 0; j < 8; j++){
        int otherBucketId = currentBucketId + offsets[j];

        if(otherBucketId >= gridCpu->getTotalBuckets() || otherBucketId < 0){
            continue;
        }

        // Vertices buckets
        if((rowLength-1)*0 == currentBucketId || (rowLength-1)*1 == currentBucketId || (rowLength-1)*rowLength == currentBucketId || (rowLength-1)*(rowLength+1)==currentBucketId){
            if((rowLength-1)*0 == currentBucketId && (j != 4 && j!=6 && j!=7)){
                continue;
            }
            else if((rowLength-1)*1 == currentBucketId && (j!= 3 && j!=5 && j!=6)){
                continue;
            }
            else if((rowLength-1)*rowLength == currentBucketId && (j!= 1 && j!=2 && j!=4)){
                continue;
            }
            else if((rowLength-1)*(rowLength+1) == currentBucketId && (j!= 0 && j!=1 && j!=3)){
                continue;
            }
        }

        if(0 == currentBucketId % rowLength && (j != 1 && j!=2 && j!=4 && j!=6 && j!=7)){
            continue;
        }

        if(rowLength-1 == currentBucketId % rowLength && (j != 0 && j!=1 && j!=3 && j!=5 && j!=6)){
            continue;
        }

        otherBucket = this->gridCpu->getBucketById(otherBucketId);

        for(size_t k = 0; k < otherBucket->getNumParticles(); k++){
            const unsigned int otherParticleId = otherBucket->getParticleId(k);
            if(otherParticleId != particleId){
                if(areColliding(particles, particleId, otherParticleId)){
                    solveCollision(particles, particleId, otherParticleId);
                }
            }
        }
    }


    // Compute collisions inside the bucket
    for(size_t j = 0; j < bucket->getNumParticles(); j++){
        const unsigned int otherParticleId = bucket->getParticleId(j);
        if(otherParticleId != particleId){
            if(areColliding(particles, particleId, otherParticleId)){
                solveCollision(particles, particleId, otherParticleId);
            }
        }
    }


    /*
    for(size_t i = 0; i < 1; i++){
        for(size_t j = 0; j < particles->size(); j++){
            if(j != particleId && areColliding(particles, particleId, j)){
#pragma omp critical
                {
                    solveCollision(particles, particleId, j);
                }
            }
        }
    }
     */
}

bool ParticleSolverCPU::areColliding(ParticleSystem *particles, const unsigned int i, const unsigned int j){
    return std::pow(particles->getMasses()[i].x + particles->getMasses()[j].x, 2) >  glm::distance2(particles->getPositions()[i], particles->getPositions()[j]);
}


bool ParticleSolverCPU::usesGPU() {return false;}
