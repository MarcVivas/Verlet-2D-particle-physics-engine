#include "Bucket.h"
#include <iostream>

Bucket::Bucket(unsigned int bucketCapacity, unsigned int id) {
    this->bucketId = id;
    this->particlesIds = new glm::uvec4[bucketCapacity];
    this->numParticles = glm::uvec4(0);
}

Bucket::Bucket() = default;

Bucket::~Bucket() {
    delete [] this->particlesIds;
}

void Bucket::addParticle(ParticleSystem* particles, unsigned int particleId) {
    this->particlesIds[this->numParticles.x].x = particleId;
    this->numParticles.x += 1;
    
    glm::vec4 particlePos = particles->getPositions()[particleId];
}

unsigned int Bucket::getNumParticles() {
    return this->numParticles.x;
}

unsigned int Bucket::getParticleId(const unsigned int pos) {
    return this->particlesIds[pos].x;
}

glm::uvec4 * Bucket::getParticles() {return this->particlesIds;}


unsigned int Bucket::getBucketId(){
    return this->bucketId;
}

void Bucket::resetBucket() {
    this->numParticles = glm::uvec4(0);
}