#include "ParticleSystem.h"
#include <iostream>
#include <random>
#include <chrono>

ParticleSystem::ParticleSystem(std::vector<Particle> &particles) {
    this->maxParticles = 1000000;
    this->numParticles = particles.size();
    this->velocities = new glm::vec4[maxParticles]();
    this->accelerations = new glm::vec4[maxParticles]();
    this->positions = new glm::vec4[maxParticles]();
    this->masses = new glm::vec4[maxParticles]();
    this->forces = new glm::vec4[maxParticles]();
    this->previousPositions = new glm::vec4[maxParticles]();

    for (int i = 0; i < maxParticles; i++) {
        unsigned int index = i % numParticles;
        this->velocities[i] = particles[index].velocity;
        this->accelerations[i] = particles[index].acceleration;
        this->positions[i] = particles[index].position;
        this->masses[i] = glm::vec4(particles[index].mass, 0, 0, 0);
        this->forces[i] = glm::vec4(0.f);
        this->previousPositions[i] = particles[index].position;
    }
}

/**
 * Updates a particle position
 * Performs Verlet integration
 * @param particleId
 * @param deltaTime
 */
void ParticleSystem::updateParticlePosition(unsigned int particleId, float deltaTime) {
    // Compute velocity
    this->velocities[particleId] = positions[particleId] - previousPositions[particleId];

    // Save the current position
    previousPositions[particleId] = positions[particleId];

    // Update acceleration
    // F = MA;
    // A = F/M
    this->accelerations[particleId] = this->forces[particleId];

    // Update position
    this->positions[particleId] += this->velocities[particleId] + this->accelerations[particleId] * deltaTime * deltaTime;
}

void ParticleSystem::addParticle(Particle particle) {
    this->numParticles++;

    std::cout << "Adding particle " <<numParticles << std::endl;

    if(numParticles < maxParticles){
        this->accelerations[numParticles-1] = particle.acceleration;
        this->previousPositions[numParticles-1] = particle.position;
        this->velocities[numParticles-1] = particle.velocity;
        this->positions[numParticles-1] = particle.position;
        this->forces[numParticles-1] = glm::vec4(0);
    }
}

unsigned int ParticleSystem::size() const {
    return this->numParticles;
}

glm::vec4* ParticleSystem::getMasses(){
    return this->masses;
}


glm::vec4* ParticleSystem::getPositions() {
    return this->positions;
}


glm::vec4* ParticleSystem::getVelocities() {
    return this->velocities;
}

glm::vec4* ParticleSystem::getAccelerations() {
    return this->accelerations;
}

glm::vec4* ParticleSystem::getForces() {
    return this->forces;
}

void ParticleSystem::setAccelerations(glm::vec4 *newAccelerations) {
    delete [] this->accelerations;
    this->accelerations = newAccelerations;
}

void ParticleSystem::setMasses(glm::vec4 *newMasses) {
    delete [] this->masses;
    this->masses = newMasses;
}

void ParticleSystem::setPositions(glm::vec4 *newPositions) {
    delete [] this->positions;
    this->positions = newPositions;
}

void ParticleSystem::setVelocities(glm::vec4 *newVelocities) {
    delete [] this->velocities;
    this->velocities = newVelocities;
}

glm::vec4* ParticleSystem::getPreviousPositions() {
    return previousPositions;
}

std::ostream& operator<<(std::ostream& os, const ParticleSystem& system) {
    os << "Particle System with " << system.numParticles << " particles:" << std::endl;
    for (unsigned int i = 0; i < system.numParticles; ++i) {
        os << "Particle ID: " << i << std::endl;
        os << "Position: (" << system.positions[i].x << ", " << system.positions[i].y << ", " << system.positions[i].z << ")" << std::endl;
        os << "Velocity: (" << system.velocities[i].x << ", " << system.velocities[i].y << ", " << system.velocities[i].z << ")" << std::endl;
        os << "Acceleration: (" << system.accelerations[i].x << ", " << system.accelerations[i].y << ", " << system.accelerations[i].z << ")" << std::endl;
        os << "Mass: " << system.masses[i].x << std::endl;
    }
    return os;
}

unsigned int ParticleSystem::capacity() const {
    return maxParticles;
}