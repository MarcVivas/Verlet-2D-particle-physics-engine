#include <glm/glm.hpp>
#include <vector>
#include <ostream>
#include <Particle.h>
#include <cuda_runtime.h>

#ifndef N_BODY_PARTICLESYSTEM_H
#define N_BODY_PARTICLESYSTEM_H


class ParticleSystem {
public:
    ParticleSystem(std::vector<Particle> &particles);
    void updateParticlePosition(unsigned int particleId, float deltaTime);
    unsigned int size() const;
    unsigned int capacity() const;
    glm::vec4* getPositions();
    glm::vec4* getVelocities();
    glm::vec4* getAccelerations();
    glm::vec4* getMasses();
    glm::vec4* getForces();
    glm::vec4* getPreviousPositions();
    void setMasses(glm::vec4* newMasses);
    void setPositions(glm::vec4* newPositions);
    void setAccelerations(glm::vec4* newAccelerations);
    void setVelocities(glm::vec4* newVelocities);
    friend std::ostream& operator<<(std::ostream& os, const ParticleSystem& system);
    void addParticle(Particle particle);
protected:
    unsigned int numParticles;
    unsigned int maxParticles;
    glm::vec4* positions;
    glm::vec4* accelerations;
    glm::vec4* velocities;
    glm::vec4* masses;
    glm::vec4* forces;
    glm::vec4* previousPositions;

    // Declare ParticleSimulation as a friend class
    friend class ParticleSimulation;

};


#endif //N_BODY_PARTICLESYSTEM_H
