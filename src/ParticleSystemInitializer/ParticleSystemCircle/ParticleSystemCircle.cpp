#include "ParticleSystemCircle.h"
#include <random>
#include <chrono>
#include <iostream>
#include <glm/gtc/matrix_transform.hpp>

ParticleSystemCircle::ParticleSystemCircle(size_t numParticles, float radius) : totalParticles(numParticles), maxParticleRadius(radius){}

ParticleSystem* ParticleSystemCircle::generateParticles(glm::vec3 worldDimensions) {
    Particle* particles = new Particle[totalParticles];

    std::mt19937 mt(std::chrono::system_clock::now().time_since_epoch().count());
    std::uniform_real_distribution<float> randMass(1.0, maxParticleRadius);

    // Define the disk center and radius
    glm::vec3 diskCenter = glm::vec3(worldDimensions.x /2, worldDimensions.y / 2, 0.f);
    float diskRadius = std::min(worldDimensions.x * 0.85f, worldDimensions.x );

    // Generate random seed
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> angleDistribution(0.f, 2 * glm::pi<float>());
    std::uniform_real_distribution<float> radiusDistribution(0.0f, diskRadius);

    int particlesC = 0;
    std::vector<glm::vec3> particlePositions;

    // Generate particle positions within the disk
    while (particlesC < totalParticles) {
        float angle = angleDistribution(gen);
        float radius = radiusDistribution(gen);

        float posX = diskCenter.x + radius * std::cos(angle);
        float posY = diskCenter.y + radius * std::sin(angle);
        glm::vec3 particlePos = glm::vec3(posX, posY, 0.f);

        /*bool isOverlapping = false;
        for (int i = 0; i < particlesC; i++) {
            float distance = glm::distance(particlePos, particlePositions[i]);
            if (distance < maxParticleRadius / 10.f) {
                isOverlapping = true;
                break;
            }
        }

        if (!isOverlapping) {
            particlePositions.push_back(particlePos);
            particlesC++;
        }*/
        particlePositions.push_back(particlePos);
        particlesC++;
    }

    // Assign particle positions to Particle objects
    for (int i = 0; i < totalParticles; i++) {
        glm::vec3 initialVel = glm::vec3(0, 0, 0);
        particles[i] = Particle(particlePositions[i], initialVel, randMass(mt));
    }

    std::vector<Particle> particleVector(particles, particles + totalParticles);


    delete[] particles; // free the allocated memory

    return new ParticleSystem(particleVector);
}