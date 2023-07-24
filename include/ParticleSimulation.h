#include <glad/glad.h>
#include <glm/glm.hpp>
#include "ParticleSystem.h"
#include "ParticleSystemInitializer.h"
#include "ParticleSolver.h"
#include "ParticleDrawer.h"
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

#ifndef PARTICLESIMULATION_H
#define PARTICLESIMULATION_H
class ParticleSimulation {
public:
    virtual void update();
    virtual void draw();
    virtual ParticleDrawer* getParticleDrawer();
    ParticleSimulation(ParticleSystemInitializer *particleSystemInitializer, ParticleSolver *particleSysSolver, glm::vec3 worldDim, glm::vec2 windowDim);
    ~ParticleSimulation();
    void addParticle(double mouseX, double mouseY);
protected:
    ParticleSolver *particleSolver;
    ParticleDrawer *particleDrawer;
    ParticleSystem *particleSystem;
    GLuint VAO;
    GLuint VBO;
    GLsync gSync = nullptr;
    GLuint postitions_SSBO, velocities_SSBO, accelerations_SSBO, masses_SSBO, forces_SSBO, previousPos_SSBO;
    cudaGraphicsResource* cudaPositionsSSBOResource;
    cudaGraphicsResource* cudaPreviousPositionsSSBOResource;
    cudaGraphicsResource* cudaVelocitiesSSBOResource;
    cudaGraphicsResource* cudaAccelerationsSSBOResource;
    cudaGraphicsResource* cudaMassesSSBOResource;
    cudaGraphicsResource* cudaForcesSSBOResource;

    void lockParticlesBuffer();
    void waitParticlesBuffer();
    void createBuffers(bool usesGPU);
    void configureGpuBuffers();
    void configureCpuBuffers();

};
#endif // PARTICLESIMULATION_H
