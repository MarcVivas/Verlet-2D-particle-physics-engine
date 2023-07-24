#include <iostream>
#include "ParticleSimulation.h"
#include "ParticleSolverGPU.h"
#include "AddParticle.h"

ParticleSimulation::ParticleSimulation(ParticleSystemInitializer *particleSystemInitializer,
                                       ParticleSolver *particleSysSolver, glm::vec3 worldDim, glm::vec2 windowDim){
    this->particleSolver = particleSysSolver;
    this->particleDrawer = new ParticleDrawer(worldDim, windowDim);
    this->particleSystem = particleSystemInitializer->generateParticles(worldDim);
    this->createBuffers(this->particleSolver->usesGPU());
}


ParticleSimulation::~ParticleSimulation() {
    if (particleSolver->usesGPU()) {
        cudaGraphicsUnmapResources(1, &cudaPositionsSSBOResource);
        cudaGraphicsUnmapResources(1, &cudaAccelerationsSSBOResource);
        cudaGraphicsUnmapResources(1, &cudaVelocitiesSSBOResource);
        cudaGraphicsUnmapResources(1, &cudaMassesSSBOResource);
        cudaGraphicsUnmapResources(1, &cudaForcesSSBOResource);
        cudaGraphicsUnmapResources(1, &cudaPreviousPositionsSSBOResource);
    }
    delete this->particleDrawer;
    delete this->particleSolver;
    delete this->particleSystem;
    glDeleteVertexArrays(1, &this->VAO);
    glDeleteBuffers(1, &this->VBO);
}

void ParticleSimulation::draw() {
    this->particleDrawer->draw(this->particleSystem->size());
    this->lockParticlesBuffer();
}

void ParticleSimulation::update() {
    if (this->particleSolver->usesGPU()) {
        // =====================================================
        // Map resources
        // =====================================================
 

        float4* cudaVelocitiesTemp;
        cudaGraphicsMapResources(1, &cudaVelocitiesSSBOResource);
        cudaGraphicsResourceGetMappedPointer((void**)&cudaVelocitiesTemp, nullptr, cudaVelocitiesSSBOResource);

        float4* cudaAccelerationsTemp;
        cudaGraphicsMapResources(1, &cudaAccelerationsSSBOResource);
        cudaGraphicsResourceGetMappedPointer((void**)&cudaAccelerationsTemp, nullptr, cudaAccelerationsSSBOResource);
        
        float4* cudaMassesTemp;
        cudaGraphicsMapResources(1, &cudaMassesSSBOResource);
        cudaGraphicsResourceGetMappedPointer((void**)&cudaMassesTemp, nullptr, cudaMassesSSBOResource);

        float4* cudaForcesTemp;
        cudaGraphicsMapResources(1, &cudaForcesSSBOResource);
        cudaGraphicsResourceGetMappedPointer((void**)&cudaForcesTemp, nullptr, cudaForcesSSBOResource);

        float4* cudaPreviousPosTemp;
        cudaGraphicsMapResources(1, &cudaPreviousPositionsSSBOResource);
        cudaGraphicsResourceGetMappedPointer((void**)&cudaPreviousPosTemp, nullptr, cudaPreviousPositionsSSBOResource);
        
        float4* cudaPositionsTemp;
        cudaGraphicsMapResources(1, &cudaPositionsSSBOResource);
        cudaGraphicsResourceGetMappedPointer((void**)&cudaPositionsTemp, nullptr, cudaPositionsSSBOResource);

        // Create an array to store the CUDA pointers
        float4* cudaParticleData[] = { cudaPositionsTemp, cudaVelocitiesTemp, cudaAccelerationsTemp, cudaMassesTemp, cudaForcesTemp, cudaPreviousPosTemp };

        
        // =====================================================
        
        this->particleSolver->updateParticlesPositions(this->particleSystem, cudaParticleData);
          
        // =====================================================
        // UnMap resources
        // =====================================================
        cudaGraphicsUnmapResources(1, &cudaPositionsSSBOResource);
        cudaGraphicsUnmapResources(1, &cudaVelocitiesSSBOResource);
        cudaGraphicsUnmapResources(1, &cudaAccelerationsSSBOResource);
        cudaGraphicsUnmapResources(1, &cudaMassesSSBOResource);
        cudaGraphicsUnmapResources(1, &cudaForcesSSBOResource);
        cudaGraphicsUnmapResources(1, &cudaPreviousPositionsSSBOResource);
        // =====================================================


    }
    else {
        this->particleSolver->updateParticlePositions(this->particleSystem);
    }
    this->waitParticlesBuffer();

 
}

void ParticleSimulation::createBuffers(bool usesGPU) {
    glGenVertexArrays(1, &this->VAO);
    glBindVertexArray(this->VAO);


    // Shader storage buffer objects

    // This SSBO stores particles positions
    glGenBuffers(1, &this->postitions_SSBO);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, this->postitions_SSBO);

    // This SSBO stores particles velocities
    glGenBuffers(1, &this->velocities_SSBO);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, this->velocities_SSBO);

    // This SSBO stores particles accelerations
    glGenBuffers(1, &this->accelerations_SSBO);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, this->accelerations_SSBO);

    // This SSBO stores particles accelerations
    glGenBuffers(1, &this->masses_SSBO);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, this->masses_SSBO);

    // This SSBO stores particles accelerations
    glGenBuffers(1, &this->forces_SSBO);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 4, this->forces_SSBO);

    // This SSBO stores particles accelerations
    glGenBuffers(1, &this->previousPos_SSBO);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 5, this->previousPos_SSBO);

    if(usesGPU){
        this->configureGpuBuffers();       
    }
    else{
        this->configureCpuBuffers();
    }

}

void ParticleSimulation::configureGpuBuffers() {
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, this->postitions_SSBO);
    glBufferData(GL_SHADER_STORAGE_BUFFER, sizeof(glm::vec4) * this->particleSystem->capacity(), this->particleSystem->getPositions(), GL_DYNAMIC_DRAW);
    cudaGraphicsGLRegisterBuffer(&cudaPositionsSSBOResource, postitions_SSBO, cudaGraphicsRegisterFlagsNone);

    
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, this->velocities_SSBO);
    glBufferData(GL_SHADER_STORAGE_BUFFER, sizeof(glm::vec4) * this->particleSystem->capacity(), this->particleSystem->getVelocities(), GL_DYNAMIC_DRAW);
    cudaGraphicsGLRegisterBuffer(&cudaVelocitiesSSBOResource, velocities_SSBO, cudaGraphicsRegisterFlagsNone);

    glBindBuffer(GL_SHADER_STORAGE_BUFFER, this->accelerations_SSBO);
    glBufferData(GL_SHADER_STORAGE_BUFFER, sizeof(glm::vec4) * this->particleSystem->capacity(), this->particleSystem->getAccelerations(), GL_DYNAMIC_DRAW);
    cudaGraphicsGLRegisterBuffer(&cudaAccelerationsSSBOResource, accelerations_SSBO, cudaGraphicsRegisterFlagsNone);
 

    glBindBuffer(GL_SHADER_STORAGE_BUFFER, this->masses_SSBO);
    glBufferData(GL_SHADER_STORAGE_BUFFER, sizeof(glm::vec4) * this->particleSystem->capacity(), this->particleSystem->getMasses(), GL_STATIC_DRAW);
    cudaGraphicsGLRegisterBuffer(&cudaMassesSSBOResource, masses_SSBO, cudaGraphicsRegisterFlagsNone);
    

    glBindBuffer(GL_SHADER_STORAGE_BUFFER, this->forces_SSBO);
    glBufferData(GL_SHADER_STORAGE_BUFFER, sizeof(glm::vec4) * this->particleSystem->capacity(), this->particleSystem->getForces(), GL_DYNAMIC_DRAW);
    cudaGraphicsGLRegisterBuffer(&cudaForcesSSBOResource, forces_SSBO, cudaGraphicsRegisterFlagsNone);


    glBindBuffer(GL_SHADER_STORAGE_BUFFER, this->previousPos_SSBO);
    glBufferData(GL_SHADER_STORAGE_BUFFER, sizeof(glm::vec4) * this->particleSystem->capacity(), this->particleSystem->getPreviousPositions(), GL_DYNAMIC_DRAW);
    cudaGraphicsGLRegisterBuffer(&cudaPreviousPositionsSSBOResource, previousPos_SSBO, cudaGraphicsRegisterFlagsNone);

    
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
}

/**
 * Creates persistent mapped shader storage buffer objects
 */
void ParticleSimulation::configureCpuBuffers() {
    GLbitfield bufferStorageFlags = GL_MAP_READ_BIT | GL_MAP_WRITE_BIT | GL_MAP_PERSISTENT_BIT | GL_MAP_COHERENT_BIT;

    glBindBuffer(GL_SHADER_STORAGE_BUFFER, this->masses_SSBO);
    glBufferData(GL_SHADER_STORAGE_BUFFER, sizeof(glm::vec4) * this->particleSystem->capacity(), this->particleSystem->getMasses(), GL_STATIC_DRAW);


    glBindBuffer(GL_SHADER_STORAGE_BUFFER, this->postitions_SSBO);
    glBufferStorage(GL_SHADER_STORAGE_BUFFER, sizeof(glm::vec4) * this->particleSystem->capacity(), this->particleSystem->getPositions(), bufferStorageFlags);
    glm::vec4* positions = (glm::vec4*) glMapBufferRange(GL_SHADER_STORAGE_BUFFER, 0, sizeof(glm::vec4) * this->particleSystem->capacity(), bufferStorageFlags);
    this->particleSystem->setPositions(positions);

    glBindBuffer(GL_SHADER_STORAGE_BUFFER, this->velocities_SSBO);
    glBufferStorage(GL_SHADER_STORAGE_BUFFER, sizeof(glm::vec4) * this->particleSystem->capacity(), this->particleSystem->getVelocities(), bufferStorageFlags);
    glm::vec4* velocities = (glm::vec4*)glMapBufferRange(GL_SHADER_STORAGE_BUFFER, 0, sizeof(glm::vec4) * this->particleSystem->capacity(), bufferStorageFlags);
    this->particleSystem->setVelocities(velocities);

    glBindBuffer(GL_SHADER_STORAGE_BUFFER, this->accelerations_SSBO);
    glBufferStorage(GL_SHADER_STORAGE_BUFFER, sizeof(glm::vec4) * this->particleSystem->capacity(), this->particleSystem->getAccelerations(), bufferStorageFlags);
    glm::vec4* accelerations = (glm::vec4*)glMapBufferRange(GL_SHADER_STORAGE_BUFFER, 0, sizeof(glm::vec4) * this->particleSystem->capacity(), bufferStorageFlags);
    this->particleSystem->setAccelerations(accelerations);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
}


void ParticleSimulation::lockParticlesBuffer()
{
    if( this->gSync )
    {
        glDeleteSync( this->gSync );
    }
    this->gSync = glFenceSync( GL_SYNC_GPU_COMMANDS_COMPLETE, 0 );
}

void ParticleSimulation::waitParticlesBuffer()
{
    // Insert a memory barrier to ensure synchronization between CPU and GPU
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
    if( this->gSync )
    {
        while(true)
        {
            GLenum waitReturn = glClientWaitSync( this->gSync, GL_SYNC_FLUSH_COMMANDS_BIT, 1 );
            if (waitReturn == GL_ALREADY_SIGNALED || waitReturn == GL_CONDITION_SATISFIED)
                return;
        }
    }
}

void ParticleSimulation::addParticle(double mouseX, double mouseY) {
    glm::vec3 mousePos(mouseX, mouseY, 0.f);
    glm::vec3 vel(0.f);
    particleSystem->addParticle(Particle(mousePos, vel, 0.f));

    if(particleSolver->usesGPU()){


        // =====================================================
        // Map resources
        // =====================================================
        float4* cudaVelocitiesTemp;
        cudaGraphicsMapResources(1, &cudaVelocitiesSSBOResource);
        cudaGraphicsResourceGetMappedPointer((void**)&cudaVelocitiesTemp, nullptr, cudaVelocitiesSSBOResource);

        float4* cudaAccelerationsTemp;
        cudaGraphicsMapResources(1, &cudaAccelerationsSSBOResource);
        cudaGraphicsResourceGetMappedPointer((void**)&cudaAccelerationsTemp, nullptr, cudaAccelerationsSSBOResource);

        float4* cudaMassesTemp;
        cudaGraphicsMapResources(1, &cudaMassesSSBOResource);
        cudaGraphicsResourceGetMappedPointer((void**)&cudaMassesTemp, nullptr, cudaMassesSSBOResource);

        float4* cudaForcesTemp;
        cudaGraphicsMapResources(1, &cudaForcesSSBOResource);
        cudaGraphicsResourceGetMappedPointer((void**)&cudaForcesTemp, nullptr, cudaForcesSSBOResource);

        float4* cudaPreviousPosTemp;
        cudaGraphicsMapResources(1, &cudaPreviousPositionsSSBOResource);
        cudaGraphicsResourceGetMappedPointer((void**)&cudaPreviousPosTemp, nullptr, cudaPreviousPositionsSSBOResource);

        float4* cudaPositionsTemp;
        cudaGraphicsMapResources(1, &cudaPositionsSSBOResource);
        cudaGraphicsResourceGetMappedPointer((void**)&cudaPositionsTemp, nullptr, cudaPositionsSSBOResource);

        // =====================================================
        int partId = particleSystem->size()-1;

        addNewParticle(
                cudaPositionsTemp,
                cudaMassesTemp,
                cudaAccelerationsTemp,
                cudaVelocitiesTemp,
                cudaForcesTemp,
                cudaPreviousPosTemp,
                make_float4(particleSystem->getPositions()[partId].x,  particleSystem->getPositions()[partId].y, 0.f, 0.f),
                make_float4(particleSystem->getAccelerations()[partId].x, particleSystem->getAccelerations()[partId].y, 0.f, 0.f),
                make_float4(particleSystem->getVelocities()[partId].x, particleSystem->getVelocities()[partId].y, 0.f, 0.f),
                make_float4(particleSystem->getForces()[partId].x, particleSystem->getForces()[partId].y, 0.f, 0.f),
                particleSystem->size()
                );

        // =====================================================
        // UnMap resources
        // =====================================================
        cudaGraphicsUnmapResources(1, &cudaPositionsSSBOResource);
        cudaGraphicsUnmapResources(1, &cudaVelocitiesSSBOResource);
        cudaGraphicsUnmapResources(1, &cudaAccelerationsSSBOResource);
        cudaGraphicsUnmapResources(1, &cudaMassesSSBOResource);
        cudaGraphicsUnmapResources(1, &cudaForcesSSBOResource);
        cudaGraphicsUnmapResources(1, &cudaPreviousPositionsSSBOResource);
        // =====================================================
    }

}

ParticleDrawer *ParticleSimulation::getParticleDrawer() {
    return this->particleDrawer;
}