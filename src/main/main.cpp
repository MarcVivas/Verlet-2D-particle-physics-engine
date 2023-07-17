#include "ArgumentsParser.h"

#include "ParticleSystemCircle.h"


#include "ParticleSolverCPU.h"
#include "ParticleSolverGPU.h"
#include "GridCPU.h"

#include "WindowInputManager.h"

int main(int argc, char *argv[])
{
    // Get the arguments
    ArgumentsParser args(argc, argv);

    glm::vec3 worldDimensions(1000.f, 1000.f, 0.f);

    glm::vec2 windowDim(1000, 1000);
    Window window(windowDim, "Particle collisions");

    RenderLoop renderLoop(window, true, false);

    ParticleSystemInitializer* particleSystemInitializer;

    float maxParticleRadius = 15.f;
    particleSystemInitializer = new ParticleSystemCircle(args.getNumParticles(), maxParticleRadius);

    ParticleSimulation* particleSimulation;

    switch (args.getVersion()){
        case Version::CPU_PARALLEL:
            particleSimulation = new ParticleSimulation(particleSystemInitializer,  new ParticleSolverCPU(new GridCPU(worldDimensions, args.getNumParticles() / 10.f, 20), args.getTimeStep(), worldDimensions), worldDimensions, windowDim);
            break;
        case Version::GPU:
            particleSimulation = new ParticleSimulation(particleSystemInitializer, new ParticleSolverGPU(320.0, args.getTimeStep(), worldDimensions), worldDimensions, windowDim);
            break;
    }

    WindowInputManager windowInputManager(&window, &renderLoop, particleSimulation);

    renderLoop.runLoop(particleSimulation);

    delete particleSimulation;
    delete particleSystemInitializer;
}
