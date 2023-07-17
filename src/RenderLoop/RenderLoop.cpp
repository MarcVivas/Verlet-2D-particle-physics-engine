
#include "RenderLoop.h"

RenderLoop::RenderLoop(const Window& win, bool showFps, bool vSync) :
    window(win),
    renderTimer(RenderTimer(showFps, vSync)),
    pauseSimulation(true)
{}

RenderLoop::RenderLoop()=default;

void RenderLoop::runLoop(ParticleSimulation *particleSimulation) {
    while (!glfwWindowShouldClose(this->window.getWindow()))
    {
        this->renderTimer.updateTime(this->window, this->pauseSimulation);

        if(!this->pauseSimulation){
            particleSimulation->update();
        }

        particleSimulation->draw();

        // ======================
        // Swap buffers: Front buffer(render) and back buffer (next render)
        glfwSwapBuffers(this->window.getWindow());
        // ======================

        // ======================
        // Checks if any events are triggered (keys pressed/released, mouse moved etc.) and calls the corresponding functions
        glfwPollEvents();
        // ======================

    }
}

void RenderLoop::setPauseSimulation(bool pause) {
    this->pauseSimulation = pause;
}

bool RenderLoop::getPauseSimulation(){
    return this->pauseSimulation;
}

int RenderLoop::getIteration() {
    return this->renderTimer.getIteration();
}


RenderLoop::~RenderLoop(){
    this->renderTimer.printFinalStats();
}



