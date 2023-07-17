
#include "ParticleSystemInitializer.h"

#ifndef PARTICLESYSTEMCUBEINITIALIZER_H
#define PARTICLESYSTEMCUBEINITIALIZER_H


class ParticleSystemCircle: public ParticleSystemInitializer{

public:
    ParticleSystemCircle(size_t numParticles, float maxParticleRadius);
    ParticleSystem* generateParticles(glm::vec3 worldDimensions);

private:
    size_t totalParticles;
    float maxParticleRadius;
};


#endif //PARTICLESYSTEMCUBEINITIALIZER_H
