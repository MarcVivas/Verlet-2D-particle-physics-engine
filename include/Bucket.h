#include <glm/glm.hpp>
#include "ParticleSystem.h"

#ifndef N_BODY_BUCKET_H
#define N_BODY_BUCKET_H

class Bucket {
public:
    unsigned int bucketId;
    glm::uvec4 numParticles;
    glm::uvec4 *particlesIds;
    Bucket(unsigned int bucketCapacity, unsigned int id);
    Bucket();
    void addParticle(ParticleSystem* particles, unsigned int particleId);
    void resetBucket();
    unsigned int getNumParticles();
    unsigned int getParticleId(const unsigned int pos);
    unsigned int getBucketId();
    ~Bucket();
    glm::uvec4* getParticles();
};


#endif //N_BODY_BUCKET_H
