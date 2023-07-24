#include "enums.h"
#include <cstddef>
#include <string>

#ifndef N_BODY_ARGUMENTSPARSER_H
#define N_BODY_ARGUMENTSPARSER_H


class ArgumentsParser {
public:
    ArgumentsParser(int argc, char *argv[]);
    Version getVersion();
    size_t getNumParticles();
    float getTimeStep();
    float getMaxParticleSize();
    int getBucketsPerRow();
    int getBucketCapacity();
private:
    Version version;
    size_t numParticles;
    float timeStep;
    float maxParticleSize;
    int bucketsPerRow; 
    int bucketCapacity;
};


#endif //N_BODY_ARGUMENTSPARSER_H
