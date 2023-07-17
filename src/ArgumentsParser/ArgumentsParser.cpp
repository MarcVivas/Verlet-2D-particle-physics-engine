#include "ArgumentsParser.h"
#include <iostream>

ArgumentsParser::ArgumentsParser(int argc, char *argv[]) {
    // Default values
    this->version = Version::GPU;
    this->numParticles = 100;
    this->timeStep = .00001f;
    this->maxParticleSize = 15.f;

    std::cout << "============================================ \n\n";
    std::cout << "Usage: " << argv[0] << " [-v version] [-n numParticles] [-t timeStep] [-s maxParticleSize]\n";
    std::cout << "Default: " << argv[0] << " -v 2 -n 100 -t 0.00001 -s 15.0\n\n";

    std::cout << "Available versions: \n";
    std::cout << "-v 1 (Grid algorithm - CPU parallel)\n";
    std::cout << "-v 2 (Grid algorithm - GPU parallel)\n";
  
    std::cout << "============================================ \n\n";

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "-v" && i + 1 < argc) {
            int value = std::stoi(argv[i + 1]);
            if (value >= static_cast<int>(Version::CPU_PARALLEL) &&
                value <= static_cast<int>(Version::GPU)) {
                this->version = static_cast<Version>(value);
            } else {
                std::cerr << "Invalid version\n";
                exit(EXIT_FAILURE);
            }
            i++;
        } else if (arg == "-n" && i + 1 < argc) {
            int value = std::stoi(argv[i + 1]);
            if (value > 0) {
                this->numParticles = value;
            } else {
                std::cerr << "Invalid number of particles\n";
                exit(EXIT_FAILURE);
            }
            i++;
        } 
        else if (arg == "-t" && i + 1 < argc) {
            this->timeStep = std::stof(argv[i+1]);
            i++;
        }
        else if (arg == "-s" && i + 1 < argc) {
            this->maxParticleSize = std::stof(argv[i+1]);
            i++;
        }
        else {
            std::cerr << "Usage: " << argv[0] << " [-v version] [-n numParticles] [-t timeStep] [-s maxParticleSize]\n";
            exit(EXIT_FAILURE);
        }
    }

    std::cout << "------------------------------------ \n\n";
    std::cout << "Now using: \n\n";
    std::cout << "Version: " << version << "\n";
    std::cout << "Num particles: " << numParticles << "\n";
    std::cout << "Time step: " << timeStep << '\n';
    std::cout << "Max particle size: " << maxParticleSize << "\n\n";
    std::cout << "------------------------------------ \n\n";
    
}

Version ArgumentsParser::getVersion() {
    return this->version;
}


size_t ArgumentsParser::getNumParticles() {
    return this->numParticles;
}

float ArgumentsParser::getTimeStep() {
    return this->timeStep;
}

float ArgumentsParser::getMaxParticleSize() {
    return this->maxParticleSize;
}