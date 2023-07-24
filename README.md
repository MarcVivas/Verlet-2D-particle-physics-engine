# 2D verlet physics engine

![ezgif com-video-to-gif (1)](https://github.com/MarcVivas/ParticlePhysics/assets/79216334/0528d203-6a2b-4ff4-a2c4-83961517b4c4)

### Requirements
- NVIDIA GPU compatible with CUDA
- The [Cuda toolkit](https://developer.nvidia.com/cuda-toolkit)
- CMake
- Visual studio

## Run the project (Windows)
```
mkdir build && cd build
```
```
cmake ..
```
Open the project `collisions.sln` with Visual Studio and execute it there.

## Program arguments
These are the available arguments you can use:
### 1. Versions
- `-v 1` CPU Parallel grid collision detection algorithm (OpenMP)
- `-v 2` GPU Parallel grid collision detection algorithm (Cuda)
  
>In some cases, the CPU version might be better than the GPU one (Not my case, AMD FX-6300 and GTX 960).
### 2. Time step
- `-t 0.02` The smaller the higher the precision but slower the simulation. 
### 3. Maximum particle size
- `-s 15.0` Particles would have radius from 1 to 15 pixels.
### 4. Buckets per row
- `-b 10` This would make a uniform grid of 10 x 10 buckets/cells.
### 5. Bucket capacity
- `-c 1000` The maximum number of particles that can be in a bucket.
### Example
```
.\collisions.exe -n 40000 -b 100 -c 5000 -s 5.0 -t 0.02 
```
