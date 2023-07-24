# 2D verlet physics engine
## Overview

## Run the project


## Program arguments
These are the available arguments you can use:
### 1. Versions
- `-v 1` CPU Parallel grid collision detection algorithm (OpenMP)
- `-v 2` GPU Parallel grid collision detection algorithm (Cuda)
### 2. Time step
- `-t 0.02` The smaller the higher the precision but slower the simulation. 
### 3. Maximum particle size
- `-s 15.0` Particles would have radius from 1 to 15 pixels.
### 4. Buckets per row
- `-b 10` This would make a uniform grid of 10 x 10 buckets/cells.
### 5. Bucket capacity
- `-c 1000` The maximum number of particles that can be in a bucket.
