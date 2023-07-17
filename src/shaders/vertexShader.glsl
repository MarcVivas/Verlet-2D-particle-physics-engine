#version 440 core

layout(std430, binding=0) buffer positionsBuffer
{
    vec4 positions[];
};

layout(std430, binding=1) buffer velocitiesBuffer
{
    vec4 velocities[];
};

layout(std430, binding=3) buffer massesBuffer
{
    vec4 masses[];
};

out vec4 particleVelocity;

out float particleSize;

float getParticleSize(){
    // set the point size based on the particle mass
    return masses[gl_VertexID].x;
}
uniform mat4 modelViewProjection;  // Aspect ratio of the window

void main()
{
    // Set the point size
    gl_PointSize = getParticleSize();
    particleSize = gl_PointSize;

    // Set the position of the particle
    gl_Position = modelViewProjection * vec4(positions[gl_VertexID].xyz, 1.f);

    // Pass the velocity to the fragment shader
    particleVelocity = velocities[gl_VertexID];
}