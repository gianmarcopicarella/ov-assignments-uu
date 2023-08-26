#include "cl/tools.cl"

__constant float2 gravity = (float2)(0, 0.003f);

float Rand(float r, uint* seed)
{
     return RandomFloat(seed) * r;
}

__kernel void integrate( 
    __global float2* positions, 
    __global float2* prevPositions, 
    __global uint* seeds, 
    __global float* magics)
{
    const int idx = get_global_id(0);

    if(idx > 255)
    {
        float2 curpos = positions[idx];
        float2 prevpos = prevPositions[idx];

        positions[idx] += (curpos - prevpos) + gravity;
        prevPositions[idx] = curpos;

        if(Rand(10, &seeds[idx]) < 0.03f)
        {
            positions[idx] += (float2)(Rand(magics[idx], &seeds[idx]), Rand(0.12f, &seeds[idx]));
        }

        magics[idx] += 0.0002f;
    }
}

// EOF