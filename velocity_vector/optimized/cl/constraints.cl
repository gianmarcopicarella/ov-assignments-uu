#include "cl/tools.cl"

__constant int OFFSETS[4] = {1, -1, 256, -256};
__constant int MASK[2] = {0x00000000, 0xffffffff};
__constant int DIM = 256;

typedef union {
    float df[2]; int di[2];
} U;

void processNeighbours(const int idx, int startNIdx, int endNIdx, float2* pointpos, float2* positions, float* restLengths)
{
    U u;
    for(int linknr = startNIdx; linknr < endNIdx; ++linknr)
    {
        float2* neighbourPos = &positions[idx + OFFSETS[linknr]];
        const float restLen = restLengths[linknr + idx * 4];
        const float distance = length(*neighbourPos - *pointpos);
        const int maskIndex = isfinite(distance) & (distance > restLen);
                
        const float extra = (distance - restLen) / (restLen);
        float2 dir = (*neighbourPos - *pointpos) * extra * 0.5f;

        // branch less operation
        u.df[0] = dir.x; u.df[1] = dir.y;
        u.di[0] &= MASK[maskIndex]; u.di[1] &= MASK[maskIndex];
        dir.x = u.df[0]; dir.y = u.df[1];

        // update
        *pointpos += dir;

        if (!(linknr == 3 && idx < 512))
        {
            *neighbourPos -= dir;
        }
    }
}

__kernel void solve(
    __global float2 * positions, 
    __global float * restLengths,
    __global int * indices)
{
    U u;

    for(int y = 0; y < 2*DIM; ++y)
    {
        const int x = get_global_id(0);
        const int idx = indices[x + y * DIM];
        
        float2 pointpos;

        if(idx > -1)
        {
            pointpos = positions[idx];
            processNeighbours(idx, 0, 2, &pointpos, positions, restLengths);
        }

        barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

        if(idx > -1)
        {
            processNeighbours(idx, 2, 4, &pointpos, positions, restLengths);
            positions[idx] = pointpos;
        }

        barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
    }
}

// EOF