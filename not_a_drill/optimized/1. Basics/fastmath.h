#pragma once

#include <math.h>
#include <intrin.h>
#include <tmplmath.h>

namespace Tmpl8
{
    namespace
    {
        constexpr float invtwopi = 0.1591549f;
        constexpr float twopi = 6.283185f;
        constexpr float threehalfpi = 4.7123889f;
        constexpr float pi = 3.141593f;
        constexpr float halfpi = 1.570796f;
        constexpr float quarterpi = 0.7853982f;

        float locCos52s(float x)
        {
            const float c1 = 0.9999932946f;
            const float c2 = -0.4999124376f;
            const float c3 = 0.0414877472f;
            const float c4 = -0.0012712095f;
            float x2;      // The input argument squared
            x2 = x * x;
            return (c1 + x2 * (c2 + x2 * (c3 + c4 * x2)));
        }
    }

    // https://www.researchgate.net/publication/349411371_Fast_Calculation_of_Cube_and_Inverse_Cube_Roots_Using_a_Magic_Constant_and_Its_Implementation_on_Microcontrollers
    inline double FastCbrt(float cube)
    {
        constexpr float k1 = 1.752319676f;
        constexpr float k2 = 1.2509524245f;
        constexpr float k3 = 0.5093818292f;
        int i = *(int*)&cube;
        i = 0x548c2b4b - i / 3;
        float y = *(float*)&i;
        float c = cube * y * y * y;
        y = y * (k1 - c * (k2 - k3 * c));
        float d = cube * y * y;
        c = fmaf(-d, y, 1.0f);
        y = d * fmaf(c, 0.333333333333f, 1.0f);
        y = y - 0.333333333333f * (y - cube / (y * y)); // Additional newton iteration, maybe there is a way to avoid that division...
        return y;
    }

    inline float FastSqrt(float squared)
    {
        return _mm_cvtss_f32(_mm_sqrt_ss(_mm_set_ss(squared)));
    }

    inline float FastCos(float angle)
    {
        // First clamp to the range [0, 2PI]
        angle = angle - floorf(angle * invtwopi) * twopi;
        angle = angle > 0 ? angle : -angle;

        if (angle < halfpi) return locCos52s(angle);
        if (angle < pi) return -locCos52s(pi - angle);
        if (angle < threehalfpi) return -locCos52s(angle - pi);
        return locCos52s(twopi - angle);
    }

    inline float FastAcos(float angle)
    {
        constexpr float k1 = 0.0187293f;
        constexpr float k2 = 0.0742610f;
        constexpr float k3 = 0.2121144f;
        constexpr float k4 = 1.5707288f;
        float negate = float(angle < 0);
        angle = fabs(angle);
        float ret = -k1;
        ret = ret * angle;
        ret = ret + k2;
        ret = ret * angle;
        ret = ret - k3;
        ret = ret * angle;
        ret = ret + k4;
        ret = ret * FastSqrt(1.0 - angle);
        ret = ret - 2 * negate * ret;
        return negate * 3.14159265358979 + ret;
    }

    inline float Dot3SSE(const __m128& a, const __m128& b)
    {
        constexpr int mask = 0x7f;
        return _mm_cvtss_f32(_mm_dp_ps(a, b, mask));
    }

    inline float HorizontalMax3SSE(__m128 x)
    {
        constexpr unsigned int firstPattern = _MM_SHUFFLE(3, 0, 2, 1);
        constexpr unsigned int secondPattern = _MM_SHUFFLE(3, 1, 0, 2);
        const __m128 max1 = _mm_shuffle_ps(x, x, firstPattern);
        const __m128 max3 = _mm_shuffle_ps(x, x, secondPattern);
        const __m128 max2 = _mm_max_ps(x, max1);
        const __m128 max4 = _mm_max_ps(max2, max3);
        return _mm_cvtss_f32(max4);
    }

    inline float HorizontalMin3SSE(__m128 x)
    {
        constexpr unsigned int firstPattern = _MM_SHUFFLE(3, 0, 2, 1);
        constexpr unsigned int secondPattern = _MM_SHUFFLE(3, 1, 0, 2);
        const __m128 min1 = _mm_shuffle_ps(x, x, firstPattern);
        const __m128 min3 = _mm_shuffle_ps(x, x, secondPattern);
        const __m128 min2 = _mm_min_ps(x, min1);
        const __m128 min4 = _mm_min_ps(min2, min3);
        return _mm_cvtss_f32(min4);
    }

    inline float HorizontalMin4SSE(__m128 v)
    {
        v = _mm_min_ps(v, _mm_shuffle_ps(v, v, _MM_SHUFFLE(2, 1, 0, 3)));
        v = _mm_min_ps(v, _mm_shuffle_ps(v, v, _MM_SHUFFLE(1, 0, 3, 2)));
        return _mm_cvtss_f32(v);
    }

    inline __m128 TransformPositionSSE_128(const __m128& a, const mat4& M)
    {
        __m128 a4 = a;
        a4.m128_f32[3] = 1;
        __m128 v0 = _mm_mul_ps(a4, _mm_load_ps(&M.cell[0]));
        __m128 v1 = _mm_mul_ps(a4, _mm_load_ps(&M.cell[4]));
        __m128 v2 = _mm_mul_ps(a4, _mm_load_ps(&M.cell[8]));
        __m128 v3 = _mm_mul_ps(a4, _mm_load_ps(&M.cell[12]));
        _MM_TRANSPOSE4_PS(v0, v1, v2, v3);
        return _mm_add_ps(_mm_add_ps(v0, v1), _mm_add_ps(v2, v3));
    }

    inline __m128 TransformVectorSSE_128(const __m128& a, const mat4& M)
    {
        __m128 v0 = _mm_mul_ps(a, _mm_load_ps(&M.cell[0]));
        __m128 v1 = _mm_mul_ps(a, _mm_load_ps(&M.cell[4]));
        __m128 v2 = _mm_mul_ps(a, _mm_load_ps(&M.cell[8]));
        __m128 v3 = _mm_mul_ps(a, _mm_load_ps(&M.cell[12]));
        _MM_TRANSPOSE4_PS(v0, v1, v2, v3);
        return _mm_add_ps(_mm_add_ps(v0, v1), v2);
    }
}