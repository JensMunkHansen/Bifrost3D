// SIMD Trigonometric functions for Bifrost.
// ---------------------------------------------------------------------------
// Copyright (C) Bifrost. See AUTHORS.txt for authors.
// Based on code from SOFUS by Jens Munk Hansen (GPL v3).
//
// This program is open source and distributed under the New BSD License.
// See LICENSE.txt for more detail.
// ---------------------------------------------------------------------------
// Cross-platform SIMD sin/cos implementation using SSE/AVX intrinsics.
// Works on both MSVC and GCC/Clang. Error is less than 6 ulps.
// ---------------------------------------------------------------------------

#ifndef _BIFROST_MATH_SIMD_TRIG_H_
#define _BIFROST_MATH_SIMD_TRIG_H_

#include <cstdint>
#include <cfloat>
#include <cmath>

// Include SIMD intrinsics
#ifdef _MSC_VER
#include <intrin.h>
#else
#include <x86intrin.h>
#endif

namespace Bifrost {
namespace Math {
namespace SIMD {

// ---------------------------------------------------------------------------
// Platform-specific alignment macros
// ---------------------------------------------------------------------------
#if defined(_MSC_VER)
#define BIFROST_ALIGN16 __declspec(align(16))
#define BIFROST_ALIGN32 __declspec(align(32))
#else
#define BIFROST_ALIGN16 __attribute__((aligned(16)))
#define BIFROST_ALIGN32 __attribute__((aligned(32)))
#endif

// ---------------------------------------------------------------------------
// Math constants
// ---------------------------------------------------------------------------
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
#ifndef M_PI_2
#define M_PI_2 1.57079632679489661923
#endif
#ifndef M_2_PI
#define M_2_PI 0.63661977236758134308
#endif

// ---------------------------------------------------------------------------
// Infinity constants
// ---------------------------------------------------------------------------
#if defined(_MSC_VER)
#define BIFROST_INFINITYf HUGE_VALF
#else
#define BIFROST_INFINITYf __builtin_inff()
#endif

// ---------------------------------------------------------------------------
// SIMD constants
// ---------------------------------------------------------------------------
static const BIFROST_ALIGN16 int _clear_signmask[4] = { 0x7FFFFFFF, 0x7FFFFFFF, 0x7FFFFFFF, 0x7FFFFFFF };
static const BIFROST_ALIGN16 int _neg_signmask[4] = { static_cast<int>(0x80000000), static_cast<int>(0x80000000),
                                                       static_cast<int>(0x80000000), static_cast<int>(0x80000000) };

// ---------------------------------------------------------------------------
// Helper intrinsics
// ---------------------------------------------------------------------------

// Absolute value of packed singles
inline __m128 _mm_fabs_ps(__m128 x) {
    return _mm_and_ps(x, _mm_load_ps(reinterpret_cast<const float*>(_clear_signmask)));
}

// Negate packed singles
inline __m128 _mm_neg_ps(__m128 x) {
    return _mm_xor_ps(x, _mm_load_ps(reinterpret_cast<const float*>(_neg_signmask)));
}

// Negate packed 32-bit integers
inline __m128i _mm_neg_epi32(__m128i x) {
    return _mm_sub_epi32(_mm_setzero_si128(), x);
}

// Select between a and b based on mask (blendv equivalent)
inline __m128 _mm_sel_ps(__m128 a, __m128 b, __m128 mask) {
#ifdef __SSE4_1__
    return _mm_blendv_ps(a, b, mask);
#else
    b = _mm_and_ps(b, mask);
    a = _mm_andnot_ps(mask, a);
    return _mm_or_ps(a, b);
#endif
}

// Select between a and b based on mask for integers
inline __m128i _mm_sel_epi32(__m128i a, __m128i b, __m128i mask) {
    b = _mm_and_si128(b, mask);
    a = _mm_andnot_si128(mask, a);
    return _mm_or_si128(a, b);
}

// Multiply and add: (a * b) + c
inline __m128 _mm_madd_ps(__m128 a, __m128 b, __m128 c) {
#ifdef __FMA__
    return _mm_fmadd_ps(a, b, c);
#else
    return _mm_add_ps(_mm_mul_ps(a, b), c);
#endif
}

// Check if value is infinity
inline __m128 _mm_is_infinity(__m128 d) {
    return _mm_cmpeq_ps(_mm_fabs_ps(d), _mm_set1_ps(BIFROST_INFINITYf));
}

// Multiply value by sign of another value
inline __m128 _mm_mulsign_ps(__m128 val, __m128 sgn) {
    __m128 signmask = _mm_castsi128_ps(_mm_set1_epi32(static_cast<int>(0x80000000)));
    return _mm_xor_ps(val, _mm_and_ps(signmask, sgn));
}

// ---------------------------------------------------------------------------
// Sincos constants (Chua's approximation)
// ---------------------------------------------------------------------------
#define PI4_Af 0.78515625f
#define PI4_Bf 0.00024187564849853515625f
#define PI4_Cf 3.7747668102383613586e-08f
#define PI4_Df 1.2816720341285448015e-12f

// ---------------------------------------------------------------------------
// Accurate SIMD sincos - computes sin and cos for 4 floats simultaneously
// Error is less than 6 ulps.
// ---------------------------------------------------------------------------
inline void sincos_ps(__m128 d, __m128* s_, __m128* c_) {
    __m128i q, m;
    __m128 u, s, t, rx, ry;
    __m128 r2x, r2y;

    const __m128 _m_2_pi_ps = _mm_set1_ps(static_cast<float>(M_2_PI));

    q = _mm_cvtps_epi32(_mm_mul_ps(d, _m_2_pi_ps));

    s = d;

    u = _mm_cvtepi32_ps(q);
    s = _mm_madd_ps(u, _mm_set1_ps(-PI4_Af * 2), s);
    s = _mm_madd_ps(u, _mm_set1_ps(-PI4_Bf * 2), s);
    s = _mm_madd_ps(u, _mm_set1_ps(-PI4_Cf * 2), s);
    s = _mm_madd_ps(u, _mm_set1_ps(-PI4_Df * 2), s);

    t = s;

    s = _mm_mul_ps(s, s);

    u = _mm_set1_ps(-0.000195169282960705459117889f);
    u = _mm_madd_ps(u, s, _mm_set1_ps(0.00833215750753879547119141f));
    u = _mm_madd_ps(u, s, _mm_set1_ps(-0.166666537523269653320312f));
    u = _mm_mul_ps(_mm_mul_ps(u, s), t);

    rx = _mm_add_ps(t, u);

    u = _mm_set1_ps(-2.71811842367242206819355e-07f);
    u = _mm_madd_ps(u, s, _mm_set1_ps(2.47990446951007470488548e-05f));
    u = _mm_madd_ps(u, s, _mm_set1_ps(-0.00138888787478208541870117f));
    u = _mm_madd_ps(u, s, _mm_set1_ps(0.0416666641831398010253906f));
    u = _mm_madd_ps(u, s, _mm_set1_ps(-0.5));

    ry = _mm_madd_ps(s, u, _mm_set1_ps(1));

    m = _mm_cmpeq_epi32(_mm_and_si128(q, _mm_set1_epi32(1)), _mm_set1_epi32(0));

    r2x = _mm_sel_ps(ry, rx, _mm_castsi128_ps(m));
    r2y = _mm_sel_ps(rx, ry, _mm_castsi128_ps(m));

    m = _mm_cmpeq_epi32(_mm_and_si128(q, _mm_set1_epi32(2)), _mm_set1_epi32(2));
    r2x = _mm_castsi128_ps(
        _mm_xor_si128(_mm_and_si128(m, _mm_castps_si128(_mm_set1_ps(-0.0f))), _mm_castps_si128(r2x)));

    m = _mm_cmpeq_epi32(
        _mm_and_si128(_mm_add_epi32(q, _mm_set1_epi32(1)), _mm_set1_epi32(2)), _mm_set1_epi32(2));
    r2y = _mm_castsi128_ps(
        _mm_xor_si128(_mm_and_si128(m, _mm_castps_si128(_mm_set1_ps(-0.0f))), _mm_castps_si128(r2y)));

    __m128 m1 = _mm_is_infinity(d);

    r2x = _mm_or_ps(m1, r2x);
    r2y = _mm_or_ps(m1, r2y);

    *s_ = r2x;
    *c_ = r2y;
}

// ---------------------------------------------------------------------------
// Scalar sincos - convenience wrapper for single values
// Uses the SIMD implementation internally for consistency and performance.
// ---------------------------------------------------------------------------
inline void sincosf(float angle, float* sin_out, float* cos_out) {
    __m128 angles = _mm_set_ss(angle);
    __m128 sins, coss;
    sincos_ps(angles, &sins, &coss);
    *sin_out = _mm_cvtss_f32(sins);
    *cos_out = _mm_cvtss_f32(coss);
}

// ---------------------------------------------------------------------------
// Batch sincos for arrays - processes 4 values at a time
// ---------------------------------------------------------------------------
inline void sincos_batch(const float* angles, float* sins, float* coss, size_t count) {
    size_t i = 0;

    // Process 4 at a time
    for (; i + 4 <= count; i += 4) {
        __m128 a = _mm_loadu_ps(angles + i);
        __m128 s, c;
        sincos_ps(a, &s, &c);
        _mm_storeu_ps(sins + i, s);
        _mm_storeu_ps(coss + i, c);
    }

    // Handle remainder
    for (; i < count; ++i) {
        sincosf(angles[i], &sins[i], &coss[i]);
    }
}

} // namespace SIMD
} // namespace Math
} // namespace Bifrost

#undef PI4_Af
#undef PI4_Bf
#undef PI4_Cf
#undef PI4_Df

#endif // _BIFROST_MATH_SIMD_TRIG_H_
