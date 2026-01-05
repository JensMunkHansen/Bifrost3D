// SIMD Matrix operations for Bifrost.
// ---------------------------------------------------------------------------
// Copyright (C) Bifrost. See AUTHORS.txt for authors.
//
// This program is open source and distributed under the New BSD License.
// See LICENSE.txt for more detail.
// ---------------------------------------------------------------------------
// SIMD matrix operations using SSE intrinsics.
// ---------------------------------------------------------------------------

#ifndef _BIFROST_MATH_SIMD_MATRIX_H_
#define _BIFROST_MATH_SIMD_MATRIX_H_

#include <Bifrost/Math/Matrix.h>

#ifdef _MSC_VER
#include <intrin.h>
#else
#include <x86intrin.h>
#endif

namespace Bifrost {
namespace Math {
namespace SIMD {

namespace detail {

// 3D dot product (lanes 0-2), broadcast result to all lanes
static inline __m128 dot3_ps(__m128 a, __m128 b) {
    __m128 m = _mm_mul_ps(a, b);
    __m128 shuf = _mm_shuffle_ps(m, m, _MM_SHUFFLE(2, 1, 0, 3));
    __m128 sum = _mm_add_ps(m, shuf);
    shuf = _mm_shuffle_ps(sum, sum, _MM_SHUFFLE(1, 0, 3, 2));
    sum = _mm_add_ps(sum, shuf);
    return _mm_shuffle_ps(sum, sum, _MM_SHUFFLE(0, 0, 0, 0));
}

// 3D cross product (lanes 0-2), lane 3 undefined
static inline __m128 cross3_ps(__m128 a, __m128 b) {
    __m128 a_yzx = _mm_shuffle_ps(a, a, _MM_SHUFFLE(3, 0, 2, 1));
    __m128 b_yzx = _mm_shuffle_ps(b, b, _MM_SHUFFLE(3, 0, 2, 1));
    __m128 c = _mm_sub_ps(_mm_mul_ps(a, b_yzx), _mm_mul_ps(a_yzx, b));
    return _mm_shuffle_ps(c, c, _MM_SHUFFLE(3, 0, 2, 1));
}

} // namespace detail

// ---------------------------------------------------------------------------
// SIMD 4x4 Affine Matrix Inversion
// Assumes bottom row is [0, 0, 0, 1]
// Row-major layout: M[row][col] = data[row*4 + col]
// ---------------------------------------------------------------------------
inline Matrix4x4f invert_affine(const Matrix4x4f& m) {
    const float* M = m.begin();

    // Load top 3 rows
    __m128 r0 = _mm_loadu_ps(M);      // [r00 r01 r02 tx]
    __m128 r1 = _mm_loadu_ps(M + 4);  // [r10 r11 r12 ty]
    __m128 r2 = _mm_loadu_ps(M + 8);  // [r20 r21 r22 tz]

    // Extract translation
    __m128 t = _mm_setr_ps(M[3], M[7], M[11], 0.0f);

    // Mask to zero out lane 3 (the translation component in loaded rows)
    __m128 mask = _mm_castsi128_ps(_mm_setr_epi32(-1, -1, -1, 0));
    __m128 a = _mm_and_ps(r0, mask);
    __m128 b = _mm_and_ps(r1, mask);
    __m128 c = _mm_and_ps(r2, mask);

    // Cofactors via cross products give COLUMNS of adjugate
    __m128 col0 = detail::cross3_ps(b, c);
    __m128 col1 = detail::cross3_ps(c, a);
    __m128 col2 = detail::cross3_ps(a, b);

    // Determinant = dot(a, col0) = a · (b×c)
    __m128 det = detail::dot3_ps(a, col0);
    __m128 invDet = _mm_div_ps(_mm_set1_ps(1.0f), det);

    col0 = _mm_mul_ps(col0, invDet);
    col1 = _mm_mul_ps(col1, invDet);
    col2 = _mm_mul_ps(col2, invDet);

    // Transpose columns to rows using _MM_TRANSPOSE4_PS
    __m128 row0 = col0, row1 = col1, row2 = col2, row3 = _mm_setzero_ps();
    _MM_TRANSPOSE4_PS(row0, row1, row2, row3);  // Now row0/row1/row2 are rows of R^-1

    // Compute tinv = -Rinv * t
    __m128 neg = _mm_set1_ps(-1.0f);
    __m128 tx = _mm_mul_ps(detail::dot3_ps(row0, t), neg);
    __m128 ty = _mm_mul_ps(detail::dot3_ps(row1, t), neg);
    __m128 tz = _mm_mul_ps(detail::dot3_ps(row2, t), neg);

    // Store rows and translation for assembly
    alignas(16) float rr0[4], rr1[4], rr2[4], ftx[4], fty[4], ftz[4];
    _mm_store_ps(rr0, row0);
    _mm_store_ps(rr1, row1);
    _mm_store_ps(rr2, row2);
    _mm_store_ps(ftx, tx);
    _mm_store_ps(fty, ty);
    _mm_store_ps(ftz, tz);

    // Assemble output rows with translation in last column
    __m128 out0 = _mm_setr_ps(rr0[0], rr0[1], rr0[2], ftx[0]);
    __m128 out1 = _mm_setr_ps(rr1[0], rr1[1], rr1[2], fty[0]);
    __m128 out2 = _mm_setr_ps(rr2[0], rr2[1], rr2[2], ftz[0]);
    __m128 out3 = _mm_setr_ps(0.0f, 0.0f, 0.0f, 1.0f);

    Matrix4x4f result;
    float* Out = result.begin();
    _mm_storeu_ps(Out, out0);
    _mm_storeu_ps(Out + 4, out1);
    _mm_storeu_ps(Out + 8, out2);
    _mm_storeu_ps(Out + 12, out3);

    return result;
}

// ---------------------------------------------------------------------------
// SIMD 4x4 General Matrix Inversion (for non-affine matrices)
// Row-major layout: M[row][col] = data[row*4 + col]
// Uses direct cofactor expansion similar to scalar version.
// ---------------------------------------------------------------------------
inline Matrix4x4f invert(const Matrix4x4f& m) {
    const float* v = m.begin();

    // Access as v[row*4 + col]
    #define M(r,c) v[(r)*4 + (c)]

    Matrix4x4f inv;
    float* o = inv.begin();

    // Compute cofactors (same formulas as scalar version)
    o[0] = M(1,2)*M(2,3)*M(3,1) - M(1,3)*M(2,2)*M(3,1) + M(1,3)*M(2,1)*M(3,2) - M(1,1)*M(2,3)*M(3,2) - M(1,2)*M(2,1)*M(3,3) + M(1,1)*M(2,2)*M(3,3);
    o[1] = M(0,3)*M(2,2)*M(3,1) - M(0,2)*M(2,3)*M(3,1) - M(0,3)*M(2,1)*M(3,2) + M(0,1)*M(2,3)*M(3,2) + M(0,2)*M(2,1)*M(3,3) - M(0,1)*M(2,2)*M(3,3);
    o[2] = M(0,2)*M(1,3)*M(3,1) - M(0,3)*M(1,2)*M(3,1) + M(0,3)*M(1,1)*M(3,2) - M(0,1)*M(1,3)*M(3,2) - M(0,2)*M(1,1)*M(3,3) + M(0,1)*M(1,2)*M(3,3);
    o[3] = M(0,3)*M(1,2)*M(2,1) - M(0,2)*M(1,3)*M(2,1) - M(0,3)*M(1,1)*M(2,2) + M(0,1)*M(1,3)*M(2,2) + M(0,2)*M(1,1)*M(2,3) - M(0,1)*M(1,2)*M(2,3);

    o[4] = M(1,3)*M(2,2)*M(3,0) - M(1,2)*M(2,3)*M(3,0) - M(1,3)*M(2,0)*M(3,2) + M(1,0)*M(2,3)*M(3,2) + M(1,2)*M(2,0)*M(3,3) - M(1,0)*M(2,2)*M(3,3);
    o[5] = M(0,2)*M(2,3)*M(3,0) - M(0,3)*M(2,2)*M(3,0) + M(0,3)*M(2,0)*M(3,2) - M(0,0)*M(2,3)*M(3,2) - M(0,2)*M(2,0)*M(3,3) + M(0,0)*M(2,2)*M(3,3);
    o[6] = M(0,3)*M(1,2)*M(3,0) - M(0,2)*M(1,3)*M(3,0) - M(0,3)*M(1,0)*M(3,2) + M(0,0)*M(1,3)*M(3,2) + M(0,2)*M(1,0)*M(3,3) - M(0,0)*M(1,2)*M(3,3);
    o[7] = M(0,2)*M(1,3)*M(2,0) - M(0,3)*M(1,2)*M(2,0) + M(0,3)*M(1,0)*M(2,2) - M(0,0)*M(1,3)*M(2,2) - M(0,2)*M(1,0)*M(2,3) + M(0,0)*M(1,2)*M(2,3);

    o[8] = M(1,1)*M(2,3)*M(3,0) - M(1,3)*M(2,1)*M(3,0) + M(1,3)*M(2,0)*M(3,1) - M(1,0)*M(2,3)*M(3,1) - M(1,1)*M(2,0)*M(3,3) + M(1,0)*M(2,1)*M(3,3);
    o[9] = M(0,3)*M(2,1)*M(3,0) - M(0,1)*M(2,3)*M(3,0) - M(0,3)*M(2,0)*M(3,1) + M(0,0)*M(2,3)*M(3,1) + M(0,1)*M(2,0)*M(3,3) - M(0,0)*M(2,1)*M(3,3);
    o[10] = M(0,1)*M(1,3)*M(3,0) - M(0,3)*M(1,1)*M(3,0) + M(0,3)*M(1,0)*M(3,1) - M(0,0)*M(1,3)*M(3,1) - M(0,1)*M(1,0)*M(3,3) + M(0,0)*M(1,1)*M(3,3);
    o[11] = M(0,3)*M(1,1)*M(2,0) - M(0,1)*M(1,3)*M(2,0) - M(0,3)*M(1,0)*M(2,1) + M(0,0)*M(1,3)*M(2,1) + M(0,1)*M(1,0)*M(2,3) - M(0,0)*M(1,1)*M(2,3);

    o[12] = M(1,2)*M(2,1)*M(3,0) - M(1,1)*M(2,2)*M(3,0) - M(1,2)*M(2,0)*M(3,1) + M(1,0)*M(2,2)*M(3,1) + M(1,1)*M(2,0)*M(3,2) - M(1,0)*M(2,1)*M(3,2);
    o[13] = M(0,1)*M(2,2)*M(3,0) - M(0,2)*M(2,1)*M(3,0) + M(0,2)*M(2,0)*M(3,1) - M(0,0)*M(2,2)*M(3,1) - M(0,1)*M(2,0)*M(3,2) + M(0,0)*M(2,1)*M(3,2);
    o[14] = M(0,2)*M(1,1)*M(3,0) - M(0,1)*M(1,2)*M(3,0) - M(0,2)*M(1,0)*M(3,1) + M(0,0)*M(1,2)*M(3,1) + M(0,1)*M(1,0)*M(3,2) - M(0,0)*M(1,1)*M(3,2);
    o[15] = M(0,1)*M(1,2)*M(2,0) - M(0,2)*M(1,1)*M(2,0) + M(0,2)*M(1,0)*M(2,1) - M(0,0)*M(1,2)*M(2,1) - M(0,1)*M(1,0)*M(2,2) + M(0,0)*M(1,1)*M(2,2);

    #undef M

    // Compute determinant from first row
    float det = v[0]*o[0] + v[1]*o[4] + v[2]*o[8] + v[3]*o[12];
    float rdet = 1.0f / det;

    // Scale by 1/det using SIMD
    __m128 scale = _mm_set1_ps(rdet);
    _mm_storeu_ps(o, _mm_mul_ps(_mm_loadu_ps(o), scale));
    _mm_storeu_ps(o + 4, _mm_mul_ps(_mm_loadu_ps(o + 4), scale));
    _mm_storeu_ps(o + 8, _mm_mul_ps(_mm_loadu_ps(o + 8), scale));
    _mm_storeu_ps(o + 12, _mm_mul_ps(_mm_loadu_ps(o + 12), scale));

    return inv;
}

// ---------------------------------------------------------------------------
// SIMD determinant computation
// ---------------------------------------------------------------------------
inline float determinant(const Matrix4x4f& m) {
    return Math::determinant(m);  // Use scalar version - determinant is rarely a bottleneck
}

} // namespace SIMD
} // namespace Math
} // namespace Bifrost

#endif // _BIFROST_MATH_SIMD_MATRIX_H_
