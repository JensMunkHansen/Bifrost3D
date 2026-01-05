// Test Bifrost SIMD Matrix operations.
// ---------------------------------------------------------------------------
// Copyright (C) Bifrost. See AUTHORS.txt for authors.
//
// This program is open source and distributed under the New BSD License.
// See LICENSE.txt for more detail.
// ---------------------------------------------------------------------------

#ifndef _BIFROST_MATH_SIMD_MATRIX_TEST_H_
#define _BIFROST_MATH_SIMD_MATRIX_TEST_H_

#include <Bifrost/Math/Matrix.h>
#include <Bifrost/Math/SIMDMatrix.h>

#include <gtest/gtest.h>

namespace Bifrost {
namespace Math {

class Math_SIMDMatrix : public ::testing::Test {
protected:
    static bool compare_matrix4x4f(Matrix4x4f lhs, Matrix4x4f rhs, unsigned short max_ulps) {
        return almost_equal(lhs, rhs, max_ulps);
    }
};

TEST_F(Math_SIMDMatrix, simd_invert_matches_scalar) {
    // General 4x4 matrix (non-affine)
    Matrix4x4f mat = { 2, 4, 0, 2,
                       0, 0, 1, 0,
                       6, 8, 0, 5,
                       4, 8, 0, 5 };

    Matrix4x4f scalar_inv = invert(mat);
    Matrix4x4f simd_inv = SIMD::invert(mat);

    unsigned short max_error = 10;
    EXPECT_PRED3(compare_matrix4x4f, scalar_inv, simd_inv, max_error);
}

TEST_F(Math_SIMDMatrix, simd_invert_produces_identity) {
    Matrix4x4f mat = { 2, 4, 0, 2,
                       0, 0, 1, 0,
                       6, 8, 0, 5,
                       4, 8, 0, 5 };

    Matrix4x4f simd_inv = SIMD::invert(mat);

    unsigned short max_error = 10;
    EXPECT_PRED3(compare_matrix4x4f, mat * simd_inv, Matrix4x4f::identity(), max_error);
    EXPECT_PRED3(compare_matrix4x4f, simd_inv * mat, Matrix4x4f::identity(), max_error);
}

TEST_F(Math_SIMDMatrix, simd_invert_affine_matches_scalar) {
    // Affine matrix (bottom row is [0, 0, 0, 1])
    Matrix4x4f mat = { 2, 0, 0, 1,
                       0, 2, 0, 2,
                       0, 0, 2, 3,
                       0, 0, 0, 1 };

    Matrix4x4f scalar_inv = invert(mat);
    Matrix4x4f simd_inv = SIMD::invert_affine(mat);

    unsigned short max_error = 10;
    EXPECT_PRED3(compare_matrix4x4f, scalar_inv, simd_inv, max_error);
}

TEST_F(Math_SIMDMatrix, simd_invert_affine_produces_identity) {
    // Affine matrix with rotation and translation
    Matrix4x4f mat = { 2, 0, 0, 1,
                       0, 2, 0, 2,
                       0, 0, 2, 3,
                       0, 0, 0, 1 };

    Matrix4x4f simd_inv = SIMD::invert_affine(mat);

    unsigned short max_error = 10;
    EXPECT_PRED3(compare_matrix4x4f, mat * simd_inv, Matrix4x4f::identity(), max_error);
    EXPECT_PRED3(compare_matrix4x4f, simd_inv * mat, Matrix4x4f::identity(), max_error);
}

TEST_F(Math_SIMDMatrix, simd_invert_affine_rotation_matrix) {
    // Rotation matrix (orthonormal 3x3 part)
    float c = 0.866025403784f;  // cos(30 deg)
    float s = 0.5f;              // sin(30 deg)
    Matrix4x4f mat = { c, -s, 0, 5,
                       s,  c, 0, 10,
                       0,  0, 1, 15,
                       0,  0, 0, 1 };

    Matrix4x4f scalar_inv = invert(mat);
    Matrix4x4f simd_inv = SIMD::invert_affine(mat);

    unsigned short max_error = 10;
    EXPECT_PRED3(compare_matrix4x4f, scalar_inv, simd_inv, max_error);
    EXPECT_PRED3(compare_matrix4x4f, mat * simd_inv, Matrix4x4f::identity(), max_error);
}

TEST_F(Math_SIMDMatrix, simd_invert_affine_scale_and_rotate) {
    // Scale + rotation + translation
    float c = 0.707106781187f;  // cos(45 deg)
    float s = 0.707106781187f;  // sin(45 deg)
    Matrix4x4f mat = { 2*c, -2*s, 0, 1,
                       3*s,  3*c, 0, 2,
                       0,    0,   4, 3,
                       0,    0,   0, 1 };

    Matrix4x4f scalar_inv = invert(mat);
    Matrix4x4f simd_inv = SIMD::invert_affine(mat);

    unsigned short max_error = 20;  // Slightly more tolerance for complex transform
    EXPECT_PRED3(compare_matrix4x4f, scalar_inv, simd_inv, max_error);
    EXPECT_PRED3(compare_matrix4x4f, mat * simd_inv, Matrix4x4f::identity(), max_error);
}

TEST_F(Math_SIMDMatrix, simd_determinant_matches_scalar) {
    Matrix4x4f mat = { 2, 4, 0, 2,
                       0, 0, 1, 0,
                       6, 8, 0, 5,
                       4, 8, 0, 5 };

    float scalar_det = determinant(mat);
    float simd_det = SIMD::determinant(mat);

    EXPECT_FLOAT_EQ(scalar_det, simd_det);
}

} // NS Math
} // NS Bifrost

#endif // _BIFROST_MATH_SIMD_MATRIX_TEST_H_
