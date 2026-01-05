// Bifrost distributions.
// ------------------------------------------------------------------------------------------------
// Copyright (C) Bifrost. See AUTHORS.txt for authors.
//
// This program is open source and distributed under the New BSD License.
// See LICENSE.txt for more detail.
// ------------------------------------------------------------------------------------------------

#ifndef _BIFROST_MATH_DISTRIBUTIONS_H_
#define _BIFROST_MATH_DISTRIBUTIONS_H_

#include <Bifrost/Core/Defines.h>
#include <Bifrost/Math/Constants.h>
#include <Bifrost/Math/SIMDTrig.h>
#include <Bifrost/Math/Vector.h>
#include <Bifrost/Math/Utils.h>

namespace Bifrost {
namespace Math {
namespace Distributions {

//=================================================================================================
// GGX distribution.
//=================================================================================================
namespace GGX {

struct Sample {
    Vector3f direction;
    float PDF;
};

__always_inline__ float D(float alpha, float abs_cos_theta) {
    float alpha_sqrd = alpha * alpha;
    float cos_theta_sqrd = abs_cos_theta * abs_cos_theta;
    float tan_theta_sqrd = fmaxf(1.0f - cos_theta_sqrd, 0.0f) / cos_theta_sqrd;
    float cos_theta_cubed = cos_theta_sqrd * cos_theta_sqrd;
    float foo = alpha_sqrd + tan_theta_sqrd; // No idea what to call this.
    return alpha_sqrd / (PI<float>() * cos_theta_cubed * foo * foo);
}

__always_inline__ float PDF(float alpha, float abs_cos_theta) {
    return D(alpha, abs_cos_theta) * abs_cos_theta;
}

__always_inline__ Sample sample(float alpha, Vector2f random_sample) {
    float phi = random_sample.y * (2.0f * PI<float>());

    float tan_theta_sqrd = alpha * alpha * random_sample.x / (1.0f - random_sample.x);
    float cos_theta = 1.0f / sqrt(1.0f + tan_theta_sqrd);

    float r = sqrt(fmaxf(1.0f - cos_theta * cos_theta, 0.0f));

    float sin_phi, cos_phi;
    SIMD::sincosf(phi, &sin_phi, &cos_phi);

    Sample res;
    res.direction = Vector3f(cos_phi * r, sin_phi * r, cos_theta);
    res.PDF = PDF(alpha, cos_theta); // We have to be able to inline this to reuse some temporaries.
    return res;
}

// Batched GGX sampling for better L1 cache utilization.
// Processes N samples with phi angles kept in L1, then batch sincos.
template<size_t N>
__always_inline__ void sample_batch(float alpha, const Vector2f* random_samples, Sample* results) {
    // Phase 1: Compute all phi angles (contiguous, L1-friendly)
    alignas(64) float phis[N];
    alignas(64) float cos_thetas[N];
    alignas(64) float rs[N];

    const float two_pi = 2.0f * PI<float>();
    const float alpha_sqrd = alpha * alpha;

    for (size_t i = 0; i < N; ++i) {
        phis[i] = random_samples[i].y * two_pi;

        float tan_theta_sqrd = alpha_sqrd * random_samples[i].x / (1.0f - random_samples[i].x);
        cos_thetas[i] = 1.0f / sqrt(1.0f + tan_theta_sqrd);
        rs[i] = sqrt(fmaxf(1.0f - cos_thetas[i] * cos_thetas[i], 0.0f));
    }

    // Phase 2: Batch sincos - all phis are hot in L1
    alignas(64) float sin_phis[N];
    alignas(64) float cos_phis[N];
    SIMD::sincos_batch(phis, sin_phis, cos_phis, N);

    // Phase 3: Assemble results
    for (size_t i = 0; i < N; ++i) {
        results[i].direction = Vector3f(cos_phis[i] * rs[i], sin_phis[i] * rs[i], cos_thetas[i]);
        results[i].PDF = PDF(alpha, cos_thetas[i]);
    }
}

} // NS GGX


//=================================================================================================
// Uniform sphere distribution.
//=================================================================================================
namespace Sphere {

__always_inline__ float PDF() { return 0.5f / PI<float>(); }

__always_inline__ Vector3f sample(Vector2f random_sample) {
    float z = 1.0f - 2.0f * random_sample.x;
    float r = sqrt(fmaxf(0.0f, 1.0f - z * z));
    float phi = 2.0f * PI<float>() * random_sample.y;
    float sin_phi, cos_phi;
    SIMD::sincosf(phi, &sin_phi, &cos_phi);
    return Vector3f(r * cos_phi, r * sin_phi, z);
}

// Batched sphere sampling for better L1 cache utilization.
template<size_t N>
__always_inline__ void sample_batch(const Vector2f* random_samples, Vector3f* results) {
    alignas(64) float phis[N];
    alignas(64) float zs[N];
    alignas(64) float rs[N];

    const float two_pi = 2.0f * PI<float>();

    for (size_t i = 0; i < N; ++i) {
        zs[i] = 1.0f - 2.0f * random_samples[i].x;
        rs[i] = sqrt(fmaxf(0.0f, 1.0f - zs[i] * zs[i]));
        phis[i] = two_pi * random_samples[i].y;
    }

    alignas(64) float sin_phis[N];
    alignas(64) float cos_phis[N];
    SIMD::sincos_batch(phis, sin_phis, cos_phis, N);

    for (size_t i = 0; i < N; ++i) {
        results[i] = Vector3f(rs[i] * cos_phis[i], rs[i] * sin_phis[i], zs[i]);
    }
}

} // NS Sphere

//=================================================================================================
// Henyey-Greenstein distribution.
// Physically Based Rendering version 4, section 11.3.1
// https://www.pbr-book.org/4ed/Volume_Scattering/Phase_Functions#TheHenyeyndashGreensteinPhaseFunction
//=================================================================================================
namespace HenyeyGreenstein {

struct Sample {
    Vector3f direction;
    float PDF;
};

// When g approximates -1 and random_sample approximates 0 or when g approximates 1 and random_sample approximates 1,
// the computation of cos_theta below is unstable and can give 0, leading to NaNs.
// For now we limit g to the range where it is stable.
__always_inline__ float safe_g(float g) {
    return clamp(g, -.99f, .99f);
}

__always_inline__ float evaluate(float g, float cos_theta) {
    g = safe_g(g);
    float denominator = 1 + pow2(g) + 2 * g * cos_theta;
    constexpr float recip_4_pi = 1.0f / (4 * PI<float>());
    return recip_4_pi * (1 - pow2(g)) / (denominator * sqrt(max(0.0f, denominator)));
}

__always_inline__ float evaluate(float g, Vector3f wo, Vector3f wi) {
    float cos_theta = dot(wo, wi);
    return evaluate(g, cos_theta);
}

// Sample the cosine of the angle for the distribution.
__always_inline__ float sample_cos_theta(float g, float random_sample) {
    g = safe_g(g);

    if (abs(g) < 1e-3f)
        return 1 - 2 * random_sample; // Use spherical distribution directly when g is close to 0.
    else
        return -1 / (2 * g) * (1 + pow2(g) - pow2((1 - pow2(g)) / (1 + g - 2 * g * random_sample)));
}

// Sample a direction in the distribution wrt [0,0,1] as wo.
__always_inline__ Vector3f sample_direction(float g, Vector2f random_sample) {
    float cos_theta = sample_cos_theta(g, random_sample.x);

    float sin_theta = sqrt(fmaxf(0.0f, 1.0f - pow2(cos_theta)));
    float phi = 2.0f * PI<float>() * random_sample.y;
    float sin_phi, cos_phi;
    SIMD::sincosf(phi, &sin_phi, &cos_phi);
    return Vector3f(sin_theta * cos_phi, sin_theta * sin_phi, cos_theta);
}

// Sample the distribution wrt [0,0,1] as wo.
__always_inline__ Sample sample(float g, Vector2f random_sample) {
    Vector3f wi = sample_direction(g, random_sample);
    float pdf = evaluate(g, wi.z);
    return { wi, pdf };
}

// Sample a direction in the distribution.
__always_inline__ Vector3f sample_direction(float g, Vector3f wo, Vector2f random_sample) {
    Vector3f local_wi = sample_direction(g, random_sample);

    Vector3f tangent, bitangent;
    compute_tangents(wo, tangent, bitangent);
    return tangent * local_wi.x + bitangent * local_wi.y + wo * local_wi.z;
}

// Sample the distribution.
__always_inline__ Sample sample(float g, Vector3f wo, Vector2f random_sample) {
    Vector3f wi = sample_direction(g, wo, random_sample);
    float pdf = evaluate(g, dot(wo, wi));
    return { wi, pdf };
}

} // NS HenyeyGreenstein

} // NS Distributions
} // NS Math
} // NS Bifrost

#endif // _BIFROST_MATH_DISTRIBUTIONS_H_
