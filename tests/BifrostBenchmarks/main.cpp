// Bifrost Core Library Benchmarks
// ---------------------------------------------------------------------------
// Copyright (C) Bifrost. See AUTHORS.txt for authors.
//
// This program is open source and distributed under the New BSD License.
// See LICENSE.txt for more detail.
// ---------------------------------------------------------------------------

#include "Benchmark.h"

#include <Bifrost/Math/Vector.h>
#include <Bifrost/Math/Matrix.h>
#include <Bifrost/Math/Quaternion.h>
#include <Bifrost/Math/Transform.h>
#include <Bifrost/Math/RNG.h>
#include <Bifrost/Math/Distribution1D.h>
#include <Bifrost/Math/Distribution2D.h>
#include <Bifrost/Assets/MeshCreation.h>
#include <Bifrost/Assets/Mesh.h>
#include <Bifrost/Scene/SceneNode.h>

#include <random>

using namespace Bifrost;
using namespace Bifrost::Math;
using namespace Bifrost::Benchmark;

// ---------------------------------------------------------------------------
// Vector operations
// ---------------------------------------------------------------------------
void benchmark_vector_operations() {
    std::cout << "\n=== Vector Operations ===" << std::endl;
    print_header();

    Vector3f a(1.0f, 2.0f, 3.0f);
    Vector3f b(4.0f, 5.0f, 6.0f);
    volatile float sink;

    BENCHMARK("Vector3f add", {
        Vector3f c = a + b;
        do_not_optimize(c);
    });

    BENCHMARK("Vector3f dot", {
        float d = dot(a, b);
        do_not_optimize(d);
    });

    BENCHMARK("Vector3f cross", {
        Vector3f c = cross(a, b);
        do_not_optimize(c);
    });

    BENCHMARK("Vector3f normalize", {
        Vector3f c = normalize(a);
        do_not_optimize(c);
    });

    BENCHMARK("Vector3f magnitude", {
        float m = magnitude(a);
        do_not_optimize(m);
    });
}

// ---------------------------------------------------------------------------
// Matrix operations
// ---------------------------------------------------------------------------
void benchmark_matrix_operations() {
    std::cout << "\n=== Matrix Operations ===" << std::endl;
    print_header();

    Matrix4x4f m1 = Matrix4x4f::identity();
    m1.set_row(0, Vector4f(1, 2, 3, 4));
    m1.set_row(1, Vector4f(5, 6, 7, 8));
    m1.set_row(2, Vector4f(9, 10, 11, 12));
    m1.set_row(3, Vector4f(13, 14, 15, 16));

    Matrix4x4f m2 = Matrix4x4f::identity();
    m2.set_row(0, Vector4f(2, 0, 0, 1));
    m2.set_row(1, Vector4f(0, 2, 0, 2));
    m2.set_row(2, Vector4f(0, 0, 2, 3));
    m2.set_row(3, Vector4f(0, 0, 0, 1));

    Vector4f v(1, 2, 3, 1);

    BENCHMARK("Matrix4x4f multiply", {
        Matrix4x4f r = m1 * m2;
        do_not_optimize(r);
    });

    BENCHMARK("Matrix4x4f * Vector4f", {
        Vector4f r = m1 * v;
        do_not_optimize(r);
    });

    BENCHMARK("Matrix4x4f invert", {
        Matrix4x4f inv = invert(m2);
        do_not_optimize(inv);
    });

    BENCHMARK("Matrix3x3f invert", {
        Matrix3x3f m3;
        m3.set_row(0, Vector3f(1, 2, 3));
        m3.set_row(1, Vector3f(0, 1, 4));
        m3.set_row(2, Vector3f(5, 6, 0));
        Matrix3x3f inv = invert(m3);
        do_not_optimize(inv);
    });
}

// ---------------------------------------------------------------------------
// Quaternion operations
// ---------------------------------------------------------------------------
void benchmark_quaternion_operations() {
    std::cout << "\n=== Quaternion Operations ===" << std::endl;
    print_header();

    Quaternionf q1 = Quaternionf::from_angle_axis(0.5f, Vector3f(0, 1, 0));
    Quaternionf q2 = Quaternionf::from_angle_axis(0.3f, Vector3f(1, 0, 0));
    Vector3f v(1, 2, 3);

    BENCHMARK("Quaternion multiply", {
        Quaternionf r = q1 * q2;
        do_not_optimize(r);
    });

    BENCHMARK("Quaternion * Vector3f", {
        Vector3f r = q1 * v;
        do_not_optimize(r);
    });

    BENCHMARK("Quaternion normalize", {
        Quaternionf r = normalize(q1);
        do_not_optimize(r);
    });

    BENCHMARK("Quaternion from_angle_axis", {
        Quaternionf r = Quaternionf::from_angle_axis(0.5f, Vector3f(0, 1, 0));
        do_not_optimize(r);
    });

}

// ---------------------------------------------------------------------------
// Transform operations
// ---------------------------------------------------------------------------
void benchmark_transform_operations() {
    std::cout << "\n=== Transform Operations ===" << std::endl;
    print_header();

    Transform t1(Vector3f(1, 2, 3), Quaternionf::from_angle_axis(0.5f, Vector3f(0, 1, 0)), 1.0f);
    Transform t2(Vector3f(4, 5, 6), Quaternionf::from_angle_axis(0.3f, Vector3f(1, 0, 0)), 2.0f);
    Vector3f p(1, 2, 3);

    BENCHMARK("Transform multiply", {
        Transform r = t1 * t2;
        do_not_optimize(r);
    });

    BENCHMARK("Transform * point", {
        Vector3f r = t1 * p;
        do_not_optimize(r);
    });

}

// ---------------------------------------------------------------------------
// RNG operations
// ---------------------------------------------------------------------------
void benchmark_rng_operations() {
    std::cout << "\n=== RNG Operations ===" << std::endl;
    print_header();

    RNG::LinearCongruential lcg(12345);
    RNG::XorShift32 xorshift(12345);

    BENCHMARK("LinearCongruential sample1f", {
        float r = lcg.sample1f();
        do_not_optimize(r);
    });

    BENCHMARK("XorShift32 sample1f", {
        float r = xorshift.sample1f();
        do_not_optimize(r);
    });

    BENCHMARK("LinearCongruential sample2f", {
        Vector2f r = lcg.sample2f();
        do_not_optimize(r);
    });
}

// ---------------------------------------------------------------------------
// Distribution sampling
// ---------------------------------------------------------------------------
void benchmark_distribution_operations() {
    std::cout << "\n=== Distribution Operations ===" << std::endl;
    print_header();

    // Create a simple distribution
    float weights[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
    Distribution1D<float> dist1d(weights, 8);

    RNG::LinearCongruential rng(12345);

    BENCHMARK("Distribution1D sample_continuous", {
        auto sample = dist1d.sample_continuous(rng.sample1f());
        do_not_optimize(sample.index);
        do_not_optimize(sample.PDF);
    });

    BENCHMARK("Distribution1D sample_discrete", {
        auto sample = dist1d.sample_discrete(rng.sample1f());
        do_not_optimize(sample.index);
        do_not_optimize(sample.PDF);
    });
}

// ---------------------------------------------------------------------------
// Mesh operations
// ---------------------------------------------------------------------------
void benchmark_mesh_operations() {
    std::cout << "\n=== Mesh Operations ===" << std::endl;
    print_header();

    Assets::Meshes::allocate(16);

    BENCHMARK("MeshCreation::plane(10)", {
        Assets::Mesh m = Assets::MeshCreation::plane(10);
        do_not_optimize(m);
        m.destroy();
    });

    BENCHMARK("MeshCreation::box(1)", {
        Assets::Mesh m = Assets::MeshCreation::box(1);
        do_not_optimize(m);
        m.destroy();
    });

    BENCHMARK("MeshCreation::cylinder(16, 4)", {
        Assets::Mesh m = Assets::MeshCreation::cylinder(16, 4);
        do_not_optimize(m);
        m.destroy();
    });

    BENCHMARK("MeshCreation::revolved_sphere(16, 8)", {
        Assets::Mesh m = Assets::MeshCreation::revolved_sphere(16, 8);
        do_not_optimize(m);
        m.destroy();
    });

    Assets::Meshes::deallocate();
}

// ---------------------------------------------------------------------------
// Scene graph operations
// ---------------------------------------------------------------------------
void benchmark_scene_operations() {
    std::cout << "\n=== Scene Graph Operations ===" << std::endl;
    print_header();

    Scene::SceneNodes::allocate(1024);

    BENCHMARK("SceneNode create/destroy", {
        Scene::SceneNode n("test");
        do_not_optimize(n);
        n.destroy();
    });

    // Create a small hierarchy for traversal benchmarks
    Scene::SceneNode root("root");
    for (int i = 0; i < 10; ++i) {
        Scene::SceneNode child("child");
        child.set_parent(root);
    }

    BENCHMARK("SceneNode get_global_transform", {
        Transform t = root.get_global_transform();
        do_not_optimize(t);
    });

    BENCHMARK("SceneNode set_global_transform", {
        root.set_global_transform(Transform(Vector3f(1, 2, 3)));
    });

    root.destroy();
    Scene::SceneNodes::deallocate();
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------
int main(int argc, char* argv[]) {
    std::cout << "Bifrost Core Library Benchmarks" << std::endl;
    std::cout << "================================" << std::endl;

#if defined(__AVX2__)
    std::cout << "AVX2: Enabled" << std::endl;
#else
    std::cout << "AVX2: Disabled" << std::endl;
#endif

#if defined(__FMA__)
    std::cout << "FMA: Enabled" << std::endl;
#else
    std::cout << "FMA: Disabled" << std::endl;
#endif

    benchmark_vector_operations();
    benchmark_matrix_operations();
    benchmark_quaternion_operations();
    benchmark_transform_operations();
    benchmark_rng_operations();
    benchmark_distribution_operations();
    benchmark_mesh_operations();
    benchmark_scene_operations();

    std::cout << "\nBenchmarks complete." << std::endl;
    return 0;
}
