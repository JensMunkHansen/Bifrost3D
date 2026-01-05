// Simple cross-platform benchmark harness for Bifrost.
// ---------------------------------------------------------------------------
// Copyright (C) Bifrost. See AUTHORS.txt for authors.
//
// This program is open source and distributed under the New BSD License.
// See LICENSE.txt for more detail.
// ---------------------------------------------------------------------------

#ifndef _BIFROST_BENCHMARK_H_
#define _BIFROST_BENCHMARK_H_

#include <algorithm>
#include <atomic>
#include <chrono>
#include <functional>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

namespace Bifrost {
namespace Benchmark {

struct Result {
    std::string name;
    double min_ns;
    double max_ns;
    double avg_ns;
    double median_ns;
    size_t iterations;
};

class Timer {
public:
    using Clock = std::chrono::high_resolution_clock;
    using TimePoint = Clock::time_point;
    using Duration = std::chrono::nanoseconds;

    void start() { m_start = Clock::now(); }
    void stop() { m_end = Clock::now(); }

    double elapsed_ns() const {
        return std::chrono::duration_cast<Duration>(m_end - m_start).count();
    }

    double elapsed_us() const { return elapsed_ns() / 1000.0; }
    double elapsed_ms() const { return elapsed_ns() / 1000000.0; }
    double elapsed_s() const { return elapsed_ns() / 1000000000.0; }

private:
    TimePoint m_start;
    TimePoint m_end;
};

// Use volatile to prevent compiler from optimizing away the result.
// This provides a cross-platform way to ensure the computation actually happens.
template<typename T>
inline void do_not_optimize(T const& value) {
    volatile T sink = value;
    (void)sink;
}

// Prevent compiler from reordering across this barrier
inline void clobber_memory() {
    std::atomic_thread_fence(std::memory_order_seq_cst);
}

class Benchmark {
public:
    Benchmark(const std::string& name, size_t warmup_iterations = 10, size_t min_iterations = 100)
        : m_name(name)
        , m_warmup_iterations(warmup_iterations)
        , m_min_iterations(min_iterations) {}

    template<typename Func>
    Result run(Func&& func) {
        // Warmup
        for (size_t i = 0; i < m_warmup_iterations; ++i) {
            func();
        }
        clobber_memory();

        // Determine iteration count for ~100ms of runtime
        Timer calibration_timer;
        calibration_timer.start();
        for (size_t i = 0; i < 100; ++i) {
            func();
        }
        calibration_timer.stop();

        double ns_per_iter = calibration_timer.elapsed_ns() / 100.0;
        size_t target_iterations = std::max(m_min_iterations,
            static_cast<size_t>(100000000.0 / ns_per_iter)); // Target ~100ms

        // Collect samples
        std::vector<double> samples;
        samples.reserve(target_iterations);

        Timer iter_timer;
        for (size_t i = 0; i < target_iterations; ++i) {
            iter_timer.start();
            func();
            iter_timer.stop();
            samples.push_back(iter_timer.elapsed_ns());
        }

        // Compute statistics
        std::sort(samples.begin(), samples.end());

        double sum = 0;
        for (double s : samples) sum += s;

        Result result;
        result.name = m_name;
        result.iterations = target_iterations;
        result.min_ns = samples.front();
        result.max_ns = samples.back();
        result.avg_ns = sum / samples.size();
        result.median_ns = samples[samples.size() / 2];

        return result;
    }

private:
    std::string m_name;
    size_t m_warmup_iterations;
    size_t m_min_iterations;
};

inline void print_result(const Result& r) {
    std::cout << std::left << std::setw(50) << r.name
              << std::right << std::setw(12) << std::fixed << std::setprecision(1) << r.median_ns << " ns"
              << "  (min: " << std::setw(10) << r.min_ns << " ns, "
              << "max: " << std::setw(10) << r.max_ns << " ns, "
              << "n=" << r.iterations << ")"
              << std::endl;
}

inline void print_header() {
    std::cout << std::string(100, '-') << std::endl;
    std::cout << std::left << std::setw(50) << "Benchmark"
              << std::right << std::setw(15) << "Median"
              << "  Details" << std::endl;
    std::cout << std::string(100, '-') << std::endl;
}

} // namespace Benchmark
} // namespace Bifrost

#define BENCHMARK(name, code) \
    { \
        Bifrost::Benchmark::Benchmark bench(name); \
        auto result = bench.run([&]() { code; }); \
        Bifrost::Benchmark::print_result(result); \
    }

#endif // _BIFROST_BENCHMARK_H_
