#include "../src/02_cuda_prepost/postprocess.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

constexpr std::size_t OUTPUT_COUNT = 300U * 6U;
constexpr std::size_t GUARD_BYTES = 64U;
constexpr unsigned char SENTINEL = 0xA5;
using Candidate = std::array<float, 6>;

void check(cudaError_t error, const char* operation) {
    if (error != cudaSuccess)
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(error));
}

struct Resources {
    float* output = nullptr;
    unsigned char* storage = nullptr;
    cudaStream_t stream = nullptr;
    ~Resources() {
        if (stream) cudaStreamSynchronize(stream);
        if (output) cudaFree(output);
        if (storage) cudaFree(storage);
        if (stream) cudaStreamDestroy(stream);
    }
};

struct Case {
    std::string name;
    int width = 1280, height = 720;
    float scale = 0.5F;
    int pad_x = 0, pad_y = 140;
    std::vector<float> output = std::vector<float>(OUTPUT_COUNT, 0.0F);
};

void set(Case& test, std::size_t index, const Candidate& candidate) {
    std::copy(candidate.begin(), candidate.end(), test.output.data() + index * 6U);
}

std::vector<GpuDetection> cpu_reference(const Case& test) {
    std::vector<GpuDetection> boxes;
    for (int i = 0; i < 300; ++i) {
        const float* det = test.output.data() + i * 6;
        if (!std::all_of(det, det + 6, [](float value) { return std::isfinite(value); })
            || det[4] < 0.25F || det[5] < 0.0F
            || static_cast<double>(det[5]) > static_cast<double>(std::numeric_limits<int>::max())) continue;
        const auto coordinate = [&test](float value, int padding, int size) {
            const float restored = (value - static_cast<float>(padding)) / test.scale;
            return static_cast<int>(std::clamp(static_cast<double>(restored),
                                              0.0, static_cast<double>(size - 1)));
        };
        const GpuDetection box{static_cast<int>(det[5]), det[4],
                               coordinate(det[0], test.pad_x, test.width),
                               coordinate(det[1], test.pad_y, test.height),
                               coordinate(det[2], test.pad_x, test.width),
                               coordinate(det[3], test.pad_y, test.height)};
        if (box.right > box.left && box.bottom > box.top) boxes.push_back(box);
    }
    return boxes;
}

void run(Resources& cuda, const Case& test) {
    const auto expected = cpu_reference(test);
    std::vector<unsigned char> actual(sizeof(GpuResults) + 2U * GUARD_BYTES);
    auto* result = reinterpret_cast<GpuResults*>(cuda.storage + GUARD_BYTES);
    try {
        check(cudaMemcpyAsync(cuda.output, test.output.data(), OUTPUT_COUNT * sizeof(float),
                              cudaMemcpyHostToDevice, cuda.stream), "copy candidates");
        check(postprocess_cuda(cuda.output, test.width, test.height, test.scale,
                               test.pad_x, test.pad_y, result, cuda.stream), "postprocess_cuda");
        check(cudaMemcpyAsync(actual.data(), cuda.storage, actual.size(),
                              cudaMemcpyDeviceToHost, cuda.stream), "copy results and guards");
        check(cudaStreamSynchronize(cuda.stream), "synchronize");
    } catch (...) {
        cudaStreamSynchronize(cuda.stream); // Keep host source/destination alive until DMA finishes.
        throw;
    }
    for (std::size_t i = 0; i < GUARD_BYTES; ++i) {
        if (actual[i] != SENTINEL || actual[GUARD_BYTES + sizeof(GpuResults) + i] != SENTINEL)
            throw std::runtime_error(test.name + ": result guard overwritten");
    }
    GpuResults decoded{};
    std::memcpy(&decoded, actual.data() + GUARD_BYTES, sizeof(decoded));
    if (decoded.count != static_cast<int>(expected.size()))
        throw std::runtime_error(test.name + ": count mismatch, expected "
                                 + std::to_string(expected.size()) + ", got " + std::to_string(decoded.count));
    for (std::size_t i = 0; i < expected.size(); ++i) {
        const auto& wanted = expected[i];
        const auto& got = decoded.boxes[i];
        if (got.class_id != wanted.class_id || got.score != wanted.score
            || got.left != wanted.left || got.top != wanted.top
            || got.right != wanted.right || got.bottom != wanted.bottom)
            throw std::runtime_error(test.name + ": ordered box mismatch at " + std::to_string(i));
    }
    std::cout << "PASS " << test.name << ": " << expected.size() << " ordered boxes\n";
}

int main() {
    try {
        Resources cuda;
        check(cudaStreamCreate(&cuda.stream), "create stream");
        check(cudaMalloc(reinterpret_cast<void**>(&cuda.output), OUTPUT_COUNT * sizeof(float)), "allocate candidates");
        check(cudaMalloc(reinterpret_cast<void**>(&cuda.storage), sizeof(GpuResults) + 2U * GUARD_BYTES), "allocate guarded result");
        check(cudaMemsetAsync(cuda.storage, SENTINEL, sizeof(GpuResults) + 2U * GUARD_BYTES, cuda.stream), "initialize guards");
        Case all_valid{"all 300 valid"};
        for (std::size_t i = 0; i < 300U; ++i)
            set(all_valid, i, {10.0F, 150.0F, 500.0F, 470.0F, 0.75F, static_cast<float>(i)});
        const Case empty{"none valid"};
        // Same device result reused: count must reset when a valid frame is followed by an empty one.
        for (int repeat = 0; repeat < 3; ++repeat) {
            run(cuda, all_valid);
            run(cuda, empty);
        }

        Case edges{"threshold, nonfinite, class and box boundaries"};
        const Candidate base{10.0F, 150.0F, 500.0F, 470.0F, 0.75F, 2.0F};
        for (std::size_t i = 0; i < 300U; ++i) set(edges, i, base);
        for (std::size_t field = 0; field < 6U; ++field) {
            auto candidate = base;
            candidate[field] = std::numeric_limits<float>::quiet_NaN();
            set(edges, field, candidate);
            candidate[field] = std::numeric_limits<float>::infinity();
            set(edges, field + 6U, candidate);
            candidate[field] = -std::numeric_limits<float>::infinity();
            set(edges, field + 12U, candidate);
        }
        const std::array<Candidate, 15> special{{
            {10, 150, 500, 470, 0.25F, 0},
            {10, 150, 500, 470, std::nextafter(0.25F, 0.0F), 0},
            {10, 150, 500, 470, std::nextafter(0.25F, 1.0F), 0},
            {10, 150, 500, 470, 0.8F, -1},
            {10, 150, 500, 470, 0.8F, 2.75F}, // Preserve the baseline's truncating class conversion.
            {10, 150, 500, 470, 0.8F, 2147483648.0F},
            {10, 150, 500, 470, 0.8F, std::nextafter(2147483648.0F, 0.0F)},
            {-100, -100, 1000, 1000, 0.8F, 7},
            {10, 150, 10, 470, 0.8F, 8},
            {10, 150, 500, 150, 0.8F, 9},
            {500, 150, 10, 470, 0.8F, 10},
            {-100, 150, -10, 470, 0.8F, 11},
            {1000, 150, 2000, 470, 0.8F, 12},
            {-std::numeric_limits<float>::max(), -std::numeric_limits<float>::max(),
             std::numeric_limits<float>::max(), std::numeric_limits<float>::max(), 0.8F, 13},
            {10.1F, 150.1F, 10.4F, 150.4F, 0.8F, 14}
        }};
        for (std::size_t i = 0; i < special.size(); ++i) set(edges, i + 18U, special[i]);
        run(cuda, edges);

        std::mt19937 random(2026U);
        std::uniform_real_distribution<float> value(-500.0F, 1200.0F);
        std::uniform_real_distribution<float> score(0.0F, 1.0F);
        for (const float scale : {1.0F, 0.33333334F, 1.7F, 0.00001F}) {
            Case test{"random, odd padding, scale=" + std::to_string(scale)};
            test.width = 641;
            test.height = 479;
            test.scale = scale;
            test.pad_x = 3;
            test.pad_y = 81;
            for (std::size_t i = 0; i < 300U; ++i)
                set(test, i, {value(random), value(random), value(random), value(random), score(random), static_cast<float>(i)});
            run(cuda, test);
        }
        Case one_pixel = all_valid;
        one_pixel.name = "one-pixel width / no nonempty rectangles";
        one_pixel.width = 1;
        run(cuda, one_pixel);
        Case overflow = edges;
        overflow.name = "division overflow clamps safely";
        overflow.scale = std::numeric_limits<float>::min();
        run(cuda, overflow);

        auto* result = reinterpret_cast<GpuResults*>(cuda.storage + GUARD_BYTES);
        const auto invalid = [](cudaError_t status) {
            if (status != cudaErrorInvalidValue) throw std::runtime_error("invalid argument not rejected");
        };
        invalid(postprocess_cuda(nullptr, 640, 640, 1.0F, 0, 0, result, cuda.stream));
        invalid(postprocess_cuda(cuda.output, 640, 640, 1.0F, 0, 0, nullptr, cuda.stream));
        invalid(postprocess_cuda(cuda.output, 0, 640, 1.0F, 0, 0, result, cuda.stream));
        invalid(postprocess_cuda(cuda.output, 640, -1, 1.0F, 0, 0, result, cuda.stream));
        for (const float scale : {0.0F, -1.0F, std::numeric_limits<float>::quiet_NaN(), std::numeric_limits<float>::infinity()})
            invalid(postprocess_cuda(cuda.output, 640, 640, scale, 0, 0, result, cuda.stream));
        std::cout << "PASS invalid arguments\nAll GPU postprocessing tests passed.\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "FAIL " << error.what() << '\n';
        return 1;
    }
}
