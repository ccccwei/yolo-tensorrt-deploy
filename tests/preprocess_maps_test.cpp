#include "../src/08_preprocess_maps/preprocess.hpp"
#include <opencv2/core.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

constexpr std::size_t INPUT_COUNT = 3U * 640U * 640U;
constexpr std::size_t GUARD = 16U;
constexpr std::size_t MAP_COUNT = 640U;
constexpr float SENTINEL = -12345.0F;
constexpr AxisMap MAP_SENTINEL{-12345, -12345, SENTINEL};

void check(cudaError_t status, const char* operation) {
    if (status != cudaSuccess)
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
}

struct Resources {
    unsigned char* image = nullptr;
    float* expected = nullptr;
    float* actual = nullptr;
    AxisMap* x = nullptr;
    AxisMap* y = nullptr;
    cudaStream_t stream = nullptr;
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t executable = nullptr;

    void clear_graph() {
        if (executable) check(cudaGraphExecDestroy(executable), "destroy executable");
        executable = nullptr;
        if (graph) check(cudaGraphDestroy(graph), "destroy graph");
        graph = nullptr;
    }
    ~Resources() {
        if (stream) cudaStreamSynchronize(stream);
        if (executable) cudaGraphExecDestroy(executable);
        if (graph) cudaGraphDestroy(graph);
        if (image) cudaFree(image);
        if (expected) cudaFree(expected);
        if (actual) cudaFree(actual);
        if (x) cudaFree(x);
        if (y) cudaFree(y);
        if (stream) cudaStreamDestroy(stream);
    }
};

void fill_image(cv::Mat& image, bool contrast, int pass, std::mt19937& random) {
    std::uniform_int_distribution<int> byte(0, 255);
    for (int y = 0; y < image.rows; ++y) {
        auto* row = image.ptr<unsigned char>(y);
        for (std::size_t x = 0; x < static_cast<std::size_t>(image.cols) * 3U; ++x)
            row[x] = contrast
                ? static_cast<unsigned char>(((x / 3U + static_cast<std::size_t>(y + pass)) % 2U) * 255U)
                : static_cast<unsigned char>(byte(random));
    }
}

void compare_output(const std::vector<float>& expected, const std::vector<float>& actual,
                    const std::string& name) {
    for (std::size_t index = 0; index < actual.size(); ++index) {
        if (std::memcmp(&actual[index], &expected[index], sizeof(float)) != 0)
            throw std::runtime_error(name + ": bitwise mismatch at " + std::to_string(index)
                + ", reference=" + std::to_string(expected[index])
                + ", maps=" + std::to_string(actual[index]));
        if ((index < GUARD || index >= GUARD + INPUT_COUNT) && actual[index] != SENTINEL)
            throw std::runtime_error(name + ": output guard overwritten");
        if (index >= GUARD && index < GUARD + INPUT_COUNT
            && (!std::isfinite(actual[index]) || actual[index] < 0.0F || actual[index] > 1.0F))
            throw std::runtime_error(name + ": output outside [0,1]");
    }
}

void test_image(Resources& resources, cv::Mat& image, bool contrast, std::mt19937& random) {
    resources.clear_graph(); // Destroy old captured pitch/pointers before reallocating.
    if (resources.image) check(cudaFree(resources.image), "free previous image");
    resources.image = nullptr;
    std::size_t pitch = 0U;
    const std::size_t row_bytes = static_cast<std::size_t>(image.cols) * 3U;
    check(cudaMallocPitch(reinterpret_cast<void**>(&resources.image), &pitch, row_bytes,
                          static_cast<std::size_t>(image.rows)), "allocate pitched image");

    std::vector<AxisMap> x_table(MAP_COUNT + 2U * GUARD, MAP_SENTINEL);
    std::vector<AxisMap> y_table = x_table;
    std::vector<float> expected(INPUT_COUNT + 2U * GUARD, SENTINEL);
    std::vector<float> actual = expected;
    const std::size_t output_bytes = actual.size() * sizeof(float);
    const std::size_t map_bytes = x_table.size() * sizeof(AxisMap);
    try {
        check(cudaMemcpyAsync(resources.x, x_table.data(), map_bytes,
                              cudaMemcpyHostToDevice, resources.stream), "initialize x guards");
        check(cudaMemcpyAsync(resources.y, y_table.data(), map_bytes,
                              cudaMemcpyHostToDevice, resources.stream), "initialize y guards");
        check(build_preprocess_maps(image.cols, image.rows, resources.x + GUARD,
                                    resources.y + GUARD, resources.stream), "build maps");
        check(cudaMemcpyAsync(x_table.data(), resources.x, map_bytes,
                              cudaMemcpyDeviceToHost, resources.stream), "read x maps");
        check(cudaMemcpyAsync(y_table.data(), resources.y, map_bytes,
                              cudaMemcpyDeviceToHost, resources.stream), "read y maps");
        check(cudaStreamSynchronize(resources.stream), "synchronize maps");
        for (std::size_t i = 0; i < x_table.size(); ++i) {
            for (const auto* table : {&x_table, &y_table}) {
                const auto& entry = (*table)[i];
                if (i < GUARD || i >= GUARD + MAP_COUNT) {
                    if (entry.lower != MAP_SENTINEL.lower || entry.upper != MAP_SENTINEL.upper
                        || entry.fraction != MAP_SENTINEL.fraction)
                        throw std::runtime_error("map guard overwritten");
                } else if (entry.lower != -1) {
                    const int size = table == &x_table ? image.cols : image.rows;
                    if (entry.lower < 0 || entry.upper < entry.lower || entry.upper >= size
                        || !std::isfinite(entry.fraction) || entry.fraction < 0.0F
                        || entry.fraction >= 1.0F)
                        throw std::runtime_error("invalid interpolation map");
                }
            }
        }

        // Reuse the same tables for new pixels; two passes replay a captured kernel.
        for (int pass = 0; pass < 3; ++pass) {
            fill_image(image, contrast, pass, random);
            check(cudaMemcpy2DAsync(resources.image, pitch, image.data, image.step[0],
                                   row_bytes, static_cast<std::size_t>(image.rows),
                                   cudaMemcpyHostToDevice, resources.stream), "upload pixels");
            std::fill(expected.begin(), expected.end(), SENTINEL);
            std::fill(actual.begin(), actual.end(), SENTINEL);
            check(cudaMemcpyAsync(resources.expected, expected.data(), output_bytes,
                                  cudaMemcpyHostToDevice, resources.stream), "initialize reference");
            check(cudaMemcpyAsync(resources.actual, actual.data(), output_bytes,
                                  cudaMemcpyHostToDevice, resources.stream), "initialize output");
            check(preprocess_cuda(resources.image, image.cols, image.rows, pitch,
                                  resources.expected + GUARD, resources.stream), "reference kernel");
            if (pass == 0) {
                check(preprocess_maps_cuda(resources.image, image.cols, image.rows, pitch,
                                           resources.x + GUARD, resources.y + GUARD,
                                           resources.actual + GUARD, resources.stream), "maps kernel");
            } else {
                if (pass == 1) {
                    check(cudaStreamSynchronize(resources.stream), "prime before capture");
                    check(cudaStreamBeginCapture(resources.stream, cudaStreamCaptureModeThreadLocal),
                          "begin capture");
                    const auto launch = preprocess_maps_cuda(
                        resources.image, image.cols, image.rows, pitch,
                        resources.x + GUARD, resources.y + GUARD,
                        resources.actual + GUARD, resources.stream);
                    const auto capture = cudaStreamEndCapture(resources.stream, &resources.graph);
                    check(launch, "capture maps kernel");
                    check(capture, "end capture");
                    check(cudaGraphInstantiate(&resources.executable, resources.graph,
                                               nullptr, nullptr, 0), "instantiate");
                }
                check(cudaGraphLaunch(resources.executable, resources.stream), "replay maps kernel");
            }
            check(cudaMemcpyAsync(expected.data(), resources.expected, output_bytes,
                                  cudaMemcpyDeviceToHost, resources.stream), "read reference");
            check(cudaMemcpyAsync(actual.data(), resources.actual, output_bytes,
                                  cudaMemcpyDeviceToHost, resources.stream), "read output");
            check(cudaStreamSynchronize(resources.stream), "synchronize inference");
            compare_output(expected, actual, std::to_string(image.cols) + "x"
                           + std::to_string(image.rows) + " pass " + std::to_string(pass));
        }
    } catch (...) {
        cudaStreamSynchronize(resources.stream); // Host vectors must outlive submitted DMA.
        throw;
    }
    std::cout << "PASS " << image.cols << 'x' << image.rows
              << (image.isContinuous() ? "" : " ROI") << ": 3 bitwise-equal passes\n";
}

int main() {
    try {
        Resources resources;
        check(cudaStreamCreateWithFlags(&resources.stream, cudaStreamNonBlocking), "create stream");
        const std::size_t output_bytes = (INPUT_COUNT + 2U * GUARD) * sizeof(float);
        const std::size_t map_bytes = (MAP_COUNT + 2U * GUARD) * sizeof(AxisMap);
        check(cudaMalloc(reinterpret_cast<void**>(&resources.expected), output_bytes), "allocate reference");
        check(cudaMalloc(reinterpret_cast<void**>(&resources.actual), output_bytes), "allocate output");
        check(cudaMalloc(reinterpret_cast<void**>(&resources.x), map_bytes), "allocate x maps");
        check(cudaMalloc(reinterpret_cast<void**>(&resources.y), map_bytes), "allocate y maps");
        struct Case { int width; int height; bool roi; bool contrast; };
        const std::array<Case, 17> cases{{
            {640, 640, false, false}, {1, 1, false, false},
            {1280, 720, false, false}, {720, 1280, false, false},
            {641, 479, false, false}, {479, 641, false, false},
            {2001, 1, false, false}, {1, 2001, false, false},
            {2, 3, false, false}, {3, 2, false, false},
            {639, 427, true, false}, {640, 640, true, false},
            {100001, 1, false, true}, {1, 100001, false, true},
            {320, 320, true, true}, {810, 1080, true, true},
            {1280, 720, false, false} // Return to an earlier size after rebuilding other maps.
        }};
        std::mt19937 random(2026U);
        for (const auto& test : cases) {
            cv::Mat storage(test.height + (test.roi ? 4 : 0),
                            test.width + (test.roi ? 7 : 0), CV_8UC3);
            cv::Mat image = test.roi ? storage(cv::Rect(3, 2, test.width, test.height)) : storage;
            test_image(resources, image, test.contrast, random);
        }
        const auto invalid = [](cudaError_t result) {
            if (result != cudaErrorInvalidValue) throw std::runtime_error("invalid argument accepted");
        };
        invalid(build_preprocess_maps(0, 1, resources.x, resources.y, resources.stream));
        invalid(build_preprocess_maps(1, -1, resources.x, resources.y, resources.stream));
        invalid(build_preprocess_maps(1, 1, nullptr, resources.y, resources.stream));
        invalid(build_preprocess_maps(1, 1, resources.x, nullptr, resources.stream));
        invalid(build_preprocess_maps(1, 1, resources.x, resources.x, resources.stream));
        invalid(preprocess_maps_cuda(nullptr, 1, 1, 3U, resources.x, resources.y, resources.actual, resources.stream));
        invalid(preprocess_maps_cuda(resources.image, 1, 1, 3U, resources.x, resources.y, nullptr, resources.stream));
        invalid(preprocess_maps_cuda(resources.image, 0, 1, 3U, resources.x, resources.y, resources.actual, resources.stream));
        invalid(preprocess_maps_cuda(resources.image, 1, -1, 3U, resources.x, resources.y, resources.actual, resources.stream));
        invalid(preprocess_maps_cuda(resources.image, 2, 1, 5U, resources.x, resources.y, resources.actual, resources.stream));
        invalid(preprocess_maps_cuda(resources.image, 1, 1, 3U, nullptr, resources.y, resources.actual, resources.stream));
        invalid(preprocess_maps_cuda(resources.image, 1, 1, 3U, resources.x, nullptr, resources.actual, resources.stream));
        invalid(preprocess_maps_cuda(resources.image, 1, 1, 3U, resources.x, resources.x, resources.actual, resources.stream));
        std::cout << "PASS invalid arguments\nAll lookup-table preprocessing tests passed.\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "FAIL " << error.what() << '\n';
        return 1;
    }
}
