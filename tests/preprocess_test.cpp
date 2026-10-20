#include <cuda_runtime_api.h>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

cudaError_t preprocess_cuda(const unsigned char* image, int width, int height,
                            std::size_t pitch, float* input, cudaStream_t stream);

constexpr int INPUT_SIZE = 640;
constexpr std::size_t PLANE_SIZE = 640U * 640U;
constexpr std::size_t INPUT_COUNT = 3U * PLANE_SIZE;
constexpr std::size_t GUARD_SIZE = 32U;
constexpr float SENTINEL = -12345.0F;

void check_cuda(cudaError_t error, const char* operation) {
    if (error != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(error));
    }
}

struct Resources {
    unsigned char* image = nullptr;
    float* output = nullptr;
    cudaStream_t stream = nullptr;

    ~Resources() {
        const auto report = [](cudaError_t error, const char* operation) {
            if (error != cudaSuccess) {
                std::cerr << operation << ": " << cudaGetErrorString(error) << '\n';
            }
        };
        if (stream) report(cudaStreamSynchronize(stream), "cleanup synchronize");
        if (image) report(cudaFree(image), "cudaFree image");
        if (output) report(cudaFree(output), "cudaFree output");
        if (stream) report(cudaStreamDestroy(stream), "cudaStreamDestroy");
    }
};

std::vector<float> cpu_reference(const cv::Mat& image) {
    const float scale = std::min(
        static_cast<float>(INPUT_SIZE) / static_cast<float>(image.cols),
        static_cast<float>(INPUT_SIZE) / static_cast<float>(image.rows));
    const int width = std::max(1, static_cast<int>(std::round(
        static_cast<float>(image.cols) * scale)));
    const int height = std::max(1, static_cast<int>(std::round(
        static_cast<float>(image.rows) * scale)));
    const int pad_x = (INPUT_SIZE - width) / 2;
    const int pad_y = (INPUT_SIZE - height) / 2;
    cv::Mat resized, padded;
    cv::resize(image, resized, cv::Size(width, height), 0.0, 0.0, cv::INTER_LINEAR);
    cv::copyMakeBorder(resized, padded,
                      pad_y, INPUT_SIZE - height - pad_y,
                      pad_x, INPUT_SIZE - width - pad_x,
                      cv::BORDER_CONSTANT, cv::Scalar(114, 114, 114));
    std::vector<float> output(INPUT_COUNT);
    for (int y = 0; y < INPUT_SIZE; ++y) {
        const cv::Vec3b* row = padded.ptr<cv::Vec3b>(y);
        for (int x = 0; x < INPUT_SIZE; ++x) {
            const std::size_t i = static_cast<std::size_t>(y) * 640U
                                + static_cast<std::size_t>(x);
            for (std::size_t c = 0; c < 3U; ++c) {
                output[c * PLANE_SIZE + i] =
                    static_cast<float>(row[x][2 - static_cast<int>(c)]) * (1.0F / 255.0F);
            }
        }
    }
    return output;
}

void test_image(Resources& resources, const cv::Mat& image, const std::string& name) {
    const auto expected = cpu_reference(image);
    std::vector<float> actual(INPUT_COUNT + 2U * GUARD_SIZE, SENTINEL);
    const std::size_t output_bytes = actual.size() * sizeof(float);
    const std::size_t row_bytes = static_cast<std::size_t>(image.cols) * 3U;
    std::size_t device_pitch = 0U;
    if (resources.image) {
        check_cuda(cudaFree(resources.image), "cudaFree previous image");
        resources.image = nullptr;
    }
    check_cuda(cudaMallocPitch(reinterpret_cast<void**>(&resources.image), &device_pitch,
                               row_bytes, static_cast<std::size_t>(image.rows)),
               "cudaMallocPitch");
    try {
        // 同时验证 CPU ROI 步长和 GPU pitched allocation，不能假定图片连续。
        check_cuda(cudaMemcpy2DAsync(resources.image, device_pitch, image.data, image.step[0],
                                     row_bytes, static_cast<std::size_t>(image.rows),
                                     cudaMemcpyHostToDevice, resources.stream),
                   "cudaMemcpy2DAsync H2D");
        check_cuda(cudaMemcpyAsync(resources.output, actual.data(), output_bytes,
                                   cudaMemcpyHostToDevice, resources.stream), "initialize guards");
        check_cuda(preprocess_cuda(resources.image, image.cols, image.rows, device_pitch,
                                    resources.output + GUARD_SIZE, resources.stream),
                   "preprocess_cuda");
        check_cuda(cudaMemcpyAsync(actual.data(), resources.output, output_bytes,
                                   cudaMemcpyDeviceToHost, resources.stream), "cudaMemcpyAsync D2H");
        check_cuda(cudaStreamSynchronize(resources.stream), "cudaStreamSynchronize");
    } catch (...) {
        // Host 缓冲区销毁前先等待已经提交的 DMA / kernel。
        cudaStreamSynchronize(resources.stream);
        throw;
    }
    for (std::size_t i = 0; i < GUARD_SIZE; ++i) {
        if (actual[i] != SENTINEL || actual[GUARD_SIZE + INPUT_COUNT + i] != SENTINEL) {
            throw std::runtime_error(name + ": output guard overwritten");
        }
    }
    // OpenCV uint8 resize 有取整；浮点双线性插值容许最多一个灰度级误差。
    const float tolerance = image.cols == INPUT_SIZE && image.rows == INPUT_SIZE
                                ? 1.0e-7F : 1.0F / 255.0F + 1.0e-6F;
    float max_error = 0.0F;
    for (std::size_t i = 0; i < INPUT_COUNT; ++i) {
        const float value = actual[GUARD_SIZE + i];
        const float error = std::abs(value - expected[i]);
        if (!std::isfinite(value) || value < 0.0F || value > 1.0F || error > tolerance) {
            throw std::runtime_error(name + ": mismatch at " + std::to_string(i)
                                     + ", expected=" + std::to_string(expected[i])
                                     + ", actual=" + std::to_string(value));
        }
        max_error = std::max(max_error, error);
    }
    std::cout << "PASS " << name << ": max_abs_error=" << max_error << '\n';
}

int main() {
    try {
        Resources resources;
        check_cuda(cudaStreamCreate(&resources.stream), "cudaStreamCreate");
        check_cuda(cudaMalloc(reinterpret_cast<void**>(&resources.output),
                               (INPUT_COUNT + 2U * GUARD_SIZE) * sizeof(float)), "cudaMalloc output");
        struct Case { int width; int height; bool roi; bool high_contrast = false; };
        const std::array<Case, 14> cases{{
            {640, 640, false}, {1, 1, false},
            {1280, 720, false}, {720, 1280, false},
            {641, 479, false}, {479, 641, false},
            {2001, 1, false}, {1, 2001, false},
            {2, 3, false}, {3, 2, false},
            {639, 427, true}, {640, 640, true},
            {100001, 1, false, true}, {1, 100001, false, true}
        }};
        std::mt19937 random(2026U);
        std::uniform_int_distribution<int> byte(0, 255);
        for (const Case& test : cases) {
            cv::Mat storage(test.height + (test.roi ? 4 : 0),
                            test.width + (test.roi ? 7 : 0), CV_8UC3);
            cv::Mat image = test.roi
                ? storage(cv::Rect(3, 2, test.width, test.height)) : storage;
            for (int y = 0; y < image.rows; ++y) {
                unsigned char* row = image.ptr<unsigned char>(y);
                for (std::size_t x = 0; x < static_cast<std::size_t>(image.cols) * 3U; ++x) {
                    row[x] = test.high_contrast
                        ? static_cast<unsigned char>(((x / 3U + static_cast<std::size_t>(y)) % 2U) * 255U)
                        : static_cast<unsigned char>(byte(random));
                }
            }
            test_image(resources, image, std::to_string(test.width) + "x"
                       + std::to_string(test.height) + (test.roi ? " ROI" : "")
                       + (test.high_contrast ? " high contrast" : ""));
        }
        const auto invalid = [](cudaError_t result, const char* name) {
            if (result != cudaErrorInvalidValue) {
                throw std::runtime_error(std::string(name) + ": expected cudaErrorInvalidValue, got "
                                         + cudaGetErrorString(result));
            }
        };
        invalid(preprocess_cuda(nullptr, 1, 1, 3U, resources.output, resources.stream), "null image");
        invalid(preprocess_cuda(resources.image, 1, 1, 3U, nullptr, resources.stream), "null output");
        invalid(preprocess_cuda(resources.image, 0, 1, 3U, resources.output, resources.stream), "zero width");
        invalid(preprocess_cuda(resources.image, 1, 0, 3U, resources.output, resources.stream), "zero height");
        invalid(preprocess_cuda(resources.image, -1, 1, 3U, resources.output, resources.stream), "negative width");
        invalid(preprocess_cuda(resources.image, 1, -1, 3U, resources.output, resources.stream), "negative height");
        invalid(preprocess_cuda(resources.image, 2, 1, 5U, resources.output, resources.stream), "short pitch");
        std::cout << "PASS invalid arguments\nAll preprocessing tests passed.\n";
    } catch (const std::exception& error) {
        std::cerr << "FAIL " << error.what() << '\n';
        return 1;
    }
    return 0;
}
