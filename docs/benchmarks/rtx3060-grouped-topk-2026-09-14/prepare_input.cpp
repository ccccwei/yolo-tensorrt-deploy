#include <cuda_runtime_api.h>
#include <opencv2/imgcodecs.hpp>
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

cudaError_t preprocess_cuda(const unsigned char*, int, int, std::size_t, float*, cudaStream_t);

void check(cudaError_t status) {
    if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}

struct Buffers {
    unsigned char* pixels = nullptr;
    float* tensor = nullptr;
    cudaStream_t stream = nullptr;
    ~Buffers() {
        if (stream) cudaStreamSynchronize(stream);
        if (pixels) cudaFree(pixels);
        if (tensor) cudaFree(tensor);
        if (stream) cudaStreamDestroy(stream);
    }
};

int main(int argc, char** argv) {
    try {
        if (argc != 3) throw std::runtime_error("IMAGE OUTPUT.bin required");
        cv::Mat image = cv::imread(argv[1], cv::IMREAD_COLOR);
        if (image.empty()) throw std::runtime_error("Cannot decode image");
        if (!image.isContinuous()) image = image.clone();
        const auto bytes = image.total() * image.elemSize();
        constexpr std::size_t count = 3U * 640U * 640U;
        Buffers buffers;
        check(cudaStreamCreateWithFlags(&buffers.stream, cudaStreamNonBlocking));
        check(cudaMalloc(reinterpret_cast<void**>(&buffers.pixels), bytes));
        check(cudaMalloc(reinterpret_cast<void**>(&buffers.tensor), count * sizeof(float)));
        check(cudaMemcpyAsync(buffers.pixels, image.data, bytes, cudaMemcpyHostToDevice, buffers.stream));
        check(preprocess_cuda(buffers.pixels, image.cols, image.rows, image.step[0], buffers.tensor, buffers.stream));
        std::vector<float> output(count);
        check(cudaMemcpyAsync(output.data(), buffers.tensor, count * sizeof(float), cudaMemcpyDeviceToHost, buffers.stream));
        check(cudaStreamSynchronize(buffers.stream));
        if (!std::all_of(output.begin(), output.end(), [](float x) { return std::isfinite(x) && x >= 0.0F && x <= 1.0F; }))
            throw std::runtime_error("Invalid preprocessed input");
        std::ofstream file(argv[2], std::ios::binary);
        file.write(reinterpret_cast<const char*>(output.data()), static_cast<std::streamsize>(count * sizeof(float)));
        if (!file) throw std::runtime_error("Cannot write tensor");
        std::cout << argv[1] << ": " << image.cols << 'x' << image.rows << " -> " << count << " FP32 values\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}

