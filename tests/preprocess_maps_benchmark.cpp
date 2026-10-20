#include "../src/08_preprocess_maps/preprocess.hpp"

#include <algorithm>
#include <array>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

constexpr int CALLS_PER_SAMPLE = 100;
constexpr int SAMPLES = 100;
constexpr int WARMUP = 20;
constexpr int ROUNDS = 3;

void check(cudaError_t status, const char* operation) {
    if (status != cudaSuccess)
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
}

struct Resources {
    unsigned char* image = nullptr;
    float* input = nullptr;
    AxisMap* x_map = nullptr;
    AxisMap* y_map = nullptr;
    cudaStream_t stream = nullptr;
    cudaEvent_t begin = nullptr;
    cudaEvent_t end = nullptr;
    std::array<cudaGraph_t, 2> graphs{};
    std::array<cudaGraphExec_t, 2> executables{};

    ~Resources() {
        if (stream) cudaStreamSynchronize(stream);
        for (auto executable : executables) if (executable) cudaGraphExecDestroy(executable);
        for (auto graph : graphs) if (graph) cudaGraphDestroy(graph);
        if (begin) cudaEventDestroy(begin);
        if (end) cudaEventDestroy(end);
        if (image) cudaFree(image);
        if (input) cudaFree(input);
        if (x_map) cudaFree(x_map);
        if (y_map) cudaFree(y_map);
        if (stream) cudaStreamDestroy(stream);
    }
};

cudaError_t launch(const Resources& resources, int width, int height, std::size_t maps) {
    const std::size_t pitch = static_cast<std::size_t>(width) * 3U;
    return maps == 1U
        ? preprocess_maps_cuda(resources.image, width, height, pitch, resources.x_map,
                               resources.y_map, resources.input, resources.stream)
        : preprocess_cuda(resources.image, width, height, pitch, resources.input, resources.stream);
}

void benchmark_shape(int width, int height) {
    Resources resources;
    const std::size_t bytes = static_cast<std::size_t>(width) * static_cast<std::size_t>(height) * 3U;
    std::vector<unsigned char> pixels(bytes);
    for (std::size_t index = 0; index < pixels.size(); ++index)
        pixels[index] = static_cast<unsigned char>((index * 37U + index / 7U) % 256U);
    check(cudaStreamCreateWithFlags(&resources.stream, cudaStreamNonBlocking), "create stream");
    check(cudaEventCreate(&resources.begin), "create begin event");
    check(cudaEventCreate(&resources.end), "create end event");
    check(cudaMalloc(reinterpret_cast<void**>(&resources.image), bytes), "allocate image");
    check(cudaMalloc(reinterpret_cast<void**>(&resources.input), 3U * 640U * 640U * sizeof(float)), "allocate input");
    check(cudaMalloc(reinterpret_cast<void**>(&resources.x_map), 640U * sizeof(AxisMap)), "allocate x map");
    check(cudaMalloc(reinterpret_cast<void**>(&resources.y_map), 640U * sizeof(AxisMap)), "allocate y map");
    check(cudaMemcpy(resources.image, pixels.data(), bytes, cudaMemcpyHostToDevice), "upload image");
    check(build_preprocess_maps(width, height, resources.x_map, resources.y_map, resources.stream), "build maps");

    for (std::size_t maps = 0; maps < 2U; ++maps) {
        check(launch(resources, width, height, maps), "prime kernel");
        check(cudaStreamSynchronize(resources.stream), "complete prime");
        check(cudaStreamBeginCapture(resources.stream, cudaStreamCaptureModeThreadLocal), "begin capture");
        cudaError_t status = cudaSuccess;
        for (int call = 0; call < CALLS_PER_SAMPLE && status == cudaSuccess; ++call)
            status = launch(resources, width, height, maps);
        const auto capture = cudaStreamEndCapture(resources.stream, &resources.graphs[maps]);
        check(status, "capture kernels");
        check(capture, "end capture");
        check(cudaGraphInstantiate(&resources.executables[maps], resources.graphs[maps],
                                   nullptr, nullptr, 0), "instantiate graph");
    }

    for (int round = 1; round <= ROUNDS; ++round) {
        for (int position = 0; position < 2; ++position) {
            const auto maps = static_cast<std::size_t>((position + round - 1) % 2);
            for (int warmup = 0; warmup < WARMUP; ++warmup)
                check(cudaGraphLaunch(resources.executables[maps], resources.stream), "warmup graph");
            check(cudaStreamSynchronize(resources.stream), "complete warmup");
            std::vector<double> times;
            times.reserve(SAMPLES);
            for (int sample = 0; sample < SAMPLES; ++sample) {
                check(cudaEventRecord(resources.begin, resources.stream), "record begin");
                check(cudaGraphLaunch(resources.executables[maps], resources.stream), "launch graph");
                check(cudaEventRecord(resources.end, resources.stream), "record end");
                check(cudaEventSynchronize(resources.end), "complete sample");
                float milliseconds = 0.0F;
                check(cudaEventElapsedTime(&milliseconds, resources.begin, resources.end), "elapsed time");
                times.push_back(static_cast<double>(milliseconds) * 1000.0 / CALLS_PER_SAMPLE);
            }
            const double mean = std::accumulate(times.begin(), times.end(), 0.0)
                / static_cast<double>(times.size());
            std::sort(times.begin(), times.end());
            const double p50 = times[(times.size() - 1U) / 2U]; // nearest-rank P50
            std::cout << round << ',' << width << ',' << height << ',' << maps << ','
                      << CALLS_PER_SAMPLE << ',' << SAMPLES << ',' << mean << ',' << p50 << '\n';
        }
    }
}

int main(int argc, char**) {
    try {
        if (argc != 1) throw std::invalid_argument("This benchmark accepts no arguments");
        std::cout << "round,width,height,maps,calls_per_sample,samples,mean_us,p50_us\n"
                  << std::setprecision(9);
        for (const auto& shape : std::array<std::array<int, 2>, 3>{{
                 {810, 1080}, {1280, 720}, {640, 640}}})
            benchmark_shape(shape[0], shape[1]);
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
