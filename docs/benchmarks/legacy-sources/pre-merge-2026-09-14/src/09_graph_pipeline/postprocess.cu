#include "postprocess.hpp"

#include <cuda_runtime.h>

#include <cmath>
#include <limits>

namespace {
constexpr int BLOCK_SIZE = 512;

__device__ int coordinate(float value, int padding, int size, float scale) {
    // Match the CPU's float subtraction/division, then double clamp and truncation.
    const float restored = __fdiv_rn(__fsub_rn(value, static_cast<float>(padding)), scale);
    const double clipped = fmin(fmax(static_cast<double>(restored), 0.0),
                                static_cast<double>(size - 1));
    return static_cast<int>(clipped);
}

__global__ void postprocess_kernel(const float* output, int width, int height,
                                   float scale, int pad_x, int pad_y, GpuResults* result) {
    const int index = static_cast<int>(threadIdx.x);
    __shared__ int prefix[BLOCK_SIZE];
    GpuDetection box{};
    bool valid = index < GPU_MAX_DETECTIONS;
    if (valid) {
        const float* det = output + index * 6;
        for (int field = 0; field < 6; ++field) valid = valid && isfinite(det[field]);
        valid = valid && det[4] >= 0.25F && det[5] >= 0.0F
            && static_cast<double>(det[5]) <= static_cast<double>(2147483647);
        if (valid) {
            box = {static_cast<int>(det[5]), det[4],
                   coordinate(det[0], pad_x, width, scale),
                   coordinate(det[1], pad_y, height, scale),
                   coordinate(det[2], pad_x, width, scale),
                   coordinate(det[3], pad_y, height, scale)};
            valid = box.right > box.left && box.bottom > box.top;
        }
    }
    prefix[index] = valid ? 1 : 0;
    __syncthreads();
    // Stable inclusive scan: no atomicAdd, so output keeps TensorRT's candidate order.
    for (int offset = 1; offset < BLOCK_SIZE; offset *= 2) {
        const int previous = index >= offset ? prefix[index - offset] : 0;
        __syncthreads();
        prefix[index] += previous;
        __syncthreads();
    }
    if (valid) result->boxes[prefix[index] - 1] = box;
    if (index == 0) result->count = prefix[GPU_MAX_DETECTIONS - 1];
}
} // namespace

cudaError_t postprocess_cuda(const float* output, int image_width, int image_height,
                            float scale, int pad_x, int pad_y,
                            GpuResults* result, cudaStream_t stream) {
    if (!output || !result || image_width <= 0 || image_height <= 0
        || !std::isfinite(scale) || scale <= 0.0F) return cudaErrorInvalidValue;
    postprocess_kernel<<<1, BLOCK_SIZE, 0, stream>>>(
        output, image_width, image_height, scale, pad_x, pad_y, result);
    return cudaGetLastError();
}
