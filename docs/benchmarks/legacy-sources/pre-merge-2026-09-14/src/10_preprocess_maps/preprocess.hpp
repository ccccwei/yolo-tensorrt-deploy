#pragma once

#include <cuda_runtime_api.h>
#include <cstddef>

// One entry per output coordinate; lower == -1 marks letterbox padding.
struct AxisMap {
    int lower;
    int upper;
    float fraction;
};

cudaError_t build_preprocess_maps(int width, int height, AxisMap* x_map,
                                  AxisMap* y_map, cudaStream_t stream);
cudaError_t preprocess_maps_cuda(const unsigned char* image, int width, int height,
                                 std::size_t pitch, const AxisMap* x_map,
                                 const AxisMap* y_map, float* input, cudaStream_t stream);
cudaError_t preprocess_cuda(const unsigned char* image, int width, int height,
                            std::size_t pitch, float* input, cudaStream_t stream);
