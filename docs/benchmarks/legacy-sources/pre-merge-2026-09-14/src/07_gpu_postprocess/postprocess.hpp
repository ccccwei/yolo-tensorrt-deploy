#pragma once

#include <cuda_runtime_api.h>

constexpr int GPU_MAX_DETECTIONS = 300;

struct GpuDetection {
    int class_id;
    float score;
    int left, top, right, bottom;
};

struct GpuResults {
    int count;
    GpuDetection boxes[GPU_MAX_DETECTIONS];
};

// output: [1,300,6] xyxy / score / class_id; buffers belong to the caller.
// Enqueue only: no allocation or synchronization. Valid entries are boxes[0..count).
cudaError_t postprocess_cuda(const float* output, int image_width, int image_height,
                            float scale, int pad_x, int pad_y,
                            GpuResults* result, cudaStream_t stream);
