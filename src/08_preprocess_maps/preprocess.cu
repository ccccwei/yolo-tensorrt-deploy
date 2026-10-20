#include "preprocess.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>

namespace {

constexpr int INPUT_SIZE = 640;
constexpr std::size_t PLANE_SIZE =
    static_cast<std::size_t>(INPUT_SIZE) * static_cast<std::size_t>(INPUT_SIZE);

__global__ void preprocess_kernel(const unsigned char* image,
                                  int width,
                                  int height,
                                  std::size_t pitch,
                                  int resized_width,
                                  int resized_height,
                                  int pad_x,
                                  int pad_y,
                                  double resize_scale_x,
                                  double resize_scale_y,
                                  float* input) {
    const int x = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    const int y = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
    if (x >= INPUT_SIZE || y >= INPUT_SIZE) return;

    const std::size_t index =
        static_cast<std::size_t>(y) * static_cast<std::size_t>(INPUT_SIZE)
        + static_cast<std::size_t>(x);
    constexpr float normalize = 1.0F / 255.0F;

    // Letterbox padding can differ by one pixel between opposite sides.
    if (x < pad_x || x >= pad_x + resized_width
        || y < pad_y || y >= pad_y + resized_height) {
        input[index] = 114.0F * normalize;
        input[PLANE_SIZE + index] = 114.0F * normalize;
        input[2U * PLANE_SIZE + index] = 114.0F * normalize;
        return;
    }

    // INTER_LINEAR's half-pixel mapping uses the rounded resize dimensions,
    // not the original letterbox scale. Clamp to replicate edge pixels.
    // Match OpenCV: calculate coordinates with double scale factors, then
    // convert the complete coordinate to float. Early float division loses
    // fractional pixels on very wide/tall images (e.g. 100001 x 1).
    const float source_x = fminf(fmaxf(static_cast<float>(
        (static_cast<double>(x - pad_x) + 0.5) * resize_scale_x - 0.5),
        0.0F), static_cast<float>(width - 1));
    const float source_y = fminf(fmaxf(static_cast<float>(
        (static_cast<double>(y - pad_y) + 0.5) * resize_scale_y - 0.5),
        0.0F), static_cast<float>(height - 1));
    const int x0 = static_cast<int>(floorf(source_x));
    const int y0 = static_cast<int>(floorf(source_y));
    const int x1 = min(x0 + 1, width - 1);
    const int y1 = min(y0 + 1, height - 1);
    const float dx = source_x - static_cast<float>(x0);
    const float dy = source_y - static_cast<float>(y0);

    const unsigned char* row0 = image + static_cast<std::size_t>(y0) * pitch;
    const unsigned char* row1 = image + static_cast<std::size_t>(y1) * pitch;
    for (int channel = 0; channel < 3; ++channel) {
        const std::size_t offset0 = static_cast<std::size_t>(x0) * 3U
            + static_cast<std::size_t>(channel);
        const std::size_t offset1 = static_cast<std::size_t>(x1) * 3U
            + static_cast<std::size_t>(channel);
        const float top = (1.0F - dx) * static_cast<float>(row0[offset0])
            + dx * static_cast<float>(row0[offset1]);
        const float bottom = (1.0F - dx) * static_cast<float>(row1[offset0])
            + dx * static_cast<float>(row1[offset1]);
        const float value = (1.0F - dy) * top + dy * bottom;

        // CPU resize produces uint8 before normalization. Round here too;
        // OpenCV's fixed-point interpolation can still differ by one level.
        input[static_cast<std::size_t>(2 - channel) * PLANE_SIZE + index] =
            floorf(value + 0.5F) * normalize;
    }
}


// Use the same GPU double half-pixel expression and float rounding as the
// reference kernel. Computing these on the CPU can introduce FMA differences.
__device__ AxisMap make_axis_map(int coordinate, int size, int resized,
                                int padding, double resize_scale) {
    if (coordinate < padding || coordinate >= padding + resized) return {-1, 0, 0.0F};
    const float source = fminf(fmaxf(static_cast<float>(
        (static_cast<double>(coordinate - padding) + 0.5) * resize_scale - 0.5),
        0.0F), static_cast<float>(size - 1));
    const int lower = static_cast<int>(floorf(source));
    return {lower, min(lower + 1, size - 1), source - static_cast<float>(lower)};
}

__global__ void build_maps_kernel(int width, int height, int resized_width,
                                 int resized_height, int pad_x, int pad_y,
                                 double scale_x, double scale_y,
                                 AxisMap* x_map, AxisMap* y_map) {
    const int coordinate = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (coordinate >= INPUT_SIZE) return;
    x_map[coordinate] = make_axis_map(coordinate, width, resized_width, pad_x, scale_x);
    y_map[coordinate] = make_axis_map(coordinate, height, resized_height, pad_y, scale_y);
}

__global__ void preprocess_maps_kernel(const unsigned char* image, std::size_t pitch,
                                      const AxisMap* x_map, const AxisMap* y_map,
                                      float* input) {
    const int x = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    const int y = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
    if (x >= INPUT_SIZE || y >= INPUT_SIZE) return;
    const std::size_t index = static_cast<std::size_t>(y) * INPUT_SIZE
        + static_cast<std::size_t>(x);
    const AxisMap horizontal = x_map[x];
    const AxisMap vertical = y_map[y];
    constexpr float normalize = 1.0F / 255.0F;
    if (horizontal.lower < 0 || vertical.lower < 0) {
        input[index] = 114.0F * normalize;
        input[PLANE_SIZE + index] = 114.0F * normalize;
        input[2U * PLANE_SIZE + index] = 114.0F * normalize;
        return;
    }
    const float dx = horizontal.fraction;
    const float dy = vertical.fraction;
    const unsigned char* row0 = image + static_cast<std::size_t>(vertical.lower) * pitch;
    const unsigned char* row1 = image + static_cast<std::size_t>(vertical.upper) * pitch;
    for (int channel = 0; channel < 3; ++channel) {
        const std::size_t offset0 = static_cast<std::size_t>(horizontal.lower) * 3U
            + static_cast<std::size_t>(channel);
        const std::size_t offset1 = static_cast<std::size_t>(horizontal.upper) * 3U
            + static_cast<std::size_t>(channel);
        const float top = (1.0F - dx) * static_cast<float>(row0[offset0])
            + dx * static_cast<float>(row0[offset1]);
        const float bottom = (1.0F - dx) * static_cast<float>(row1[offset0])
            + dx * static_cast<float>(row1[offset1]);
        const float value = (1.0F - dy) * top + dy * bottom;
        input[static_cast<std::size_t>(2 - channel) * PLANE_SIZE + index] =
            floorf(value + 0.5F) * normalize;
    }
}

}  // namespace

cudaError_t preprocess_cuda(const unsigned char* image,
                            int width,
                            int height,
                            std::size_t pitch,
                            float* input,
                            cudaStream_t stream) {
    if (!image || !input || width <= 0 || height <= 0
        || pitch < static_cast<std::size_t>(width) * 3U) {
        return cudaErrorInvalidValue;
    }

    const float scale = std::min(
        static_cast<float>(INPUT_SIZE) / static_cast<float>(width),
        static_cast<float>(INPUT_SIZE) / static_cast<float>(height));
    const int resized_width = std::max(
        1, static_cast<int>(std::round(static_cast<float>(width) * scale)));
    const int resized_height = std::max(
        1, static_cast<int>(std::round(static_cast<float>(height) * scale)));
    const int pad_x = (INPUT_SIZE - resized_width) / 2;
    const int pad_y = (INPUT_SIZE - resized_height) / 2;
    const double resize_scale_x = static_cast<double>(width) / resized_width;
    const double resize_scale_y = static_cast<double>(height) / resized_height;

    const dim3 block(32U, 8U);
    const dim3 grid((static_cast<unsigned int>(INPUT_SIZE) + block.x - 1U) / block.x,
                    (static_cast<unsigned int>(INPUT_SIZE) + block.y - 1U) / block.y);
    preprocess_kernel<<<grid, block, 0, stream>>>(
        image, width, height, pitch, resized_width, resized_height, pad_x, pad_y,
        resize_scale_x, resize_scale_y, input);
    return cudaGetLastError();  // Caller checks asynchronous errors when synchronizing.
}

cudaError_t build_preprocess_maps(int width, int height, AxisMap* x_map,
                                  AxisMap* y_map, cudaStream_t stream) {
    if (width <= 0 || height <= 0 || !x_map || !y_map || x_map == y_map)
        return cudaErrorInvalidValue;
    const float scale = std::min(
        static_cast<float>(INPUT_SIZE) / static_cast<float>(width),
        static_cast<float>(INPUT_SIZE) / static_cast<float>(height));
    const int resized_width = std::max(
        1, static_cast<int>(std::round(static_cast<float>(width) * scale)));
    const int resized_height = std::max(
        1, static_cast<int>(std::round(static_cast<float>(height) * scale)));
    build_maps_kernel<<<3U, 256U, 0, stream>>>(
        width, height, resized_width, resized_height,
        (INPUT_SIZE - resized_width) / 2, (INPUT_SIZE - resized_height) / 2,
        static_cast<double>(width) / resized_width,
        static_cast<double>(height) / resized_height, x_map, y_map);
    return cudaGetLastError();
}

cudaError_t preprocess_maps_cuda(const unsigned char* image, int width, int height,
                                 std::size_t pitch, const AxisMap* x_map,
                                 const AxisMap* y_map, float* input, cudaStream_t stream) {
    if (!image || !input || !x_map || !y_map || x_map == y_map || width <= 0 || height <= 0
        || pitch < static_cast<std::size_t>(width) * 3U) return cudaErrorInvalidValue;
    // The caller must rebuild both tables whenever the source dimensions change.
    const dim3 block(32U, 8U);
    const dim3 grid(20U, 80U);
    preprocess_maps_kernel<<<grid, block, 0, stream>>>(image, pitch, x_map, y_map, input);
    return cudaGetLastError();
}
