// 直接验证 04 的实际 staging 路径，不为测试另建一套生产接口。
#define main pinned_demo_main
#include "../src/03_pinned_memory/main.cpp"
#undef main

void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

void check_cuda(cudaError_t error, const char* operation) {
    if (error != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(error));
    }
}

void require_pinned(const void* pointer) {
    cudaPointerAttributes attributes{};
    check_cuda(cudaPointerGetAttributes(&attributes, pointer), "cudaPointerGetAttributes");
    require(attributes.type == cudaMemoryTypeHost, "Expected pinned host memory");
}

void test_image(cv::Mat& image, CudaResources& cuda, CudaResources& reference) {
    const std::size_t row_bytes = static_cast<std::size_t>(image.cols) * 3U;
    const std::size_t bytes = row_bytes * static_cast<std::size_t>(image.rows);
    const auto old_capacity = cuda.image_capacity;
    const auto* old_host = cuda.host_image;
    const auto* old_device = cuda.image;
    require(prepare_image(cuda, bytes), "prepare_image failed");
    require(cuda.image_capacity >= bytes, "Image capacity is too small");
    if (bytes <= old_capacity) {
        require(cuda.host_image == old_host && cuda.image == old_device
                    && cuda.image_capacity == old_capacity, "Buffers were not reused");
    }
    require_pinned(cuda.host_image);
    std::vector<float> actual(INPUT_COUNT), expected(INPUT_COUNT);
    // 第二次调用前更改全部像素，防止提前 staging 的实现错误地通过测试。
    for (int pass = 0; pass < 2; ++pass) {
        if (pass != 0) cv::bitwise_not(image, image);
        float scale = 0.0F;
        int pad_x = 0, pad_y = 0;
        require(preprocess(image, cuda, scale, pad_x, pad_y), "Pinned preprocess failed");
        for (int y = 0; y < image.rows; ++y) {
            require(std::memcmp(cuda.host_image + static_cast<std::size_t>(y) * row_bytes,
                                image.ptr<unsigned char>(y), row_bytes) == 0,
                    "Staging bytes differ from the current image");
        }
        const float expected_scale = std::min(
            static_cast<float>(INPUT_SIZE) / static_cast<float>(image.cols),
            static_cast<float>(INPUT_SIZE) / static_cast<float>(image.rows));
        const int width = std::max(1, static_cast<int>(std::round(
            static_cast<float>(image.cols) * expected_scale)));
        const int height = std::max(1, static_cast<int>(std::round(
            static_cast<float>(image.rows) * expected_scale)));
        require(scale == expected_scale && pad_x == (INPUT_SIZE - width) / 2
                    && pad_y == (INPUT_SIZE - height) / 2, "Letterbox metadata differs");
        check_cuda(cudaMemcpy(actual.data(), cuda.input, INPUT_COUNT * sizeof(float),
                              cudaMemcpyDeviceToHost), "Read pinned-path tensor");
        // 与 03 相同的 pageable 上传，直接使用源图步长，不经过 04 的 host staging。
        check_cuda(cudaMemcpy2D(reference.image, row_bytes, image.data, image.step[0],
                                row_bytes, static_cast<std::size_t>(image.rows),
                                cudaMemcpyHostToDevice), "Upload pageable reference");
        check_cuda(preprocess_cuda(reference.image, image.cols, image.rows, row_bytes,
                                    static_cast<float*>(reference.input), reference.stream),
                   "Reference preprocessing kernel");
        check_cuda(cudaStreamSynchronize(reference.stream), "Synchronize reference kernel");
        check_cuda(cudaMemcpy(expected.data(), reference.input, INPUT_COUNT * sizeof(float),
                              cudaMemcpyDeviceToHost), "Read reference tensor");
        require(std::memcmp(actual.data(), expected.data(), INPUT_COUNT * sizeof(float)) == 0,
                "Pinned and pageable preprocessing tensors differ");
    }
    std::cout << "PASS " << image.cols << 'x' << image.rows
              << (image.isContinuous() ? " contiguous" : " ROI") << " (including updated pixels)\n";
}

void test_output(CudaResources& cuda) {
    require_pinned(cuda.host_output);
    std::vector<float> expected(OUTPUT_COUNT), pageable(OUTPUT_COUNT);
    for (std::size_t i = 0; i < OUTPUT_COUNT; ++i) expected[i] = static_cast<float>(i) / 16.0F;
    check_cuda(cudaMemcpy(cuda.output, expected.data(), OUTPUT_COUNT * sizeof(float),
                          cudaMemcpyHostToDevice), "Initialize output tensor");
    try {
        check_cuda(cudaMemcpyAsync(cuda.host_output, cuda.output, OUTPUT_COUNT * sizeof(float),
                                   cudaMemcpyDeviceToHost, cuda.stream), "Pinned output D2H");
        check_cuda(cudaMemcpyAsync(pageable.data(), cuda.output, OUTPUT_COUNT * sizeof(float),
                                   cudaMemcpyDeviceToHost, cuda.stream), "Pageable output D2H");
        check_cuda(cudaStreamSynchronize(cuda.stream), "Synchronize output D2H");
    } catch (...) {
        cudaStreamSynchronize(cuda.stream); // 异常时也必须等 DMA 完成再销毁 pageable。
        throw;
    }
    require(std::memcmp(cuda.host_output, expected.data(), OUTPUT_COUNT * sizeof(float)) == 0
                && std::memcmp(cuda.host_output, pageable.data(), OUTPUT_COUNT * sizeof(float)) == 0,
            "Pinned output D2H differs from pageable reference");
    std::cout << "PASS pinned output allocation and D2H\n";
}

int main() {
    try {
        CudaResources cuda, reference;
        check_cuda(cudaStreamCreate(&cuda.stream), "Create production stream");
        check_cuda(cudaStreamCreate(&reference.stream), "Create reference stream");
        check_cuda(cudaMalloc(&cuda.input, INPUT_COUNT * sizeof(float)), "Allocate input");
        check_cuda(cudaMalloc(&cuda.output, OUTPUT_COUNT * sizeof(float)), "Allocate output");
        check_cuda(cudaMallocHost(reinterpret_cast<void**>(&cuda.host_output),
                                  OUTPUT_COUNT * sizeof(float)), "Allocate pinned output");
        check_cuda(cudaMalloc(&reference.input, INPUT_COUNT * sizeof(float)), "Allocate reference tensor");
        check_cuda(cudaMalloc(reinterpret_cast<void**>(&reference.image), 1600U * 900U * 3U),
                   "Allocate reference image");
        struct Case { int width; int height; bool roi; };
        const std::array<Case, 8> cases{{
            {16, 16, false}, {640, 640, false}, {1280, 720, false},
            {720, 1280, false}, {641, 479, false}, {639, 427, true},
            {1600, 900, true}, {1, 1, false}
        }};
        for (const auto& test : cases) {
            cv::Mat storage(test.height + (test.roi ? 4 : 0),
                            test.width + (test.roi ? 7 : 0), CV_8UC3);
            cv::Mat image = test.roi ? storage(cv::Rect(3, 2, test.width, test.height)) : storage;
            for (int y = 0; y < image.rows; ++y) {
                auto* row = image.ptr<unsigned char>(y);
                for (std::size_t x = 0; x < static_cast<std::size_t>(image.cols) * 3U; ++x) {
                    row[x] = static_cast<unsigned char>((x * 17U + static_cast<std::size_t>(y) * 29U) % 256U);
                }
            }
            test_image(image, cuda, reference);
        }
        test_output(cuda);
        float scale = 0.0F;
        int pad_x = 0, pad_y = 0;
        require(!prepare_image(cuda, 0U), "Zero-size allocation should fail");
        require(!preprocess(cv::Mat(), cuda, scale, pad_x, pad_y), "Empty image should fail");
        require(!preprocess(cv::Mat(4, 4, CV_8UC1), cuda, scale, pad_x, pad_y), "Grayscale image should fail");
        CudaResources unprepared;
        require(!preprocess(cv::Mat(4, 4, CV_8UC3), unprepared, scale, pad_x, pad_y),
                "Unprepared resources should fail");
        cuda.image_capacity = 1U;
        require(!preprocess(cv::Mat(4, 4, CV_8UC3), cuda, scale, pad_x, pad_y),
                "Insufficient capacity should fail");
        std::cout << "PASS invalid inputs\nAll pinned-memory tests passed.\n";
    } catch (const std::exception& error) {
        std::cerr << "FAIL " << error.what() << '\n';
        return 1;
    }
    return 0;
}
