// 测实际 11 注册、上传、前处理路径；不需要 TensorRT engine 文件。
#define main registered_demo_main
#include "../src/09_registered_input/main.cpp"
#undef main

void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

void check_cuda(cudaError_t error, const char* operation) {
    if (error != cudaSuccess)
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(error));
}

void initialize(CudaResources& cuda) {
    check_cuda(cudaStreamCreateWithFlags(&cuda.stream, cudaStreamNonBlocking), "Create stream");
    check_cuda(cudaMalloc(&cuda.input, INPUT_COUNT * sizeof(float)), "Allocate input");
}

void require_pinned(void* pointer) {
    cudaPointerAttributes attributes{};
    check_cuda(cudaPointerGetAttributes(&attributes, pointer), "Read host pointer attributes");
    require(attributes.type == cudaMemoryTypeHost, "Expected pinned/registered host memory");
    unsigned int flags = 0;
    check_cuda(cudaHostGetFlags(&flags, pointer), "Read host allocation/registration flags");
}

void require_unregistered(void* pointer) {
    cudaPointerAttributes attributes{};
    // CUDA 11.8 can query ordinary host memory without intentionally failing an API call.
    check_cuda(cudaPointerGetAttributes(&attributes, pointer), "Read unregistered host attributes");
    require(attributes.type == cudaMemoryTypeUnregistered, "Host memory is still registered");
}

void capture_preprocessing(CudaResources& cuda, const cv::Mat& image) {
    float scale = 0.0F;
    int pad_x = 0, pad_y = 0;
    require(stage_image(image, cuda, scale, pad_x, pad_y), "Stage before capture");
    require(upload_image(cuda, image.cols, image.rows), "Prime preprocessing");
    check_cuda(cudaStreamSynchronize(cuda.stream), "Finish prime");
    check_cuda(cudaStreamBeginCapture(cuda.stream, cudaStreamCaptureModeThreadLocal), "Begin capture");
    const bool submitted = upload_image(cuda, image.cols, image.rows);
    const auto status = cudaStreamEndCapture(cuda.stream, &cuda.graph);
    require(submitted, "Capture upload/preprocess submission failed");
    check_cuda(status, "End capture");
    check_cuda(cudaGraphInstantiate(&cuda.executable, cuda.graph, nullptr, nullptr, 0), "Instantiate graph");
}

void test_image(cv::Mat& image, CudaResources& direct, CudaResources& staging) {
    require(prepare_image(direct, image, true), "Register image");
    require(prepare_image(staging, image, false), "Prepare staging reference");
    require(direct.registered && direct.host_source == image.data
                && direct.host_image == nullptr && direct.source_pitch == image.step[0],
            "Registered path must read the original pixels with their actual pitch");
    require(!staging.registered && staging.host_source == staging.host_image
                && staging.source_pitch == static_cast<std::size_t>(image.cols) * 3U,
            "Staging path must use its own packed buffer");
    require_pinned(image.data);
    require_pinned(staging.host_image);
    capture_preprocessing(direct, image);
    capture_preprocessing(staging, image);
    std::vector<float> actual(INPUT_COUNT), expected(INPUT_COUNT), previous;
    for (int pass = 0; pass < 3; ++pass) {
        if (pass != 0) cv::bitwise_not(image, image); // 同地址新像素，不能重复读取捕获时的旧答案。
        float direct_scale = 0.0F, staging_scale = 0.0F;
        int direct_x = 0, direct_y = 0, staging_x = 0, staging_y = 0;
        require(stage_image(image, direct, direct_scale, direct_x, direct_y), "Stage registered input");
        require(stage_image(image, staging, staging_scale, staging_x, staging_y), "Stage reference input");
        require(direct_scale == staging_scale && direct_x == staging_x && direct_y == staging_y,
                "Letterbox metadata differs");
        check_cuda(cudaGraphLaunch(direct.executable, direct.stream), "Replay registered graph");
        check_cuda(cudaStreamSynchronize(direct.stream), "Finish registered replay");
        check_cuda(cudaGraphLaunch(staging.executable, staging.stream), "Replay staging graph");
        check_cuda(cudaStreamSynchronize(staging.stream), "Finish staging replay");
        check_cuda(cudaMemcpy(actual.data(), direct.input, INPUT_COUNT * sizeof(float),
                              cudaMemcpyDeviceToHost), "Read registered tensor");
        check_cuda(cudaMemcpy(expected.data(), staging.input, INPUT_COUNT * sizeof(float),
                              cudaMemcpyDeviceToHost), "Read staging tensor");
        require(std::memcmp(actual.data(), expected.data(), INPUT_COUNT * sizeof(float)) == 0,
                "Registered and staging tensors differ");
        if (pass != 0) require(actual != previous, "Graph replay did not use updated source pixels");
        previous = actual;
    }
    require(direct.clear_source(), "Destroy graph before unregistering image");
    require_unregistered(image.data);
    require(!direct.registered && direct.source_image.empty() && !direct.host_source
                && !direct.graph && !direct.executable, "Source cleanup left live references");
    require(staging.clear_source(), "Release staging source");
    std::cout << "PASS " << image.cols << 'x' << image.rows
              << (image.isContinuous() ? " contiguous" : " ROI") << " (3 graph replays)\n";
}

void test_lifetime() {
    CudaResources cuda;
    initialize(cuda);
    {
        cv::Mat local(17, 29, CV_8UC3, cv::Scalar(7, 19, 31));
        require(prepare_image(cuda, local, true), "Register local image");
        capture_preprocessing(cuda, local);
    }
    require(!cuda.source_image.empty() && cuda.source_image.at<cv::Vec3b>(0, 0)[0] == 7U,
            "Resource owner did not retain the source allocation");
    cv::Mat retained = cuda.source_image; // 取消注册后保留地址，方便查询是否真的已解除注册。
    check_cuda(cudaGraphLaunch(cuda.executable, cuda.stream), "Launch before source cleanup");
    require(cuda.clear_source(), "Cleanup must wait for pending graph before unregister");
    require_unregistered(retained.data);

    // 异常分支也必须清图/解注册；外部 cv::Mat 继续正常拥有像素。
    struct InjectedFailure {};
    try {
        CudaResources temporary;
        initialize(temporary);
        require(prepare_image(temporary, retained, true), "Register exception-path image");
        capture_preprocessing(temporary, retained);
        check_cuda(cudaGraphLaunch(temporary.executable, temporary.stream), "Launch before exception");
        throw InjectedFailure{};
    } catch (const InjectedFailure&) {
        require_unregistered(retained.data);
    }
    std::cout << "PASS retained cv::Mat, pending replay cleanup, exception cleanup\n";
}

int main() {
    try {
        CudaResources direct, staging;
        initialize(direct);
        initialize(staging);
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
                for (std::size_t x = 0; x < static_cast<std::size_t>(image.cols) * 3U; ++x)
                    row[x] = static_cast<unsigned char>((x * 17U + static_cast<std::size_t>(y) * 29U) % 256U);
            }
            test_image(image, direct, staging);
        }
        float scale = 0.0F;
        int pad_x = 0, pad_y = 0;
        require(!prepare_image(direct, cv::Mat(), true), "Empty image should fail");
        require(!prepare_image(direct, cv::Mat(4, 4, CV_8UC1), true), "Grayscale should fail");
        require(!stage_image(cv::Mat(4, 4, CV_8UC3), direct, scale, pad_x, pad_y),
                "Unprepared image should fail");
        test_lifetime();
        std::cout << "All registered-input tests passed.\n";
    } catch (const std::exception& error) {
        std::cerr << "FAIL " << error.what() << '\n';
        return 1;
    }
    return 0;
}
