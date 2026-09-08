#include <NvInferRuntime.h>
#include <cuda_runtime_api.h>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

// 固定模型：images=[1,3,640,640]
// 固定输出：output0=[1,300,6]
// 每行数据：[x1, y1, x2, y2, score, class_id]
constexpr char INPUT_NAME[] = "images";
constexpr char OUTPUT_NAME[] = "output0";
constexpr int INPUT_SIZE = 640;
constexpr int MAX_DETECTIONS = 300;
constexpr int DETECTION_SIZE = 6;
constexpr float SCORE_THRESHOLD = 0.25F;
constexpr std::size_t INPUT_COUNT = 1U * 3U * INPUT_SIZE * INPUT_SIZE;
constexpr std::size_t OUTPUT_COUNT = 1U * MAX_DETECTIONS * DETECTION_SIZE;

// TensorRT 要求提供 ILogger
class Logger final : public nvinfer1::ILogger {
public:
    void log(Severity severity, const char* message) noexcept override {
        if (severity <= Severity::kWARNING) {
            std::cerr << "[TensorRT] " << message << '\n';
        }
    }
};

cv::Mat preprocess(const cv::Mat& image, float& scale, int& pad_x, int& pad_y) {
    scale = std::min(
        static_cast<float>(INPUT_SIZE) / static_cast<float>(image.cols),
        static_cast<float>(INPUT_SIZE) / static_cast<float>(image.rows));

    const int width = static_cast<int>(std::round(static_cast<float>(image.cols) * scale));
    const int height = static_cast<int>(std::round(static_cast<float>(image.rows) * scale));
    pad_x = (INPUT_SIZE - width) / 2;
    pad_y = (INPUT_SIZE - height) / 2;

    cv::Mat resized, padded;
    cv::resize(image, resized, cv::Size(width, height));
    cv::copyMakeBorder(
        resized, padded,
        pad_y, INPUT_SIZE - height - pad_y,
        pad_x, INPUT_SIZE - width - pad_x,
        cv::BORDER_CONSTANT, cv::Scalar(114, 114, 114));

    const int shape[] = {1, 3, INPUT_SIZE, INPUT_SIZE};
    cv::Mat input(4, shape, CV_32F);
    float* input_data = input.ptr<float>();
    constexpr std::size_t plane_size =
        static_cast<std::size_t>(INPUT_SIZE) * static_cast<std::size_t>(INPUT_SIZE);
    constexpr float normalize = 1.0F / 255.0F;

    for (int y = 0; y < INPUT_SIZE; ++y) {
        const cv::Vec3b* row = padded.ptr<cv::Vec3b>(y);
        for (int x = 0; x < INPUT_SIZE; ++x) {
            const std::size_t index =
                static_cast<std::size_t>(y) * static_cast<std::size_t>(INPUT_SIZE)
                + static_cast<std::size_t>(x);
            input_data[index] = static_cast<float>(row[x][2]) * normalize;
            input_data[plane_size + index] = static_cast<float>(row[x][1]) * normalize;
            input_data[2U * plane_size + index] = static_cast<float>(row[x][0]) * normalize;
        }
    }
    return input;
}

int postprocess(const std::vector<float>& output,
                 cv::Mat& image,
                 float scale,
                 int pad_x,
                 int pad_y) {
    int count = 0;
    for (int i = 0; i < MAX_DETECTIONS; ++i) {
        const float* det = output.data() + i * DETECTION_SIZE;
        if (det[4] < SCORE_THRESHOLD) continue;

        const int left = std::clamp(
            static_cast<int>((det[0] - static_cast<float>(pad_x)) / scale), 0, image.cols - 1);
        const int top = std::clamp(
            static_cast<int>((det[1] - static_cast<float>(pad_y)) / scale), 0, image.rows - 1);
        const int right = std::clamp(
            static_cast<int>((det[2] - static_cast<float>(pad_x)) / scale), 0, image.cols - 1);
        const int bottom = std::clamp(
            static_cast<int>((det[3] - static_cast<float>(pad_y)) / scale), 0, image.rows - 1);
        if (right <= left || bottom <= top) continue;

        ++count;
        const int class_id = static_cast<int>(det[5]);
        cv::rectangle(image, {left, top}, {right, bottom}, {0, 255, 0}, 2);
        cv::putText(
            image, cv::format("class=%d %.2f", class_id, det[4]),
            {left, std::max(20, top - 5)}, cv::FONT_HERSHEY_SIMPLEX,
            0.6, {0, 255, 0}, 2);
    }
    return count;
}

bool infer(nvinfer1::IExecutionContext& context,
           void* device_input,
           void* device_output,
           cudaStream_t stream,
           const cv::Mat& input,
           std::vector<float>& output) {
    return cudaMemcpyAsync(device_input, input.ptr<float>(),
                           INPUT_COUNT * sizeof(float),
                           cudaMemcpyHostToDevice, stream) == cudaSuccess
        && context.enqueueV3(stream)
        && cudaMemcpyAsync(output.data(), device_output,
                           OUTPUT_COUNT * sizeof(float),
                           cudaMemcpyDeviceToHost, stream) == cudaSuccess
        && cudaStreamSynchronize(stream) == cudaSuccess;
}

void release_cuda(void* device_input, void* device_output, cudaStream_t stream) {
    if (stream) {
        cudaStreamSynchronize(stream);
        cudaStreamDestroy(stream);
    }
    if (device_output) cudaFree(device_output);
    if (device_input) cudaFree(device_input);
}

bool is_image(const std::filesystem::path& path) {
    std::string extension = path.extension().string();
    std::transform(extension.begin(), extension.end(), extension.begin(),
                   [](unsigned char value) {
                       return static_cast<char>(std::tolower(value));
                   });
    return extension == ".jpg" || extension == ".jpeg"
        || extension == ".png" || extension == ".bmp";
}

int main(int argc, char** argv) {
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " yolo.engine image_folder\n";
        return 1;
    }

    // 1. 查找文件夹中的图片
    const std::filesystem::path image_folder(argv[2]);
    if (!std::filesystem::is_directory(image_folder)) {
        std::cerr << "Cannot open image folder\n";
        return 1;
    }

    std::vector<std::filesystem::path> image_paths;
    for (const auto& entry : std::filesystem::directory_iterator(image_folder)) {
        if (entry.is_regular_file() && is_image(entry.path())) {
            image_paths.push_back(entry.path());
        }
    }
    std::sort(image_paths.begin(), image_paths.end());
    std::cout << "Images: " << image_paths.size() << '\n';
    if (image_paths.empty()) {
        return 1;
    }

    // 2. 读取 TensorRT engine
    std::ifstream file(argv[1], std::ios::binary | std::ios::ate);
    if (!file || file.tellg() <= 0) {
        std::cerr << "Cannot open engine\n";
        return 1;
    }

    const std::streamsize engine_size = file.tellg();
    std::vector<char> engine_data(static_cast<std::size_t>(engine_size));
    file.seekg(0);
    if (!file.read(engine_data.data(), engine_size)) {
        std::cerr << "Cannot read engine\n";
        return 1;
    }

    // 3. 创建 runtime、engine 和 context
    Logger logger;
    std::unique_ptr<nvinfer1::IRuntime> runtime(
        nvinfer1::createInferRuntime(logger));
    if (!runtime) return 1;

    std::unique_ptr<nvinfer1::ICudaEngine> engine(
        runtime->deserializeCudaEngine(engine_data.data(), engine_data.size()));
    if (!engine) return 1;

    std::unique_ptr<nvinfer1::IExecutionContext> context(
        engine->createExecutionContext());
    if (!context) return 1;

    const auto input_shape = engine->getTensorShape(INPUT_NAME);
    const auto output_shape = engine->getTensorShape(OUTPUT_NAME);
    if (input_shape.nbDims != 4
        || input_shape.d[0] != 1
        || input_shape.d[1] != 3
        || input_shape.d[2] != INPUT_SIZE
        || input_shape.d[3] != INPUT_SIZE
        || output_shape.nbDims != 3
        || output_shape.d[0] != 1
        || output_shape.d[1] != MAX_DETECTIONS
        || output_shape.d[2] != DETECTION_SIZE
        || engine->getTensorDataType(INPUT_NAME) != nvinfer1::DataType::kFLOAT
        || engine->getTensorDataType(OUTPUT_NAME) != nvinfer1::DataType::kFLOAT) {
        std::cerr << "Unsupported engine input or output\n";
        return 1;
    }

    // 4. 申请 GPU 内存
    void* device_input = nullptr;
    void* device_output = nullptr;
    cudaStream_t stream = nullptr;

    const bool cuda_ready =
        cudaMalloc(&device_input, INPUT_COUNT * sizeof(float)) == cudaSuccess
        && cudaMalloc(&device_output, OUTPUT_COUNT * sizeof(float)) == cudaSuccess
        && cudaStreamCreate(&stream) == cudaSuccess;

    if (!cuda_ready
        || !context->setTensorAddress(INPUT_NAME, device_input)
        || !context->setTensorAddress(OUTPUT_NAME, device_output)) {
        release_cuda(device_input, device_output, stream);
        std::cerr << "Cannot prepare CUDA\n";
        return 1;
    }

    // 5. 循环推理文件夹中的图片
    std::filesystem::create_directories("results");
    std::vector<float> output(OUTPUT_COUNT);
    using Clock = std::chrono::steady_clock;
    using Milliseconds = std::chrono::duration<double, std::milli>;
    for (const auto& image_path : image_paths) {
        cv::Mat image = cv::imread(image_path.string());
        if (image.empty()) {
            std::cerr << "Cannot open image: " << image_path << '\n';
            continue;
        }

        float scale = 0.0F;
        int pad_x = 0;
        int pad_y = 0;
        const auto start = Clock::now();
        const cv::Mat input = preprocess(image, scale, pad_x, pad_y);
        const auto preprocess_end = Clock::now();

        if (!infer(*context, device_input, device_output, stream, input, output)) {
            release_cuda(device_input, device_output, stream);
            std::cerr << "Inference failed: " << image_path << '\n';
            return 1;
        }
        // infer() 已同步 CUDA Stream，计时包含拷贝和 GPU 执行完成。
        const auto infer_end = Clock::now();

        const int count = postprocess(output, image, scale, pad_x, pad_y);
        const auto postprocess_end = Clock::now();

        std::cout << image_path.filename() << " | Detections: " << count
                  << std::fixed << std::setprecision(3)
                  << " | Preprocess: " << Milliseconds(preprocess_end - start).count() << " ms"
                  << " | Infer (H2D+TRT+D2H): " << Milliseconds(infer_end - preprocess_end).count() << " ms"
                  << " | Postprocess: " << Milliseconds(postprocess_end - infer_end).count() << " ms\n";

        const std::filesystem::path result_path =
            std::filesystem::path("results") / image_path.filename();
        if (!cv::imwrite(result_path.string(), image)) {
            release_cuda(device_input, device_output, stream);
            std::cerr << "Cannot save: " << result_path << '\n';
            return 1;
        }
        std::cout << "Saved: " << result_path << '\n';
    }

    release_cuda(device_input, device_output, stream);
    return 0;
}
