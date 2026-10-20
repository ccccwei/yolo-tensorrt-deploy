#include "postprocess.hpp"

#include <NvInferRuntime.h>
#include <cuda_runtime_api.h>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <array>
#include <cctype>
#include <charconv>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

constexpr char INPUT_NAME[] = "images";
constexpr char OUTPUT_NAME[] = "output0";
constexpr int INPUT_SIZE = 640;
constexpr int MAX_DETECTIONS = 300;
constexpr int DETECTION_SIZE = 6;
constexpr float SCORE_THRESHOLD = 0.25F;
constexpr std::size_t INPUT_COUNT = 3U * INPUT_SIZE * INPUT_SIZE;
constexpr std::size_t OUTPUT_COUNT = 1U * MAX_DETECTIONS * DETECTION_SIZE;
using Clock = std::chrono::steady_clock;
using Milliseconds = std::chrono::duration<double, std::milli>;
using Timing = std::array<double, 4>; // preprocess, infer, postprocess, total (ms)

cudaError_t preprocess_cuda(const unsigned char* image, int width, int height,
                            std::size_t pitch, float* input, cudaStream_t stream);

class Logger final : public nvinfer1::ILogger {
public:
    void log(Severity severity, const char* message) noexcept override {
        if (severity <= Severity::kWARNING) std::cerr << "[TensorRT] " << message << '\n';
    }
};

// 在 return 或异常退出时也释放已申请的 CUDA 资源。
struct CudaResources {
    unsigned char* image = nullptr;
    std::size_t image_capacity = 0;
    void* input = nullptr;
    void* output = nullptr;
    GpuResults* gpu_results = nullptr;
    GpuResults host_results{}; // 普通主机内存；锁页缓冲区留给 03。
    cudaStream_t stream = nullptr;
    ~CudaResources() {
        if (stream) {
            cudaStreamSynchronize(stream);
            cudaStreamDestroy(stream);
        }
        if (gpu_results) cudaFree(gpu_results);
        if (image) cudaFree(image);
        if (output) cudaFree(output);
        if (input) cudaFree(input);
    }
};

struct Detection {
    int class_id;
    float score;
    cv::Rect box;
};

bool preprocess(const cv::Mat& image, const CudaResources& cuda,
                float& scale, int& pad_x, int& pad_y) {
    scale = std::min(static_cast<float>(INPUT_SIZE) / static_cast<float>(image.cols),
                     static_cast<float>(INPUT_SIZE) / static_cast<float>(image.rows));
    const int width = std::max(1, static_cast<int>(std::round(static_cast<float>(image.cols) * scale)));
    const int height = std::max(1, static_cast<int>(std::round(static_cast<float>(image.rows) * scale)));
    pad_x = (INPUT_SIZE - width) / 2;
    pad_y = (INPUT_SIZE - height) / 2;

    // 原始 BGR uint8 图片传入 GPU，缩放、补边、RGB/CHW 和归一化由一个 kernel 完成。
    const std::size_t row_bytes = static_cast<std::size_t>(image.cols) * 3U;
    const bool submitted = cudaMemcpy2DAsync(
        cuda.image, row_bytes, image.data, image.step[0], row_bytes,
        static_cast<std::size_t>(image.rows), cudaMemcpyHostToDevice, cuda.stream) == cudaSuccess
        && preprocess_cuda(cuda.image, image.cols, image.rows, row_bytes,
                           static_cast<float*>(cuda.input), cuda.stream) == cudaSuccess;
    // 同步后才结束前处理计时；本版先不做跨阶段重叠。
    const auto status = cudaStreamSynchronize(cuda.stream);
    return submitted && status == cudaSuccess;
}

std::vector<Detection> postprocess(const std::vector<float>& output, const cv::Mat& image,
                                   float scale, int pad_x, int pad_y) {
    std::vector<Detection> detections;
    for (int i = 0; i < MAX_DETECTIONS; ++i) {
        const float* det = output.data() + i * DETECTION_SIZE;
        if (!std::all_of(det, det + DETECTION_SIZE, [](float value) { return std::isfinite(value); })
            || det[4] < SCORE_THRESHOLD || det[5] < 0.0F
            || static_cast<double>(det[5]) > static_cast<double>(std::numeric_limits<int>::max())) continue;

        const auto coordinate = [scale](float value, int padding, int size) {
            const float restored = (value - static_cast<float>(padding)) / scale;
            return static_cast<int>(std::clamp(
                static_cast<double>(restored), 0.0, static_cast<double>(size - 1)));
        };
        const int left = coordinate(det[0], pad_x, image.cols);
        const int top = coordinate(det[1], pad_y, image.rows);
        const int right = coordinate(det[2], pad_x, image.cols);
        const int bottom = coordinate(det[3], pad_y, image.rows);
        if (right <= left || bottom <= top) continue;
        detections.push_back({static_cast<int>(det[5]), det[4], {left, top, right - left, bottom - top}});
    }
    return detections;
}

void draw_detections(cv::Mat& image, const std::vector<Detection>& detections) {
    for (const auto& detection : detections) {
        const auto& box = detection.box;
        cv::rectangle(image, box.tl(), box.br(), {0, 255, 0}, 2);
        cv::putText(image, cv::format("class=%d %.2f", detection.class_id, detection.score),
                    {box.x, std::max(20, box.y - 5)}, cv::FONT_HERSHEY_SIMPLEX,
                    0.6, {0, 255, 0}, 2);
    }
}

std::vector<Detection> unpack(const GpuResults& result) {
    if (result.count < 0 || result.count > GPU_MAX_DETECTIONS)
        throw std::runtime_error("Invalid GPU detection count");
    std::vector<Detection> detections;
    detections.reserve(static_cast<std::size_t>(result.count));
    for (int i = 0; i < result.count; ++i) {
        const auto& box = result.boxes[i];
        detections.push_back({box.class_id, box.score,
                             {box.left, box.top, box.right - box.left, box.bottom - box.top}});
    }
    return detections;
}

bool infer(nvinfer1::IExecutionContext& context, CudaResources& cuda,
           std::vector<float>& output, const cv::Mat& image,
           float scale, int pad_x, int pad_y, bool gpu_post) {
    // 两种模式都保留前处理末尾和此处末尾的同步，只改变后处理位置。
    bool submitted = context.enqueueV3(cuda.stream);
    if (gpu_post) {
        submitted = submitted
            && postprocess_cuda(static_cast<const float*>(cuda.output), image.cols, image.rows,
                                scale, pad_x, pad_y, cuda.gpu_results, cuda.stream) == cudaSuccess
            && cudaMemcpyAsync(&cuda.host_results, cuda.gpu_results, sizeof(GpuResults),
                               cudaMemcpyDeviceToHost, cuda.stream) == cudaSuccess;
    } else {
        submitted = submitted
            && cudaMemcpyAsync(output.data(), cuda.output, OUTPUT_COUNT * sizeof(float),
                               cudaMemcpyDeviceToHost, cuda.stream) == cudaSuccess;
    }
    // 失败时也先等待已经提交的操作，保证主机缓冲区可以安全释放。
    const auto status = cudaStreamSynchronize(cuda.stream);
    return submitted && status == cudaSuccess;
}

bool valid_engine(const nvinfer1::ICudaEngine& engine) {
    const auto in = engine.getTensorShape(INPUT_NAME);
    const auto out = engine.getTensorShape(OUTPUT_NAME);
    if (engine.getNbIOTensors() != 2 || engine.hasImplicitBatchDimension()
        || in.nbDims != 4 || in.d[0] != 1 || in.d[1] != 3
        || in.d[2] != INPUT_SIZE || in.d[3] != INPUT_SIZE
        || out.nbDims != 3 || out.d[0] != 1
        || out.d[1] != MAX_DETECTIONS || out.d[2] != DETECTION_SIZE
        || engine.getTensorIOMode(INPUT_NAME) != nvinfer1::TensorIOMode::kINPUT
        || engine.getTensorIOMode(OUTPUT_NAME) != nvinfer1::TensorIOMode::kOUTPUT) return false;
    for (const char* name : {INPUT_NAME, OUTPUT_NAME}) {
        if (engine.getTensorDataType(name) != nvinfer1::DataType::kFLOAT
            || engine.getTensorFormat(name) != nvinfer1::TensorFormat::kLINEAR
            || engine.getTensorVectorizedDim(name) != -1
            || engine.getTensorLocation(name) != nvinfer1::TensorLocation::kDEVICE) return false;
    }
    return true;
}

int parse_count(const char* text, bool allow_zero) {
    const std::string value(text);
    int count = 0;
    const auto result = std::from_chars(value.data(), value.data() + value.size(), count);
    if (value.empty() || value.front() < '0' || value.front() > '9'
        || result.ec != std::errc{} || result.ptr != value.data() + value.size()
        || count < (allow_zero ? 0 : 1)) {
        throw std::invalid_argument("ITERATIONS must be a positive integer; WARMUP must be a nonnegative integer");
    }
    return count;
}

bool is_image(const std::filesystem::path& path) {
    std::string extension = path.extension().string();
    std::transform(extension.begin(), extension.end(), extension.begin(), [](unsigned char value) {
        return static_cast<char>(std::tolower(value));
    });
    return extension == ".jpg" || extension == ".jpeg" || extension == ".png" || extension == ".bmp";
}

std::string csv_quote(const std::string& text) {
    std::string quoted = "\"";
    for (const char value : text) {
        if (value == '"') quoted += '"';
        quoted += value;
    }
    return quoted + '"';
}

void print_statistics(const std::vector<Timing>& timings, bool gpu_post) {
    const char* names[] = {"Preprocess (H2D+CUDA+sync)",
                          gpu_post ? "Infer (TRT+GPU post+D2H+sync)" : "Infer (TRT+D2H+sync)",
                          gpu_post ? "CPU unpack" : "CPU postprocess", "Pipeline"};
    std::cout << std::fixed << std::setprecision(3);
    for (std::size_t stage = 0; stage < 4U; ++stage) {
        std::vector<double> values;
        values.reserve(timings.size());
        double sum = 0.0;
        for (const auto& timing : timings) {
            values.push_back(timing[stage]);
            sum += timing[stage];
        }
        std::sort(values.begin(), values.end());
        const auto percentile = [&values](double fraction) {
            const auto rank = static_cast<std::size_t>(std::ceil(fraction * static_cast<double>(values.size())));
            return values[rank - 1U]; // nearest-rank, 1-based
        };
        std::cout << "  " << names[stage] << ": mean=" << sum / static_cast<double>(values.size())
                  << " p50=" << percentile(0.50) << " p95=" << percentile(0.95)
                  << " p99=" << percentile(0.99) << " ms\n";
    }
}

int main(int argc, char** argv) {
    try {
        if (argc < 3 || argc > 6) {
            std::cerr << "Usage: " << argv[0] << " ENGINE IMAGE_FOLDER [ITERATIONS=200] [WARMUP=50] [GPU_POST=0]\n";
            return 1;
        }
        const int iterations = argc >= 4 ? parse_count(argv[3], false) : 200;
        const int warmup = argc >= 5 ? parse_count(argv[4], true) : 50;
        const int gpu_post_flag = argc >= 6 ? parse_count(argv[5], true) : 0;
        if (gpu_post_flag > 1) throw std::invalid_argument("GPU_POST must be 0 or 1");
        const bool gpu_post = gpu_post_flag != 0;
        const std::filesystem::path folder(argv[2]);
        if (!std::filesystem::is_directory(folder)) throw std::runtime_error("Cannot open image folder");
        std::vector<std::filesystem::path> paths;
        for (const auto& entry : std::filesystem::directory_iterator(folder)) {
            if (entry.is_regular_file() && is_image(entry.path())) paths.push_back(entry.path());
        }
        std::sort(paths.begin(), paths.end());
        if (paths.empty()) throw std::runtime_error("No images found");

        std::ifstream file(argv[1], std::ios::binary | std::ios::ate);
        if (!file || file.tellg() <= 0) throw std::runtime_error("Cannot open engine");
        const std::streamsize size = file.tellg();
        std::vector<char> engine_data(static_cast<std::size_t>(size));
        file.seekg(0);
        if (!file.read(engine_data.data(), size)) throw std::runtime_error("Cannot read engine");
        Logger logger;
        std::unique_ptr<nvinfer1::IRuntime> runtime(nvinfer1::createInferRuntime(logger));
        if (!runtime) throw std::runtime_error("Cannot create TensorRT runtime");
        std::unique_ptr<nvinfer1::ICudaEngine> engine(
            runtime->deserializeCudaEngine(engine_data.data(), engine_data.size()));
        if (!engine || !valid_engine(*engine)) throw std::runtime_error("Unsupported engine: expected FP32 linear [1,3,640,640] -> [1,300,6]");
        std::unique_ptr<nvinfer1::IExecutionContext> context(engine->createExecutionContext());
        if (!context) throw std::runtime_error("Cannot create execution context");

        CudaResources cuda;
        if (cudaMalloc(&cuda.input, INPUT_COUNT * sizeof(float)) != cudaSuccess
            || cudaMalloc(&cuda.output, OUTPUT_COUNT * sizeof(float)) != cudaSuccess
            || cudaStreamCreate(&cuda.stream) != cudaSuccess
            || !context->setTensorAddress(INPUT_NAME, cuda.input)
            || !context->setTensorAddress(OUTPUT_NAME, cuda.output)) throw std::runtime_error("Cannot prepare CUDA");

        if (gpu_post && cudaMalloc(reinterpret_cast<void**>(&cuda.gpu_results), sizeof(GpuResults)) != cudaSuccess)
            throw std::runtime_error("Cannot allocate GPU postprocess buffer");

        const auto result_folder = std::filesystem::path("results/02_cuda_prepost")
            / (gpu_post ? "gpu_post" : "cpu_post");
        std::filesystem::create_directories(result_folder);
        std::ofstream timing_file(result_folder / "timings.csv");
        std::ofstream detection_file(result_folder / "detections.csv");
        if (!timing_file || !detection_file) throw std::runtime_error("Cannot open result CSV files");
        timing_file << "image,iteration,preprocess_ms,infer_ms,postprocess_ms,total_ms\n" << std::setprecision(9);
        detection_file << "image,class_id,score,left,top,right,bottom\n" << std::setprecision(9);
        std::vector<float> output(OUTPUT_COUNT);
        std::size_t processed = 0;
        bool failed_image = false;
        std::cout << "Images: " << paths.size() << " | Iterations per image: " << iterations
                  << " | Warmup per image: " << warmup
                  << " | GPU postprocessing: " << gpu_post << '\n';

        for (const auto& path : paths) {
            cv::Mat image = cv::imread(path.string()); // 解码不计时，每张图片只读取一次。
            if (image.empty()) {
                std::cerr << "Cannot open image: " << path << '\n';
                failed_image = true;
                continue;
            }
            // 原图显存按需扩容，预热和正式测量期间不分配显存。
            const std::size_t image_bytes = static_cast<std::size_t>(image.cols)
                * static_cast<std::size_t>(image.rows) * 3U;
            if (image_bytes > cuda.image_capacity) {
                if (cuda.image && cudaFree(cuda.image) != cudaSuccess) throw std::runtime_error("Cannot free image buffer");
                cuda.image = nullptr;
                if (cudaMalloc(reinterpret_cast<void**>(&cuda.image), image_bytes) != cudaSuccess) {
                    throw std::runtime_error("Cannot allocate image buffer");
                }
                cuda.image_capacity = image_bytes;
            }
            std::vector<Timing> timings;
            timings.reserve(static_cast<std::size_t>(iterations));
            std::vector<Detection> detections;
            // 预热与正式测量走相同流程；负数索引只用于预热，避免 warmup + iterations 溢出。
            for (int iteration = -warmup; iteration < iterations; ++iteration) {
                float scale = 0.0F;
                int pad_x = 0, pad_y = 0;
                const auto start = Clock::now();
                if (!preprocess(image, cuda, scale, pad_x, pad_y)) throw std::runtime_error("Preprocessing failed: " + path.string());
                const auto pre_end = Clock::now();
                if (!infer(*context, cuda, output, image, scale, pad_x, pad_y, gpu_post))
                    throw std::runtime_error("Inference failed: " + path.string());
                const auto infer_end = Clock::now(); // infer 已同步，不只是 CPU 提交时间。
                detections = gpu_post ? unpack(cuda.host_results)
                                      : postprocess(output, image, scale, pad_x, pad_y);
                const auto post_end = Clock::now();
                if (iteration >= 0) {
                    timings.push_back({Milliseconds(pre_end - start).count(),
                                       Milliseconds(infer_end - pre_end).count(),
                                       Milliseconds(post_end - infer_end).count(),
                                       Milliseconds(post_end - start).count()});
                }
            }
            ++processed;
            std::cout << path.filename() << " | Detections: " << detections.size() << '\n';
            print_statistics(timings, gpu_post);
            const std::string filename = csv_quote(path.filename().string());
            for (std::size_t i = 0; i < timings.size(); ++i) {
                timing_file << filename << ',' << i + 1U;
                for (const double value : timings[i]) timing_file << ',' << value;
                timing_file << '\n';
            }
            for (const auto& detection : detections) {
                const auto& box = detection.box;
                detection_file << filename << ',' << detection.class_id << ',' << detection.score
                               << ',' << box.x << ',' << box.y
                               << ',' << box.x + box.width << ',' << box.y + box.height << '\n';
            }
            draw_detections(image, detections); // 绘图、保存、日志和 CSV 写入都在测量之外。
            if (!cv::imwrite((result_folder / path.filename()).string(), image)) throw std::runtime_error("Cannot save annotated image");
            timing_file.flush();
            detection_file.flush();
            if (!timing_file || !detection_file) throw std::runtime_error("Cannot write result CSV files");
        }
        timing_file.close();
        detection_file.close();
        if (!timing_file || !detection_file) throw std::runtime_error("Cannot close result CSV files");
        if (processed == 0U || failed_image) return 1;
        std::cout << "Saved: " << result_folder << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
