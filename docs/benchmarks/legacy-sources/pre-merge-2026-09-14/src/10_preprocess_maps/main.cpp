#include "preprocess.hpp"

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
#include <cstring>
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
using Timing = std::array<double, 4>; // CPU staging, GPU events, CPU postprocess, CPU total (ms)

class Logger final : public nvinfer1::ILogger {
public:
    void log(Severity severity, const char* message) noexcept override {
        if (severity <= Severity::kWARNING) std::cerr << "[TensorRT] " << message << '\n';
    }
};

// 在 return 或异常退出时也释放已申请的 CUDA 资源。
struct CudaResources {
    unsigned char* image = nullptr;
    unsigned char* host_image = nullptr;
    float* host_output = nullptr;
    std::size_t image_capacity = 0;
    AxisMap* x_map = nullptr;
    AxisMap* y_map = nullptr;
    bool maps_enabled = true;
    void* input = nullptr;
    void* output = nullptr;
    cudaStream_t stream = nullptr;
    cudaEvent_t begin = nullptr;
    cudaEvent_t end = nullptr;
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t executable = nullptr;
    bool clear_graph() {
        if (executable && cudaGraphExecDestroy(executable) != cudaSuccess) return false;
        executable = nullptr;
        if (graph && cudaGraphDestroy(graph) != cudaSuccess) return false;
        graph = nullptr;
        return true;
    }
    ~CudaResources() {
        if (stream) cudaStreamSynchronize(stream);
        clear_graph(); // Graph 节点引用的 context 和 buffer 此时仍有效。
        if (begin) cudaEventDestroy(begin);
        if (end) cudaEventDestroy(end);
        if (stream) cudaStreamDestroy(stream);
        if (host_image) cudaFreeHost(host_image);
        if (host_output) cudaFreeHost(host_output);
        if (image) cudaFree(image);
        if (x_map) cudaFree(x_map);
        if (y_map) cudaFree(y_map);
        if (output) cudaFree(output);
        if (input) cudaFree(input);
    }
};

// Pinned 内存和显存均按最大原图容量复用；调用位置在预热与计时之外。
bool prepare_image(CudaResources& cuda, std::size_t bytes) {
    if (bytes == 0U) return false;
    if (bytes <= cuda.image_capacity) return true;
    if (cuda.stream && cudaStreamSynchronize(cuda.stream) != cudaSuccess) return false;
    if (cuda.image && cudaFree(cuda.image) != cudaSuccess) return false;
    cuda.image = nullptr;
    if (cuda.host_image && cudaFreeHost(cuda.host_image) != cudaSuccess) return false;
    cuda.host_image = nullptr;
    cuda.image_capacity = 0;
    if (cudaMalloc(reinterpret_cast<void**>(&cuda.image), bytes) != cudaSuccess
        || cudaMallocHost(reinterpret_cast<void**>(&cuda.host_image), bytes) != cudaSuccess) return false;
    cuda.image_capacity = bytes;
    return true;
}

struct Detection {
    int class_id;
    float score;
    cv::Rect box;
};

bool stage_image(const cv::Mat& image, const CudaResources& cuda,
                float& scale, int& pad_x, int& pad_y) {
    if (image.empty() || image.type() != CV_8UC3) return false;
    const std::size_t row_bytes = static_cast<std::size_t>(image.cols) * 3U;
    const std::size_t image_bytes = row_bytes * static_cast<std::size_t>(image.rows);
    if (!cuda.host_image || !cuda.image || !cuda.input || !cuda.stream
        || image_bytes > cuda.image_capacity) return false;
    scale = std::min(static_cast<float>(INPUT_SIZE) / static_cast<float>(image.cols),
                     static_cast<float>(INPUT_SIZE) / static_cast<float>(image.rows));
    const int width = std::max(1, static_cast<int>(std::round(static_cast<float>(image.cols) * scale)));
    const int height = std::max(1, static_cast<int>(std::round(static_cast<float>(image.rows) * scale)));
    pad_x = (INPUT_SIZE - width) / 2;
    pad_y = (INPUT_SIZE - height) / 2;

    // imread 返回普通内存：每次推理先复制到 pinned buffer，这段 CPU 复制也计时。
    if (image.isContinuous()) {
        std::memcpy(cuda.host_image, image.data, image_bytes);
    } else {
        for (int row = 0; row < image.rows; ++row) {
            std::memcpy(cuda.host_image + static_cast<std::size_t>(row) * row_bytes,
                        image.ptr<unsigned char>(row), row_bytes);
        }
    }
    return true; // CPU memcpy 不进入 Graph，每次 launch 前都填入当前图像。
}

// 不分配内存、不同步：两个模式都捕获整个提交段，只有前处理 kernel 不同。
bool submit(nvinfer1::IExecutionContext& context, const CudaResources& cuda,
            int width, int height) {
    const std::size_t row_bytes = static_cast<std::size_t>(width) * 3U;
    return cudaMemcpy2DAsync(
        cuda.image, row_bytes, cuda.host_image, row_bytes, row_bytes,
        static_cast<std::size_t>(height), cudaMemcpyHostToDevice, cuda.stream) == cudaSuccess
        && (cuda.maps_enabled
            ? preprocess_maps_cuda(cuda.image, width, height, row_bytes, cuda.x_map, cuda.y_map,
                                   static_cast<float*>(cuda.input), cuda.stream)
            : preprocess_cuda(cuda.image, width, height, row_bytes,
                              static_cast<float*>(cuda.input), cuda.stream)) == cudaSuccess
        && context.enqueueV3(cuda.stream)
        && cudaMemcpyAsync(cuda.host_output, cuda.output, OUTPUT_COUNT * sizeof(float),
                           cudaMemcpyDeviceToHost, cuda.stream) == cudaSuccess;
}

bool capture_graph(nvinfer1::IExecutionContext& context, CudaResources& cuda,
                   int width, int height) {
    if (cudaStreamBeginCapture(cuda.stream, cudaStreamCaptureModeThreadLocal) != cudaSuccess) return false;
    const bool submitted = submit(context, cuda, width, height);
    cudaGraph_t captured = nullptr;
    // 即使提交失败，也必须结束捕获，清理失效图，而不是悄悄退回普通提交。
    const auto status = cudaStreamEndCapture(cuda.stream, &captured);
    if (!submitted || status != cudaSuccess) {
        if (captured) cudaGraphDestroy(captured);
        return false;
    }
    cuda.graph = captured;
    return cudaGraphInstantiate(&cuda.executable, cuda.graph, nullptr, nullptr, 0) == cudaSuccess;
}

std::vector<Detection> postprocess(const float* output, const cv::Mat& image,
                                   float scale, int pad_x, int pad_y) {
    std::vector<Detection> detections;
    for (int i = 0; i < MAX_DETECTIONS; ++i) {
        const float* det = output + i * DETECTION_SIZE;
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

void print_statistics(const std::vector<Timing>& timings) {
    constexpr const char* names[] = {"CPU staging", "GPU (H2D+CUDA+TRT+D2H)", "CPU postprocess", "Pipeline"};
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
            std::cerr << "Usage: " << argv[0] << " ENGINE IMAGE_FOLDER [ITERATIONS=200] [WARMUP=50] [MAPS=1]\n";
            return 1;
        }
        const int iterations = argc >= 4 ? parse_count(argv[3], false) : 200;
        const int warmup = argc >= 5 ? parse_count(argv[4], true) : 50;
        if (argc >= 6 && std::string(argv[5]) != "0" && std::string(argv[5]) != "1")
            throw std::invalid_argument("MAPS must be 0 or 1");
        const bool maps_enabled = argc < 6 || std::string(argv[5]) == "1";
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
        cuda.maps_enabled = maps_enabled;
        if (cudaMalloc(reinterpret_cast<void**>(&cuda.x_map), INPUT_SIZE * sizeof(AxisMap)) != cudaSuccess
            || cudaMalloc(reinterpret_cast<void**>(&cuda.y_map), INPUT_SIZE * sizeof(AxisMap)) != cudaSuccess
            || cudaMalloc(&cuda.input, INPUT_COUNT * sizeof(float)) != cudaSuccess
            || cudaMalloc(&cuda.output, OUTPUT_COUNT * sizeof(float)) != cudaSuccess
            || cudaMallocHost(reinterpret_cast<void**>(&cuda.host_output), OUTPUT_COUNT * sizeof(float)) != cudaSuccess
            || cudaStreamCreateWithFlags(&cuda.stream, cudaStreamNonBlocking) != cudaSuccess
            || cudaEventCreate(&cuda.begin) != cudaSuccess
            || cudaEventCreate(&cuda.end) != cudaSuccess
            || !context->setTensorAddress(INPUT_NAME, cuda.input)
            || !context->setTensorAddress(OUTPUT_NAME, cuda.output)) throw std::runtime_error("Cannot prepare CUDA");

        const std::filesystem::path result_folder("results/10_preprocess_maps");
        std::filesystem::create_directories(result_folder);
        std::ofstream timing_file(result_folder / "timings.csv");
        std::ofstream detection_file(result_folder / "detections.csv");
        if (!timing_file || !detection_file) throw std::runtime_error("Cannot open result CSV files");
        timing_file << "image,iteration,staging_ms,gpu_ms,postprocess_ms,total_ms\n" << std::setprecision(9);
        detection_file << "image,class_id,score,left,top,right,bottom\n" << std::setprecision(9);
        std::size_t processed = 0;
        bool failed_image = false;
        std::size_t samples = 0;
        double total_wall_ms = 0.0;
        std::cout << "Images: " << paths.size() << " | Iterations per image: " << iterations
                  << " | Warmup per image: " << warmup << " | Maps: " << maps_enabled << " | Graph: 1" << '\n';

        for (const auto& path : paths) {
            cv::Mat image = cv::imread(path.string()); // 解码不计时，每张图片只读取一次。
            if (image.empty()) {
                std::cerr << "Cannot open image: " << path << '\n';
                failed_image = true;
                continue;
            }
            const auto setup_start = Clock::now();
            // 先销毁引用旧地址的图，再扩容；尺寸不同的图片重新捕获，不混进稳态计时。
            if (!cuda.clear_graph()) throw std::runtime_error("Cannot destroy previous graph");
            const std::size_t image_bytes = static_cast<std::size_t>(image.cols)
                * static_cast<std::size_t>(image.rows) * 3U;
            if (!prepare_image(cuda, image_bytes)) throw std::runtime_error("Cannot prepare pinned image buffers");
            float scale = 0.0F;
            int pad_x = 0, pad_y = 0;
            if (!stage_image(image, cuda, scale, pad_x, pad_y)) throw std::runtime_error("Cannot stage image");
            // 表只依赖尺寸，不依赖像素；在捕获与稳态计时前生成，后续重放仅查表。
            if (maps_enabled && build_preprocess_maps(image.cols, image.rows, cuda.x_map,
                                                      cuda.y_map, cuda.stream) != cudaSuccess)
                throw std::runtime_error("Cannot build preprocessing maps");
            // TensorRT 首次执行可能更新内部状态/分配资源，必须在捕获前完成。两种模式都执行。
            const bool primed = submit(*context, cuda, image.cols, image.rows);
            const auto prime_status = cudaStreamSynchronize(cuda.stream);
            if (!primed || prime_status != cudaSuccess) throw std::runtime_error("Initial inference failed");
            if (!capture_graph(*context, cuda, image.cols, image.rows))
                throw std::runtime_error("CUDA Graph capture/instantiate failed; no automatic fallback");
            const double setup_ms = Milliseconds(Clock::now() - setup_start).count();
            std::cout << path.filename() << " | setup_ms=" << setup_ms << '\n';
            std::vector<Timing> timings;
            timings.reserve(static_cast<std::size_t>(iterations));
            std::vector<Detection> detections;
            Clock::time_point measurement_start;
            for (int iteration = -warmup; iteration < iterations; ++iteration) {
                if (iteration == 0) measurement_start = Clock::now();
                const auto start = Clock::now();
                if (!stage_image(image, cuda, scale, pad_x, pad_y)) throw std::runtime_error("Staging failed");
                const auto stage_end = Clock::now();
                if (cudaEventRecord(cuda.begin, cuda.stream) != cudaSuccess) throw std::runtime_error("Cannot record begin event");
                const bool submitted = cudaGraphLaunch(cuda.executable, cuda.stream) == cudaSuccess;
                if (!submitted || cudaEventRecord(cuda.end, cuda.stream) != cudaSuccess
                    || cudaEventSynchronize(cuda.end) != cudaSuccess) throw std::runtime_error("GPU inference failed");
                const auto gpu_end = Clock::now(); // 两种模式都只在整段 GPU 工作末尾同步一次。
                detections = postprocess(cuda.host_output, image, scale, pad_x, pad_y);
                const auto post_end = Clock::now();
                float gpu_ms = 0.0F;
                if (cudaEventElapsedTime(&gpu_ms, cuda.begin, cuda.end) != cudaSuccess)
                    throw std::runtime_error("Cannot measure GPU events");
                if (iteration >= 0) {
                    timings.push_back({Milliseconds(stage_end - start).count(),
                                       static_cast<double>(gpu_ms),
                                       Milliseconds(post_end - gpu_end).count(),
                                       Milliseconds(post_end - start).count()});
                }
            }
            // 真正串行吞吐率：测量区间的墙钟时间，包含循环/统计开销，不使用 1000/p50。
            total_wall_ms += Milliseconds(Clock::now() - measurement_start).count();
            samples += timings.size();
            ++processed;
            std::cout << path.filename() << " | Detections: " << detections.size() << '\n';
            print_statistics(timings);
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
        std::ofstream throughput_file(result_folder / "throughput.csv");
        throughput_file << "samples,wall_ms,fps\n" << std::setprecision(9)
                        << samples << ',' << total_wall_ms << ','
                        << static_cast<double>(samples) * 1000.0 / total_wall_ms << '\n';
        throughput_file.close();
        if (!throughput_file) throw std::runtime_error("Cannot write throughput CSV");
        std::cout << "Serial throughput: " << static_cast<double>(samples) * 1000.0 / total_wall_ms << " images/s\n";
        std::cout << "Saved: " << result_folder << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
