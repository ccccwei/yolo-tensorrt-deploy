#include <NvInferRuntime.h>
#include <cuda_runtime_api.h>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include "postprocess.hpp"

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
#include <utility>
#include <vector>

constexpr char INPUT_NAME[] = "images";
constexpr char OUTPUT_NAME[] = "output0";
constexpr int INPUT_SIZE = 640;
constexpr int MAX_DETECTIONS = 300;
constexpr int DETECTION_SIZE = 6;
constexpr std::size_t INPUT_COUNT = 3U * INPUT_SIZE * INPUT_SIZE;
constexpr std::size_t OUTPUT_COUNT = 1U * MAX_DETECTIONS * DETECTION_SIZE;
using Clock = std::chrono::steady_clock;
using Milliseconds = std::chrono::duration<double, std::milli>;
using Timing = std::array<double, 4>; // staging, GPU, postprocess, request latency (ms)

cudaError_t preprocess_cuda(const unsigned char* image, int width, int height,
                            std::size_t pitch, float* input, cudaStream_t stream);

class Logger final : public nvinfer1::ILogger {
public:
    void log(Severity severity, const char* message) noexcept override {
        if (severity <= Severity::kWARNING) std::cerr << "[TensorRT] " << message << '\n';
    }
};

void cuda_check(cudaError_t error, const char* operation) {
    if (error != cudaSuccess) throw std::runtime_error(
        std::string(operation) + ": " + cudaGetErrorString(error));
}

struct Detection { int class_id; float score; cv::Rect box; };
struct Image { std::filesystem::path path; cv::Mat pixels; };
struct Record { std::size_t image = 0, iteration = 0; Timing timing{}; };

struct GraphResource {
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t executable = nullptr;
    GraphResource() = default;
    GraphResource(const GraphResource&) = delete;
    GraphResource& operator=(const GraphResource&) = delete;
    ~GraphResource() {
        if (executable) cudaGraphExecDestroy(executable);
        if (graph) cudaGraphDestroy(graph);
    }
};

// 每个槽独占 context、stream、events 和所有可变缓冲区；仅共享只读 engine。
struct Slot {
    std::unique_ptr<nvinfer1::IExecutionContext> context;
    std::vector<std::unique_ptr<GraphResource>> graphs; // 此槽每张预加载图片一个图。
    cudaStream_t stream = nullptr;
    cudaEvent_t start_event = nullptr, done_event = nullptr;
    unsigned char *host_image = nullptr, *device_image = nullptr;
    GpuResults *host_results = nullptr, *device_results = nullptr;
    void *input = nullptr, *output = nullptr;
    std::size_t image_index = 0, task = 0, image_capacity = 0;
    float scale = 0.0F;
    int pad_x = 0, pad_y = 0;
    bool busy = false;
    Clock::time_point start;
    double staging_ms = 0.0;

    ~Slot() {
        if (stream) cudaStreamSynchronize(stream);
        graphs.clear(); // 先销毁引用 context 与固定 buffer 地址的图。
        context.reset(); // context 必须活到 GPU 不再访问它之后。
        if (start_event) cudaEventDestroy(start_event);
        if (done_event) cudaEventDestroy(done_event);
        if (host_image) cudaFreeHost(host_image);
        if (host_results) cudaFreeHost(host_results);
        if (device_results) cudaFree(device_results);
        if (device_image) cudaFree(device_image);
        if (input) cudaFree(input);
        if (output) cudaFree(output);
        if (stream) cudaStreamDestroy(stream);
    }
};

struct Slots {
    std::array<std::unique_ptr<Slot>, 2> values;
    ~Slots() {
        // 出错时先等所有 stream，再让各槽析构，engine 在更外层最后释放。
        for (const auto& slot : values) {
            if (slot && slot->stream) cudaStreamSynchronize(slot->stream);
        }
    }
};

void prepare_slot(Slot& slot, nvinfer1::ICudaEngine& engine, std::size_t image_bytes) {
    slot.context.reset(engine.createExecutionContext());
    if (!slot.context) throw std::runtime_error("Cannot create execution context");
    cuda_check(cudaStreamCreateWithFlags(&slot.stream, cudaStreamNonBlocking), "Create stream");
    cuda_check(cudaEventCreate(&slot.start_event), "Create start event");
    cuda_check(cudaEventCreate(&slot.done_event), "Create done event");
    cuda_check(cudaMallocHost(reinterpret_cast<void**>(&slot.host_image), image_bytes), "Allocate pinned image");
    cuda_check(cudaMallocHost(reinterpret_cast<void**>(&slot.host_results), sizeof(GpuResults)), "Allocate pinned results");
    cuda_check(cudaMalloc(reinterpret_cast<void**>(&slot.device_results), sizeof(GpuResults)), "Allocate device results");
    cuda_check(cudaMemsetAsync(slot.device_results, 0, sizeof(GpuResults), slot.stream), "Initialize result buffer");
    cuda_check(cudaMalloc(reinterpret_cast<void**>(&slot.device_image), image_bytes), "Allocate device image");
    cuda_check(cudaMalloc(&slot.input, INPUT_COUNT * sizeof(float)), "Allocate input");
    cuda_check(cudaMalloc(&slot.output, OUTPUT_COUNT * sizeof(float)), "Allocate output");
    slot.image_capacity = image_bytes;
    if (!slot.context->setTensorAddress(INPUT_NAME, slot.input)
        || !slot.context->setTensorAddress(OUTPUT_NAME, slot.output)) {
        throw std::runtime_error("Cannot bind engine input/output");
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

// CPU 拷贝始终逐请求执行，不进入 CUDA Graph，且包含在 staging / total 计时中。
void stage_image(Slot& slot, const cv::Mat& image) {
    if (slot.busy) throw std::logic_error("Cannot overwrite an in-flight slot");
    const std::size_t row_bytes = static_cast<std::size_t>(image.cols) * 3U;
    const std::size_t bytes = row_bytes * static_cast<std::size_t>(image.rows);
    if (image.empty() || image.type() != CV_8UC3 || bytes > slot.image_capacity) {
        throw std::runtime_error("Invalid image or slot capacity");
    }
    slot.scale = std::min(static_cast<float>(INPUT_SIZE) / static_cast<float>(image.cols),
                          static_cast<float>(INPUT_SIZE) / static_cast<float>(image.rows));
    const int width = std::max(1, static_cast<int>(std::round(static_cast<float>(image.cols) * slot.scale)));
    const int height = std::max(1, static_cast<int>(std::round(static_cast<float>(image.rows) * slot.scale)));
    slot.pad_x = (INPUT_SIZE - width) / 2;
    slot.pad_y = (INPUT_SIZE - height) / 2;
    slot.start = Clock::now();
    if (image.isContinuous()) {
        std::memcpy(slot.host_image, image.data, bytes);
    } else {
        for (int row = 0; row < image.rows; ++row) {
            std::memcpy(slot.host_image + static_cast<std::size_t>(row) * row_bytes,
                        image.ptr<unsigned char>(row), row_bytes);
        }
    }
    slot.staging_ms = Milliseconds(Clock::now() - slot.start).count();
}

// 同一提交函数供 GRAPH=0 和捕获使用：无分配、无同步、无 CPU 输出访问。
void enqueue_commands(Slot& slot, const cv::Mat& image) {
    const std::size_t row_bytes = static_cast<std::size_t>(image.cols) * 3U;
    const std::size_t bytes = row_bytes * static_cast<std::size_t>(image.rows);
    cuda_check(cudaMemcpyAsync(slot.device_image, slot.host_image, bytes,
                               cudaMemcpyHostToDevice, slot.stream), "Submit H2D");
    cuda_check(preprocess_cuda(slot.device_image, image.cols, image.rows, row_bytes,
                               static_cast<float*>(slot.input), slot.stream), "Submit preprocessing");
    if (!slot.context->enqueueV3(slot.stream)) throw std::runtime_error("Submit inference failed");
    cuda_check(postprocess_cuda(static_cast<const float*>(slot.output), image.cols, image.rows,
                                slot.scale, slot.pad_x, slot.pad_y, slot.device_results, slot.stream), "Submit postprocessing");
    cuda_check(cudaMemcpyAsync(slot.host_results, slot.device_results, sizeof(GpuResults),
                               cudaMemcpyDeviceToHost, slot.stream), "Submit D2H");
}

std::unique_ptr<GraphResource> capture_graph(Slot& slot, const cv::Mat& image) {
    auto captured = std::make_unique<GraphResource>();
    cuda_check(cudaStreamBeginCapture(slot.stream, cudaStreamCaptureModeThreadLocal), "Begin graph capture");
    try {
        enqueue_commands(slot, image);
    } catch (...) {
        // 提交异常也必须 EndCapture，不能留下失效捕获状态或偷偷回退。
        cudaStreamEndCapture(slot.stream, &captured->graph);
        throw;
    }
    cuda_check(cudaStreamEndCapture(slot.stream, &captured->graph), "End graph capture (no automatic fallback)");
    cuda_check(cudaGraphInstantiate(&captured->executable, captured->graph, nullptr, nullptr, 0), "Instantiate graph");
    return captured;
}

// submit 不等待 GPU。再次使用同一个槽前，调用方必须先 finish。
void submit(Slot& slot, const cv::Mat& image, std::size_t image_index, std::size_t task, bool graph_enabled) {
    slot.image_index = image_index;
    slot.task = task;
    stage_image(slot, image);
    cuda_check(cudaEventRecord(slot.start_event, slot.stream), "Record GPU start");
    if (graph_enabled) {
        cuda_check(cudaGraphLaunch(slot.graphs.at(image_index)->executable, slot.stream), "Launch graph");
    } else {
        enqueue_commands(slot, image);
    }
    cuda_check(cudaEventRecord(slot.done_event, slot.stream), "Record GPU completion");
    slot.busy = true;
}

void finish(Slot& slot, const std::vector<Image>& images,
            std::vector<std::vector<Detection>>& last, std::vector<Record>* records) {
    if (!slot.busy) return;
    cuda_check(cudaEventSynchronize(slot.done_event), "Wait for slot");
    float gpu_ms = 0.0F;
    cuda_check(cudaEventElapsedTime(&gpu_ms, slot.start_event, slot.done_event), "Measure GPU span");
    const auto post_start = Clock::now();
    auto detections = unpack(*slot.host_results); // 只组装 C++ 结果，不重新过滤或恢复坐标。
    const auto end = Clock::now();
    if (records) {
        (*records)[slot.task] = {slot.image_index, slot.task / images.size() + 1U,
            {slot.staging_ms, static_cast<double>(gpu_ms), Milliseconds(end - post_start).count(),
             Milliseconds(end - slot.start).count()}};
    }
    last[slot.image_index] = std::move(detections);
    slot.busy = false;
}

// 按 image 0, image 1, ... 轮转。用 3 张不同尺寸图可验证两槽的尺寸切换。
void run(Slots& slots, std::size_t slot_count, const std::vector<Image>& images,
         std::size_t samples, std::vector<std::vector<Detection>>& last, std::vector<Record>* records,
         bool graph_enabled) {
    for (std::size_t task = 0; task < samples; ++task) {
        auto& slot = *slots.values[task % slot_count];
        finish(slot, images, last, records);
        submit(slot, images[task % images.size()].pixels, task % images.size(), task, graph_enabled);
    }
    // 从最早未消费的任务开始 drain；不足一个完整循环时从槽 0 开始。
    const std::size_t oldest = samples >= slot_count ? samples % slot_count : 0U;
    for (std::size_t offset = 0; offset < slot_count; ++offset) {
        finish(*slots.values[(oldest + offset) % slot_count], images, last, records);
    }
}

bool valid_engine(const nvinfer1::ICudaEngine& engine) {
    const auto in = engine.getTensorShape(INPUT_NAME);
    const auto out = engine.getTensorShape(OUTPUT_NAME);
    if (engine.getNbIOTensors() != 2 || engine.hasImplicitBatchDimension()
        || in.nbDims != 4 || in.d[0] != 1 || in.d[1] != 3 || in.d[2] != INPUT_SIZE || in.d[3] != INPUT_SIZE
        || out.nbDims != 3 || out.d[0] != 1 || out.d[1] != MAX_DETECTIONS || out.d[2] != DETECTION_SIZE
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
        || count < (allow_zero ? 0 : 1)) throw std::invalid_argument("Invalid integer argument");
    return count;
}

std::size_t checked_product(std::size_t left, std::size_t right) {
    if (right != 0U && left > std::numeric_limits<std::size_t>::max() / right) {
        throw std::overflow_error("Image size or sample count overflow");
    }
    return left * right;
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

void print_statistics(const std::vector<Record>& records) {
    constexpr const char* names[] = {"CPU staging", "GPU span (H2D+pre+TRT+post+D2H)", "CPU result unpack", "Request latency"};
    std::cout << std::fixed << std::setprecision(3);
    for (std::size_t stage = 0; stage < 4U; ++stage) {
        std::vector<double> values;
        values.reserve(records.size());
        double sum = 0.0;
        for (const auto& record : records) { values.push_back(record.timing[stage]); sum += record.timing[stage]; }
        std::sort(values.begin(), values.end());
        const auto percentile = [&values](double fraction) {
            const auto rank = static_cast<std::size_t>(std::ceil(fraction * static_cast<double>(values.size())));
            return values[rank - 1U];
        };
        std::cout << "  " << names[stage] << ": mean=" << sum / static_cast<double>(values.size())
                  << " p50=" << percentile(0.50) << " p95=" << percentile(0.95)
                  << " p99=" << percentile(0.99) << " ms\n";
    }
}

void draw_detections(cv::Mat& image, const std::vector<Detection>& detections) {
    for (const auto& detection : detections) {
        const auto& box = detection.box;
        cv::rectangle(image, box.tl(), box.br(), {0, 255, 0}, 2);
        cv::putText(image, cv::format("class=%d %.2f", detection.class_id, detection.score),
                    {box.x, std::max(20, box.y - 5)}, cv::FONT_HERSHEY_SIMPLEX, 0.6, {0, 255, 0}, 2);
    }
}

int main(int argc, char** argv) {
    try {
        if (argc < 3 || argc > 7) {
            std::cerr << "Usage: " << argv[0] << " ENGINE IMAGE_FOLDER [ITERATIONS=200] [WARMUP=50] [SLOTS=2] [GRAPH=1]\n";
            return 1;
        }
        const int iterations = argc >= 4 ? parse_count(argv[3], false) : 200;
        const int warmup = argc >= 5 ? parse_count(argv[4], true) : 50;
        const int slot_argument = argc >= 6 ? parse_count(argv[5], false) : 2;
        if (slot_argument > 2) throw std::invalid_argument("SLOTS must be 1 or 2");
        const int graph_argument = argc >= 7 ? parse_count(argv[6], true) : 1;
        if (graph_argument > 1) throw std::invalid_argument("GRAPH must be 0 or 1");
        const bool graph_enabled = graph_argument != 0;
        const auto slot_count = static_cast<std::size_t>(slot_argument);
        const std::filesystem::path folder(argv[2]);
        if (!std::filesystem::is_directory(folder)) throw std::runtime_error("Cannot open image folder");
        std::vector<std::filesystem::path> paths;
        for (const auto& entry : std::filesystem::directory_iterator(folder)) {
            if (entry.is_regular_file() && is_image(entry.path())) paths.push_back(entry.path());
        }
        std::sort(paths.begin(), paths.end());
        std::vector<Image> images;
        bool failed_image = false;
        std::size_t image_capacity = 0;
        for (const auto& path : paths) {
            cv::Mat image = cv::imread(path.string()); // 全部预加载；计时期间保持原图不变。
            if (image.empty()) { std::cerr << "Cannot read image: " << path << '\n'; failed_image = true; continue; }
            image_capacity = std::max(image_capacity, checked_product(
                checked_product(static_cast<std::size_t>(image.cols), 3U), static_cast<std::size_t>(image.rows)));
            images.push_back({path, std::move(image)});
        }
        if (images.empty()) throw std::runtime_error("No readable images found");
        const auto samples = checked_product(images.size(), static_cast<std::size_t>(iterations));
        const auto warmup_samples = checked_product(images.size(), static_cast<std::size_t>(warmup));
        const auto graph_count = graph_enabled ? checked_product(slot_count, images.size()) : 0U;
        checked_product(samples, sizeof(Record));
        std::vector<Record> records(samples);
        std::vector<std::vector<Detection>> last(images.size());

        std::ifstream file(argv[1], std::ios::binary | std::ios::ate);
        if (!file || file.tellg() <= 0) throw std::runtime_error("Cannot open engine");
        const std::streamsize size = file.tellg();
        std::vector<char> engine_data(static_cast<std::size_t>(size));
        file.seekg(0);
        if (!file.read(engine_data.data(), size)) throw std::runtime_error("Cannot read engine");
        Logger logger;
        std::unique_ptr<nvinfer1::IRuntime> runtime(nvinfer1::createInferRuntime(logger));
        if (!runtime) throw std::runtime_error("Cannot create TensorRT runtime");
        std::unique_ptr<nvinfer1::ICudaEngine> engine(runtime->deserializeCudaEngine(engine_data.data(), engine_data.size()));
        if (!engine || !valid_engine(*engine)) throw std::runtime_error("Unsupported engine: expected FP32 linear [1,3,640,640] -> [1,300,6]");
        const auto setup_start = Clock::now();
        Slots slots;
        for (std::size_t index = 0; index < slot_count; ++index) {
            slots.values[index] = std::make_unique<Slot>();
            auto& slot = *slots.values[index];
            prepare_slot(slot, *engine, image_capacity);
            if (graph_enabled) slot.graphs.resize(images.size());
            for (std::size_t image_index = 0; image_index < images.size(); ++image_index) {
                const auto& image = images[image_index].pixels;
                stage_image(slot, image);
                // 两种模式均 prime；TRT 首次分配/状态更新必须在捕获之前完成。
                enqueue_commands(slot, image);
                cuda_check(cudaStreamSynchronize(slot.stream), "Prime inference before capture");
                if (graph_enabled) slot.graphs[image_index] = capture_graph(slot, image);
            }
        }
        const double setup_ms = Milliseconds(Clock::now() - setup_start).count();
        const std::filesystem::path result_folder("results/09_graph_pipeline");
        std::filesystem::create_directories(result_folder);
        std::ofstream timing_file(result_folder / "timings.csv"), throughput_file(result_folder / "throughput.csv");
        std::ofstream detection_file(result_folder / "detections.csv");
        if (!timing_file || !throughput_file || !detection_file) throw std::runtime_error("Cannot open result CSV files");
        std::cout << "Images: " << images.size() << " | Iterations per image: " << iterations
                  << " | Warmup per image: " << warmup << " | Slots: " << slot_count
                  << " | Graph: " << graph_enabled << " | GraphExecs: " << graph_count
                  << " | Setup: " << setup_ms << " ms\n";
        run(slots, slot_count, images, warmup_samples, last, nullptr, graph_enabled); // 完整 drain，正式测量从空队列开始。
        const auto start = Clock::now();
        run(slots, slot_count, images, samples, last, &records, graph_enabled);
        const double wall_ms = Milliseconds(Clock::now() - start).count();
        const double fps = static_cast<double>(samples) * 1000.0 / wall_ms;

        print_statistics(records);
        std::cout << "Throughput: " << fps << " FPS (" << samples << " samples / " << wall_ms << " ms)\n";
        throughput_file << "samples,wall_ms,fps\n" << std::setprecision(9) << samples << ',' << wall_ms << ',' << fps << '\n';
        timing_file << "image,iteration,staging_ms,gpu_ms,postprocess_ms,total_ms\n" << std::setprecision(9);
        for (const auto& record : records) {
            timing_file << csv_quote(images[record.image].path.filename().string()) << ',' << record.iteration;
            for (const double value : record.timing) timing_file << ',' << value;
            timing_file << '\n';
        }
        detection_file << "image,class_id,score,left,top,right,bottom\n" << std::setprecision(9);
        for (std::size_t index = 0; index < images.size(); ++index) {
            const auto& image = images[index];
            for (const auto& detection : last[index]) {
                const auto& box = detection.box;
                detection_file << csv_quote(image.path.filename().string()) << ',' << detection.class_id << ',' << detection.score
                               << ',' << box.x << ',' << box.y << ',' << box.x + box.width << ',' << box.y + box.height << '\n';
            }
            cv::Mat annotated = image.pixels.clone();
            draw_detections(annotated, last[index]);
            if (!cv::imwrite((result_folder / image.path.filename()).string(), annotated)) throw std::runtime_error("Cannot save annotated image");
        }
        timing_file.close(); throughput_file.close(); detection_file.close();
        if (!timing_file || !throughput_file || !detection_file) throw std::runtime_error("Cannot write result CSV files");
        std::cout << "Saved: " << result_folder << '\n';
        return failed_image ? 1 : 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
