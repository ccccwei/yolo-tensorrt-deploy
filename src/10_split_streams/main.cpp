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
#include <utility>
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

// 槽只管理张量和事件；所有槽共享一个 context 和同一条 compute stream。
struct Slot {
    cudaEvent_t start_event = nullptr, input_ready = nullptr;
    cudaEvent_t inference_done = nullptr, done_event = nullptr;
    unsigned char *host_image = nullptr, *device_image = nullptr;
    float* host_output = nullptr;
    void *input = nullptr, *output = nullptr;
    std::size_t image_index = 0, task = 0, image_capacity = 0;
    float scale = 0.0F;
    int pad_x = 0, pad_y = 0;
    bool busy = false;
    Clock::time_point start;
    double staging_ms = 0.0;

    ~Slot() {
        if (start_event) cudaEventDestroy(start_event);
        if (input_ready) cudaEventDestroy(input_ready);
        if (inference_done) cudaEventDestroy(inference_done);
        if (done_event) cudaEventDestroy(done_event);
        if (host_image) cudaFreeHost(host_image);
        if (host_output) cudaFreeHost(host_output);
        if (device_image) cudaFree(device_image);
        if (input) cudaFree(input);
        if (output) cudaFree(output);
    }
};

struct Pipeline {
    std::unique_ptr<nvinfer1::IExecutionContext> context;
    cudaStream_t input_stream = nullptr, compute_stream = nullptr, output_stream = nullptr;
    std::array<std::unique_ptr<Slot>, 4> slots;
    bool split = true;

    ~Pipeline() {
        // 即使 submit 中途抛错，也先 drain 所有已提交的工作，再释放 context 和槽。
        for (const auto stream : {input_stream, compute_stream, output_stream}) {
            if (stream) cudaStreamSynchronize(stream);
        }
        context.reset();
        for (const auto stream : {input_stream, compute_stream, output_stream}) {
            if (stream) cudaStreamDestroy(stream);
        }
    }
};

void prepare_slot(Slot& slot, std::size_t image_bytes) {
    cuda_check(cudaEventCreate(&slot.start_event), "Create start event");
    cuda_check(cudaEventCreateWithFlags(&slot.input_ready, cudaEventDisableTiming), "Create input-ready event");
    cuda_check(cudaEventCreateWithFlags(&slot.inference_done, cudaEventDisableTiming), "Create inference-done event");
    cuda_check(cudaEventCreate(&slot.done_event), "Create done event");
    cuda_check(cudaMallocHost(reinterpret_cast<void**>(&slot.host_image), image_bytes), "Allocate pinned image");
    cuda_check(cudaMallocHost(reinterpret_cast<void**>(&slot.host_output), OUTPUT_COUNT * sizeof(float)), "Allocate pinned output");
    cuda_check(cudaMalloc(reinterpret_cast<void**>(&slot.device_image), image_bytes), "Allocate device image");
    cuda_check(cudaMalloc(&slot.input, INPUT_COUNT * sizeof(float)), "Allocate input");
    cuda_check(cudaMalloc(&slot.output, OUTPUT_COUNT * sizeof(float)), "Allocate output");
    slot.image_capacity = image_bytes;
}

void prepare_pipeline(Pipeline& pipeline, nvinfer1::ICudaEngine& engine,
                      std::size_t slot_count, std::size_t image_bytes) {
    pipeline.context.reset(engine.createExecutionContext());
    if (!pipeline.context) throw std::runtime_error("Cannot create execution context");
    cuda_check(cudaStreamCreateWithFlags(&pipeline.input_stream, cudaStreamNonBlocking), "Create input stream");
    cuda_check(cudaStreamCreateWithFlags(&pipeline.compute_stream, cudaStreamNonBlocking), "Create compute stream");
    cuda_check(cudaStreamCreateWithFlags(&pipeline.output_stream, cudaStreamNonBlocking), "Create output stream");
    for (std::size_t index = 0; index < slot_count; ++index) {
        pipeline.slots[index] = std::make_unique<Slot>();
        prepare_slot(*pipeline.slots[index], image_bytes);
    }
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

// submit 不等待 GPU。再次使用同一个槽前，调用方必须先 finish。
void submit(Pipeline& pipeline, Slot& slot, const cv::Mat& image,
            std::size_t image_index, std::size_t task) {
    if (slot.busy) throw std::logic_error("Cannot overwrite an in-flight slot");
    const std::size_t row_bytes = static_cast<std::size_t>(image.cols) * 3U;
    const std::size_t bytes = row_bytes * static_cast<std::size_t>(image.rows);
    if (image.empty() || image.type() != CV_8UC3 || bytes > slot.image_capacity) {
        throw std::runtime_error("Invalid image or slot capacity");
    }
    slot.image_index = image_index;
    slot.task = task;
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
    const auto input_stream = pipeline.split ? pipeline.input_stream : pipeline.compute_stream;
    const auto output_stream = pipeline.split ? pipeline.output_stream : pipeline.compute_stream;
    cuda_check(cudaEventRecord(slot.start_event, input_stream), "Record GPU start");
    cuda_check(cudaMemcpyAsync(slot.device_image, slot.host_image, bytes,
                               cudaMemcpyHostToDevice, input_stream), "Submit H2D");
    cuda_check(cudaEventRecord(slot.input_ready, input_stream), "Record input ready");
    cuda_check(cudaStreamWaitEvent(pipeline.compute_stream, slot.input_ready, 0), "Wait for H2D");
    cuda_check(preprocess_cuda(slot.device_image, image.cols, image.rows, row_bytes,
                               static_cast<float*>(slot.input), pipeline.compute_stream), "Submit preprocessing");
    // 只切换下一次 enqueue 的地址，不改写旧槽；同一 CPU 线程、同一 compute stream 串行提交。
    if (!pipeline.context->setTensorAddress(INPUT_NAME, slot.input)
        || !pipeline.context->setTensorAddress(OUTPUT_NAME, slot.output)) {
        throw std::runtime_error("Cannot bind engine input/output");
    }
    if (!pipeline.context->enqueueV3(pipeline.compute_stream)) throw std::runtime_error("Submit inference failed");
    cuda_check(cudaEventRecord(slot.inference_done, pipeline.compute_stream), "Record inference done");
    cuda_check(cudaStreamWaitEvent(output_stream, slot.inference_done, 0), "Wait for inference");
    cuda_check(cudaMemcpyAsync(slot.host_output, slot.output, OUTPUT_COUNT * sizeof(float),
                               cudaMemcpyDeviceToHost, output_stream), "Submit D2H");
    cuda_check(cudaEventRecord(slot.done_event, output_stream), "Record GPU completion");
    slot.busy = true;
}

void finish(Slot& slot, const std::vector<Image>& images,
            std::vector<std::vector<Detection>>& last, std::vector<Record>* records) {
    if (!slot.busy) return;
    cuda_check(cudaEventSynchronize(slot.done_event), "Wait for slot");
    float gpu_ms = 0.0F;
    cuda_check(cudaEventElapsedTime(&gpu_ms, slot.start_event, slot.done_event), "Measure GPU span");
    const auto post_start = Clock::now();
    auto detections = postprocess(slot.host_output, images[slot.image_index].pixels,
                                 slot.scale, slot.pad_x, slot.pad_y);
    const auto end = Clock::now();
    if (records) {
        (*records)[slot.task] = {slot.image_index, slot.task / images.size() + 1U,
            {slot.staging_ms, static_cast<double>(gpu_ms), Milliseconds(end - post_start).count(),
             Milliseconds(end - slot.start).count()}};
    }
    last[slot.image_index] = std::move(detections);
    slot.busy = false;
}

// 按 image 0, image 1, ... 轮转；槽循环复用，最后按提交顺序 drain。
void run(Pipeline& pipeline, std::size_t slot_count, const std::vector<Image>& images,
         std::size_t samples, std::vector<std::vector<Detection>>& last, std::vector<Record>* records) {
    for (std::size_t task = 0; task < samples; ++task) {
        auto& slot = *pipeline.slots[task % slot_count];
        finish(slot, images, last, records);
        submit(pipeline, slot, images[task % images.size()].pixels, task % images.size(), task);
    }
    // 从最早未消费的任务开始 drain；不足一个完整循环时从槽 0 开始。
    const std::size_t oldest = samples >= slot_count ? samples % slot_count : 0U;
    for (std::size_t offset = 0; offset < slot_count; ++offset) {
        finish(*pipeline.slots[(oldest + offset) % slot_count], images, last, records);
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
    constexpr const char* names[] = {"CPU staging", "GPU span (H2D+CUDA+TRT+D2H)", "CPU postprocess", "Request latency"};
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
            std::cerr << "Usage: " << argv[0] << " ENGINE IMAGE_FOLDER [ITERATIONS=200] [WARMUP=50] [SLOTS=2] [SPLIT=1]\n";
            return 1;
        }
        const int iterations = argc >= 4 ? parse_count(argv[3], false) : 200;
        const int warmup = argc >= 5 ? parse_count(argv[4], true) : 50;
        const int slot_argument = argc >= 6 ? parse_count(argv[5], false) : 2;
        if (slot_argument > 4) throw std::invalid_argument("SLOTS must be in [1,4]");
        const std::string split_argument = argc >= 7 ? argv[6] : "1";
        if (split_argument != "0" && split_argument != "1") throw std::invalid_argument("SPLIT must be 0 or 1");
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
        Pipeline pipeline;
        pipeline.split = split_argument == "1";
        prepare_pipeline(pipeline, *engine, slot_count, image_capacity);
        const std::filesystem::path result_folder("results/10_split_streams");
        std::filesystem::create_directories(result_folder);
        std::ofstream timing_file(result_folder / "timings.csv"), throughput_file(result_folder / "throughput.csv");
        std::ofstream detection_file(result_folder / "detections.csv");
        if (!timing_file || !throughput_file || !detection_file) throw std::runtime_error("Cannot open result CSV files");
        std::cout << "Images: " << images.size() << " | Iterations per image: " << iterations
                  << " | Warmup per image: " << warmup << " | Slots: " << slot_count
                  << " | Split streams: " << pipeline.split << " | Contexts: 1\n";
        std::cout << "Per-slot buffers: pinned=" << image_capacity + OUTPUT_COUNT * sizeof(float)
                  << " device=" << image_capacity + (INPUT_COUNT + OUTPUT_COUNT) * sizeof(float)
                  << " bytes (engine/context/driver memory excluded)\n";
        run(pipeline, slot_count, images, warmup_samples, last, nullptr); // 完整 drain，正式测量从空队列开始。
        const auto start = Clock::now();
        run(pipeline, slot_count, images, samples, last, &records);
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
