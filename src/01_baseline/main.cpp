// 01 基础版：CPU 前处理 → 输入拷入 GPU → TensorRT 推理 → 输出拷回 CPU → CPU 后处理。
// 每张图片串行执行，可选择预热和重复测量；画框、文件读写不计入流水线耗时。
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

// ==================== 1. 模型约定与计时类型 ====================
// 固定 batch=1：输入为 FP32 RGB [1,3,640,640]，输出为 FP32 [1,300,6]。
// 每条输出按 [x1, y1, x2, y2, score, class_id] 解释，坐标相对于补边后的输入图。
// 张量名称、形状和数据类型必须与加载的 engine 一致，后续由 valid_engine 检查。
constexpr char INPUT_NAME[] = "images";
constexpr char OUTPUT_NAME[] = "output0";
constexpr int INPUT_SIZE = 640;
constexpr int MAX_DETECTIONS = 300;
constexpr int DETECTION_SIZE = 6;
constexpr float SCORE_THRESHOLD = 0.25F;
// COUNT 表示 float 元素个数；申请显存和传输数据时还要乘 sizeof(float)。
constexpr std::size_t INPUT_COUNT = 3U * INPUT_SIZE * INPUT_SIZE;
constexpr std::size_t OUTPUT_COUNT = 1U * MAX_DETECTIONS * DETECTION_SIZE;
// 使用单调时钟测量 CPU 墙钟耗时，避免系统时间调整影响统计。
using Clock = std::chrono::steady_clock;
using Milliseconds = std::chrono::duration<double, std::milli>;
using Timing = std::array<double, 4>; // 前处理、推理、后处理、总耗时，单位均为 ms。

// ==================== 2. 日志、资源管理与检测结果 ====================
// TensorRT 通过此接口输出日志；只打印 WARNING 及更严重的消息。
class Logger final : public nvinfer1::ILogger {
public:
    void log(Severity severity, const char* message) noexcept override {
        if (severity <= Severity::kWARNING) std::cerr << "[TensorRT] " << message << '\n';
    }
};

// 用对象析构管理 GPU 输入/输出缓冲区和 CUDA stream（任务执行队列）。
// return 或异常退出时，先等队列中的任务结束，再销毁 stream 并释放显存。
struct CudaResources {
    void* input = nullptr;
    void* output = nullptr;
    cudaStream_t stream = nullptr;
    ~CudaResources() {
        if (stream) {
            cudaStreamSynchronize(stream);
            cudaStreamDestroy(stream);
        }
        if (output) cudaFree(output);
        if (input) cudaFree(input);
    }
};

// 后处理得到的单个检测结果：类别编号、置信度、原图坐标系中的矩形框。
struct Detection {
    int class_id;
    float score;
    cv::Rect box;
};

// ==================== 3. CPU 前处理 ====================
// 将 OpenCV 解码得到的 BGR 图像转换为模型输入；通过引用返回缩放比例和左/上补边量，
// 供后处理把模型输出坐标映射回原图。
cv::Mat preprocess(const cv::Mat& image, float& scale, int& pad_x, int& pad_y) {
    // 3.1 Letterbox：保持宽高比缩放，使长边适配 640，短边居中补齐。
    scale = std::min(static_cast<float>(INPUT_SIZE) / static_cast<float>(image.cols),
                     static_cast<float>(INPUT_SIZE) / static_cast<float>(image.rows));
    const int width = std::max(1, static_cast<int>(std::round(static_cast<float>(image.cols) * scale)));
    const int height = std::max(1, static_cast<int>(std::round(static_cast<float>(image.rows) * scale)));
    pad_x = (INPUT_SIZE - width) / 2;
    pad_y = (INPUT_SIZE - height) / 2;

    // 3.2 用灰色 114 补边到 640×640；总补边量为奇数时，多出的像素放到右侧或下侧。
    cv::Mat resized, padded;
    cv::resize(image, resized, cv::Size(width, height));
    cv::copyMakeBorder(resized, padded,
                      pad_y, INPUT_SIZE - height - pad_y,
                      pad_x, INPUT_SIZE - width - pad_x,
                      cv::BORDER_CONSTANT, cv::Scalar(114, 114, 114));

    // 3.3 分配连续的 NCHW float 输入：N 为批次，C 为通道，H/W 为高/宽。
    // 三个通道各占一个 H×W 平面，plane 用于计算相邻通道的起始偏移。
    const int shape[] = {1, 3, INPUT_SIZE, INPUT_SIZE};
    cv::Mat input(4, shape, CV_32F);
    float* data = input.ptr<float>();
    constexpr std::size_t plane = static_cast<std::size_t>(INPUT_SIZE) * INPUT_SIZE;
    constexpr float normalize = 1.0F / 255.0F;
    // 一次遍历完成 BGR→RGB、像素值除以 255 归一化到 [0,1]、HWC→CHW 排布转换。
    for (int y = 0; y < INPUT_SIZE; ++y) {
        const cv::Vec3b* row = padded.ptr<cv::Vec3b>(y);
        for (int x = 0; x < INPUT_SIZE; ++x) {
            const std::size_t index = static_cast<std::size_t>(y) * INPUT_SIZE
                + static_cast<std::size_t>(x);
            data[index] = static_cast<float>(row[x][2]) * normalize;
            data[plane + index] = static_cast<float>(row[x][1]) * normalize;
            data[2U * plane + index] = static_cast<float>(row[x][0]) * normalize;
        }
    }
    return input;
}

// ==================== 4. CPU 后处理与绘图 ====================
// 按模型约定逐行读取检测结果，筛选后还原到原图；这里不执行额外的 NMS（非极大值抑制）。
std::vector<Detection> postprocess(const std::vector<float>& output, const cv::Mat& image,
                                   float scale, int pad_x, int pad_y) {
    std::vector<Detection> detections;
    for (int i = 0; i < MAX_DETECTIONS; ++i) {
        const float* det = output.data() + i * DETECTION_SIZE;
        // 跳过 NaN/无穷值、低置信度结果，以及不能安全转为非负 int 的类别值。
        if (!std::all_of(det, det + DETECTION_SIZE, [](float value) { return std::isfinite(value); })
            || det[4] < SCORE_THRESHOLD || det[5] < 0.0F
            || static_cast<double>(det[5]) > static_cast<double>(std::numeric_limits<int>::max())) continue;

        // 逆向撤销 Letterbox：先减去补边，再除以缩放比例，最后限制在原图像素范围内。
        const auto coordinate = [scale](float value, int padding, int size) {
            const float restored = (value - static_cast<float>(padding)) / scale;
            return static_cast<int>(std::clamp(
                static_cast<double>(restored), 0.0, static_cast<double>(size - 1)));
        };
        const int left = coordinate(det[0], pad_x, image.cols);
        const int top = coordinate(det[1], pad_y, image.rows);
        const int right = coordinate(det[2], pad_x, image.cols);
        const int bottom = coordinate(det[3], pad_y, image.rows);
        // 裁剪后宽高必须为正；OpenCV 矩形保存的是左上角坐标和宽高。
        if (right <= left || bottom <= top) continue;
        detections.push_back({static_cast<int>(det[5]), det[4], {left, top, right - left, bottom - top}});
    }
    return detections;
}

// 直接在原图上画绿色框和类别/置信度标签；由 main 在计时结束后调用。
void draw_detections(cv::Mat& image, const std::vector<Detection>& detections) {
    for (const auto& detection : detections) {
        const auto& box = detection.box;
        cv::rectangle(image, box.tl(), box.br(), {0, 255, 0}, 2);
        cv::putText(image, cv::format("class=%d %.2f", detection.class_id, detection.score),
                    {box.x, std::max(20, box.y - 5)}, cv::FONT_HERSHEY_SIMPLEX,
                    0.6, {0, 255, 0}, 2);
    }
}

// ==================== 5. GPU 推理与引擎兼容性检查 ====================
// 同一 stream 按提交顺序执行 H2D（主机→设备）、推理、D2H（设备→主机）。
// 输入输出使用普通主机内存；虽然调用 Async 接口，函数末尾仍同步等待整条队列完成，
// 成功返回后 CPU 可安全读取 output，本版也没有让多张图片的处理相互重叠。
bool infer(nvinfer1::IExecutionContext& context, const CudaResources& cuda,
           const cv::Mat& input, std::vector<float>& output) {
    // && 短路求值：某一步提交失败后，不再继续提交后续步骤。
    const bool submitted = cudaMemcpyAsync(cuda.input, input.ptr<float>(), INPUT_COUNT * sizeof(float),
                           cudaMemcpyHostToDevice, cuda.stream) == cudaSuccess
        && context.enqueueV3(cuda.stream)
        && cudaMemcpyAsync(output.data(), cuda.output, OUTPUT_COUNT * sizeof(float),
                           cudaMemcpyDeviceToHost, cuda.stream) == cudaSuccess;
    // 即使提交失败，也先等待已提交的操作，再让调用方释放 host 缓冲区。
    const auto status = cudaStreamSynchronize(cuda.stream);
    return submitted && status == cudaSuccess;
}

// 本程序按固定大小分配缓冲区，因此只接受下述静态输入输出约定。
// 这里限制的是 I/O 为 FP32，并不要求 engine 内部所有算子都使用 FP32 计算。
bool valid_engine(const nvinfer1::ICudaEngine& engine) {
    const auto in = engine.getTensorShape(INPUT_NAME);
    const auto out = engine.getTensorShape(OUTPUT_NAME);
    // 检查显式 batch、张量数量、固定形状，以及两个张量的输入/输出角色。
    if (engine.getNbIOTensors() != 2 || engine.hasImplicitBatchDimension()
        || in.nbDims != 4 || in.d[0] != 1 || in.d[1] != 3
        || in.d[2] != INPUT_SIZE || in.d[3] != INPUT_SIZE
        || out.nbDims != 3 || out.d[0] != 1
        || out.d[1] != MAX_DETECTIONS || out.d[2] != DETECTION_SIZE
        || engine.getTensorIOMode(INPUT_NAME) != nvinfer1::TensorIOMode::kINPUT
        || engine.getTensorIOMode(OUTPUT_NAME) != nvinfer1::TensorIOMode::kOUTPUT) return false;
    // 两个张量都必须是设备端的 FP32 线性、非向量化布局，才能按连续 float 数组访问。
    for (const char* name : {INPUT_NAME, OUTPUT_NAME}) {
        if (engine.getTensorDataType(name) != nvinfer1::DataType::kFLOAT
            || engine.getTensorFormat(name) != nvinfer1::TensorFormat::kLINEAR
            || engine.getTensorVectorizedDim(name) != -1
            || engine.getTensorLocation(name) != nvinfer1::TensorLocation::kDEVICE) return false;
    }
    return true;
}

// ==================== 6. 参数、文件名与统计辅助函数 ====================
// 严格解析十进制整数，拒绝空串、符号、尾随字符和溢出；仅预热次数允许为 0。
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

// 按扩展名筛选候选图片（忽略大小写）；文件能否真正解码由后续 imread 判断。
bool is_image(const std::filesystem::path& path) {
    std::string extension = path.extension().string();
    std::transform(extension.begin(), extension.end(), extension.begin(), [](unsigned char value) {
        return static_cast<char>(std::tolower(value));
    });
    return extension == ".jpg" || extension == ".jpeg" || extension == ".png" || extension == ".bmp";
}

// CSV 字段用双引号包裹，内部双引号重复一次，避免文件名中的逗号等破坏列结构。
std::string csv_quote(const std::string& text) {
    std::string quoted = "\"";
    for (const char value : text) {
        if (value == '"') quoted += '"';
        quoted += value;
    }
    return quoted + '"';
}

// 对当前图片的正式测量分别统计四项耗时的均值和 P50/P95/P99；调用时 timings 非空。
// Infer 包含双向传输、TensorRT 执行和同步等待，并非纯 GPU 计算耗时。
void print_statistics(const std::vector<Timing>& timings) {
    constexpr const char* names[] = {"Preprocess", "Infer (H2D+TRT+D2H+sync)", "Postprocess", "Pipeline"};
    std::cout << std::fixed << std::setprecision(3);
    for (std::size_t stage = 0; stage < 4U; ++stage) {
        std::vector<double> values;
        values.reserve(timings.size());
        double sum = 0.0;
        for (const auto& timing : timings) {
            values.push_back(timing[stage]);
            sum += timing[stage];
        }
        // 分位数采用 nearest-rank：排序后取 ceil(比例×样本数) 对应的值，不做插值。
        std::sort(values.begin(), values.end());
        const auto percentile = [&values](double fraction) {
            const auto rank = static_cast<std::size_t>(std::ceil(fraction * static_cast<double>(values.size())));
            return values[rank - 1U]; // rank 从 1 开始，数组下标从 0 开始。
        };
        std::cout << "  " << names[stage] << ": mean=" << sum / static_cast<double>(values.size())
                  << " p50=" << percentile(0.50) << " p95=" << percentile(0.95)
                  << " p99=" << percentile(0.99) << " ms\n";
    }
}

// ==================== 7. 主流程：初始化 → 逐图测量 → 保存结果 ====================
int main(int argc, char** argv) {
    try {
        // 7.1 解析 engine 路径、图片目录、每图测量次数和预热次数。
        // 默认每张图片测量一次且不预热；需要基准统计时显式传入 ITERATIONS/WARMUP。
        if (argc < 3 || argc > 5) {
            std::cerr << "Usage: " << argv[0] << " ENGINE IMAGE_FOLDER [ITERATIONS=1] [WARMUP=0]\n";
            return 1;
        }
        const int iterations = argc >= 4 ? parse_count(argv[3], false) : 1;
        const int warmup = argc >= 5 ? parse_count(argv[4], true) : 0;
        // 7.2 只遍历指定目录的直接子文件，按路径排序以固定处理顺序。
        const std::filesystem::path folder(argv[2]);
        if (!std::filesystem::is_directory(folder)) throw std::runtime_error("Cannot open image folder");
        std::vector<std::filesystem::path> paths;
        for (const auto& entry : std::filesystem::directory_iterator(folder)) {
            if (entry.is_regular_file() && is_image(entry.path())) paths.push_back(entry.path());
        }
        std::sort(paths.begin(), paths.end());
        if (paths.empty()) throw std::runtime_error("No images found");

        // 7.3 二进制读取序列化 engine：先在文件末尾获取大小，再回到开头读取全部内容。
        std::ifstream file(argv[1], std::ios::binary | std::ios::ate);
        if (!file || file.tellg() <= 0) throw std::runtime_error("Cannot open engine");
        const std::streamsize size = file.tellg();
        std::vector<char> engine_data(static_cast<std::size_t>(size));
        file.seekg(0);
        if (!file.read(engine_data.data(), size)) throw std::runtime_error("Cannot read engine");
        // runtime 负责反序列化，engine 保存模型执行信息，context 持有本次执行的状态。
        // 局部对象按声明逆序销毁，使 context、engine、runtime 在 logger 之前释放。
        Logger logger;
        std::unique_ptr<nvinfer1::IRuntime> runtime(nvinfer1::createInferRuntime(logger));
        if (!runtime) throw std::runtime_error("Cannot create TensorRT runtime");
        std::unique_ptr<nvinfer1::ICudaEngine> engine(
            runtime->deserializeCudaEngine(engine_data.data(), engine_data.size()));
        if (!engine || !valid_engine(*engine)) throw std::runtime_error("Unsupported engine: expected FP32 linear [1,3,640,640] -> [1,300,6]");
        std::unique_ptr<nvinfer1::IExecutionContext> context(engine->createExecutionContext());
        if (!context) throw std::runtime_error("Cannot create execution context");

        // 7.4 一次性申请 GPU 输入/输出显存和 stream，并按张量名称把地址绑定到 context。
        // 后续所有图片与迭代复用这些资源；cuda 先于 context 析构，确保任务结束后再释放 TRT 对象。
        CudaResources cuda;
        if (cudaMalloc(&cuda.input, INPUT_COUNT * sizeof(float)) != cudaSuccess
            || cudaMalloc(&cuda.output, OUTPUT_COUNT * sizeof(float)) != cudaSuccess
            || cudaStreamCreate(&cuda.stream) != cudaSuccess
            || !context->setTensorAddress(INPUT_NAME, cuda.input)
            || !context->setTensorAddress(OUTPUT_NAME, cuda.output)) throw std::runtime_error("Cannot prepare CUDA");

        // 7.5 在当前工作目录下准备结果目录和两个 CSV；重复运行会覆盖同名结果文件。
        const std::filesystem::path result_folder("results/01_baseline");
        std::filesystem::create_directories(result_folder);
        std::ofstream timing_file(result_folder / "timings.csv");
        std::ofstream detection_file(result_folder / "detections.csv");
        if (!timing_file || !detection_file) throw std::runtime_error("Cannot open result CSV files");
        timing_file << "image,iteration,preprocess_ms,infer_ms,postprocess_ms,total_ms\n" << std::setprecision(9);
        detection_file << "image,class_id,score,left,top,right,bottom\n" << std::setprecision(9);
        // CPU 输出数组同样只分配一次，每轮 D2H 拷贝覆盖其内容。
        std::vector<float> output(OUTPUT_COUNT);
        std::size_t processed = 0;
        bool failed_image = false;
        std::cout << "Images: " << paths.size() << " | Iterations per image: " << iterations
                  << " | Warmup per image: " << warmup << '\n';

        // 7.6 逐图执行：每张图片只解码一次，然后在同一张原图上完成预热和重复测量。
        for (const auto& path : paths) {
            cv::Mat image = cv::imread(path.string()); // 解码不计时，每张图片只读取一次。
            if (image.empty()) {
                // 坏图不阻止后续图片处理，但最终退出码仍标记此次运行存在失败。
                std::cerr << "Cannot open image: " << path << '\n';
                failed_image = true;
                continue;
            }
            std::vector<Timing> timings;
            timings.reserve(static_cast<std::size_t>(iterations));
            std::vector<Detection> detections;
            // 预热与正式测量走相同流程；负数索引只用于预热，避免 warmup + iterations 溢出。
            for (int iteration = -warmup; iteration < iterations; ++iteration) {
                float scale = 0.0F;
                int pad_x = 0, pad_y = 0;
                // 四个时间点划出前处理、推理和后处理；总耗时覆盖这三段。
                const auto start = Clock::now();
                const cv::Mat input = preprocess(image, scale, pad_x, pad_y);
                const auto pre_end = Clock::now();
                if (!infer(*context, cuda, input, output)) throw std::runtime_error("Inference failed: " + path.string());
                const auto infer_end = Clock::now(); // infer 已同步，不只是 CPU 提交时间。
                detections = postprocess(output, image, scale, pad_x, pad_y);
                const auto post_end = Clock::now();
                // 预热也完整执行上述流程，但只有正式迭代才写入统计样本。
                if (iteration >= 0) {
                    timings.push_back({Milliseconds(pre_end - start).count(),
                                       Milliseconds(infer_end - pre_end).count(),
                                       Milliseconds(post_end - infer_end).count(),
                                       Milliseconds(post_end - start).count()});
                }
            }
            // 7.7 输出当前图片的统计及逐次耗时；检测框只保存最后一次迭代的结果。
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
            // 每张图片完成后刷新 CSV 并检查写入状态，尽早发现磁盘等写入错误。
            timing_file.flush();
            detection_file.flush();
            if (!timing_file || !detection_file) throw std::runtime_error("Cannot write result CSV files");
        }
        // 7.8 检查最终写入状态；至少成功处理一张图片且没有坏图，才以 0 表示成功。
        timing_file.close();
        detection_file.close();
        if (!timing_file || !detection_file) throw std::runtime_error("Cannot close result CSV files");
        if (processed == 0U || failed_image) return 1;
        std::cout << "Saved: " << result_folder << '\n';
        return 0;
    } catch (const std::exception& error) {
        // 统一报告异常；栈展开会自动析构智能指针、CUDA 资源和文件流等局部对象。
        std::cerr << error.what() << '\n';
        return 1;
    }
}
