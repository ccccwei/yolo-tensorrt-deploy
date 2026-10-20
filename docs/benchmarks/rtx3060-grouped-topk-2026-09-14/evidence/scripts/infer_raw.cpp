// Diagnostic only: exercise the unchanged 06 runtime and save full FP32 outputs.
#define main original_demo_main
#include "../../src/06_cuda_graph/main.cpp"
#undef main

int main(int argc, char** argv) {
    try {
        if (argc != 4) throw std::runtime_error("ENGINE IMAGE_FOLDER NEW_OUTPUT_FOLDER required");
        const std::filesystem::path out(argv[3]);
        if (!std::filesystem::create_directory(out)) throw std::runtime_error("Output directory must be new");
        std::ifstream file(argv[1], std::ios::binary | std::ios::ate);
        if (!file || file.tellg() <= 0) throw std::runtime_error("Cannot open engine");
        const std::streamsize size = file.tellg();
        std::vector<char> bytes(static_cast<std::size_t>(size));
        file.seekg(0);
        if (!file.read(bytes.data(), size)) throw std::runtime_error("Cannot read engine");
        Logger logger;
        std::unique_ptr<nvinfer1::IRuntime> runtime(nvinfer1::createInferRuntime(logger));
        if (!runtime) throw std::runtime_error("Cannot create runtime");
        std::unique_ptr<nvinfer1::ICudaEngine> engine(runtime->deserializeCudaEngine(bytes.data(), bytes.size()));
        if (!engine || !valid_engine(*engine)) throw std::runtime_error("Unsupported engine");
        std::unique_ptr<nvinfer1::IExecutionContext> context(engine->createExecutionContext());
        if (!context) throw std::runtime_error("Cannot create context");
        CudaResources cuda;
        if (cudaMalloc(&cuda.input, INPUT_COUNT * sizeof(float)) != cudaSuccess
            || cudaMalloc(&cuda.output, OUTPUT_COUNT * sizeof(float)) != cudaSuccess
            || cudaMallocHost(reinterpret_cast<void**>(&cuda.host_output), OUTPUT_COUNT * sizeof(float)) != cudaSuccess
            || cudaStreamCreateWithFlags(&cuda.stream, cudaStreamNonBlocking) != cudaSuccess
            || !context->setTensorAddress(INPUT_NAME, cuda.input)
            || !context->setTensorAddress(OUTPUT_NAME, cuda.output)) throw std::runtime_error("CUDA init failed");
        std::vector<std::filesystem::path> paths;
        for (const auto& entry : std::filesystem::directory_iterator(argv[2]))
            if (entry.is_regular_file() && is_image(entry.path())) paths.push_back(entry.path());
        std::sort(paths.begin(), paths.end());
        if (paths.empty()) throw std::runtime_error("No images");
        std::ofstream detections(out / "detections.csv");
        detections << "image,class_id,score,left,top,right,bottom\n" << std::setprecision(9);
        for (const auto& path : paths) {
            const cv::Mat image = cv::imread(path.string());
            if (image.empty() || !cuda.clear_graph() || !prepare_image(cuda, image.total() * image.elemSize()))
                throw std::runtime_error("Image preparation failed");
            float scale = 0.0F;
            int px = 0, py = 0;
            if (!stage_image(image, cuda, scale, px, py) || !submit(*context, cuda, image.cols, image.rows)
                || cudaStreamSynchronize(cuda.stream) != cudaSuccess
                || !capture_graph(*context, cuda, image.cols, image.rows)) throw std::runtime_error("Prime/capture failed");
            std::vector<float> first;
            for (int pass = 0; pass < 3; ++pass) {
                if (!stage_image(image, cuda, scale, px, py) || cudaGraphLaunch(cuda.executable, cuda.stream) != cudaSuccess
                    || cudaStreamSynchronize(cuda.stream) != cudaSuccess) throw std::runtime_error("Replay failed");
                if (pass == 0) first.assign(cuda.host_output, cuda.host_output + OUTPUT_COUNT);
                else if (std::memcmp(first.data(), cuda.host_output, OUTPUT_COUNT * sizeof(float)) != 0)
                    throw std::runtime_error("Non-deterministic replay output");
            }
            std::ofstream raw(out / (path.filename().string() + ".bin"), std::ios::binary);
            raw.write(reinterpret_cast<const char*>(cuda.host_output), static_cast<std::streamsize>(OUTPUT_COUNT * sizeof(float)));
            if (!raw) throw std::runtime_error("Cannot save raw output");
            for (const auto& d : postprocess(cuda.host_output, image, scale, px, py))
                detections << csv_quote(path.filename().string()) << ',' << d.class_id << ',' << d.score << ','
                           << d.box.x << ',' << d.box.y << ',' << d.box.x + d.box.width << ',' << d.box.y + d.box.height << '\n';
        }
        if (!detections) throw std::runtime_error("Cannot write detections");
        std::cout << paths.size() << " images, each three deterministic Graph replays, full output saved\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
