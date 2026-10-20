<p align="center">
  <img src="docs/assets/logo.svg" alt="YOLO TensorRT Deploy logo" width="80" height="80">
</p>

<h1 align="center">YOLO TensorRT Deploy</h1>

<p align="center">
  从简洁部署到极致速度
</p>

<p align="center">
  <a href="README.md">English</a> | <strong>简体中文</strong>
</p>

YOLO26m 的 C++ / TensorRT / CUDA 部署与优化系列。从单文件推理开始，逐步优化前处理、内存传输和执行流程。

[推理流程](docs/architecture.md) · [系列目录](ROADMAP.md#series) · [优化笔记](docs/notes/README.md)

## 版本与性能

| 版本 | 内容 | 笔记 | 26m 加速比（P50） | 26n 历史加速比（P50） |
| --- | --- | --- | --- | --- |
| [01 基础版](src/01_baseline/) | CPU 前后处理、GPU 推理，支持预热与重复测量 | [01](docs/notes/01_baseline.md) | 1.00× | 1.00× |
| [02 CUDA 前后处理](src/02_cuda_prepost/) | 融合前处理；可选 GPU 过滤与坐标还原 | [02](docs/notes/02_cuda_prepost.md) | 1.20× | 1.44× |
| [03 Pinned Memory](src/03_pinned_memory/) | 复用页锁定传输缓冲区，保留 CPU 后处理 | [03](docs/notes/03_pinned_memory.md) | 1.18× | 1.38× |
| [04 异步流水线](src/04_async_pipeline/) | 双槽独立资源，允许请求重叠；保留单槽对照 | [04](docs/notes/04_async_pipeline.md) | 0.72× | 1.06× |
| [05 CUDA Graph](src/05_cuda_graph/) | 捕获 H2D → 前处理 → TensorRT → D2H，后处理仍在 CPU | [05](docs/notes/05_cuda_graph.md) | 1.25× | 1.57× |
| [06 完整 Graph](src/06_full_graph/) | 将 GPU 前处理、推理、后处理一起捕获 | [06](docs/notes/06_full_graph.md) | 1.25× | 1.61× |
| [07 Graph 流水线](src/07_graph_pipeline/) | 完整 Graph 与双槽组合；保留槽数与 Graph 开关 | [07](docs/notes/07_graph_pipeline.md) | 0.70× | 0.93× |
| [08 前处理映射表](src/08_preprocess_maps/) | 按原图尺寸预计算一维插值表；保留原始 kernel 对照 | [08](docs/notes/08_preprocess_maps.md) | 1.25× | 1.58× |
| [09 注册原图内存](src/09_registered_input/) | 直接注册解码后的原图，省掉每次请求的 CPU staging 拷贝 | [09](docs/notes/09_registered_input.md) | 1.29× | 1.70× |
| [10 分阶段 Stream](src/10_split_streams/) | 一个 TensorRT context，拆分 H2D / compute / D2H，支持 1–4 槽 | [10](docs/notes/10_split_streams.md) | 0.64× | 0.83× |
| [11 Grouped TopK](tools/grouped_topk.py) | 改写两次 TopK，保留权重与 IO；复用 05 或 09 | [11](docs/notes/11_grouped_topk.md) | 1.31× | 1.76× |

加速比 = 同模型的 01 P50 ÷ 本版 P50；各版使用默认模式，11 使用 09 + Grouped TopK。默认使用 YOLO26m。[26m 实测](docs/benchmarks/rtx3060-yolo26m-2026-10-15/README.md) · [26n 历史结果](docs/benchmarks/rtx3060-unified-baseline-2026-09-15/README.md)，两个批次分别报告。

## 快速开始

依赖：NVIDIA GPU 及兼容驱动、C++17 编译器、CMake 3.24+、Make、CUDA Toolkit 11.8+、TensorRT 8.6.x、OpenCV 3.4+（core、imgproc、imgcodecs）。模型导出另需 Python 3.10 和 `uv`；C++ 推理不依赖 Python。

```bash
git clone https://github.com/ccccwei/yolo-tensorrt-deploy.git
cd yolo-tensorrt-deploy

# 1. 下载权重并导出 ONNX
uv venv .venv-export --python 3.10
uv pip install --python .venv-export/bin/python -r tools/export-requirements.txt
.venv-export/bin/python tools/export_onnx.py yolo26m.pt

# 2. 构建 FP16 Engine
sh tools/build_engine.sh weights/yolo26m.onnx

# 3. 在目标 GPU 上编译十个推理程序
cmake --preset release -DCMAKE_CUDA_ARCHITECTURES=native
cmake --build --preset release

# 4. 运行基础版
./build/release/src/yolo_baseline weights/yolo26m.fp16.engine images
```

PT、ONNX 和 Engine 不随仓库提交；导出脚本自动下载官方权重到 `weights/`。Engine 请在目标 GPU 和 TensorRT 环境中生成。

`native` 使用编译机器的 GPU 架构。为另一台机器构建时，显式指定目标计算能力，例如 RTX 3060 使用 `-DCMAKE_CUDA_ARCHITECTURES=86`。

默认 `trtexec` 路径为 `/usr/local/TensorRT/bin/trtexec`。安装位置不同时，替换对应的构建与配置命令：

```bash
TRTEXEC=/path/to/TensorRT/bin/trtexec sh tools/build_engine.sh weights/yolo26m.onnx
cmake --preset release -DCMAKE_CUDA_ARCHITECTURES=native -DTensorRT_ROOT=/path/to/TensorRT
```

## 使用说明

十个推理程序都支持 JPG、JPEG、PNG、BMP，按路径顺序遍历，不递归子目录。

01 的参数为 `ENGINE IMAGE_FOLDER [ITERATIONS=1] [WARMUP=0]`，默认每图推理一次，可显式启用基准测量。02 为 `ENGINE IMAGE_FOLDER [ITERATIONS=200] [WARMUP=50] [GPU_POST=0]`；03 沿用 200/50 默认次数，不带后处理开关。测量次数必须大于 0，预热次数可以为 0。对照实验应显式使用相同次数，01 默认的单次运行不作为稳态基准。

```bash
# 01 默认演示，以及带预热的基准测试。
./build/release/src/yolo_baseline weights/yolo26m.fp16.engine images
./build/release/src/yolo_baseline weights/yolo26m.fp16.engine images 1000 200
# 02 最后一个参数：0 = CPU 后处理（默认），1 = GPU 后处理。
./build/release/src/yolo_cuda_prepost weights/yolo26m.fp16.engine images 1000 200 0
./build/release/src/yolo_cuda_prepost weights/yolo26m.fp16.engine images 1000 200 1
./build/release/src/yolo_pinned_memory weights/yolo26m.fp16.engine images 1000 200
```

输出保存在当前工作目录的 `results/<版本目录>/`，例如 `results/01_baseline/`、`results/10_split_streams/`。02 的两种模式分别保存到 `results/02_cuda_prepost/cpu_post/` 和 `results/02_cuda_prepost/gpu_post/`。所有当前版本都保存标框图片、`timings.csv` 和 `detections.csv`；04/05/06/07/08/09/10 再增加 `throughput.csv`；09 另存 `setup.csv`。重复运行会覆盖对应版本和模式的同名文件。
01、02、03 为每张图片打印各阶段的均值、p50、p95、p99（最近秩法，单位 ms）。`timings.csv` 保存每次正式测量的四项耗时，`detections.csv` 和标框图片保存每张图片最后一次推理的检测结果。

03 用 `cudaMallocHost` / `cudaFreeHost` 管理并复用原图 `host_image` 和输出 `host_output` 缓冲区；原图缓冲区仅在计时外按需扩容。每次前处理都将 OpenCV 普通内存中的原图复制到页锁定缓冲区，这次 CPU 拷贝也计入耗时。D2H 直接写入页锁定输出，后处理直接读取该缓冲区。该版本只改变主机内存与拷贝路径，不保证所有负载都能加速。

### 04–07 · 运行与消融开关

同样先传 `ENGINE IMAGE_FOLDER [ITERATIONS=200] [WARMUP=50]`，后面再加对应开关：

| 可执行文件 | 额外参数 |
| --- | --- |
| `yolo_async_pipeline` | `[SLOTS=2]`，只允许 1 / 2 |
| `yolo_cuda_graph` | `[GRAPH=1]`，只允许 0 / 1 |
| `yolo_full_graph` | `[GRAPH=1]`，只允许 0 / 1 |
| `yolo_graph_pipeline` | `[SLOTS=2] [GRAPH=1]` |

```bash
./build/release/src/yolo_async_pipeline weights/yolo26m.fp16.engine images 1000 200 2
./build/release/src/yolo_cuda_graph weights/yolo26m.fp16.engine images 1000 200 1
./build/release/src/yolo_full_graph weights/yolo26m.fp16.engine images 1000 200 1
./build/release/src/yolo_graph_pipeline weights/yolo26m.fp16.engine images 1000 200 2 1
```

使用单槽或关闭对应优化进行同源码对照；07 支持槽数与 Graph 的全部四种组合。下一轮前先另存结果，完整消融命令和解释见各版笔记。

每次请求仍把解码后的原图复制到 pinned staging，CPU 拷贝计入耗时。05、06 逐图串行处理；04/07 预加载图片文件夹，再通过槽位轮转请求，每槽独占 context、stream 和可变缓冲区。

### 08–10 · 新增消融实验

Engine、文件夹、测量次数和预热次数的参数不变，最后增加这些开关：

| 可执行文件 | 额外参数 |
| --- | --- |
| `yolo_preprocess_maps` | `[MAPS=1]`，只允许 0 / 1，Graph 始终开启 |
| `yolo_registered_input` | `[REGISTERED=1]`，只允许 0 / 1，Graph 始终开启 |
| `yolo_split_streams` | `[SLOTS=2] [SPLIT=1]`，槽数 1–4、SPLIT 为 0 / 1，不使用 Graph |

```bash
./build/release/src/yolo_preprocess_maps weights/yolo26m.fp16.engine images 1000 200 1
./build/release/src/yolo_registered_input weights/yolo26m.fp16.engine images 1000 200 1
# 固定两槽，对比调度方式；运行下一组前先另存结果。
./build/release/src/yolo_split_streams weights/yolo26m.fp16.engine images 1000 200 2 0
./build/release/src/yolo_split_streams weights/yolo26m.fp16.engine images 1000 200 2 1
```

08 在 setup 生成几何映射表，每次重放仍然暂存、上传并前处理当前像素。09 的对照组使用 pinned staging，注册组在 H2D 时直接读取原始 `cv::Mat`；两组每次都计算 CPU 几何参数，只有对照组额外执行像素复制。注册地址必须保持稳定，GPU 工作完成前不能释放或改写输入；每张新分配图片只推理一次时，注册成本可能超过省掉的拷贝。

10 预加载原图并复用每槽缓冲区，但只使用**一个** TensorRT context。事件连接上传、计算、下载三个阶段；SPLIT=0 保留相同槽数、事件和工作，只放在一条 stream；SPLIT=1 将传输与计算拆开。加槽只增加缓冲深度，不改变模型 batch，也不让两次 TensorRT 推理同时执行。

## 计时范围

01、02、03 使用同一计时边界，画框均在测量区间外：

| 阶段 | 01 基础版 | 02，CPU 后处理 | 03 Pinned Memory |
| --- | --- | --- | --- |
| `Preprocess` | CPU 缩放、补边、BGR → RGB、HWC → CHW、归一化 | 原始 uint8 图片 H2D + 融合 CUDA 前处理 + 同步 | 原图复制到页锁定缓冲区 + H2D + 同一 CUDA 前处理 + 同步 |
| `Infer` | FP32 输入 H2D + TensorRT + 结果 D2H + 同步 | TensorRT + 结果 D2H + 同步 | 与 02 同范围，D2H 写入页锁定输出 |
| `Postprocess` | CPU 置信度过滤与坐标还原，不含画框 | 与 01 相同 | 算法同 02，直接读取页锁定输出 |
| `Pipeline` | `Preprocess + Infer + Postprocess` | 与 01 相同 | 与 01 相同，包含 CPU 暂存拷贝 |

02 开启 `GPU_POST=1` 后，CUDA 后处理与压紧结果 D2H 计入 **Infer**，**Postprocess** 只计 CPU 将结果组装为检测框的 unpack。两种模式都保留前处理末尾、推理末尾的同步，使用普通主机内存。应比较 **Pipeline 总时间**，不能只看 Postprocess 列变小。

这里使用包含同步等待的 CPU 墙钟时间，不是纯 GPU kernel 时间。初始化、模型和图片加载、预热、画图、保存及日志均不计入测量。01、02 使用普通主机内存，03 使用页锁定内存；这三版没有跨图并发、双缓冲或 Graph。

比较 01、02、03 时应使用相同条件下的 `Pipeline`，不要单独比较 `Infer` 来计算整体加速比：02/03 把输入传输移到了 `Preprocess`，03 还包含 CPU 暂存拷贝。研究 pinned memory 时，以 02 的 `GPU_POST=0` 对照 03。相同图片反复运行测得的是该负载的稳态结果，不包含磁盘 I/O，也不是完整数据集吞吐量。

04/05/06/07/08/09/10 的 CSV 使用 `staging_ms / gpu_ms / postprocess_ms / total_ms`：分别记录 CPU 输入准备、CUDA 事件包围的整段 GPU 工作、CPU 后处理或结果组装，以及从输入准备开始到 CPU 消费结果完成的时间。除 09 注册模式只计算几何参数外，输入准备都包含逐次 CPU pinned staging 拷贝。GPU 区间包含传输，启用 GPU 后处理时也包含其 kernel；10 还包含阶段间事件等待，不是纯 TensorRT 计算时间，不能要求阶段相加等于 CPU 总时间。

`throughput.csv` 使用 **完成样本数 ÷ 正式测量墙钟时间**，包含循环记录和流水线填充/drain，不是 `1000/P50`。吞吐率变高时，请求延迟可能也会变高；04/07/10 的请求延迟不包含入槽前的上游排队。读写文件、预热和 setup 不计入稳态测量；Graph 版本会另外打印 setup 成本。

09 的 `setup.csv` 逐图记录 `setup_ms` 与 `teardown_ms`：前者包含注册或 staging 资源准备、prime 与 Graph 创建，后者包含 Graph 销毁、取消注册与源图引用释放。这两项都不计入稳态延迟，不能在每图只推一次的应用中忽略；08 的建表成本也包含在 setup 中，不是免费的预计算。

## 测试

编译后运行五个 GPU 组件对照测试（不需要 Engine 文件）：

```bash
ctest --test-dir build/release --output-on-failure
```

测试需要 NVIDIA GPU，覆盖常见及极端长宽比、奇数补边、非连续 ROI、输出越界和非法参数。CUDA 浮点双线性插值与 OpenCV uint8 缩放并非逐位一致：归一化后的最大绝对误差容许 `1/255 + 1e-6`，不缩放时使用更严格的 `1e-7`。这不能替代真实数据集上的检测精度验证。

`pinned_memory_test` 另检查 CPU 暂存拷贝、非连续 ROI、更新输入后的缓冲区复用，以及输入输出缓冲区的页锁定属性；`gpu_postprocess_test` 检查 CPU/GPU 筛选、坐标还原、结果顺序和边界条件。`preprocess_maps_test` 检查参考与查表输出逐位一致、重建映射表、pitched 内存、边界哨兵，以及改像素后的 Graph 重放；`registered_input_test` 检查注册状态、ROI 步长、更新输入后的重放和正常/异常清理。可用 `-DBUILD_TESTING=OFF` 跳过测试程序构建。

另有仅依赖 Python 标准库的可选集成检查，使用已编译程序和本机 Engine：

```bash
python3 tests/basic_integration.py --bin-dir build/release/src \
  --engine weights/yolo26m.fp16.engine --images images
python3 tests/series_integration.py --bin-dir build/release/src \
  --engine weights/yolo26m.fp16.engine --images images
python3 tests/advanced_integration.py --bin-dir build/release/src \
  --engine weights/yolo26m.fp16.engine --images images
```

基础脚本验证 01 与 02 的两种模式；系列脚本比较 02 的 CPU/GPU 模式和 03/04/05/06/07；进阶脚本以 03 为参考，比较 08/09 的两种模式、10 的全部八种槽数/SPLIT 组合，并验证 09 的 setup CSV。检查覆盖输出、CSV 格式和非法输入；系列及进阶脚本还检查混合图片尺寸与槽位 drain，10 包含「1 个请求、4 个槽」的收尾。测试图片及输出放在临时目录；它们是回归测试，不能替代数据集精度评估。

## 当前限制

当前 C++ 示例仅支持静态 Batch 1、640 × 640 的 YOLO26 检测模型，输入输出约定如下：

| 张量 | 名称 | 形状 | 类型 / 内容 |
| --- | --- | --- | --- |
| 输入 | `images` | `[1, 3, 640, 640]` | FP32，RGB，像素归一化至 0–1 |
| 输出 | `output0` | `[1, 300, 6]` | FP32，每行为 `[x1, y1, x2, y2, score, class_id]` |

Engine 内部允许 FP16，输入输出仍须为 FP32。默认置信度阈值 `0.25`：CPU 路径在 `main.cpp` 修改 `SCORE_THRESHOLD`，02/06/07 的 GPU 路径在 `postprocess.cu` 修改阈值，02 做对照时须同步修改两条路径并重新编译。NMS-free 不等于没有后处理，仍需筛选置信度并还原坐标。

当前构建不支持 TensorRT 10+。导出脚本可导出 YOLO11n，但 C++ 后处理不支持其输出格式；动态尺寸、多 Batch、分割和姿态模型也暂不支持。

04/07/10 将整个文件夹预加载到内存，每槽另分配最大原图容量的缓冲区；04/07 每槽独立 context，10 则共用一个 context 串行推理。07 还为每槽、每张图片缓存 Graph，setup 和内存随图片数增加。Graph 依赖固定地址和参数，不支持捕获时会明确失败。08 每图准备时重建映射表，09 要求主机支持注册且源内存寿命稳定，这些 setup 成本都单独报告。上述示例不是无限大图库、动态 shape、摄像头或生产服务框架。

模型权重及 `images/` 中的两张测试图片来自 [Ultralytics](https://github.com/ultralytics/ultralytics)。
