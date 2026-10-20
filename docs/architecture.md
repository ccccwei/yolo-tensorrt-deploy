# 推理流程与版本结构

当前系列连续编号为 01–11：01–10 是独立可编译的推理版本，11 改写模型图、复用 05 或 09。每篇保留完整主程序和自己的 CUDA kernel 副本，共用模型、图片、转换脚本与 CMake，不互相包含源码。编号表示学习顺序，不表示速度排名。

历史测量保留原编号与原始数字，见[新旧编号对照](chapter-numbering.md)；原 01/02/03/07 源码见[合并前快照](benchmarks/legacy-sources/pre-merge-2026-09-14/)。旧 07 的 pinned staging 和更少同步点不属于当前 02，不能移用旧 07 数字作为 02 的性能结果。

## 目录和运行入口

所有命令从仓库根目录运行，程序位于 `build/release/src/`。输出通常与源码目录同名，例如 01 写入 `results/01_baseline/`；02 按模式写入 `results/02_cuda_prepost/cpu_post/` 或 `results/02_cuda_prepost/gpu_post/`。

| 目录 | 可执行程序 | 本版研究的变化 | 额外开关 |
| --- | --- | --- | --- |
| [01_baseline](../src/01_baseline/) | `yolo_baseline` | CPU 前后处理、单次演示与预热基准 | 无 |
| [02_cuda_prepost](../src/02_cuda_prepost/) | `yolo_cuda_prepost` | 融合 CUDA 前处理，可选 GPU 后处理 | `GPU_POST=0/1`，默认 0 |
| [03_pinned_memory](../src/03_pinned_memory/) | `yolo_pinned_memory` | pinned 输入 staging 与输出复用 | 无 |
| [04_async_pipeline](../src/04_async_pipeline/) | `yolo_async_pipeline` | 跨请求双缓冲 | `SLOTS=1/2`，默认 2 |
| [05_cuda_graph](../src/05_cuda_graph/) | `yolo_cuda_graph` | 单槽 GPU 链捕获与重放，CPU 后处理 | `GRAPH=0/1`，默认 1 |
| [06_full_graph](../src/06_full_graph/) | `yolo_full_graph` | 将 GPU 后处理加入 Graph | `GRAPH=0/1`，默认 1 |
| [07_graph_pipeline](../src/07_graph_pipeline/) | `yolo_graph_pipeline` | 两槽 + GPU 后处理 + Graph cache | `SLOTS=1/2 GRAPH=0/1`，默认 `2 1` |
| [08_preprocess_maps](../src/08_preprocess_maps/) | `yolo_preprocess_maps` | 按原图尺寸预计算插值映射，Graph 固定开启 | `MAPS=0/1`，默认 1 |
| [09_registered_input](../src/09_registered_input/) | `yolo_registered_input` | 注册原图内存，省去 CPU staging 复制，Graph 固定开启 | `REGISTERED=0/1`，默认 1 |
| [10_split_streams](../src/10_split_streams/) | `yolo_split_streams` | 单 context，H2D / compute / D2H 分阶段流水线 | `SLOTS=1..4 SPLIT=0/1`，默认 `2 1` |

```bash
cmake --preset release
cmake --build --preset release -j

# 01：ENGINE IMAGE_FOLDER [ITERATIONS=1] [WARMUP=0]。
./build/release/src/yolo_baseline weights/yolo26m.fp16.engine images
./build/release/src/yolo_baseline weights/yolo26m.fp16.engine images 1000 200

# 其余版本：ENGINE IMAGE_FOLDER [ITERATIONS=200] [WARMUP=50]，之后接本版开关。
./build/release/src/yolo_cuda_prepost weights/yolo26m.fp16.engine images 1000 200 0
./build/release/src/yolo_cuda_prepost weights/yolo26m.fp16.engine images 1000 200 1
./build/release/src/yolo_graph_pipeline weights/yolo26m.fp16.engine images 1000 200 2 1
./build/release/src/yolo_preprocess_maps weights/yolo26m.fp16.engine images 1000 200 1
./build/release/src/yolo_registered_input weights/yolo26m.fp16.engine images 1000 200 1
./build/release/src/yolo_split_streams weights/yolo26m.fp16.engine images 1000 200 2 1
```

次数按每张图片计算。04 / 07 先预加载所有可读图片、按图片顺序轮转；01 / 02 / 03 / 05 / 06 逐张解码，每张图片连续完成预热和测量。01 默认每图一次、无预热，显式传入相同的 `1000 200` 才用于稳态对照。每次运行会覆盖本版同模式同名结果，消融测试应分别保存 CSV。

08 / 09 延续逐图预热和测量，10 延续全部预加载后的轮转调度。08 / 09 的末尾开关不是 Graph 开关；两版都固定捕获和重放，只分别改变前处理查表、原图内存来源。

## 输入输出约定

当前验证环境为 Linux、CUDA 11.8、TensorRT 8.6.1、C++17，模型为静态 Batch 1 的 YOLO26 end-to-end 检测模型。

| Tensor | 类型与布局 | 语义 |
| --- | --- | --- |
| `images` | FP32 linear `[1,3,640,640]` | RGB、CHW、归一化到 `[0,1]` |
| `output0` | FP32 linear `[1,300,6]` | 每行 `x1,y1,x2,y2,score,class_id` |

Engine 内部可使用 FP16，I/O 仍为 FP32。YOLO26 的这条输出路径不再执行 NMS，但仍需要过滤低分框、去除非法结果、恢复原图坐标和裁边。当前不是 YOLO11、动态 shape、动态 batch、分割或姿态的通用适配器。

所有当前版本保存 `timings.csv`、最后一次每图的 `detections.csv` 和标框图片；04 / 05 / 06 / 07 另保存实际测量墙钟的 `throughput.csv`。01、02、03 打印阶段均值和 P50 / P95 / P99。

08–10 同样保存上述三个 CSV 与标框图片；09 额外保存 `setup.csv`（`image,setup_ms,teardown_ms`），单列注册、prime、Graph 捕获与资源释放的成本。

<a id="basic-versions"></a>

## 01–03：从 CPU 流程到 CUDA 前后处理与 pinned 缓冲区

01 的流程：CPU Letterbox、BGR→RGB、归一化、HWC→CHW → FP32 输入 H2D → TensorRT → 原始结果 D2H → CPU 后处理。单次演示和基准测试使用同一路径，绘图都在计时区间外。[基础部署与测量笔记](notes/01_baseline.md)。

02 将原始 BGR uint8 图传到 GPU，用一个 kernel 完成缩放、补边、归一化和 RGB/CHW，直接写 TensorRT 输入。默认 `GPU_POST=0` 下载原始结果后在 CPU 过滤；`GPU_POST=1` 在 TensorRT 后另启动后处理 kernel，完成过滤、坐标还原和稳定压紧，再下载固定容量结果。CPU / GPU 路径的检测规则相同；它们是两个 kernel，中间执行 TensorRT，不是将前后处理融合成一个 kernel。

02 两种模式均使用普通 pageable 主机内存，不引入旧 07 的 pinned staging、减少同步或 Graph。前处理末尾和推理末尾的 stream 同步均保留。OpenCV uint8 插值与 CUDA 取整并非逐位一致，像素对照允许归一化后 `1/255` 加浮点容差；完整数据集精度仍需另查。[CUDA 前后处理笔记](notes/02_cuda_prepost.md)。

03 对照的是 02 的 CPU 后处理路径：增加 `cudaMallocHost` 的原图 staging 与输出缓冲区，每次先把普通 `cv::Mat` 复制到 pinned 内存。分配和扩容在计时循环外，这次 CPU 复制在前处理计时内。02 / 03 仍在前处理后和推理后同步，不做跨图重叠。[Pinned Memory 笔记](notes/03_pinned_memory.md)。

## 04–07：执行位置、提交方式和并发分别对照

```text
CPU staging
    ↓
H2D → CUDA 前处理 → TensorRT → [可选 GPU 后处理] → D2H
    ↓
完成事件 / 同步 → CPU 后处理或仅 unpack → 可用检测结果
```

04 / 07 采用 1 或 2 个槽。每槽独占 context、stream、pinned/device 缓冲区和 events，只共享 engine。双槽时先提交两个请求，再在复用某槽前等待并消费它的结果；预热和测量结束均完整 drain。不会在 DMA 期间覆盖 pinned 输入，也不会让两条 stream 同时使用同一个 context。

05 / 06 串行处理，每张图按自身原图尺寸捕获；切换图片前先销毁旧图，再按需扩容。07 为每槽每张预加载图片保留一个 GraphExec。07 的网络 shape、context 状态和绑定地址固定，改变原图尺寸对应另一缓存项；这是静态模型示例，不是任意动态 shape 的缓存方案。

Graph 模式先正常提交一次并同步（prime），再捕获与实例化；每次正式请求仍执行 CPU staging。关闭 Graph 只改提交方式，不改算法和末尾等待点；捕获失败报错，不自动回退。Graph、context 和缓冲区只在所属 stream 工作结束后销毁。

02 / 06 / 07 的 GPU 后处理按候选顺序稳定压紧有效框，CPU 只将其转成 `Detection`。输出为固定容量 `GpuResults`，D2H 仍复制整个结构，不是按 count 动态缩短传输。06 / 07 保留各自独立副本和原有 pinned 路径，合并章节不改变这些后续版本。

## 08–10：继续隔离剩余开销

08 在 05 的单槽 Graph 路径上比较直接计算与查表。两张一维映射表保存横纵源坐标、相邻索引与插值权重；setup 时在 GPU 上按当前原图尺寸生成，Graph 重放只查表并完成插值与布局转换。像素变化不要求重建表，尺寸变化则必须重建；修改表之前先确保引用它的 Graph 已完成。`MAPS=0` 仍使用相同 Graph、传输与 CPU 后处理，不把 Graph 收益算成查表收益。

09 同样固定单槽 Graph，`REGISTERED=0` 是复制到 `cudaMallocHost` staging；`REGISTERED=1` 用 `cudaHostRegister` 注册当前 `cv::Mat`，直接从该原图执行每次 H2D。它省掉的是 CPU→CPU 复制，不是 PCIe H2D，也不是让 TensorRT 直接读 host 内存。Graph 执行完毕并销毁后才能注销或释放原图，DMA 期间不可改写像素。注册、捕获和注销不能算作免费操作：本例每图注册一次，稳态延迟与 `setup.csv` 应一起看；每张新图只推一次的业务必须另测包含这些成本的路径。

10 将数据拷贝与计算分到三条 non-blocking stream，以 `input_ready` 和 `inference_done` 事件连接依赖。每槽有独立的 pinned/device 缓冲区，但整个程序只有一个 context，所有 `enqueueV3` 始终由同一 CPU 线程提交到同一 compute stream。它只尝试重叠不同请求的拷贝与计算，不并发执行两个 TensorRT 推理；区别于 04 / 07 的每槽独立 context。

10 允许 1–4 槽。`SPLIT=0` 保留相同的缓冲区、事件与等待调用，只把三个阶段排入同一 compute stream；固定槽数比较 `SPLIT=0/1`，再单独考察队列深度。槽在 D2H 后的 `done` 完成并消费结果后才能复用；析构和异常路径先 drain 三条 stream。更多槽会增加内存与排队时间，不保证吞吐更高。

## 如何理解测量列

| 版本 | `timings.csv` 阶段列 | 测量边界 |
| --- | --- | --- |
| 01 / 02 / 03 | `preprocess_ms,infer_ms,postprocess_ms,total_ms` | 同步后的 CPU 墙钟；02 / 03 的原图 H2D 属于前处理 |
| 04 / 05 / 06 / 07 | `staging_ms,gpu_ms,postprocess_ms,total_ms` | CPU staging、完整 GPU 链 events、CPU 后处理或 unpack、请求墙钟延迟 |
| 08 / 09 | 同上 | 串行请求；09 注册模式的 staging 只含几何准备，不再复制像素 |
| 10 | 同上 | CPU 复制、跨阶段 GPU 事件跨度、CPU 后处理、含入槽排队的请求延迟 |

02 的 GPU 模式将后处理 kernel、结果 D2H 及末尾同步计入 `infer_ms`，`postprocess_ms` 只剩 CPU unpack；CPU 模式的 `postprocess_ms` 则包含过滤和坐标还原。比较 CPU / GPU 后处理时看 `total_ms`，不能把某列移走的工作当作消失。

`gpu_ms` 包含 H2D、CUDA 前处理、TRT、可选 GPU 后处理和 D2H，不是纯网络推理耗时。04 / 07 的 `staging_ms` 只测 CPU 复制；05 / 06 还包含计算 Letterbox 参数。不同区间不能直接相加，也不能只比较局部阶段宣称端到端加速。

所有当前版本均不把读图、绘图、日志和输出文件 IO 计入请求延迟。04 / 07 的 `total_ms` 从入槽 staging 到 CPU 消费完成，包含已入槽排队与等待，不包含入槽前的上游等待；其测量墙钟包含流水线填充、drain 和内存中记录开销。

`throughput.csv` 格式为 `samples,wall_ms,fps`，`fps = samples × 1000 / wall_ms`。05 / 06 对各图片的正式测量区间求和，04 / 07 测整段轮转任务；均排除预热及结果文件 IO，不能叫视频解码端到端 FPS。并发时尤其不能用 `1000 / P50` 代替实测 FPS。

05 / 06 / 07 的 setup 包含 prime 和可选捕获 / 实例化，但不包括 engine 加载；它另列，不混入稳态延迟。详细开关对照和资源代价见[学习笔记](notes/README.md)，统一结果见[旧编号 05–09 实验记录](benchmarks/rtx3060-series-05-09-2026-09-14/README.md)。

08 的 setup 还包含尺寸映射表生成；09 的 setup / teardown 包含注册 / 注销等生命周期操作，均排除 engine 加载。08 / 09 的 `throughput.csv` 对逐图正式测量墙钟求和，10 则使用含填充和 drain 的整段轮转墙钟。10 的 GPU 事件可以位于不同 stream，跨度包含阶段等待与其他请求竞争，不是各 kernel 独占时长。08–10 的历史结果单独保存在[旧编号 10–12 对照实验](benchmarks/rtx3060-series-10-12-2026-09-14/README.md)，不与旧编号 01–09 的历史批次拼接排名。

## 验证入口与尚未覆盖的能力

组件测试包括 [CPU/CUDA 前处理](../tests/preprocess_test.cpp)、[pinned staging 与传输](../tests/pinned_memory_test.cpp)、[CPU/GPU 后处理](../tests/gpu_postprocess_test.cpp)；[基础集成检查](../tests/basic_integration.py)验证 01 与 02 的命令、计时和结果，[系列集成检查](../tests/series_integration.py)对照 02 的两种模式与 03 / 04 / 05 / 06 / 07。

08–10 增加[映射表前处理对照](../tests/preprocess_maps_test.cpp)、[注册原图与 Graph 重放](../tests/registered_input_test.cpp)，以及[进阶集成检查](../tests/advanced_integration.py)。组件测试不需要模型文件，集成检查需要本机生成的 engine；具体命令见脚本 `--help`。

```bash
ctest --test-dir build/release --output-on-failure
```

动态 shape、INT8、视频解码、服务化、通用库 API、完整数据集 mAP 等属于[长期路线](../ROADMAP.md)，不是当前版本已实现功能。
