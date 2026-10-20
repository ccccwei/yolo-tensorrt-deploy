# 06：让 GPU 完成前后处理，再返回检测框

01 把像素整理和检测框筛选留在 CPU，GPU 负责模型推理。06 把这两端计算都搬到 GPU，并记录中间整段操作来重复提交。按请求 P50 计算，本次默认配置相对 01 的加速比为 **1.25×**；这反映整套实现的效果，不能只归功于 GPU 后处理。

## 从 01 等待输出的地方看起

[01 的源码](../../src/01_baseline/main.cpp)先在 CPU 等比例缩放原图、补边到 640 × 640，再转为 RGB、归一化，并把三个通道分别连续存放。整理好的输入复制到 GPU，称为 H2D；TensorRT 推理后，输出复制回 CPU，称为 D2H。

```text
01：CPU 前处理 → H2D → TensorRT → D2H → CPU 后处理
```

模型输出是 `[1,300,6]`：每张图最多 300 个候选，每个候选包含四个坐标、分数和类别。CPU 等待 D2H 完成，才开始过滤低分框、恢复原图坐标、裁剪边界。当前 YOLO26 输出已经完成模型内的筛选，程序不再执行额外 NMS。

这些逐像素、逐候选的操作可以让 GPU 线程并行完成。06 将 CPU 的工作缩减为准备当前原图，以及在 GPU 完成后组装 C++ 检测结果。

## 一次请求如何走完

CPU 先把原图复制到预先申请的主机缓冲区。这块内存不会被操作系统换出，称为页锁定内存（Pinned Memory）；从普通图片复制到它的步骤称为 staging，每次请求都会执行并计时。

H2D 现在传输原始 BGR 字节。GPU 前处理函数让多个线程同时生成输入像素，一次完成缩放、补边、转 RGB、归一化和通道重排，直接写入固定的模型输入显存。这样的 GPU 函数称为 CUDA kernel。原图传输量取决于原始分辨率，不能默认小于 01 的浮点输入。

TensorRT 完成后，另一个 kernel 检查候选分数和有限值，撤销补边与缩放，再把坐标裁剪到原图范围，过滤无效框。最后回传已经整理好的结果：

```text
CPU：普通原图 → memcpy 到 pinned 输入
GPU：H2D → CUDA 前处理 → TensorRT → CUDA 后处理 → D2H
CPU：等待完成 → 将有效结果组装为 Detection
```

这些 GPU 操作进入同一条按顺序执行的工作队列，也就是 CUDA stream。输入输出和原图缓冲区反复复用，CPU 只在 GPU 工作末尾等待一次。01 也在推理段末尾同步；这里的变化是计算位置和提交内容。

源码入口为 [main.cpp](../../src/06_full_graph/main.cpp) 的 `stage_image()`、`submit()` 和 `unpack()`；两个 GPU 计算分别位于 [preprocess.cu](../../src/06_full_graph/preprocess.cu) 与 [postprocess.cu](../../src/06_full_graph/postprocess.cu)。

## 有效框怎样排好，CPU 怎样读取

假设候选依次为“有效、无效、有效”，输出应连续保存第一个和第三个框。每个线程先判断自己的候选是否有效，再统计当前位置之前共有多少个有效项，就知道自己应写入哪个位置。这种累积计数叫前缀和；本版用它压紧有效结果，同时保留原候选顺序。

因此，结果顺序不依赖线程谁先执行。`GpuResults` 保存一个 `count` 和最多 300 个框，只有 `boxes[0..count)` 有效。每次 kernel 都重写 `count`，没有检测时写 0，CPU 不会误读上一轮数量。

结果缓冲区在初始化时清零一次，之后每次 D2H 都复制整个固定容量的 `GpuResults`。当前布局占 7204 字节，原始浮点输出占 7200 字节；这里的紧密排列便于 CPU 读取，并没有减少 D2H 字节数。

CPU 必须等 D2H 完成，才能调用 `unpack()`。它检查 `count` 是否在 0–300 内，把有效项转成 `Detection` 和 `cv::Rect`，不再过滤分数或计算坐标。

## 把这段固定工作记录成图

每次请求都提交同一组 GPU 操作，CPU 可以先记录操作与依赖，再生成一个能重复执行的对象。记录过程叫 capture，生成可执行对象叫 instantiate；这就是本版使用的 CUDA Graph。

`submit()` 包含 `H2D → CUDA 前处理 → TensorRT → CUDA 后处理 → D2H`，既供普通运行使用，也供捕获使用。启用 Graph 后，每轮通过一次 `cudaGraphLaunch()` 重放这段流程；CPU staging 和 `unpack()` 仍在图外。

每次重放都会重新读取当前 pinned 输入、执行模型和后处理。Graph 试图减少的是 CPU 重复提交开销，模型和后处理计算仍要完成，GPU 后处理自身也有 kernel 启动与执行成本。

捕获前先准备缓冲区，普通执行一次并同步，让 TensorRT 完成可能的首次初始化；这称为 prime。随后捕获、实例化、预热，最后才计入正式样本。`GRAPH=0` 也会 prime，`WARMUP=0` 因此不代表冷启动测试。

## 地址和尺寸为何不能随便改

图记录了缓冲区地址、H2D 长度、原图宽高，以及前后处理所需的缩放和补边参数。输入内容可在上一轮完成后更新；这些地址或参数变化后，必须先停止使用旧图。

本版每换一张图片都销毁旧图，必要时扩容原图缓冲区，再为当前图片重新捕获，即使网络输入一直是 `[1,3,640,640]`。它没有实现图缓存或参数更新。

捕获期间不分配、不同步、不由 CPU 读取输出。若提交失败，仍结束捕获并清理图，随后报错退出。资源销毁先等待 stream 完成，再释放图、事件和缓冲区，最后释放外层 TensorRT context，避免图引用失效资源。

## 和 01 比较

统一测试条件：RTX 3060、YOLO26m FP16 / FP32 I/O、640 × 640、batch 1、Release；`bus.jpg`、`zidane.jpg`，3 轮，每轮每图预热 200 次、测量 1000 次，每版 6000 条样本。

| 版本 | 请求 P50（ms） | 相对 01 加速比 |
| --- | ---: | ---: |
| 01 · CPU 前后处理 | 4.547317 | 1.00× |
| 06 · 完整 Graph 开启 | 3.640479 | **1.25×** |

加速比是 `01 的请求 P50 ÷ 本版请求 P50`，本次 P50 降低约 19.9%。相对 01，变化包含 CUDA 前后处理、原图与 pinned 传输路径、缓冲区复用和 Graph 提交方式；这个比较不能单独证明哪一项贡献最大。

01 的 `total_ms` 从 CPU 前处理开始，06 从 staging 开始，两者都到 CPU 检测结果可用为止。06 的 `gpu_ms` 包含传输、前处理、模型和后处理，`postprocess_ms` 只表示 CPU `unpack()`；不要把它与 01 的 CPU 后处理时间理解为相同工作量。

[统一报告与原始数据](../benchmarks/rtx3060-yolo26m-2026-10-15/README.md)提供完整口径。这是两张图片的重复测试，排除了初始化、预热、解码、画框和文件读写，不是完整数据集精度验证。CUDA 前处理与 OpenCV 可能存在插值舍入差异；[YOLO26n 历史开关实验和检查记录](../benchmarks/rtx3060-series-05-09-2026-09-14/README.md)单独保留。

## 运行与观察

在仓库根目录使用同一 engine、图片目录和迭代次数：

```bash
cmake --preset release -DCMAKE_CUDA_ARCHITECTURES=native
cmake --build --preset release --target yolo_baseline yolo_full_graph -j

./build/release/src/yolo_baseline weights/yolo26m.fp16.engine images 1000 200
./build/release/src/yolo_full_graph weights/yolo26m.fp16.engine images 1000 200 1
```

06 参数为 `ENGINE IMAGE_FOLDER [ITERATIONS=200] [WARMUP=50] [GRAPH=1]`，次数按每张图计算。最后的 `1` 改为 `0`，GPU 前后处理保持不变，只改为普通提交，适合检查 Graph 的贡献。表中 engine 哈希和精确复现路径见统一报告。

结果分别保存在 `results/01_baseline/`、`results/06_full_graph/`，请求比较使用 `timings.csv` 的 `total_ms`。同一版本重跑会覆盖同名输出，切换开关前需保存结果。

每图 `setup_ms` 包含准备缓冲区、prime 和可选的图捕获，不属于上表稳态延迟。图创建成本需要多次请求摊薄。`throughput.csv` 按正式测量完成数除以实际墙钟时间计算吞吐率，不使用 P50 的倒数。

动手自检：沿着 `postprocess_kernel()` 到 `unpack()` 追踪一张没有有效检测的图片，说明 `count` 如何变成 0，以及为什么 CPU 必须等待 D2H 完成后才能读取它。

返回：[笔记目录](README.md) · [01 基准](01_baseline.md)。
