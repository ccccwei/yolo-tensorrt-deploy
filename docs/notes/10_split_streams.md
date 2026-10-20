# 10：拆分传输与计算，让不同请求的阶段重叠

01 按顺序处理每个请求：CPU 完成缩放、补边、通道转换和归一化，把浮点输入拷到 GPU（H2D），执行 TensorRT，再把输出拷回主机（D2H），最后由 CPU 筛选检测框并还原坐标。当前请求处理完，才开始下一次。

这条路径容易理解，但 GPU 计算和数据传输之间也有等待。例如，第一张图片正在执行 TensorRT 时，第二张图片尚未上传。如果提前上传第二张图，就有机会让传输和计算同时进行，缩短连续完成一批请求的时间。

10 围绕这个机会重新安排请求：CUDA 完成前处理，页锁定内存支持异步传输，多组缓冲区保留在途请求，上传、计算和下载分别进入任务队列。**默认双槽模式的请求 P50 为 7.151163 ms，相对 01 的加速比为 0.64×，延迟增加约 57.3%。**本版展示的取舍是阶段重叠与请求等待，不能把并发直接理解成每张图更早返回。

## 先为第二个请求留出位置

01 只需等待当前请求结束后复用输入输出。要提前提交第二个请求，就必须为它保留另一套数据，避免覆盖 GPU 正在读取的第一套输入。本版把一组输入、输出和完成标记称为一个槽（slot）。

每个槽包含主机上的原图暂存区和输出区，以及 GPU 上的原图、模型输入与模型输出。主机暂存区使用页锁定内存（Pinned Memory），即 CUDA 可以直接用于异步传输的主机内存。普通原图每次仍要先复制进去，这段 CPU 复制也计时。

上传的是 BGR 原图。随后一个 CUDA kernel（GPU 并行函数）完成缩放、补边、BGR 转 RGB、归一化和通道排列，直接写出模型输入。模型与 CPU 后处理沿用 01 的约定；本版没有使用 CUDA Graph，也没有注册原图或预计算映射表。

```text
01：CPU 前处理 → H2D 浮点输入 → TensorRT → D2H → CPU 后处理
10：CPU 复制到当前槽 → H2D 原图 → CUDA 前处理 → TensorRT → D2H → CPU 后处理
                         不同请求的传输与计算尝试重叠
```

两个槽轮流使用：先提交槽 0，再提交槽 1；第三次请求回到槽 0 时，必须先等待并消费槽 0 的旧结果。槽数表示最多保留多少个在途请求，不是模型的 Batch，本版每次推理仍为 Batch 1。

## 三条队列，两个不能省的依赖

CUDA stream 是按提交顺序执行 GPU 工作的队列。同一 stream 内，后一项等待前一项；不同 stream 可以独立推进，但没有自动的数据依赖。本版为上传、计算、下载分别创建一条 stream。

同一个请求仍必须先上传、后计算、再下载。CUDA event 是记录在队列中的完成标记，另一条队列可以等到标记完成后再继续：

```text
CPU             复制当前图片到槽的 pinned 输入
                         │
input stream    H2D → input_ready
                         │ 等待上传完成
compute stream           └→ CUDA 前处理 → TensorRT → inference_done
                                                          │ 等待输出生成
output stream                                             └→ D2H → done
                                                                    │ 等待下载完成
CPU                                                                 └→ 后处理
```

`input_ready` 保证前处理读到完整原图；`inference_done` 保证 D2H 复制的是已经生成的模型输出。CPU 等 `done` 后才读取输出并复用该槽，不能只看到 TensorRT 已提交就提前改写输入。

这些依赖只约束同一请求。硬件允许时，`H2D(n+1)` 可以与 `TensorRT(n)` 重叠，下载也可以与后续计算重叠。是否真的重叠，还受拷贝引擎、PCIe 与显存带宽、CPU 提交速度限制。

三条 stream 使用 non-blocking 创建方式，避免默认 stream 的隐式同步干扰这种安排；它不会让等待事件失效，也不保证硬件一定并行执行所有操作。

## 多个槽，共用一个 TensorRT context

TensorRT context 保存执行模型时所需的状态与资源。本版只有一个 context，所有前处理和 `enqueueV3()` 都进入同一条 compute stream，因此两次 TensorRT 推理始终串行执行。

提交来自一条 CPU 线程。每次把下一次推理的输入输出地址指向当前槽，再提交到共同的计算队列：

```cpp
context->setTensorAddress(INPUT_NAME, slot.input);
context->setTensorAddress(OUTPUT_NAME, slot.output);
context->enqueueV3(compute_stream);
```

更新下一次提交的地址，不会把旧槽的内容搬到新槽。旧输入输出必须保留到相关 GPU 工作结束，本实现采用更保守的规则，等整个槽的 `done` 后才复用。运行期间不改变 shape、profile 或 context 激活内存，也不把同一个 context 放到多条 stream 并发推理。

正常退出或提交中途失败时，`Pipeline` 都先等待三条 stream，再释放 context，最后释放槽里的事件和内存。仅检查“这个槽是否标成忙”不够，因为错误也可能发生在 GPU 工作已启动、状态尚未更新的时候。

## 更多槽为什么可能只增加等待

只有一个槽时，提交下一次请求前就要等当前请求完成，没有跨请求重叠空间。两个槽允许另一张图提前进入；但计算仍然串行，后来者可能在 compute stream 中等前一张完成。

例如，每次计算需要 1 ms，即使第二张图提前上传完，也不能跳过第一张图剩余的计算时间。第三、第四张继续排进来，可能让队列更长，却没有提高计算阶段每秒完成的数量。

槽还占内存。若最大原图需要 `B = width × height × 3` 字节，每槽主机页锁定缓冲区占 `B + 300 × 6 × 4` 字节，GPU 缓冲区占 `B + (3 × 640 × 640 + 300 × 6) × 4` 字节。槽数增加，这部分占用也成比例增加；它不包含 engine、context、驱动和其他进程内存。

因此要区分两个指标：请求延迟回答“这一张进入槽以后，多久得到结果”，吞吐率回答“整段时间一共完成多少张”。在途请求越多，二者越不能互相代替。

## 相对 01 的性能

统一测试条件：RTX 3060、YOLO26m FP16 / FP32 输入输出、640 × 640、Batch 1、Release；`bus.jpg` 和 `zidane.jpg` 各跑 3 轮，每轮每图预热 200 次、测量 1000 次，每版共 6000 条记录。

| 版本 | 请求 P50（ms） | 相对 01 加速比 |
| --- | ---: | ---: |
| 01 · CPU 前后处理 | 4.547317 | 1.00× |
| 10 · SLOTS=2，SPLIT=1 | 7.151163 | 0.64× |

P50 是全部请求样本的中位延迟，加速比为 `01 P50 ÷ 本版 P50`，小于 1 表示这次请求延迟更高。10 从入槽后 CPU 复制开始，到 CPU 后处理结束，包含提交后的执行等待及等待消费结果的时间，不含入槽前排队；解码、初始化、预热、画框和文件读写也不计入。完整数据见[统一基准报告](../benchmarks/rtx3060-yolo26m-2026-10-15/README.md)。

相对 01，这个结果包含 CUDA 前处理、Pinned Memory、多槽和拆流的整体变化。仅考察拆流机制时，应固定槽数比较 `SPLIT=0/1`；关闭拆流仍保留相同缓冲区和事件调用，只把所有 GPU 工作排入同一 compute stream。旧 YOLO26n 的[历史开关实验](../benchmarks/rtx3060-series-10-12-2026-09-14/README.md)单独存档，不用其中的吞吐倍率代替本表的 0.64×。

`timings.csv` 的 `gpu_ms` 是从上传前到下载后的事件跨度，包含阶段等待和竞争，并非 TensorRT 独占 GPU 的时间；`total_ms` 才是表中使用的请求延迟。吞吐率另见 `throughput.csv`，按完成请求数除以正式循环墙钟时间计算，包含流水线填充与排空，不能用 `1000/P50` 推算。

## 运行与观察

先按[项目首页](../../README.zh-CN.md#快速开始)准备依赖与 engine，再在仓库根目录执行。两次运行使用同一份 engine、相同图片和次数；下面用于本地对照，报告所用 engine 的确切哈希见统一基准报告。

```bash
cmake --preset release -DCMAKE_CUDA_ARCHITECTURES=native
cmake --build --preset release --target yolo_baseline yolo_split_streams -j

ENGINE=weights/yolo26m.fp16.engine
./build/release/src/yolo_baseline "$ENGINE" images 1000 200
./build/release/src/yolo_split_streams "$ENGINE" images 1000 200 2 1
```

10 的参数为 `ENGINE IMAGE_FOLDER [ITERATIONS=200] [WARMUP=50] [SLOTS=2] [SPLIT=1]`，槽数支持 1–4。两版分别写入 `results/01_baseline/`、`results/10_split_streams/`；重跑会覆盖各自同名结果，换配置前先另存 CSV 和图片。以上是一轮运行，报告中的 6000 条来自三轮合并。

阅读源码可从 [main.cpp](../../src/10_split_streams/main.cpp) 的 `Slot` 看一份请求的数据，再依次读 `submit()`、`finish()` 和 `run()`。前者建立事件依赖，后两者决定何时消费和复用槽；[preprocess.cu](../../src/10_split_streams/preprocess.cu)处理 GPU 像素转换。[集成回归](../../tests/advanced_integration.py)还覆盖请求数不足整圈时的排空，防止最后几个请求漏记。

动手观察时，固定 `SPLIT=1`，分别使用 1、2、4 个槽，并分别保存结果。把请求 P50、实际 images/s 和打印的缓冲区占用一起比较，判断增加的槽是在帮助阶段重叠，还是主要让请求等得更久。不要仅凭其中一个数字判断整套调度更合适。

返回：[笔记目录](README.md) · [01 基准](01_baseline.md)。
