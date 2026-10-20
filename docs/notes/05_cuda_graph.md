# 05：把一张图片的 GPU 工作记录下来，反复执行

01 每处理一张图片，都先在 CPU 整理像素，再把输入送到 GPU 推理，取回输出，最后由 CPU 筛出检测框。05 保留一次处理一张图的方式，把前处理移到 GPU，并把重复的 GPU 提交过程记录下来。按请求 P50 计算，本次默认配置相对 01 的加速比为 **1.25×**；这是整套实现的结果。

## 从 01 的一次请求开始

[01 的源码](../../src/01_baseline/main.cpp)先把原图等比例缩放、补边到 640 × 640，再完成 BGR 转 RGB、除以 255、按三个通道分别存放像素。完成后，把浮点输入从主机复制到显存，也就是 H2D；TensorRT 推理后，把输出从显存复制回主机，也就是 D2H。

CPU 等待输出就绪，再检查每个候选框的分数和坐标，把框还原到原图。整个过程是：

```text
01：CPU 前处理 → H2D → TensorRT → D2H → CPU 后处理
```

这里有两部分重复工作：CPU 每轮都要遍历像素，且每轮都要重新提交相同顺序的 GPU 操作。05 分别处理这两个问题，理解收益时也要把它们分开。

## 先让 GPU 从原图生成输入

05 直接把 BGR 原图送到 GPU。一个 GPU 函数让许多线程同时生成输出像素，在一次处理中完成缩放、补边、通道转换和归一化。这种 GPU 函数称为 CUDA kernel；模型输入格式仍是 `[1,3,640,640]` 浮点张量。

传输的数据也随之改变：01 传的是整理好的浮点张量，05 传的是原图字节。原图越大，传输量也越大，不能把这个变化简单理解为永远少搬数据。

为了让主机侧传输缓冲区保持可用，程序预先申请一块不会被操作系统换出的内存，称为页锁定内存（Pinned Memory）。OpenCV 解码出的图片仍在普通内存，每次请求先用 CPU `memcpy` 复制到这块缓冲区；这一步称为 staging，并计入请求延迟。输出也使用页锁定内存，原图和模型输入输出缓冲区反复复用。

此时，一次请求变成：

```text
CPU：原图 → memcpy 到 pinned 输入
GPU：H2D → CUDA 前处理 → TensorRT → D2H 到 pinned 输出
CPU：等待完成 → 筛选检测框、恢复原图坐标
```

GPU 操作提交到同一条按顺序执行的队列中，这条队列就是 CUDA stream。与 01 相同，CPU 只在整段 GPU 工作末尾等待一次；05 将前处理也放进这条队列。

## 再把固定的提交顺序记录下来

上面的 GPU 工作每轮顺序相同，TensorRT 内部还可能提交多个 kernel。CPU 可以先记录这些操作和依赖，整理成可执行对象，之后每轮只发出一次执行请求。这套机制叫 CUDA Graph。

记录过程叫 capture，生成可执行对象叫 instantiate，重复执行使用 `cudaGraphLaunch()`。05 记录的范围是 `H2D → CUDA 前处理 → TensorRT → D2H`。CPU staging 和 CPU 后处理仍在图外，每次照常执行。

Graph 保存的是操作及其参数，并没有缓存检测答案。每次重放时，H2D 都会读取 pinned 输入中的当前像素；模型计算也会重新执行。它尝试减少重复提交和调度的开销，并没有减少 YOLO 的数学计算量。

[main.cpp](../../src/05_cuda_graph/main.cpp)中，`submit()` 写出了整段 GPU 操作；普通运行和捕获都调用它。`capture_graph()` 负责记录和实例化，正式循环则在直接 `submit()` 与 `cudaGraphLaunch()` 之间选择。前处理实现见 [preprocess.cu](../../src/05_cuda_graph/preprocess.cu)。

## 为什么先准备，再捕获

图中记录了内存地址、传输长度和原图尺寸。可以在上一轮完成后更新同一地址里的像素；若释放该内存、换地址或改变尺寸，旧图就不能继续按原参数使用。

因此，每张图片先销毁旧图，必要时扩容原图缓冲区，再普通执行一次完整 GPU 流程并同步。这次准备执行称为 prime，用来完成 TensorRT 可能发生的首次资源分配和状态更新。随后才捕获、实例化、预热，进入正式测量。

网络输入虽然固定为 640 × 640，原图尺寸仍会改变 H2D 长度和前处理参数。本版每换一张图片就重新捕获，包括同尺寸图片；它没有实现图参数更新或图缓存。

捕获阶段的 `submit()` 不分配、不等待，也不读取 GPU 结果。提交失败时仍结束捕获并清理图，随后报错退出。退出或异常时先等待 stream 完成，再销毁图和缓冲区，最后释放外层 TensorRT context，保证图引用的资源一直有效。

## 和 01 比较

统一测试条件：RTX 3060、YOLO26m FP16 / FP32 I/O、640 × 640、batch 1、Release；`bus.jpg`、`zidane.jpg`，3 轮，每轮每图预热 200 次、测量 1000 次，每版 6000 条样本。

| 版本 | 请求 P50（ms） | 相对 01 加速比 |
| --- | ---: | ---: |
| 01 · CPU 前后处理 | 4.547317 | 1.00× |
| 05 · Graph 开启 | 3.629839 | **1.25×** |

加速比是 `01 的请求 P50 ÷ 本版请求 P50`。本次 P50 降低约 20.2%，包含 CUDA 前处理、传输路径与缓冲区复用、Graph 提交方式的组合效果。若想知道 Graph 自身贡献，应在 05 内只切换 `GRAPH`，不能把相对 01 的全部差值归给它。

两版都用 CPU 墙钟测一次请求：01 从 CPU 前处理开始，05 从 staging 开始，直到 CPU 后处理完成。初始化、预热、图片解码、画框和文件读写不计入。05 的 `gpu_ms` 是 H2D 到 D2H 的事件区间，不是纯模型时间，也不能与 CPU 分段简单相加来代替 `total_ms`。

[统一报告与原始数据](../benchmarks/rtx3060-yolo26m-2026-10-15/README.md)保留完整口径。这里只是两张图的重复测试；CUDA 插值与 OpenCV 可能存在像素舍入差异，性能比较也不等于完整数据集精度验证。[YOLO26n 历史 Graph 开关实验](../benchmarks/rtx3060-series-05-09-2026-09-14/README.md)单独保留，不与本次基准拼接。

## 运行与观察

在仓库根目录使用同一 engine、图片目录和迭代次数：

```bash
cmake --preset release -DCMAKE_CUDA_ARCHITECTURES=native
cmake --build --preset release --target yolo_baseline yolo_cuda_graph -j

./build/release/src/yolo_baseline weights/yolo26m.fp16.engine images 1000 200
./build/release/src/yolo_cuda_graph weights/yolo26m.fp16.engine images 1000 200 1
```

05 的参数为 `ENGINE IMAGE_FOLDER [ITERATIONS=200] [WARMUP=50] [GRAPH=1]`。前两个次数按每张图计算，最后的 `1` 表示启用 Graph；改为 `0` 可观察普通提交。该示例用于本机对照，表中 engine 的哈希和精确复现路径见统一报告。

输出位于 `results/01_baseline/` 和 `results/05_cuda_graph/`。比较 `timings.csv` 的 `total_ms`；同一版本再次运行会覆盖同名文件，切换开关前需保存上一组结果。

日志中的 `setup_ms` 包含每图缓冲区准备、prime 和可选捕获，不属于稳态 P50。若只处理一张图一次，这笔成本很难摊薄。`throughput.csv` 则按完成样本数除以正式测量总墙钟时间计算吞吐率，不能用 `1000/P50` 代替。

动手观察：保存 Graph 开、关两组结果，检查检测结果和 `total_ms`，再找出每次 launch 前的 `stage_image()`。解释为什么删掉这次拷贝后，图仍能运行，却可能一直处理上一张图片。

返回：[笔记目录](README.md) · [01 基准](01_baseline.md)。
