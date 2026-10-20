# 07：让两张图片同时在途，并重放各自的 GPU 流程

01 完成一张图片后，才开始下一张。07 准备两套独立资源，让两张图片的处理有机会重叠，同时把前后处理移到 GPU，减少重复提交。按请求 P50 计算，本次默认双槽配置相对 01 的加速比为 **0.70×**，也就是延迟高约 43.6%。增加并发并不保证一张图更早返回。

## 从 01 的串行请求开始

[01 的源码](../../src/01_baseline/main.cpp)先在 CPU 缩放、补边、转换颜色和数据布局，再把浮点输入复制到显存（H2D）。TensorRT 完成推理后，把输出复制回主机（D2H），CPU 等待完成并筛出检测框、恢复原图坐标。

```text
01：CPU 前处理 → H2D → TensorRT → D2H → CPU 后处理
    当前图片完成后，才开始下一张
```

CPU 等待 GPU 时，下一张图还没有进入处理。若准备第二套输入输出，CPU 就有机会在前一个请求未结束时提交后一个请求，传输和计算也可能重叠。代价是两张图会竞争 GPU 资源，而且每套资源都占用内存。

## 先把一张图的流程准备好

07 每次先把普通内存中的 BGR 原图复制到一块不会被操作系统换出的主机内存，称为页锁定内存（Pinned Memory）。这次 CPU 拷贝称为 staging，发生在每次请求中，不能因为图片重复使用就省略。

随后 H2D 传输原图。GPU 线程并行完成缩放、补边、转 RGB、除以 255 和通道重排，直接生成 `[1,3,640,640]` 输入。这段 GPU 函数称为 CUDA kernel；它替代了 01 的 CPU 前处理，传输量也从固定的浮点输入改为随原图尺寸变化的字节数。

模型输出仍为 `[1,300,6]`。另一个 kernel 过滤低分和非有限值、撤销缩放与补边、裁剪坐标，按原候选顺序连续保存有效框。每个候选先判断有效性，再累积此前有效数量来确定写入位置，这种方法叫前缀和。它保持顺序稳定，不额外执行 NMS。

```text
CPU：原图 → memcpy 到 pinned 输入
GPU：H2D → CUDA 前处理 → TensorRT → CUDA 后处理 → D2H
CPU：等待完成 → 将有效结果组装为 Detection
```

回传的 `GpuResults` 包含数量 `count` 和最多 300 个框，每次重写 `count`，CPU 只读取有效范围。结果缓冲区初始化时清零，D2H 仍复制完整结构体：当前布局为 7204 字节，原始输出为 7200 字节，所以没有通过减少回传字节数来提速。

源码入口：[main.cpp](../../src/07_graph_pipeline/main.cpp)、[preprocess.cu](../../src/07_graph_pipeline/preprocess.cu)、[postprocess.cu](../../src/07_graph_pipeline/postprocess.cu)、[结果结构](../../src/07_graph_pipeline/postprocess.hpp)。

## 两套资源怎样互不覆盖

一张图尚未处理完，CPU 就改写它的输入，会让 GPU 读到错误数据。因此，07 把一个请求需要的推理状态、队列、缓冲区和完成标记放在一起，称为一个槽（slot）。默认有两个槽，仅共享只读 TensorRT engine。

每槽有独立的 TensorRT 执行状态对象（context）、原图 staging、输入输出显存和结果缓冲区。GPU 操作进入该槽自己的 CUDA stream，即按顺序执行的工作队列。不同 stream 提供重叠执行的机会，实际能重叠多少仍由 GPU 资源决定。

`submit()` 先填入本槽原图，再提交整段 GPU 工作，在末尾记录完成事件 `done_event`，随后立即返回。CPU 可接着使用另一个空槽。事件是 GPU 执行到特定位置的标记，CPU 通过它判断本槽是否已经完成。

再次使用同一个槽前，`finish()` 等待它的完成事件，读取回传结果，调用 `unpack()` 组装检测框，再将槽标为空闲。`stage_image()` 会拒绝覆盖仍然忙碌的槽。

调度过程可以这样读：

```text
提交 A 到槽 0 → 提交 B 到槽 1
等待并消费 A → 槽 0 提交 C
等待并消费 B → 槽 1 提交 D → ……
```

最后仍可能有请求在途，`run()` 按提交顺序等待并取完剩余结果，这叫 drain。预热也完整 drain，再从空槽开始正式计时，单个任务和奇数任务都要走完这个收尾过程。

## 每个槽为何还要保存自己的图

每个请求都重复提交同一顺序的 GPU 操作。可以先记录这些操作及依赖，再生成可重复执行的对象，之后通过一次 `cudaGraphLaunch()` 提交。这就是 CUDA Graph：记录过程称为 capture，生成可执行对象称为 instantiate。

本版记录 `H2D → CUDA 前处理 → TensorRT → CUDA 后处理 → D2H`。图外仍需每次 CPU staging，以及完成后的 `unpack()`。重放会读取当前输入并重新推理，不会复用上一张图的检测答案。

图中记录了缓冲区地址、传输大小和前后处理参数。槽 0 与槽 1 的地址不同，同一槽处理不同原图尺寸时参数也不同。因此，初始化时按“槽 × 图片”准备图：两槽、三张图共六个可执行图，运行时用图片编号查找本槽的图。

这就是代码中的 Graph cache。每个槽的模型输入输出地址固定，网络 shape 始终是 `[1,3,640,640]`；不同图片只改变网络外部的尺寸、拷贝长度和前后处理参数。同槽内的多个图共享一个状态固定的 context。

这份缓存面向预先读入的静态图片，不能直接当作动态网络 shape 的通用方案。改变图片尺寸、缓冲区地址或 context 状态后，需要先完成在途工作，销毁旧图，再重新配置并捕获。相同尺寸的不同图片也各存一张图，图库变大时图数量和 setup 成本都会增长。

## 准备和退出也属于实现的一部分

`prepare_slot()` 为每槽申请能容纳最大原图的缓冲区。随后对每张图片先普通执行一次并同步，完成 TensorRT 可能的首次资源初始化，这一步称为 prime；启用 Graph 才继续捕获和实例化。关闭 Graph 或预热次数为 0 时，prime 仍会执行。

普通运行与捕获共用 `enqueue_commands()`，其中没有内存分配、CPU 等待或读取 GPU 结果。捕获中出现异常时仍结束捕获并清理图，随后退出，避免把普通提交误标为 Graph 运行。

`GraphResource` 由 `unique_ptr` 独占持有，防止同一图被重复释放。退出或异常时先等待所有 stream，再依次释放各槽的图、context、事件和缓冲区，外层 engine 最后释放。

## 和 01 比较

统一测试条件：RTX 3060、YOLO26m FP16 / FP32 I/O、640 × 640、batch 1、Release；`bus.jpg`、`zidane.jpg`，3 轮，每轮每图预热 200 次、测量 1000 次，每版 6000 条样本。

| 版本 | 请求 P50（ms） | 相对 01 加速比 |
| --- | ---: | ---: |
| 01 · CPU 前后处理 | 4.547317 | 1.00× |
| 07 · 双槽、Graph 开启 | 6.528452 | **0.70×** |

加速比是 `01 的请求 P50 ÷ 本版请求 P50`。默认 07 在这个指标上慢于 01。它同时改变前后处理位置、主机传输内存、资源复用、提交方式和请求调度；这个结果不能单独归因于 Graph 或某个 kernel。

07 的 `total_ms` 从本槽 CPU 原图拷贝开始，到 CPU 取出并组装结果为止，包含入槽后的 GPU 资源竞争和等待消费时间，不含入槽前排队；缩放参数计算发生在开始计时之前。01 则从 CPU 前处理开始计时。两者都排除初始化、预热、解码、画框和文件读写。

吞吐率回答“一段时间完成多少张”，请求延迟回答“一张图多久返回”。并发可能提高前者，同时增加后者。07 的 `throughput.csv` 按完成数除以正式测量总墙钟时间计算，包含填充与 drain，不能用 `1000/P50` 推算，也不能拿旧批次吞吐倍率替换上表的延迟倍率。

[统一报告与原始数据](../benchmarks/rtx3060-yolo26m-2026-10-15/README.md)记录本次结果。[YOLO26n 历史 Graph / 槽数实验](../benchmarks/rtx3060-series-05-09-2026-09-14/README.md)单独保留。两张示例图的重复测量不代表完整数据集性能与精度，CUDA 前处理与 OpenCV 也可能存在插值舍入差异。

## 运行与观察

在仓库根目录使用同一 engine、图片目录和迭代次数：

```bash
cmake --preset release -DCMAKE_CUDA_ARCHITECTURES=native
cmake --build --preset release --target yolo_baseline yolo_graph_pipeline -j

./build/release/src/yolo_baseline weights/yolo26m.fp16.engine images 1000 200
./build/release/src/yolo_graph_pipeline weights/yolo26m.fp16.engine images 1000 200 2 1
```

07 的参数为 `ENGINE IMAGE_FOLDER [ITERATIONS=200] [WARMUP=50] [SLOTS=2] [GRAPH=1]`，次数按每张图计算。槽数只允许 1、2，Graph 只允许 0、1。表中 engine 哈希和精确复现路径见统一报告。

基准输出在 `results/01_baseline/`，本版输出在 `results/07_graph_pipeline/`。查看 `timings.csv` 的 `total_ms` 和独立的 `throughput.csv`；切换开关前保存结果，同一版本重跑会覆盖同名文件。

保持槽数不变、切换 Graph，可以观察提交方式；保持 Graph 不变、切换槽数，可以观察并发。日志中的 `Setup` 包含槽资源准备、prime、捕获和实例化，`GraphExecs` 给出可执行图数量。这些额外成本未计入上表稳态 P50，需要按真实请求量考虑。

动手观察：准备三张不同尺寸图片，使用双槽各运行一次。按 A→B→C 追踪 `run()`，说明 C 为什么必须等槽 0 完成、为什么要选择 C 对应的图，以及最后哪几个结果由 drain 收回。

返回：[笔记目录](README.md) · [01 基准](01_baseline.md)。
