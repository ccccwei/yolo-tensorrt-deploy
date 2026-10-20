> 历史版本笔记：此处描述合并前的独立旧 07。当前 CUDA 前后处理请阅读 [02 笔记](../../../notes/02_cuda_prepost.md)；本文其余编号均为历史编号。

# 07：把后处理搬到 GPU

这一步只研究一个问题：TensorRT 已经把检测结果写在显存里，能不能让 GPU 顺手处理完，再交给 CPU？

## YOLO26 不需要 NMS，为什么还有后处理？

本仓库使用的 YOLO26 NMS-free 导出，每张图片输出 `[1, 300, 6]`，每行是 `x1, y1, x2, y2, score, class_id`。这条路径不需要 NMS，但还要过滤低分框、恢复原图坐标、裁剪越界坐标。不要把“不需要 NMS”理解成“输出可以直接画到原图”。不同导出配置可能走另一条检测路径，具体以模型输出合同为准。[Ultralytics 官方说明](https://docs.ultralytics.com/guides/end2end-detection/)

07 把原来 CPU 上的这些工作放入一个 CUDA kernel：

1. 检查六个字段是否为有限值，拒绝 NaN 和无穷大。
2. 保留 `score >= 0.25`，检查类别值非负且能转成 `int`。
3. 对坐标执行 `(坐标 - padding) / scale`，还原到原图。
4. 裁到 `[0, 宽 - 1]`、`[0, 高 - 1]`，丢弃没有面积的框。
5. 将有效框按原来的候选顺序排在输出数组前面。

类别值的转换沿用 04 的规则：允许可表示的非负数再转 `int`，不是另加 COCO 类别范围检查，也不要求整数输入。这样实验只改变执行位置，不顺带改变结果定义。

## 代码怎么读

- [main.cpp](src/07_gpu_postprocess/main.cpp)：准备 pinned 内存，提交完整流水线，记录耗时和保存结果。
- [postprocess.hpp](src/07_gpu_postprocess/postprocess.hpp)：CPU / GPU 共用的数据结构和函数声明，不依赖 OpenCV。
- [postprocess.cu](src/07_gpu_postprocess/postprocess.cu)：过滤、还原、裁剪以及稳定压紧。
- [preprocess.cu](src/07_gpu_postprocess/preprocess.cu)：与 04 相同的 CUDA 前处理，独立保留一份，方便逐版阅读。

提交顺序是：

```text
CPU staging → H2D → CUDA 前处理 → TensorRT → CUDA 后处理 → D2H → CPU 容器转换
               └──────── 同一条 stream，末尾只等待一次 ──────────┘
```

CPU 仍负责普通图片到 pinned buffer 的复制，以及把已经过滤好的 `GpuDetection` 转成 OpenCV 的 `cv::Rect`。后者只转换数据结构，不再重复筛选和坐标变换。画图、解码、写文件均在测量之外。

## 为什么不直接 atomicAdd？

一个 block 启动 512 个线程，前 300 个线程各处理一个候选框，其余线程的有效标记为 0。共享内存中的标记经过前缀和，得到每个有效框的写入下标。例如：

```text
有效标记：0 1 0 1 1
前缀和：  0 1 1 2 3
写入下标：  0   1 2
```

这样能保持候选框原来的顺序。用 `atomicAdd` 竞争下标也能压紧，但输出顺序会受线程执行顺序影响，检测 CSV 的对比就不稳定。这里选择容易理解的共享内存扫描，没有额外引入扫描库。

所有线程都经过 `__syncthreads()`；无效候选也不能提前退出，否则其他线程可能永远等不到它。扫描每一轮先读取旧值，再同步，写回后再同步，避免线程读到同一轮刚刚写入的新值。[CUDA 编程指南](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/writing-cuda-kernels.html)

## 这里没有“大幅减少传输量”

原始输出是 `300 × 6 × 4 = 7200` 字节；当前输出结构是一个 `int count` 加 300 个 24 字节的框，在本项目平台上是 7204 字节。虽然有效框被放在前面，D2H 仍一次复制整个固定容量结构。

这是为了只提交一次 D2H：CPU 提交复制时还不知道 GPU 最后留下几个框。如果先复制 count、等待，再按 count 复制框，会增加依赖和一次等待。因而本步研究的是“后处理执行位置”，不是压缩传输流量。300 条结果本来就很少，增加 kernel 和 block 内同步的成本也可能比 CPU 处理更高，不能假定一定加速。

## 怎么做公平对比

同一个程序提供 `GPU_POST` 开关。`0` 保留 CPU 后处理，`1` 使用 GPU 后处理；两者使用相同的 pinned 输入、stream、事件与末尾等待方式：

```bash
./build/release/src/yolo_gpu_postprocess weights/yolo26n.fp16.engine images 1000 200 0
./build/release/src/yolo_gpu_postprocess weights/yolo26n.fp16.engine images 1000 200 1
```

默认最后一个参数是 `1`。结果保存在 `results/07_gpu_postprocess/`，下一次运行会覆盖同名文件，因此需要对比两次数据时应先保存上一组 CSV。

| 字段 | 含义 |
| --- | --- |
| `staging_ms` | CPU 计算 letterbox 参数、把图片复制到 pinned 内存 |
| `gpu_ms` | CUDA Event 测量 H2D、前处理、TRT、可选 GPU 后处理、D2H 的整段 stream 时间 |
| `postprocess_ms` | `GPU_POST=0` 时是 CPU 后处理；`1` 时仅是 CPU 容器转换，不是 GPU kernel 耗时 |
| `total_ms` | CPU 实际观察到的单次完整延迟，包含 staging、提交、等待和容器转换 |

GPU 时间与 CPU 时间来自不同的计时区间，`staging_ms + gpu_ms + postprocess_ms` 不一定等于 `total_ms`：CPU 提交、驱动与等待返回等成本也存在。吞吐率另用测量段实际墙钟时间计算 `samples / wall_seconds`，不把 `1000 / P50` 叫作吞吐率。该版仍是单图串行，尚未构建跨图片并发。

对比 04 与 07 时还有“去掉中途同步、调整计时边界”等变化，所以隔离 GPU 后处理收益时优先看同一个 07 程序的 `GPU_POST=0 / 1`，不要把两个版本的差值全部归功于后处理 kernel。

## 正确性与边界

[gpu_postprocess_test.cpp](../../../../tests/gpu_postprocess_test.cpp) 直接对照 04 的 CPU 规则，不加载 TensorRT。检查全有效 / 全无效、阈值附近、NaN / Inf、非法类别、坐标越界、零面积框、奇数 padding、多种缩放比例、输出越界保护和稳定顺序；连续复用同一结果缓冲区也要正确更新 count，不能把上张图片的数量遗留下来。

坐标除法使用明确的浮点舍入指令，没有启用 fast-math。测试要求框的整数坐标、类别与顺序精确一致，而不只是“看起来差不多”。分数直接拷贝，没有另算一次置信度。

```bash
ctest --test-dir build/release -R gpu_postprocess --output-on-failure
```

性能数字以实际实验记录为准。本步是执行位置的对照实验；如果测下来变慢，保留这个结果也有价值，因为它说明当前小输出量并不值得独立启动一个后处理 kernel。


## 实验记录

本次使用 RTX 3060、CUDA 11.8、TensorRT 8.6.1、YOLO26n FP16 / FP32 I/O、640×640、batch 1、Release 构建。三轮交替顺序，每轮每图预热 200 次、测量 1000 次，使用 bus / zidane 两图，共 **6000 个样本 / 配置**。P50 / P95 合并全部样本计算；FPS 使用总样本数除以实际测量墙钟，不是 P50 的倒数。

| 配置 | 请求 P50 (ms) | P95 (ms) | 实测 images/s |
| --- | ---: | ---: | ---: |
| GPU_POST=0（CPU） | 1.565 | 1.582 | 640.0 |
| GPU_POST=1（GPU） | 1.575 | 1.599 | 638.4 |

本次 GPU 路径的 P50 **高 0.6%**，没有测到明显端到端收益；差值很小，不把三轮微基准当作统计显著性结论。CPU 后处理 / 容器转换均值从 **1.434 μs** 降至 **0.313 μs**，但缩短 CPU 的这一步并不保证整个请求更快。新增 kernel 的开销与 stream 调度仍需计入。

后处理组件通过 13 次 CPU / GPU 对照和非法参数检查；`compute-sanitizer` 的 memcheck 为 **0 errors**，racecheck 为 **0 hazards**。这验证了边界和共享内存扫描的正确性，不意味着所有模型都适合 GPU 后处理。

[完整实验条件、逐轮数据与复现脚本](../../rtx3060-series-05-09-2026-09-14/README.md)。集成测试覆盖三种尺寸图片轮换、奇数任务数和不足两槽的收尾；03–09 在性能实验中每图最后一次的类别、分数、框与顺序也精确一致。这不是完整数据集 mAP 验证。
