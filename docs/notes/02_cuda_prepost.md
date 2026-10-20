# 02：CUDA 前后处理

01 在 CPU 上把图片整理成模型输入，上传到 GPU 推理，再把结果取回 CPU 筛选。02 主要改变这条路径中的计算位置：**原图先上传，由 GPU 完成前处理；后处理默认仍在 CPU，也可以切换到 GPU。** 本篇直接以 01 为基准。

源码：[main.cpp](../../src/02_cuda_prepost/main.cpp)、[preprocess.cu](../../src/02_cuda_prepost/preprocess.cu)、[postprocess.cu](../../src/02_cuda_prepost/postprocess.cu)。基准实现见 [01 源码](../../src/01_baseline/main.cpp)。

## 01 在上传之前做了什么

假设输入是一张 1280 × 720 的图片，而模型需要 640 × 640。01 先把图片缩成 640 × 360，再在上下各补 140 行灰色像素。这种保留宽高比、用补边凑齐尺寸的处理叫 **letterbox**。

接着，CPU 把 OpenCV 的 BGR 通道顺序改成 RGB，将像素值除以 255，并把每个像素紧挨着的三个通道拆成三个连续平面。模型需要的就是这种 CHW 布局：先存整张 R 平面，再存 G、B 平面。完成这些步骤后，CPU 才将 FP32 输入上传到 GPU。

```text
01：CPU 缩放、补边、转换输入 → H2D → TensorRT → D2H → 等待 → CPU 后处理
02：原图 H2D → CUDA 前处理 → 等待 → TensorRT → D2H → 等待 → CPU 后处理
```

H2D 是主机内存到显存的复制，D2H 是反方向的复制。02 将前处理放在 H2D 之后：CPU 只计算缩放比例和补边位置，GPU 直接生成 TensorRT 要读取的输入缓冲区。

这样既改变了计算位置，也改变了上传内容。640 × 640 的三通道原图占 1,228,800 字节，同尺寸 FP32 输入占 4,915,200 字节；但原图分辨率并不固定，大图上传量也可能超过模型输入量，所以不能把传输收益一概写成四倍。

## 一个线程完成一个输出像素

以输出图中某个有效像素为例，GPU 线程先扣掉补边位置，再计算它在原图中的对应位置。这个位置通常落在四个原像素之间，线程按距离加权求值，这就是**双线性插值**。如果输出像素位于灰边，直接写入 `114/255` 即可。

得到像素值后，同一个线程继续完成舍入、归一化和通道重排，并把三个值写到 R、G、B 平面的对应位置。这个运行在 GPU 上的函数称为 **kernel**。

这里没有分别保存“缩放后的图”“RGB 图”“浮点图”，而是读取原图、连续完成几步计算、直接写最终输入。这就是本篇的**前处理融合**：省去中间图像的生成与读写。融合范围只在前处理内部，TensorRT 和后处理仍各自执行。

在 `preprocess_kernel()` 中，输出位置 `(x,y)` 对应的平面下标是：

```text
index = y × 640 + x
R 写到 input[index]
G 写到 input[640 × 640 + index]
B 写到 input[2 × 640 × 640 + index]
```

插值使用取整后的缩放尺寸做半像素坐标映射，边缘坐标限制在原图范围内。当前 kernel 会先执行 `floorf(value + 0.5F)` 再归一化；OpenCV 的固定点插值仍可能带来一个灰度级的差异。组件测试因此允许归一化后最大误差 `1/255 + 1e-6`，无需缩放时使用更严格的 `1e-7`。这不等于已经证明完整数据集的检测精度相同。

## 后处理为什么还保留 CPU 开关

本项目的 YOLO26 end-to-end engine 输出 `[1,300,6]`：最多 300 行，每行是 `x1,y1,x2,y2,score,class_id`。模型已经完成候选选择，不需要另做 NMS，但仍要检查数值、过滤低分框，再恢复原图坐标。

例如前面的图片缩放比例为 0.5，上边补了 140 行；模型给出的 `y=190` 要恢复成 `(190-140)/0.5=100`。CPU 和 GPU 两条后处理路径都使用这套规则：分数至少 0.25，类别非负且可转成 `int`，坐标裁剪到原图范围，去掉没有面积的框。

设置 `GPU_POST=1` 后，每个 GPU 线程处理一行候选。有效框不能随意争抢输出位置，否则每次结果顺序可能变化。代码先统计每个位置之前共有多少个有效框，再决定写到哪里：

```text
候选有效标记：0 1 0 1 1
累计有效数量：0 1 1 2 3
三个有效框依次写到下标 0、1、2
```

这种累计计算称为**前缀和**。实现使用一个含 512 个线程的线程组（block），前 300 个线程检查候选，其余线程也参与同步。无效候选也必须参与 `__syncthreads()` 组内同步；每轮扫描也必须先读完旧值再写新值，才能保证结果和顺序稳定。

GPU 将有效框压紧到 `GpuResults` 前部，D2H 仍复制整个固定容量结构。CPU 模式下载 7200 字节，GPU 模式在当前布局下下载 7204 字节，并没有变成“只传几个框”。先读 count 再按数量下载会多一次等待，本实现保留一次完整下载。

因此，CPU 原本只处理 300 行时，额外 kernel 和同步未必划算。默认 `GPU_POST=0`；开启后，CPU 的 `unpack()` 只负责把有效结果转成 OpenCV 检测结构，不再重复筛选。两个模式保持相同的 CUDA 前处理、普通主机内存和两次等待，适合在本篇内部观察后处理位置的影响。

## 资源和计时怎样对应到代码

阅读 `main.cpp` 时，可以按 `CudaResources` → `preprocess()` → `infer()` → `postprocess()` / `unpack()` 的顺序跟踪一次请求。

`CudaResources` 保存 stream 和显存缓冲区。stream 是 GPU 工作的有序队列，同一队列中的复制、前处理和推理按提交顺序执行。原图显存只在容量不足时扩容，模型输入输出只在初始化时分配，计时循环复用这些空间；退出时先等待 GPU 完成，再释放资源。

图片仍在普通主机内存中。`cudaMemcpy2DAsync()` 用 `image.step[0]` 表示源图每行跨度，只复制每行有效像素，因此也能处理从大图截取、各行之间有空隙的 ROI。函数名带 `Async` 不代表 CPU 一定立即返回；本版随后也会明确等待 stream 完成。

`preprocess_ms` 包含原图上传、CUDA 前处理和第一次等待。`infer_ms` 在默认模式包含 TensorRT、输出下载和第二次等待；开启 GPU 后处理后，它还包含后处理 kernel。此时 `postprocess_ms` 只剩 CPU 的 `unpack()` 时间，不能据此判断后处理整体快了多少。

**统一比较 `total_ms`：从前处理开始，到 CPU 检测结果可用为止。** 它不含读图、engine 加载、显存分配、预热、画框和文件写入。阶段的 P50 不能相加当作总 P50。

## 相对 01 的结果

统一实验使用 RTX 3060、YOLO26m FP16 / FP32 I/O、640 × 640、batch 1、Release；`bus.jpg` 和 `zidane.jpg` 共 3 轮，每轮每图预热 200 次、测量 1000 次，每版 6000 条请求记录。

| 版本 | 请求 P50（ms） | 相对 01 加速比 |
| --- | ---: | ---: |
| 01 · CPU 前后处理 | 4.547317 | 1.00× |
| 02 · CUDA 前处理、CPU 后处理 | 3.802120 | **1.20×** |

加速比是 `01 的请求 P50 ÷ 02 的请求 P50`。这说明本轮默认 02 的整个请求比 01 更快，不能把 1.20× 全部归因于某个 kernel，也不代表 GPU 后处理模式的成绩。计时包含上传内容与同步位置变化带来的整体影响。

[统一测试报告](../benchmarks/rtx3060-yolo26m-2026-10-15/README.md)保留逐轮数据、环境和 engine 哈希。两张示例图的重复测量用于观察稳态延迟，不代表完整数据集精度。

## 自己运行一次

先按仓库 README 准备依赖与 engine，然后在仓库根目录执行；01 和 02 使用同一 engine、图片与次数：

```bash
cmake --preset release -DCMAKE_CUDA_ARCHITECTURES=native
cmake --build --preset release --target yolo_baseline yolo_cuda_prepost -j
./build/release/src/yolo_baseline weights/yolo26m.fp16.engine images 1000 200
./build/release/src/yolo_cuda_prepost weights/yolo26m.fp16.engine images 1000 200 0
```

结果分别写到 `results/01_baseline/` 和 `results/02_cuda_prepost/cpu_post/`。同一种配置再次运行会覆盖自身结果，多轮比较时需先另存输出。上面的命令用于本地对照；要复现已记录数字，应使用统一报告中相同哈希的 engine。

只观察后处理位置时，可以再执行：

```bash
./build/release/src/yolo_cuda_prepost weights/yolo26m.fp16.engine images 1000 200 1
```

结果写到 `results/02_cuda_prepost/gpu_post/`。一个具体练习是：比较两种模式的检测 CSV 和 `total_ms`，再解释为什么 GPU 模式的 `postprocess_ms` 很小，却不能单凭这一列说它更快。

返回：[笔记目录](README.md) · [01 基准](01_baseline.md)。
