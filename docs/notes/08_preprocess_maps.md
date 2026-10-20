# 08：预计算映射表，复用前处理的坐标计算

01 处理一张图片时，先在 CPU 上缩放、补边、调整通道顺序并归一化，再把浮点输入从主机内存拷到 GPU（H2D）。TensorRT 完成推理后，把输出拷回主机（D2H），CPU 再筛选检测框、还原坐标。一次请求完整走完这些步骤，才处理下一次。

例如，1280 × 720 的图片会缩到 640 × 360，上下各补 140 行灰色像素，得到模型需要的 640 × 640 输入。每次输入尺寸相同，缩放时各个输出像素对应的原图位置也相同。08 利用这个规律，把重复求坐标改成读取事先算好的表。

相对 01，本版还把前处理搬到 CUDA，使用页锁定内存上传原图，并用 CUDA Graph 提交 GPU 工作。**相对 01 的加速比 1.25× 是这套实现的整体结果，不能全部归给映射表。**下面先解释这条处理路径，再看表具体省掉了什么。

## 从 CPU 前处理改为上传原图

01 在 CPU 上生成三个浮点通道。08 先把原图复制到一块可重复使用的页锁定内存，再上传原始 BGR 像素。页锁定内存（Pinned Memory）是 CUDA 可以直接用于异步传输的主机内存；这里的临时输入区称为 staging，每次填充它的 CPU 复制也计入请求延迟。

上传后，一个 CUDA kernel 完成缩放、补边、BGR 转 RGB、通道排列和归一化。kernel 是在 GPU 上并行执行的函数：本版让每个线程负责一个输出位置，直接把结果写到模型的浮点输入中，减少中间图像的生成和遍历。

```text
01：CPU 前处理 → H2D 浮点输入 → TensorRT → D2H → CPU 后处理
08：CPU 复制到 pinned → H2D 原图 → CUDA 查表前处理 → TensorRT → D2H → CPU 后处理
```

08 把中间的 H2D、CUDA 前处理、TensorRT、D2H 捕获为 CUDA Graph。它记录这组操作及依赖，后续用一次重放提交；CPU 仍在重放前填入当前像素，在 GPU 工作结束后完成后处理。Graph 不会省掉这些操作本身，也不包含 CPU 复制。

## 同一列的坐标，只算一次

缩放通常要把一个输出位置映射回原图，再取附近四个像素加权求值，这就是双线性插值。横坐标映射使用下面的半像素规则，其中 `resize_scale_x` 是原图宽度除以实际缩放宽度：

```text
source_x = float((double(x - pad_x) + 0.5) × resize_scale_x - 0.5)
```

仍以宽度 1280 缩到 640 为例，输出第 100 列对应原图横坐标 200.5，需要读取第 200、201 列，横向权重各为 0.5。输出第 100 列上的每一行都重复使用这一组横向结果；纵向同理。

本版因此准备两张长度为 640 的表：`x_map` 保存每列的左右邻点和权重，`y_map` 保存每行的上下邻点和权重。每项是一个 `AxisMap`，补边位置用 `lower == -1` 标记，读到它就写灰色 114。

```cpp
struct AxisMap {
    int lower;       // 较小的源坐标；-1 表示补边
    int upper;       // 相邻的较大源坐标
    float fraction;  // 距离 lower 的小数部分
};
```

两张表共约 15 KiB。满幅有效输入原本每帧要做约 `2 × 640 × 640` 次轴向坐标计算，现在准备阶段最多生成 1280 项，每帧按行列复用。像素内容仍逐帧读取，四邻点插值、通道转换和完整张量写出仍需执行；查表也增加了内存读取。

## 复用计算结果，也要保留计算规则

查表只改变坐标结果的复用方式。`make_axis_map()` 与逐像素 CUDA 路径都先用 double 计算完整坐标，再转成 float，随后钳制边界、向下取整并取得权重。若提前把比例改成 float，极宽图片上的舍入就可能改变。

表在 GPU 上生成，沿用原表达式，避免 CPU 与 GPU 的算术处理差异。插值结果也保留 `floorf(value + 0.5F)` 后再除以 255 的顺序，不开启 `--use_fast_math` 来换取另一套精度行为。

这里的精确对照是“查表 CUDA”与“逐像素计算 CUDA”。01 的 OpenCV uint8 缩放采用另一套舍入实现，两条 CUDA 路径与它仍可能相差一个灰度级，不能把查表路径内部的逐位一致理解成与 01 所有输入都完全一致。

[前处理测试](../../tests/preprocess_maps_test.cpp)覆盖普通尺寸、单像素、奇数补边、极端长宽及非连续布局。ROI 是原图中的一块视图，它的相邻两行可能有间隙；`pitch` 表示两行起始地址的字节距离，kernel 必须用它寻址。测试还在 Graph 重放之间改变像素，检查表复用时仍读取新内容。

## 尺寸和地址决定表能用多久

表仅依赖图片尺寸和缩放参数。同尺寸的不同画面可以共用一份表；换成不同尺寸后必须重建。本实现按图片准备资源，每张图片都重新生成表并捕获 Graph，没有额外做跨图片缓存。

```text
每张图片准备：结束旧 GPU 工作 → 销毁旧 Graph → 按需扩容输入
             → 填入像素 → 生成映射表 → 普通推理一次并同步 → 捕获 Graph
每次请求：   更新 pinned 像素 → 重放 Graph → 等待完成 → CPU 后处理
```

捕获前普通执行一次，是让 TensorRT 完成首次执行所需的准备。表生成、资源准备及 Graph 创建归入 `setup_ms`，不在稳态请求内；同尺寸只推理一次时，需要把这些成本加回来。

Graph 会记住缓冲区地址。执行尚未完成时不能改表或复用其输入；释放资源时先等 stream（GPU 任务队列）完成，再销毁 Graph，最后释放内存。表指针本身不携带尺寸，调用者还必须保证当前图片与表匹配。

阅读源码可从 [main.cpp](../../src/08_preprocess_maps/main.cpp) 的 `stage_image()`、`submit()` 和 `capture_graph()` 跟进完整请求，再到 [preprocess.cu](../../src/08_preprocess_maps/preprocess.cu) 看 `make_axis_map()`、`build_preprocess_maps()` 和 `preprocess_maps_cuda()`。

## 相对 01 的性能

统一测试条件：RTX 3060、YOLO26m FP16 / FP32 输入输出、640 × 640、Batch 1、Release；`bus.jpg` 和 `zidane.jpg` 各跑 3 轮，每轮每图预热 200 次、测量 1000 次，每版共 6000 条记录。

| 版本 | 请求 P50（ms） | 相对 01 加速比 |
| --- | ---: | ---: |
| 01 · CPU 前后处理 | 4.547317 | 1.00× |
| 08 · MAPS=1 | 3.627002 | 1.25× |

P50 是全部请求样本的中位延迟，加速比为 `01 P50 ÷ 本版 P50`。本版总延迟从 CPU staging 开始，到 CPU 后处理结束，包含传输、推理与等待；解码、setup、预热、画框和文件读写不计入。完整条件和数据见[统一基准报告](../benchmarks/rtx3060-yolo26m-2026-10-15/README.md)。

这说明本轮 08 的完整路径比 01 延迟更低。若只想观察映射表的作用，应在本程序内比较 `MAPS=0/1`；两组都保留 CUDA 前处理、Pinned Memory 和 Graph。旧 YOLO26n 批次的[开关实验与前处理微基准](../benchmarks/rtx3060-series-10-12-2026-09-14/README.md)单独存档，局部 kernel 的倍数不能代替请求加速比。

## 运行与观察

先按[项目首页](../../README.zh-CN.md#快速开始)准备依赖与 engine，再在仓库根目录执行。两次运行使用同一份 engine、相同图片和次数；下面用于本地对照，报告所用 engine 的确切哈希见统一基准报告。

```bash
cmake --preset release -DCMAKE_CUDA_ARCHITECTURES=native
cmake --build --preset release --target yolo_baseline yolo_preprocess_maps -j

ENGINE=weights/yolo26m.fp16.engine
./build/release/src/yolo_baseline "$ENGINE" images 1000 200
./build/release/src/yolo_preprocess_maps "$ENGINE" images 1000 200 1
```

08 的参数为 `ENGINE IMAGE_FOLDER [ITERATIONS=200] [WARMUP=50] [MAPS=1]`。两版分别写入 `results/01_baseline/`、`results/08_preprocess_maps/`；重跑会覆盖各自同名结果，换配置前先另存 CSV 和图片。以上是一轮运行，报告中的 6000 条来自三轮合并。

动手观察时，先保存 `MAPS=1` 的结果，再把末尾参数改成 `0` 重跑。比较 `timings.csv` 中的 `gpu_ms` 与 `total_ms`：前者覆盖整段 GPU 工作，后者还包含 CPU 复制和后处理。尝试根据这两列解释，为什么少算一些前处理坐标，未必让整次请求按相同比例缩短。

返回：[笔记目录](README.md) · [01 基准](01_baseline.md)。
