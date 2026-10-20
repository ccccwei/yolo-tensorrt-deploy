# 09：注册原图内存，省掉上传前的 CPU 复制

01 的一次请求从 CPU 前处理开始：缩放、补边、交换通道并归一化，得到模型需要的浮点输入。随后把输入从主机拷到 GPU（H2D），执行 TensorRT，再把输出拷回主机（D2H），由 CPU 筛选检测框并还原坐标。

09 把前处理放到 GPU，因此可以先上传原始 BGR 图片。为了让上传使用异步传输，常见做法是先把普通图片复制到一块页锁定内存，再上传它。以 1280 × 720 的 BGR 图为例，每次请求都要额外在 CPU 上复制约 2.76 MB，随后才开始 H2D。

如果原图已经位于一块可以长期复用的内存中，就可以把这块内存直接注册给 CUDA，省掉那次主机内部复制。**本版相对 01 的加速比 1.29× 还包含 CUDA 前处理、Graph 提交等变化，不是一次注册操作单独带来的倍数。**

## 先让原图本身适合异步上传

页锁定内存（Pinned Memory）是在传输期间保持物理页面稳定的主机内存，CUDA 可以用它进行异步传输。程序有两种获得方式：申请一块新的页锁定缓冲区，或用 `cudaHostRegister()` 注册一块已存在的内存。

新申请的缓冲区没有当前图片，需要每次先复制像素；本版称这一步为 staging。注册已有原图则保持像素地址不变，让 H2D 直接以它为源。注册放在图片准备阶段，不在每次请求里重复执行。

```text
01：CPU 前处理 → H2D 浮点输入 → TensorRT → D2H → CPU 后处理

09 REGISTERED=0：普通原图 → CPU memcpy 到 pinned 缓冲区 → H2D 原图
09 REGISTERED=1：已注册原图 ─────────────────────────→ H2D 原图
随后均执行：     CUDA 前处理 → TensorRT → D2H → CPU 后处理
```

CUDA 前处理用一个 kernel（GPU 并行函数）完成缩放、灰色补边、BGR 转 RGB、归一化以及按通道排列。模型与 01 相同，CPU 后处理仍解释 `[1,300,6]` 的检测输出；注册不会改变模型计算或置信度阈值。

两种 09 模式都使用 CUDA Graph：先记录 H2D、CUDA 前处理、TensorRT、D2H 及依赖，再通过一次重放提交这组操作。注册模式只是把其中 H2D 的源地址换成原图；Graph 每次重放仍读取该地址里的当前像素并上传。

因此，本版省掉的是 CPU 上的一次额外复制，**不是 GPU zero-copy**。前处理仍读取显存中的原图，没有直接在主机内存上计算，也没有把图片预上传一次后一直复用 GPU 像素。

## 一块图片内存，不一定紧密排列

`cv::Mat` 可以是一张完整图片，也可以是原图中的一个 ROI（局部视图）。ROI 只取部分列时，两行有效像素之间仍可能隔着原图剩余区域。`step` 就是相邻两行起始地址的字节距离，不能一律当成 `width × 3`。

例如，一块 100 × 2 的 BGR ROI，每行有效像素为 300 字节，但 `step=3000`。两行总共只有 600 字节像素，从首行起点到末行有效数据结束却跨越 3300 字节；只注册 600 字节会漏掉第二行。

```text
注册跨度 = (rows - 1) × step + cols × 3
上传宽度 = cols × 3
上传高度 = rows
```

`prepare_image()` 先检查跨度计算是否溢出、是否超出 OpenCV 记录的可用范围，再执行注册。`upload_image()` 使用 `cudaMemcpy2DAsync()`，以真实 `step` 逐行取有效像素，GPU 端仍按紧凑行宽存放。注册范围可以包含行间空隙，上传不必复制这些空隙。

程序逐张图片注册、释放，不同时注册重叠 ROI。注册失败会输出 CUDA 错误并退出，不会无提示地切回 staging；否则运行结果就无法说明实际走了哪条路径。

## Graph 记住地址，也带来了寿命约束

Graph 捕获时记住源地址、行步长、图片尺寸和前处理参数。同一地址里可以换成新画面，但必须先等上一次上传完成；地址或尺寸变了，就需要重新准备。本实现选择按图片重新捕获 Graph。

```text
每张图片准备：等旧 GPU 工作完成 → 销毁旧 Graph → 解除旧注册
             → 注册当前原图 → 普通推理一次并同步 → 捕获新 Graph
每次请求：   校验图片、计算缩放参数 → 重放 Graph → 等待完成 → CPU 后处理
图片结束：   等待 stream → 销毁 Graph → 解除注册 → 释放原图引用
```

stream 是 GPU 的任务队列。末尾等待确保 D2H 完成后 CPU 才读取检测结果，也让下一次请求可以安全更新输入。捕获前的普通推理用于完成 TensorRT 首次执行的准备。

资源对象保留一份 `cv::Mat` 浅拷贝：它增加 OpenCV 所有像素内存的引用计数，不复制像素。即使局部变量提前离开作用域，内存也要等同步、销毁 Graph、解除注册后才释放。

若 `cv::Mat` 包装的是相机 SDK 提供的外部 buffer，浅拷贝不能替它取得所有权。调用者仍要保证相机不会提前回收或覆盖正在上传的那块内存。这是把示例接入采集程序时必须补上的约束。

## 注册成本需要被多少次请求分摊

适合本方案的输入是地址稳定、会被重复填充的缓冲区，例如一组长期使用的相机帧缓冲。一次注册后可以处理多帧，最后统一解除注册。当前示例为每张已解码图片注册一次，再重复推理，模拟的是这种地址复用条件。

若每张图片都分配新内存且只推理一次，注册成本也要算进每次请求。`setup.csv` 单独记录 `setup_ms` 和 `teardown_ms`：前者包含缓冲区准备、注册、首次普通推理与 Graph 创建，后者包含同步、销毁 Graph 与取消注册，不能把它们当成纯 `cudaHostRegister()` 耗时。

要估算少量请求的成本，可将“正式运行墙钟时间 + setup + teardown”除以请求数。这仍不含 engine 加载、解码和输出文件，不能当作整个进程的冷启动时间。换分配器、采集接口或图片尺寸后，需要重新观察摊销结果。

阅读源码可从 [main.cpp](../../src/09_registered_input/main.cpp) 的 `prepare_image()` 看注册范围，到 `stage_image()` 看省掉的复制，再沿 `upload_image()`、`capture_graph()` 和 `clear_source()` 跟进地址的使用与释放。[preprocess.cu](../../src/09_registered_input/preprocess.cu)负责 GPU 上的像素转换。

[注册输入测试](../../tests/registered_input_test.cpp)比较注册与 staging 路径的浮点输入，覆盖非连续 ROI、改变像素后的 Graph 重放以及正常和异常退出时的清理。两种 09 模式的输入应逐字节一致；这个对照不代表 CUDA 插值与 01 的 OpenCV 缩放在所有图片上逐位一致，更不能代替完整数据集精度评估。

## 相对 01 的性能

统一测试条件：RTX 3060、YOLO26m FP16 / FP32 输入输出、640 × 640、Batch 1、Release；`bus.jpg` 和 `zidane.jpg` 各跑 3 轮，每轮每图预热 200 次、测量 1000 次，每版共 6000 条记录。

| 版本 | 请求 P50（ms） | 相对 01 加速比 |
| --- | ---: | ---: |
| 01 · CPU 前后处理 | 4.547317 | 1.00× |
| 09 · REGISTERED=1 | 3.519440 | 1.29× |

P50 是全部请求样本的中位延迟，加速比为 `01 P50 ÷ 本版 P50`。09 从校验当前输入和计算缩放参数开始，到 CPU 后处理完成，计入 H2D、GPU 前处理、TensorRT、D2H 和等待；解码、setup、teardown、预热、画框及文件读写在外。完整条件和原始数据见[统一基准报告](../benchmarks/rtx3060-yolo26m-2026-10-15/README.md)。

若只看注册减少了多少 CPU 复制，应在 09 内比较 `REGISTERED=0/1`，两组都开启 Graph 并使用同一 CUDA 前处理。旧 YOLO26n 的[开关实验与少量请求摊销记录](../benchmarks/rtx3060-series-10-12-2026-09-14/README.md)保留在历史报告中，不与本表拼接计算。

## 运行与观察

先按[项目首页](../../README.zh-CN.md#快速开始)准备依赖与 engine，再在仓库根目录执行。两次运行使用同一份 engine、相同图片和次数；下面用于本地对照，报告所用 engine 的确切哈希见统一基准报告。

```bash
cmake --preset release -DCMAKE_CUDA_ARCHITECTURES=native
cmake --build --preset release --target yolo_baseline yolo_registered_input -j

ENGINE=weights/yolo26m.fp16.engine
./build/release/src/yolo_baseline "$ENGINE" images 1000 200
./build/release/src/yolo_registered_input "$ENGINE" images 1000 200 1
```

09 的参数为 `ENGINE IMAGE_FOLDER [ITERATIONS=200] [WARMUP=50] [REGISTERED=1]`。两版分别写入 `results/01_baseline/`、`results/09_registered_input/`；重跑会覆盖各自同名结果，换配置前先另存 CSV 和图片。以上是一轮运行，报告中的 6000 条来自三轮合并。

动手观察时，保存注册模式的结果，再把末尾参数改成 `0` 重跑，重点比较 `staging_ms` 和 `gpu_ms`。前者应体现主机复制的变化，后者依然包含 H2D。结合这两列复述一次请求的路线，确认自己能指出省掉的是哪一段复制，以及仍然存在的是哪一段。

返回：[笔记目录](README.md) · [01 基准](01_baseline.md)。
