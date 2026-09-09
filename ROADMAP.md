# YOLO TensorRT Deploy：从简洁部署到极致速度

> 从一份简洁可读的部署代码出发，每篇解释一类优化，用对照实验记录收益与代价。

当前状态：**系列第 01 篇的基础代码已可运行，后续优化与正式性能基线待实现**。已有 ONNX 导出、FP16 Engine 构建、C++ 文件夹推理、CPU 前后处理和逐图分阶段计时；尚未建立预热、重复测试和精度回归组成的完整 Benchmark。

以[第 10 节的系列目录](#series)作为近期内容顺序。下文 M0–M10 是长期技术选题与验收参考，不代表已经完成，也不要求每篇都扩展为框架或推理库。当前系列先固定 YOLO26n、640 × 640、Batch 1、TensorRT 8.6.1 / CUDA 11.8，跨版本迁移另开对照，不与优化收益混算。

返回项目首页：[README.md](README.md)

## 1. 项目定位

系列面向有 C++ 基础的学生、开发者和专业部署工程师：基础篇把流程写清楚，进阶篇把优化原理、代码差异和测量边界写清楚。保持轻量、显式，只有实际优化或复用需要时才增加封装。

早期的推理库与 API 分层设想见 [ADR-0001：目标用户与 API 定位](docs/adr/0001-target-users-and-api-positioning.md)，作为长期工程化参考，不作为当前基础示例必须实现的接口要求。

这不是一个“能跑起来就结束”的 TensorRT Demo，而是一个长期维护的 YOLO 性能工程项目：

- 每次只优化一个主要变量；
- 每个性能结论都能由代码、配置和原始数据复现；
- 同时约束速度、精度、显存和稳定性；
- 先完成 NVIDIA GPU / TensorRT 主线，再复用同一套接口与评测协议扩展 CPU；
- 最终沉淀为可复用的 C++ 推理库、Benchmark 工具和一套公开文章。

项目不会脱离条件宣称“全网最快”。这里的“最快”始终表示：

> 在固定硬件、软件栈、模型、输入尺寸、精度、Batch、前后处理语义和测试协议下，得到当前可复现的最优结果。

### 默认优化目标

首条主线优先追求 **Batch=1 的端到端最低延迟**，并分别维护三种性能配置：

| 配置 | 主要目标 | 典型场景 | 核心指标 |
| --- | --- | --- | --- |
| `latency_b1` | 单请求最低延迟 | 摄像头、机器人、实时交互 | P50 / P90 / P99 延迟 |
| `throughput` | 单卡最大吞吐 | 离线批处理、多路视频 | FPS、GPU 利用率 |
| `service_slo` | 满足尾延迟约束下的最大并发 | 在线服务 | QPS@P99 SLO、排队时间 |

## 2. 首版边界：先做窄，再做深

### v0.1 要完成

- Linux x86_64 + NVIDIA GPU；
- 一个固定的 YOLO Detect 参考模型；
- 固定输入，例如 `1x3x640x640`，Batch=1；
- 锁定一个 TensorRT 主版本和完整依赖矩阵，不在首版同时兼容多个大版本；
- PyTorch 参考输出、ONNX 导出与正确性校验；
- TensorRT FP32 / FP16 Engine 构建、保存和加载；
- C++17 单图推理 CLI；
- 分阶段 Benchmark：预处理、H2D、推理、后处理、D2H、端到端；
- 首版可先使用 CPU 后处理，下一阶段再迁移到 GPU；
- 检测结果 JSON 和可视化图片输出；
- 完整记录环境、命令、权重哈希和原始结果。

### v0.1 暂不做

- 训练、蒸馏和模型结构设计；
- 一开始兼容所有 YOLO 仓库和版本；
- 分割、姿态、旋转框等任务；
- 动态 Shape、多 Batch、多 GPU；
- INT8、自定义 TensorRT Plugin；
- 视频解码、服务化、Windows 和 CPU 后端。

这些能力会逐步加入，但不应阻塞第一条“从模型到检测结果”的完整链路。

### TensorRT 版本策略

TensorRT 10 与 11 之间不只是函数签名变化，精度与 Plugin 工作流也有结构性差异，因此不能假设一个轻量 API 宏就能完全兼容：

| 路线 | 精度控制 | 量化 | Plugin |
| --- | --- | --- | --- |
| TensorRT 11.x 主线 | Strongly Typed 网络；通过显式类型、ModelOpt AutoCast 等准备模型 | 显式 Q/DQ / ModelOpt，不能依赖隐式 INT8 校准 | 使用 V3 接口；旧 V2 Plugin 和部分旧内置 NMS Plugin 已移除 |
| TensorRT 8.x / 10.x 兼容线 | 可使用对应版本的 Builder Flag 或显式类型工作流 | 旧版 Calibrator 仅留在兼容实现中 | 按目标版本隔离 V2 / V3 实现 |

对新项目，建议以 **TensorRT 11.x** 作为长期主线；如果必须复用 TensorRT 8.6 / CUDA 11.8 等旧环境，则先明确建立 legacy 兼容线，不能把两条路线生成的 Engine 和 Benchmark 混在一起。正式进入 M2 前，在 `docs/compatibility.md` 中钉死首版的 GPU、驱动、CUDA、TensorRT、ONNX Opset 和编译器版本。

## 3. 先统一：我们到底在测什么

完整数据流如下：

```text
编码图像/视频帧
      │
      ▼
   图像解码 ──► CPU HWC 图像
                    │
                    ▼
        resize / letterbox / normalize / layout
                    │
                    ▼
                  H2D
                    │
                    ▼
             TensorRT enqueue
                    │
                    ▼
             GPU 原始输出 Tensor
                    │
          ┌─────────┴─────────┐
          │                   │
          ▼                   ▼
 v0.1：D2H raw        高速路径：GPU decode / NMS
          │                   │
          ▼                   ▼
 CPU decode / NMS        D2H compact results
          │                   │
          └─────────┬─────────┘
                    ▼
          坐标恢复 / 结构化输出
```

项目必须区分以下测试，禁止把不同边界的数字放在同一张排行榜里：

| Benchmark | 起点 → 终点 | 用途 |
| --- | --- | --- |
| `engine_only` | GPU 输入 Tensor → GPU 原始输出 | 观察 TensorRT Engine 理论性能 |
| `with_transfer` | Host Tensor → Host 输出 | 观察 H2D / D2H 影响 |
| `pipeline_e2e` | 已解码 CPU 图像 → 检测结果 | **项目默认端到端指标** |
| `file_e2e` | 编码图片 → 检测结果 | 观察图片解码成本 |
| `video_steady_state` | 连续帧输入 → 连续结果 | 观察流水线吞吐、丢帧和抖动 |
| `server_load` | 请求进入 → 响应返回 | 观察排队、并发和 P99 SLO |

“1.2 ms / 833 FPS”本身没有意义。每个公开数字都必须说明：

- 是否包含图像解码；
- 是否包含预处理、H2D、后处理、NMS 和 D2H；
- 输入是随机 Tensor 还是真实图像；
- 使用同步、异步、CUDA Graph 还是多流；
- 是单请求延迟、稳态吞吐，还是二者的混合值。

## 4. 技术路线总览

| 里程碑 | 目标 | 主要交付物 | 验收重点 |
| --- | --- | --- | --- |
| M0 / v0.0.0 | 建立性能契约 | Benchmark 协议、环境采集、结果 Schema | 在声明环境与容差内复现同一结论 |
| M1 / v0.0.1 | 正确性与 ONNX 基线 | 导出、校验、golden cases、mAP 基线 | PyTorch / ONNX 语义对齐 |
| M2 / v0.0.2 | TensorRT 工具基线 | FP32 / FP16 Engine、`trtexec` 脚本 | 得到 Engine 理论上限和层级 Profile |
| M3 / v0.1.0 | 最小 C++ Runtime | Engine loader、RAII Buffer、推理 CLI | C++ Engine-only 接近工具基线 |
| M4 / v0.2 | 全 GPU 前后处理 | CUDA 预处理、Decode、NMS | 热路径无逐帧分配，只回传紧凑结果 |
| M5 / v0.3 | 极致低延迟 | Pinned Memory、异步流、Buffer Pool、CUDA Graph | Nsight 验证无意外同步和空洞 |
| M6 / v0.4 | 动态与并发 | Shape Bucket、Context Pool、多流调度 | 动态场景下 P99 可控 |
| M7 / v0.5 | INT8 与混合精度 | 校准、Q/DQ、精度回归、Pareto 数据 | 加速收益大于精度与维护成本 |
| M8 / v0.6 | 真实业务链路 | 视频流水线、背压、服务压测 | 报告 load-to-load P99、QPS、丢帧率 |
| M9 / v0.7 | CPU 后端 | ONNX Runtime / OpenVINO、线程与 NUMA 调优 | 共用输入输出语义和评测协议 |
| M10 / v1.0 | 可生产使用 | 稳定 API、容器、CI、兼容矩阵 | 可安装、可验证、可回归、可观测 |

## 5. 一步一步实现

### M0：先建立性能契约

在写优化代码之前先完成：

- 固定一个参考权重、输入尺寸、类别表、置信度阈值、IoU 阈值和 `max_det`；
- 记录模型来源、许可证、权重 SHA-256 和 ONNX SHA-256；
- 编写环境采集脚本，记录 GPU、驱动、CUDA、TensorRT、编译器、功耗模式和 Git commit；
- 定义统一的 Benchmark JSON Schema；
- 同时保留真实图像输入与 synthetic input 测试；
- 建立小规模快速回归集和完整精度评测集。

完成标准：任何一个结果都能回答“在哪台机器、用什么代码、什么命令、什么输入、测了多少次”。

### M1：建立正确性基线

1. 使用原始 PyTorch 实现生成参考输出；
2. 导出 ONNX，运行 ONNX Checker，并用 ONNX Runtime 执行参考推理；
3. 对比 raw head Tensor，而不是只看画框结果；
4. 对固定样本对比框、分数、类别、NMS 后结果；
5. 在完整验证集上记录 mAP50-95；
6. 保存少量合法可分发的 golden input / output 用于 CI。

必须固定 letterbox 的缩放、Padding、颜色通道、归一化、坐标恢复和 NMS 语义。很多“部署精度下降”并不是 TensorRT 造成的，而是前后处理不一致。

### M2：用 `trtexec` 找到 TensorRT 工具基线

1. 锁定 TensorRT 大版本后，分别构建 FP32 参考 Engine 和对应版本工作流下的 FP16 静态 Shape Engine；
2. 保存完整构建命令、构建日志、Engine Inspector 信息和层级 Profile；
3. 固定一条 **CUDA Graph off、Data Transfer off** 基线，供 M3 的普通 enqueue Runtime 验收；
4. 固定一条 **CUDA Graph on、Data Transfer off** 基线，作为 M5 的低延迟上限参考；
5. 使用明确的数据传输开关再测一条 H2D / D2H 路径；
6. 每组重复多轮，观察抖动、功耗与温度；
7. 将 `trtexec --help`、完整参数和 TensorRT 版本一并归档，避免默认启用 CUDA Graph、默认排除数据传输等行为造成误比。

这里的结果不是最终应用性能，而是 C++ Runtime 应努力接近的“工具上限”。

### M3：写最小 C++ TensorRT Runtime

第一版只追求结构清晰、结果正确和计时可信：

- Builder 与 Runtime 分离，部署进程只加载 Engine；
- 使用 RAII 管理 Runtime、Engine、Context、Stream 和显存；
- 所有输入输出 Buffer 启动时一次性分配；
- 轻量兼容层只隔离 Runtime API 差异，精度与 Plugin 工作流按 TensorRT 大版本分别实现；
- 易用同步 API 与高性能异步 API 分离；
- Benchmark 热路径禁止日志、文件 IO、图片保存和逐帧对象分配；
- 用 CUDA Event 测 GPU 阶段，用单调时钟测真正端到端。

初始工程目标：相同 Engine、Graph off、Transfer off 和同步设置下，C++ `engine_only` 与对应 `trtexec` 基线的差距控制在约 5% 内。它是调优目标，不是跨硬件保证；Graph on 的工具结果留到 M5 再比较。

### M4：把预处理和后处理搬到 GPU

按 Profile 结果依次处理：

1. 融合 resize / letterbox / normalize / HWC→CHW；
2. 预处理直接写入 TensorRT 输入 Buffer，减少中间 Tensor；
3. 将 decode、阈值过滤和 NMS 移到 GPU；
4. 在 GPU 端压缩结果，只把 `num_dets + boxes + scores + classes` 回传 Host；
5. 对稀疏目标和密集目标分别测试后处理耗时；
6. 同时保留 `raw_head` 和 `end_to_end` 两种输出，方便正确性分析和公平对比。

不要因为“Plugin 看起来更专业”就先写 Plugin。只有 Profile 证明普通 CUDA Kernel 或现有算子无法满足需求时，才承担自定义 Plugin 的版本兼容成本。

### M5：压榨 Batch=1 低延迟

- 固定 Shape、固定 Buffer 地址、固定执行路径；
- 使用 Pinned Host Memory 与 `cudaMemcpyAsync`；
- 使用非默认 CUDA Stream；
- 使用双缓冲或小型 Buffer Pool；
- 清除每帧 `cudaMalloc/cudaFree`、Context 创建和 Stream 创建；
- 清除不必要的 `cudaDeviceSynchronize`；
- 对稳定执行路径捕获并复用 CUDA Graph；
- 使用 Nsight Systems 查看 CPU 提交、Memcpy、Kernel 和同步空洞；
- 只有在单个 Kernel 成为证据明确的瓶颈时，再使用 Nsight Compute 深挖。

必须分别报告首帧、冷启动和稳态性能。CUDA Graph 的收益也必须提供 on/off 对照。

### M6：从最快单图扩展到动态 Shape 和并发

- 优先使用有限的 Shape Bucket，例如 320 / 480 / 640，而不是一个跨度巨大的 Profile；
- 每个 Profile 明确 `min / opt / max`，让真实输入尽量靠近 `opt`；
- 每个并发执行流使用独立 Execution Context；
- 建立 Context / Stream / Buffer Pool；
- 动态 Shape 下按执行状态维护 CUDA Graph Cache；
- 分别测 Shape 切换、并发数、Batch 和 P99 抖动；
- 不盲目增加多流，只有吞吐提升且尾延迟可接受时才保留。

### M7：INT8 与混合精度

1. 使用和真实输入分布一致的代表性校准集；
2. TensorRT 11.x 主线使用显式 Q/DQ / ModelOpt；旧 Calibrator 只存在于 8.x / 10.x 兼容线；
3. 量化产物与缓存必须绑定模型哈希、预处理、输入 Shape、TensorRT 大版本和软件栈；
4. 对敏感层允许回退到 FP16；
5. 重新跑完整 mAP 与逐类别指标；
6. 绘制 FP32 / FP16 / INT8 的速度—精度 Pareto 图。

可先把 FP16 `ΔmAP50-95 ≤ 0.1 AP`、INT8 `ΔmAP50-95 ≤ 0.5 AP` 作为默认回归门槛，但门槛必须可配置，并由实际业务要求决定。

### M8：进入真实业务链路

- 多路视频解码与推理流水线；
- 可选 NVDEC / 硬件解码，单独报告是否计入性能；
- 背压、队列上限、超时、丢帧策略；
- Engine 预热、Context 池和优雅关闭；
- 自研 Runtime 与 Triton 等服务方案做相同口径对比；
- 记录 QPS、P50 / P99、排队时间、GPU 显存、功耗和丢帧率。

### M9：CPU 极速部署支线

CPU 后端复用同一模型契约、前后处理语义、公共 API 和 Benchmark Schema：

1. 以 ONNX Runtime CPU Execution Provider 建立跨平台基线；
2. 以 OpenVINO 建立 Intel CPU 优化主线；
3. 分别测试 latency / throughput 模式；
4. 系统调节 intra-op、inter-op、线程亲和性和物理核数量；
5. 处理 NUMA、内存带宽和嵌套线程池问题；
6. 进行 INT8 静态量化与完整精度回归；
7. ARM 设备再按目标平台评估 ONNX Runtime、OpenVINO 或 NCNN。

CPU 与 GPU 的结果按硬件分组展示，不做没有约束条件的横向 FPS 比较。

## 6. Benchmark 协议

### 6.1 测试流程

- 预热至少 3 秒或 200 次，以较晚结束者为准；
- 稳态测试建议 30～60 秒，且不少于 1000 次；
- 独立重复 5 轮；
- 报告 P50 / P90 / P95 / P99、均值、最小值和轮间变异系数；
- 同机重复的初始稳定性目标为 `CV ≤ 3%`；跨同型号机器的差异单独报告，不承诺数值完全一致；
- 冷启动、Engine 反序列化、首次 enqueue 单独统计；
- GPU 分阶段计时使用 CUDA Event；
- 真正端到端使用 `std::chrono::steady_clock`；
- 异步调用必须在正确边界同步，禁止把“提交耗时”当作“执行耗时”；
- 测试期间记录温度、时钟、功耗和是否发生降频；
- 吞吐与单请求延迟单独测试，不能简单用 `1000 / 平均延迟` 代替并发吞吐。

### 6.2 正确性护栏

- 固定预处理、置信度阈值、IoU 阈值和 `max_det`；
- 对齐 raw Tensor 时记录绝对误差、相对误差和异常值；
- 对齐检测结果时处理框顺序和浮点并列情况；
- 每次图优化、精度切换、Kernel 或 Plugin 修改后运行完整回归；
- 性能结果必须与对应的 mAP 结果绑定；
- 不允许通过偷偷缩小输入、改变 NMS 或减少类别获得“免费加速”。

### 6.3 每条结果至少记录

```text
run_id, timestamp, git_commit, command,
benchmark_scope, input_source, output_mode,
model, model_hash, onnx_hash, input_shape, shape_profile, batch, precision, backend,
gpu, compute_capability, driver, cuda, cudnn, tensorrt, power_mode,
cuda_graph, sync_mode, stream_count, concurrency,
postprocess_device, nms_impl, conf_threshold, iou_threshold, max_det,
preprocess_ms, h2d_ms, enqueue_ms, gpu_compute_ms, postprocess_ms, d2h_ms,
e2e_p50_ms, e2e_p90_ms, e2e_p99_ms, throughput_fps,
map_50_95, gpu_memory_mb, host_memory_mb, power_w, temperature_c,
engine_build_s, engine_size_mb, deserialize_ms, first_infer_ms,
warmup, iterations, repeats
```

### 6.4 结果表模板

每张结果表只放同一个 `benchmark_scope` 的数据，其他配置差异必须作为列展示或拆表：

| 版本 | 后端 | 精度 | Shape / Batch | Pre | H2D | Enqueue | GPU Compute | Post + D2H | E2E P50 | E2E P99 | FPS | mAP50-95 |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| M1 | PyTorch | FP32 | 640 / 1 | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| M2 | TensorRT | FP32 | 640 / 1 | — | — | TBD | TBD | — | — | — | TBD | TBD |
| M2 | TensorRT | FP16 | 640 / 1 | — | — | TBD | TBD | — | — | — | TBD | TBD |
| M3 | C++ TensorRT | FP16 | 640 / 1 | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| M5 | C++ TRT + CUDA | FP16 | 640 / 1 | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| M7 | C++ TRT + CUDA | INT8 | 640 / 1 | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |

## 7. 长期工程化架构参考

以下目录仅在后续确有库化需求时考虑，不是当前目录，也不是系列开篇的搭建任务。基础篇保留单文件实现，后续每篇只增加该实验需要的文件；篇章版本通过明确的 commit 或发布后的 Tag 复现。

```text
yolo_deploy/
├── README.md
├── LICENSE
├── CHANGELOG.md
├── CMakeLists.txt
├── CMakePresets.json
├── pyproject.toml
├── include/yolo_deploy/         # 稳定公共 C++ API，不暴露 TRT 类型
│   ├── detector.hpp
│   ├── backend.hpp
│   ├── tensor.hpp
│   └── types.hpp
├── src/
│   ├── core/                    # 流程编排
│   ├── models/yolo/             # YOLO 输出适配与元数据
│   ├── backends/
│   │   ├── tensorrt/            # Builder、Runtime、Context、Engine Cache
│   │   └── cpu/                 # 后续接 ORT / OpenVINO
│   ├── kernels/cuda/            # Preprocess、Decode、NMS
│   └── utils/                   # Profile、图像 IO、可视化
├── python/yolo_deploy/
│   ├── export.py
│   ├── reference.py
│   ├── validate.py
│   └── benchmark.py
├── configs/
│   ├── models/
│   └── benchmarks/
├── tests/
│   ├── unit/
│   ├── cuda/
│   ├── integration/
│   └── data/
├── benchmarks/
│   ├── schema.json
│   ├── scripts/
│   └── results/                 # 提交 JSON / CSV，不提交 Engine
├── docs/
│   ├── architecture.md
│   ├── benchmark_protocol.md
│   ├── compatibility.md
│   ├── adr/
│   └── zhihu/
├── examples/
├── docker/
├── tools/
└── artifacts/                   # 模型、Engine、数据集；默认 Git Ignore
```

核心边界保持为：

```text
Input → Preprocess → Backend::enqueue → YOLO Decode/NMS → Detections
```

约束如下：

- ONNX 是统一中间表示，C++ Runtime 不依赖训练框架；
- `backend.hpp` 不暴露 TensorRT 类型，CPU 后端才能自然接入；
- YOLO 输出解析属于 `models/yolo/`，不能散落在 TensorRT Runtime 中；
- Builder 与 Runtime 分离，Engine 不在推理进程中重复构建；
- 公共 API 同时提供易用同步入口与面向设备内存的异步入口。

概念 API：

```cpp
// 易用层：负责完整 pipeline
std::vector<Detection> Detector::infer(const ImageView& image);

// 高性能层：调用方管理设备 Tensor 和 Stream
void Detector::enqueue(
    const DeviceTensorView& input,
    DeviceTensorView& output,
    StreamHandle stream);
```

## 8. TensorRT Engine 与缓存策略

`.engine` / `.plan` 不是通用模型文件，不应直接提交到 Git 或假设能跨机器复用。Engine Cache Key 至少包含：

- ONNX 与权重哈希；
- GPU 型号和 Compute Capability；
- 驱动、TensorRT / CUDA 版本；
- 输入 Profile；
- FP32 / FP16 / INT8；
- Builder Flags、Workspace / Memory Pool 配置；
- Plugin 名称、版本和哈希；
- 校准缓存元数据。

缓存不命中时应明确重建，不能静默加载一个来源不明的 Engine。

## 9. 常见性能陷阱

- 异步 enqueue 后未正确同步，测到的只是提交时间；
- 只报平均值或最好的一次，不报 P99 和波动；
- GPU 尚未升频或已经因为功耗、温度降频；
- Benchmark 使用全零输入，和真实数据的功耗、缓存行为不一致；
- 动态 Profile 范围太大，实际 Shape 又远离 `opt`；
- 每帧分配显存、创建 Context 或创建 Stream；
- 使用 Pageable Memory 和同步 `cudaMemcpy`；
- 每个阶段都调用一次全设备同步；
- 把完整预测 Tensor 拷回 CPU 后再做 NMS；
- 在热路径中使用日志、文件 IO、图片保存或频繁 Python 对象转换；
- FP16 / INT8 只看速度，不跑完整精度回归；
- ONNX 简化、图融合或 Plugin 修改后没有重新校验语义；
- Workspace 太小，限制 TensorRT 的 Tactic 选择；
- 盲目增加 Auxiliary Stream，反而增加同步和显存成本；
- 多线程错误共享同一个 Execution Context；
- CUDA Graph 捕获后改变 Shape、Buffer 地址或 Context 状态；
- 在不同 GPU 或软件栈之间直接分发旧 Engine；
- CPU 路线里 ORT、OpenVINO、OpenCV 各自开满线程，造成过度并行；
- CPU 跨 NUMA 访问，或把逻辑核数量当成物理核数量。

<a id="series"></a>

## 10. 系列目录：从简洁部署到极致速度

系列名：**《YOLO TensorRT 部署：从简洁部署到极致速度》**

第一篇先提供易读、可运行的起点，再逐篇说明性能问题如何被发现、优化如何实现、效果如何验证。文章可用于知乎等平台，仓库保留对应代码与复现实验。

固定原则：**每篇只研究一类主要因素，优化前后使用同一口径，同时交代正确性、收益和代价。** 没有收益的实验也保留，不预设每种技术都会更快。

### 主线篇章

| 篇章 | 核心问题 | 代码 / 实验产物 | 状态 |
| --- | --- | --- | --- |
| 01 · 简洁部署 | 如何用清晰的 C++ 代码跑通 YOLO26n？ | 单文件推理、模型转换脚本、画框与基础计时 | 基础代码已实现，文章待整理 |
| 02 · 先把时间测准 | 时间花在哪里，结果是否正确？ | 预热与重复测试、分阶段计时、正确性对照、原始结果 | 待实现 |
| 03 · CUDA 前处理 | 将缩放、补边、归一化和布局转换融合后有何收益？ | CPU / CUDA 对照、像素与检测结果误差检查 | 待实现 |
| 04 · 内存与数据拷贝 | 哪些分配与传输可以复用或减少？ | CPU 缓冲区复用、Pinned Memory 对照、拷贝耗时记录 | 待实现 |
| 05 · 异步流水线 | 如何组织相邻图片的拷贝与计算？ | Stream / Event、双缓冲实验，分别报告延迟与吞吐 | 待实现 |
| 06 · CUDA Graph | 当前负载是否受重复提交开销限制？ | 固定 Shape 与地址的捕获 / 重放、Graph on/off 对照 | 待实现 |
| 07 · 深入性能瓶颈 | 剩余耗时是否值得进一步优化？ | Nsight 诊断、针对瓶颈的单项实验与适用边界 | 待实现 |
| 08 · 极致速度复盘 | 从基础版到优化版，哪些改动真正有效？ | 同环境重测各篇版本、延迟对比、精度与复杂度总结 | 待实现 |

篇章顺序是实验计划，不是固定的提速阶梯。主线优先优化 Batch 1 端到端延迟；异步流水线等实验另报吞吐收益，不把吞吐提升当作单张延迟下降。

当前 `infer()` 虽使用 `cudaMemcpyAsync`，但每张图片末尾都会同步，且 Host 缓冲区没有使用 Pinned Memory，不能标为已实现多图重叠流水线。Graph 与异步执行的实验边界参考 [TensorRT 8.6.1 开发指南](https://docs.nvidia.com/deeplearning/tensorrt/archives/tensorrt-861/developer-guide/index.html)，是否有收益以本项目对照结果为准。

### 可选专题

INT8 与混合精度、Python / C++ 对照、动态 Shape、多 Batch、视频、服务化和 CPU 部署保留为后续专题，不阻塞主线。当前 YOLO26 输出已经是 `[1, 300, 6]` 检测结果，不额外执行 NMS；只有扩展到需要 NMS 的模型时，才开展对应的 CPU / CUDA / Plugin 对照。

### 每篇文章固定结构

1. **结论卡片**：硬件、模型、输入、精度、Batch、优化前后数字；
2. **本篇问题**：瓶颈是什么，为什么怀疑它；
3. **可复现基线**：命令、配置、代码版本和输入；
4. **唯一变量实验**：本轮只改变一类因素；
5. **正确性检查**：Tensor 误差、检测差异和 mAP；
6. **性能结果**：P50 / P99、吞吐、显存，必要时包含功耗；
7. **失败与边界**：哪些场景没有收益，为什么；
8. **复现入口**：实际存在的 Git commit / Tag、原始结果和下一篇预告。

第 01 篇只交付基础流程与运行示例，不把两张测试图片的计时当作正式 Benchmark；从第 02 篇起补齐测量与正确性基线，之后再发布优化结论。每篇发布时保留明确的代码版本，避免后续优化覆盖读者正在复现的基础版。

### 长期维护的标志性图表

- 端到端延迟堆叠图；
- 每一步优化的阶梯图；
- FP32 / FP16 / INT8 速度—精度 Pareto 图；
- Batch / 并发—吞吐—P99 曲线；
- 同步与异步 Pipeline 的 GPU 时间轴；
- 目标数量—NMS 耗时曲线；
- GPU / CPU 利用率、显存、功耗和温度曲线；
- “有效 / 无效 / 有条件有效”的优化总结矩阵。

文章标题中的倍率或毫秒数必须能由仓库数据直接复算。保留失败实验，比只展示最终答案更有可信度，也更容易形成个人技术标签。

## 11. v0.1 Definition of Done

- [ ] 仓库骨架、构建系统和依赖说明完成；
- [ ] 默认模型、数据集、输入输出语义和许可证确认；
- [ ] PyTorch → ONNX 导出可重复执行；
- [ ] PyTorch / ONNX 的固定样本与完整 mAP 对齐；
- [ ] FP32 / FP16 TensorRT Engine 可由脚本构建；
- [ ] C++ Runtime 能输出可验证的检测 JSON；
- [ ] Benchmark 能输出阶段耗时与机器可读 JSON；
- [ ] 性能测试不在热路径分配内存或保存图片；
- [ ] README 中存在一条从零可复制的运行命令；
- [ ] 整理并发布系列第 01、02 篇，标注对应代码版本；版本验收通过后再打 Git Tag。

## 12. 接下来马上做什么

下一步围绕系列第 01、02 篇推进，不重复搭建已有的基础代码：

1. 整理第一篇的 Engine 初始化、前处理、推理、后处理与计时说明，发布时记录对应 commit；
2. 固定参考权重、测试输入和预处理语义，补充与原始模型的正确性对照；
3. 在独立测试入口增加预热、重复推理和原始结果保存，避免把基础示例变成复杂的 Benchmark 框架；
4. 区分 Engine-only、含拷贝推理和图像 Pipeline 的计时边界，画框、文件读写单独说明；
5. 在固定环境下形成基线，再根据测量结果选择下一项优化。

当前阶段不创建空的多版本工程、不实现未来篇章的功能，也不预先承诺加速倍数。

## 13. 官方参考资料

- [NVIDIA TensorRT：Performance Benchmarking](https://docs.nvidia.com/deeplearning/tensorrt/latest/performance/benchmarking.html)
- [NVIDIA TensorRT：Optimizing TensorRT Performance](https://docs.nvidia.com/deeplearning/tensorrt/latest/performance/optimization.html)
- [NVIDIA TensorRT：Runtime API Quick Start](https://docs.nvidia.com/deeplearning/tensorrt/latest/getting-started/quick-start-runtime-tutorial.html)
- [NVIDIA TensorRT：Migrating from TensorRT 10.x to 11.x](https://docs.nvidia.com/deeplearning/tensorrt/latest/api/migration/tensorrt-10x-to-11x.html)
- [ONNX Runtime：Thread Management](https://onnxruntime.ai/docs/performance/tune-performance/threading.html)
- [OpenVINO：Benchmark Tool](https://docs.openvino.ai/2026/get-started/learn-openvino/openvino-samples/benchmark-tool.html)

---

如果这个项目最终能留下一个核心方法论，希望是：**先证明结果正确，再证明数字可信，最后才谈最快。**
