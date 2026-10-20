# RTX 3060：01–11 统一基准

2026-09-15 使用当前源码重新测量。所有加速比统一为 **01 的请求 P50 ÷ 对应版本的请求 P50**，不混入吞吐倍率，也不与旧批次拼接。

## 条件

- RTX 3060，驱动 525.147.05，CUDA 11.8，TensorRT 8.6.1，Release 构建。
- YOLO26n，内部 FP16、FP32 I/O，batch 1，640 × 640；输入为 `bus.jpg` 和 `zidane.jpg`。
- 01–10 使用同一原模型 engine；11 使用配对 Grouped TopK engine，复用 09 注册输入模式。两份 engine 来自先前相同初始 timing cache 的独立构建，详见 [11 笔记](../../notes/11_grouped_topk.md)。
- 各版使用默认优化开关；每轮每图预热 200 次、测量 1000 次，顺序执行并轮换 3 轮。每版 6000 条，合计 66,000 条记录。
- P50 采用全部样本的 nearest-rank；不平均各轮 P50。计时读取程序的 `total_ms`，包含前后处理、传输、推理和同步，排除解码、初始化、预热与文件读写。

## 结果

| 版本 | 测量模式 | 请求 P50（ms） | 相对 01 加速比 |
| --- | --- | ---: | ---: |
| 01 | CPU 前后处理 | 2.166601 | 1.00× |
| 02 | CPU 后处理 | 1.509208 | 1.44× |
| 03 | Pinned Memory | 1.573842 | 1.38× |
| 04 | 双槽 | 2.047407 | 1.06× |
| 05 | Graph 开 | 1.377862 | 1.57× |
| 06 | 完整 Graph 开 | 1.343122 | 1.61× |
| 07 | 双槽 + Graph | 2.321121 | 0.93× |
| 08 | 映射表开 | 1.371012 | 1.58× |
| 09 | 注册原图 | 1.275870 | 1.70× |
| 10 | 双槽 + 拆流 | 2.606408 | 0.83× |
| 11 | 09 + Grouped TopK | 1.230703 | 1.76× |

04、07、10 的默认双槽请求延迟包含入槽后的执行等待，不含入槽前排队。倍率低于 1 表示请求 P50 高于 01，不表示吞吐率更低；本表只比较请求延迟。

GPU 未锁频，3 轮顺序不完全均衡（01 分别排第 1、4、4 位）。这些数字是本轮观测，不用于断言小幅差异具有统计显著性，也不是完整数据集精度评价。

## 数据与复现

[汇总](data/summary.json) · [命令、环境与哈希](data/metadata.json) · [逐轮 CSV 与日志](data/) · [运行脚本](run.py)

已核对 33 次运行的样本数量、合并 P50、相对 01 倍率，以及模型、源码、程序和图片哈希；每个配置三轮保存的检测结果一致。不同配置之间不据此宣称完整精度一致。

Engine 不入库；下面路径是本轮使用的本地配对文件，SHA-256 已记录在元数据中。重新构建 engine 应视作新一轮实验。

```bash
cmake --build --preset release --target yolo_baseline yolo_cuda_prepost yolo_pinned_memory yolo_async_pipeline yolo_cuda_graph yolo_full_graph yolo_graph_pipeline yolo_preprocess_maps yolo_registered_input yolo_split_streams -j4
python3 docs/benchmarks/rtx3060-unified-baseline-2026-09-15/run.py \
  --engine artifacts/grouped-topk-IAqxawtr/fixed_cache/control.engine \
  --grouped-engine artifacts/grouped-topk-IAqxawtr/fixed_cache/grouped.engine \
  --output results/unified-baseline-new-run
```

输出目录须为新路径。运行使用临时工作目录，不覆盖各版本已有结果。
