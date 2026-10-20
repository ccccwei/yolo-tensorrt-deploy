# RTX 3060：YOLO26m 的 01–11 复测

2026-10-15 改用中等规模的 **YOLO26m**，重新测量全部 11 种默认配置。官方当前正式发布的最新系列仍为 YOLO26；YOLO27 尚未提供权重，见 [Ultralytics 模型目录](https://docs.ultralytics.com/models/)。

本次各版都与 **YOLO26m 的 01** 比较。此前的 [YOLO26n 基准](../rtx3060-unified-baseline-2026-09-15/README.md)独立保留，不跨模型、跨批次计算加速比。

## 测试条件

- RTX 3060，驱动 525.147.05，CUDA 11.8，TensorRT 8.6.1，Release。
- 官方 `yolo26m.pt`，静态 640 × 640、Batch 1、80 类；FP16 构建，FP32 输入输出。
- 01–10 共用原模型 engine；11 使用分组模型 engine，运行方式与 09 相同。两份 engine 使用相同参数、各自独立的空 timing cache 构建，构建过程串行执行。
- `bus.jpg`、`zidane.jpg`，每轮每图预热 200 次、测量 1000 次，轮换顺序测 3 轮。每配置 6000 条，33 次运行合计 **66,000 条记录**。
- 请求时间取 `total_ms`，包括前后处理、传输、推理和同步等待；不含解码、初始化、注册、Graph 创建、预热、画框及文件写入。合并原始样本后求 P50，加速比 = **01 P50 ÷ 本版 P50**。

## 结果

| 版本 | 测量模式 | 请求 P50（ms） | 相对本模型 01 加速比 |
| --- | --- | ---: | ---: |
| 01 | CPU 前后处理 | 4.547317 | 1.00× |
| 02 | CUDA 前处理、CPU 后处理 | 3.802120 | 1.20× |
| 03 | Pinned Memory | 3.861314 | 1.18× |
| 04 | 双槽 | 6.330046 | 0.72× |
| 05 | Graph 开 | 3.629839 | 1.25× |
| 06 | 完整 Graph 开 | 3.640479 | 1.25× |
| 07 | 双槽 + Graph | 6.528452 | 0.70× |
| 08 | 映射表开 | 3.627002 | 1.25× |
| 09 | 注册原图 | 3.519440 | 1.29× |
| 10 | 双槽 + 拆流 | 7.151163 | 0.64× |
| 11 | 09 + Grouped TopK | 3.484134 | 1.31× |

11 的完整方案从 01 的 **4.547317 ms** 降至 **3.484134 ms**，加速比 **1.31×**，请求延迟降低约 **23.4%**。它包含 CUDA 前处理、注册输入、Graph 和模型图改写的共同效果。

09 为 **3.519440 ms / 1.29×**；11 比它低约 **1.0%**。这点差异不能单独证明 TopK 的稳定收益：engine 构建可能选择不同实现，GPU 未锁频，3 轮运行顺序也未完全均衡。小幅差异不作统计显著性结论。

04、07、10 默认使用双槽，请求延迟包含入槽后的执行等待，不含入槽前排队。它们的倍率低于 1，表示单请求更慢，不代表吞吐率一定更低；实测吞吐另保存在各轮 `throughput.csv` 中，不能用 `1000/P50` 替代。

本轮最大的请求加速比约 1.31×，此前 26n 为 1.76×。两个批次的结果说明，本项目在较大模型上的整体相对收益更小；具体差异仍受模型、构建及运行状态共同影响。

## 校验

- 官方权重 SHA-256 与 GitHub 发布资产一致；记录见[模型来源与导出环境](evidence/provenance.json)。
- 分组改写保留全部权重、非目标节点及模型 IO，见[结构检查](evidence/structural-check.json)。两份 engine 均确认是 FP32 `images [1,3,640,640] → output0 [1,300,6]`。
- Grouped TopK CPU 测试 5/5 通过；01–02、02–07、08–10 三组集成测试均通过，包括开关、结果、CSV、槽位回收和错误输入处理。见[测试记录](evidence/integration-checks.json)及同目录日志。
- 独立重算了全部样本的均值、P50/P95/P99 和倍率，核对了全部记录文件的哈希。各配置三轮检测结果稳定；09 与 11 在两张图上的检测 CSV 完全一致。见[数据复核](evidence/data-audit.json)。

以上是两张示例图的重复测试，**尚未验证完整数据集 mAP**。

## 数据与复现

[汇总 JSON](data/summary.json) · [运行命令、环境与哈希](data/metadata.json) · [逐轮 CSV 与输出](data/) · [运行脚本](run.py) · [配对构建脚本](build_engines.py)

[原模型构建记录](evidence/control.build.json) · [分组模型构建记录](evidence/grouped.build.json)。构建分别约 300 秒和 289 秒，均不计入请求时间。Engine 不入库，重新构建视作新实验。

准备好项目依赖后，在仓库根目录运行；下面的模型、engine 和结果目录应为新路径：

```bash
.venv-export/bin/python tools/export_onnx.py yolo26m.pt
mkdir -p artifacts/yolo26m-retest/models
.venv-export/bin/python tools/grouped_topk.py weights/yolo26m.onnx artifacts/yolo26m-retest/models/control.onnx --groups 1
.venv-export/bin/python tools/grouped_topk.py weights/yolo26m.onnx artifacts/yolo26m-retest/models/grouped.onnx --groups 8
python3 docs/benchmarks/rtx3060-yolo26m-2026-10-15/build_engines.py \
  --models artifacts/yolo26m-retest/models --output artifacts/yolo26m-retest/engines
cmake --build --preset release -j4
python3 docs/benchmarks/rtx3060-yolo26m-2026-10-15/run.py \
  --engine artifacts/yolo26m-retest/engines/control.engine \
  --grouped-engine artifacts/yolo26m-retest/engines/grouped.engine \
  --output results/yolo26m-retest
```
