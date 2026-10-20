# 章节编号对照

当前系列连续编号为 **01–11**：01–10 各有独立推理程序，11 修改模型图并复用 05 / 09。目录、笔记、CMake 和新运行的结果目录统一使用当前编号；可执行文件名称和命令行参数保持不变。

| 原始编号（历史报告） | 当前编号 | 当前内容 |
| --- | --- | --- |
| 01 + 02 | [01](notes/01_baseline.md) | 基础部署与基准测试 |
| 03 + 07 | [02](notes/02_cuda_prepost.md) | CUDA 前后处理，CPU/GPU 后处理开关 |
| 04 | [03](notes/03_pinned_memory.md) | Pinned Memory |
| 05 | [04](notes/04_async_pipeline.md) | 异步流水线 |
| 06 | [05](notes/05_cuda_graph.md) | CUDA Graph |
| 08 | [06](notes/06_full_graph.md) | 完整 GPU Graph |
| 09 | [07](notes/07_graph_pipeline.md) | Graph 流水线 |
| 10 | [08](notes/08_preprocess_maps.md) | 前处理插值映射表 |
| 11 | [09](notes/09_registered_input.md) | 注册原图内存 |
| 12 | [10](notes/10_split_streams.md) | 分阶段流水线 |
| 13 | [11](notes/11_grouped_topk.md) | Grouped TopK 模型图优化 |

合并曾短暂保留原编号，所以上一布局中的 `03_cuda_prepost` 现在是 `02_cuda_prepost`；其余活动目录也按上表连续排列。不再保留单独的 02 基准程序或 07 GPU 后处理章节入口。

## 历史数据怎么读

`docs/benchmarks/` 下的历史报告目录、原始 CSV / JSON、命令记录、脚本和测量哈希保留原编号，不把旧标签替换成新实验。已有 `results/` 数据也不自动搬动或覆盖。之后运行程序，会写入 `results/<当前编号与名称>/`；CUDA 前后处理继续区分 `cpu_post/` 与 `gpu_post/`。

例如，历史报告中的“07 GPU 后处理”不是现在的“07 Graph 流水线”。原 07 使用 pinned staging 和末尾单次同步，而现在 02 使用普通内存与两处同步；不能只改表头就把历史 07 的性能当成当前 02 的成绩。

为了继续检查和复现旧实验，[历史源码快照](benchmarks/legacy-sources/pre-merge-2026-09-14/README.md)保留原 01–12 的目录和输出约定，并可独立构建。原始 benchmark runner 应使用这组旧程序；当前程序由仓库根目录的 CMake 构建，并由当前的组件 / 集成测试验证。重编号不代表重新做了性能测试。

返回：[当前笔记目录](notes/README.md) · [English README](../README.md) · [中文 README](../README.zh-CN.md)
