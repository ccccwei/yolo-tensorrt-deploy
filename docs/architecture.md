# 当前推理流程

`main()` 加载 Engine，并创建可重复使用的 ExecutionContext、输入输出显存和 CUDA Stream。`infer()` 只执行输入拷贝、`enqueueV3`、输出拷贝和同步。

```text
图片
  → OpenCV 读取
  → CPU Letterbox、BGR 转 RGB、归一化、HWC 转 CHW
  → CUDA H2D
  → TensorRT enqueueV3
  → CUDA D2H
  → YOLO 后处理
  → 检测结果
```

每张图片只执行前处理、内存拷贝、`enqueueV3` 和后处理，不会重新加载 Engine 或重新申请显存。

当前只支持 YOLO26 的 `[1, 300, 6]` 输出，直接解析 `xyxy + score + class_id`，不再执行 NMS。

输入 Tensor 名为 `images`，输出 Tensor 名为 `output0`，二者均使用 FP32 I/O。TensorRT Engine 内部允许采用 FP16 tactic。
