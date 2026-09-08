#!/bin/sh
set -eu

onnx=${1:?"Usage: $0 MODEL.onnx [OUTPUT.engine]"}
engine=${2:-${onnx%.onnx}.fp16.engine}
trtexec=${TRTEXEC:-/usr/local/TensorRT/bin/trtexec}

"$trtexec" \
    "--onnx=$onnx" \
    "--saveEngine=$engine" \
    --fp16 \
    --memPoolSize=workspace:1024 \
    --skipInference
