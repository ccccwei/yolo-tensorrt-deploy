#!/usr/bin/env python3
"""Download YOLO checkpoints and export static ONNX models into weights/."""

import argparse
from pathlib import Path

import onnx
from ultralytics import YOLO


WEIGHTS_DIR = Path(__file__).resolve().parent.parent / "weights"
DEFAULT_MODELS = ("yolo11n.pt", "yolo26n.pt")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "models",
        nargs="*",
        default=DEFAULT_MODELS,
        help="checkpoint names, for example yolo11n.pt",
    )
    args = parser.parse_args()

    WEIGHTS_DIR.mkdir(exist_ok=True)

    for model_name in args.models:
        pt_path = WEIGHTS_DIR / Path(model_name).name
        onnx_path = Path(
            YOLO(str(pt_path)).export(
                format="onnx",
                imgsz=640,
                batch=1,
                opset=16,
                dynamic=False,
                simplify=False,
                nms=False,
                device="cpu",
            )
        )

        onnx.checker.check_model(onnx.load(onnx_path))
        print(f"{pt_path.name} -> {onnx_path.name}")


if __name__ == "__main__":
    main()
