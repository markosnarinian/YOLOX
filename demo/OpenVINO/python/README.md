# YOLOX with OpenVINO Runtime in Python

See the [shared guide](../README.md) for the supported **OpenVINO 2025.4.1**
baseline, OS/hardware limits, model contract and optional IR conversion.

## Install

Run from the repository root. This inference-only environment needs no PyTorch,
training dependencies, `openvino-dev`, or native SDK setup script. It imports the
existing `yolox.deploy` helpers directly from the checkout.

```shell
python3.12 -m venv .venv-openvino
source .venv-openvino/bin/activate
python -m pip install 'openvino==2025.4.1' 'numpy>=1.26,<2.4' 'opencv-python-headless>=4.10,<5'
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
python -c 'import openvino as ov; print(ov.__version__); print(ov.Core().available_devices)'
```

Use Python 3.12 or 3.13. `opencv-python` can replace the headless wheel if a GUI
is needed; do not install both wheels in the same environment.

## Run

```shell
python demo/OpenVINO/python/openvino_inference.py -m yolox.onnx -i assets/dog.jpg
python demo/OpenVINO/python/openvino_inference.py -m yolox.xml -i assets/dog.jpg -o demo_output -s 0.3 -d CPU
```

The model must have one static `[1, 3, H, W]` input and one undecoded
`[1, anchors, 85]` COCO output with strides 8/16/32. The image is written to
`<output_dir>/<input_basename>`; `demo_output` and visualization confidence 0.3
remain the defaults. NMS uses IoU 0.45 and candidate score 0.1 as before.
`--labels` and `--number_top` remain accepted for CLI compatibility but, as in
the original demo, do not change the COCO labels or detections.

The implementation uses `ov.Core.read_model`, `compile_model`, an explicit
synchronous infer request, and FP32 Runtime input/output tensors. Python adds
the batch dimension explicitly and copies the output before in-place decoding.
No preprocessing is embedded in the model and no RGB conversion is performed.
