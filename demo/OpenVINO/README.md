# YOLOX with OpenVINO Runtime

* [Python demo](./python/README.md)
* [C++ demo](./cpp/README.md)

## Supported baseline and scope

These demos use the **OpenVINO 2025.4.1 Runtime API**, not the removed legacy
Inference Engine API. Pin the Python package and native SDK to 2025.4.1 for a
reproducible baseline. CMake accepts 2025.4.1 or newer compatible SDKs; newer
versions must be validated separately.

The documented deployment target is **64-bit Linux x86-64, CPU**, with Python
3.12 or 3.13. For a vendor-supported OS use Ubuntu 22.04 (kernel 5.15+) or 24.04
(kernel 6.8+) and a supported Intel CPU. See the SDK's
[system requirements](https://docs.openvino.ai/2025/about-openvino/release-notes-openvino/system-requirements.html)
for the exact processor, OS, driver and compiler combinations.

Python CPU inference has been smoke-tested with 2025.4.1, Python 3.12 and a
YOLOX-Nano 416 ONNX model on Linux x86-64 in a Debian 12 orb. That is not
certification of Debian as a vendor-supported OS. The full C++ demo also compiled,
linked and ran CPU inference with the Ubuntu 22 native SDK and OpenCV 4.6.0;
see the [C++ guide](./cpp/README.md) for exact build and ABI requirements.
Windows, macOS, ARM, GPU and NPU are not validated by these demos. OpenVINO's GPU
plugin targets supported Intel graphics, **not CUDA/NVIDIA GPUs**. Other devices
and `AUTO` require the appropriate plugins, hardware and drivers; inspect
`openvino.Core().available_devices` before selecting one. Legacy MYRIAD/HDDL
devices are not supported by this baseline.

## Model contract

Both demos accept an ONNX file directly or an OpenVINO IR `.xml` with its `.bin`
beside it. Use the repository's [ONNX export instructions](../ONNXRuntime/README.md)
and export **without embedded decoding** (the default). There is no legacy
opset-10-only restriction; use the current exporter's supported default opset.

* Single static batch-one NCHW BGR image input, unnormalized 0-255 values.
* Python reads H/W from the model; C++ retains its 416x416 input constants.
  To use a different C++ size, change `INPUT_H`/`INPUT_W` and rebuild.
* Single raw COCO output `[1, anchors, 85]`, with strides 8, 16 and 32.
  P6/stride-64 models, dynamic shapes, embedded decode/NMS and other class counts
  are outside this demo contract.
* Preprocessing is unchanged: aspect-preserving resize, top-left placement,
  padding with 114, BGR CHW packing, no normalization. Decoding, thresholds and
  NMS remain in the demos. Runtime boundary tensors are FP32, including when
  the model uses FP16 internally; Python no longer forces FP16 output rounding.

## Optional ONNX to IR conversion

ONNX does not need a separate conversion step. For IR, after installing the
Python dependencies from the [Python guide](./python/README.md), run from the
repository root:

```shell
python -c 'import openvino as ov; ov.save_model(ov.convert_model("yolox.onnx"), "yolox.xml", compress_to_fp16=False)'
```

This produces `yolox.xml` and `yolox.bin` and retains FP32 weights. To compress
weights to FP16, set `compress_to_fp16=True` (this does not promise FP16 execution
on every device). No Model Optimizer `mo.py`, `openvino-dev`, `ngraph`, or 2021
`setupvars.sh` is needed. Historical preconverted model archives are not
validated with this baseline; re-export ONNX and convert with the pinned Runtime
instead of relying on an old IR.
