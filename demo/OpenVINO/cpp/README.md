# YOLOX with OpenVINO Runtime in C++

See the [shared guide](../README.md) for the supported **OpenVINO 2025.4.1**
baseline, OS/hardware limits, model contract and optional IR conversion. The
full C++ demo was built and run on CPU in the Linux development orb. Do not
infer GPU/NPU or other OS support from successful CPU inference.

## Build on Linux x86-64

Requirements:

* OpenVINO **2025.4.1** native Runtime SDK (headers, libraries, CPU plugin and
  `OpenVINOConfig.cmake`), from the
  [official installation instructions](https://docs.openvino.ai/2025/get-started/install-openvino.html).
* OpenCV 4 development package, including core, imgproc and imgcodecs (on
  Ubuntu: `sudo apt install libopencv-dev`). A Python `cv2` wheel alone is not a
  C++ OpenCV SDK.
* CMake 3.16+ and a C++17 compiler supported by the SDK, e.g. GCC 11 on Ubuntu
  22.04 or GCC 13 on Ubuntu 24.04.

From the repository root, configure using the installed SDK's CMake directory
(replace the example path with the directory containing `OpenVINOConfig.cmake`):

```shell
cmake -S demo/OpenVINO/cpp -B /tmp/yolox-openvino-build \
  -DOpenVINO_DIR=/path/to/openvino/runtime/cmake
cmake --build /tmp/yolox-openvino-build --parallel
```

Alternatively source the **2025.4.1** SDK's `setupvars.sh` before configuring,
if provided by your distribution. Ensure its shared libraries and CPU plugin
remain discoverable when executing the program. CMake links the imported
`openvino::runtime` target; there is no separate InferenceEngine or ngraph
dependency. Older SDKs are rejected; newer SDKs need independent validation.

On Linux, OpenVINO and OpenCV must use compatible libstdc++ ABIs. The OpenVINO
Python wheel's CMake package sets `_GLIBCXX_USE_CXX11_ABI=0`; pairing that wheel
with Debian's system OpenCV (ABI 1) compiled but failed to link `cv::imread`,
`cv::putText` and `cv::imwrite`. Use the native Ubuntu 22 SDK archive with system
OpenCV instead; do not override the ABI macro to hide a binary incompatibility.

### Executed build baseline

On 2026-10-25, the complete demo configured, compiled, linked and ran using
Debian 12 x86-64, GCC 12.2, CMake 3.25.1, Ninja, OpenCV 4.6.0 and the official
Ubuntu 22 OpenVINO 2025.4.1 native SDK:

* [SDK archive](https://storage.openvinotoolkit.org/repositories/openvino/packages/2025.4.1/linux/openvino_toolkit_ubuntu22_2025.4.1.20426.82bbf0292c5_x86_64.tgz)
* SHA-256: `4e9501655a30e93ebdd1c21944c16eb094012cc541653f8151ed981061466d84`,
  verified against the publisher's accompanying `.sha256` file.

Using a Nano 416 ONNX model exported by this repository, CPU inference on
`assets/dog.jpg` reported class 16 (dog) 0.85507, class 1 (bicycle) 0.80264 and
class 2 (car) 0.79819, and wrote `_demo.jpg`. This is an executed Linux CPU
smoke test, not vendor certification of Debian or an accuracy benchmark.

## Run

```shell
/tmp/yolox-openvino-build/yolox_openvino yolox.onnx assets/dog.jpg CPU
# Or use yolox.xml, keeping yolox.bin beside it:
/tmp/yolox-openvino-build/yolox_openvino yolox.xml assets/dog.jpg CPU
```

The positional CLI stays `<model> <image> <device>`. Detection records are
printed to stderr and the visualization is written to **`_demo.jpg` in the
current working directory**, as before.

The C++ demo retains `INPUT_W = INPUT_H = 416`, 80 COCO classes, strides
8/16/32, confidence 0.3, IoU 0.45 and class-agnostic NMS. It validates a static
`[1, 3, 416, 416]` input and raw `[1, 3549, 85]` output rather than reading or
writing incompatible buffers. For another input size, update `INPUT_W` and
`INPUT_H` in `yolox_openvino.cpp`, rebuild and use a matching model.

The Runtime path uses `ov::Core`, `ov::CompiledModel`, `ov::InferRequest` and
`ov::Tensor`. A `PrePostProcessor` declares FP32 host tensors (and converts to
the model's types when needed); the demo still performs BGR resize/padding/CHW
packing itself. Output tensor/request lifetimes cover decoding. Unsupported
models and unreadable images fail with a nonzero exit status.
