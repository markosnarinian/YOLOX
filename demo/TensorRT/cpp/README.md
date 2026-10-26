# YOLOX-TensorRT in C++

As YOLOX models are easy to convert to tensorrt using [torch2trt gitrepo](https://github.com/NVIDIA-AI-IOT/torch2trt), 
our C++ demo does not include the model converting or constructing like other tenorrt demos.


## Step 1: Prepare serialized engine file

Follow the trt [python demo README](https://github.com/Megvii-BaseDetection/YOLOX/blob/main/demo/TensorRT/python/README.md) to convert and save the serialized engine file.

Check the 'model_trt.engine' file generated from Step 1, which will be automatically saved at the current demo dir.


## Step 2: build the demo

Please follow the [TensorRT Installation Guide](https://docs.nvidia.com/deeplearning/tensorrt/install-guide/index.html) to install TensorRT.

This source uses TensorRT 7/8's binding API; it is not a TensorRT 10 port.
Configure SDK roots through CMake rather than editing source paths.

### Validated build baseline

On 2026-10-25, the complete executable configured, compiled and linked on
Debian 12 x86-64 with GCC 12.2, CMake 3.25.1, Ninja, OpenCV 4.6.0,
**TensorRT 8.6.1.6 (CUDA 12.0)** and CUDA Toolkit **12.0.140**
(CUDA runtime 12.0.146). NVIDIA's Ubuntu 22 packages were extracted into a
disposable SDK root; their SHA-256 values were checked against the official
HTTPS repository's `Packages.gz` metadata. No fake headers or stub inference
libraries were used. The executable also loaded and returned its expected
nonzero usage exit status without an engine argument.

That initial orb check was **build validation, not GPU inference validation**.
The subsequent Asus GPU check below adds qualified execution evidence; Windows
builds remain unverified. The build emits legacy binding-API deprecation
warnings; success does not make this sample compatible with modern
TensorRT 10+. Install the matching SDK, runtime libraries and GPU driver on the
deployment host. This SDK baseline is separate from the Python package's uv
CUDA 13.0 profile; do not assume their binaries or serialized engines interchange.

The actual build used explicit `TENSORRT_INCLUDE_DIR` and `TENSORRT_LIBRARY`
cache entries for the extracted distribution's multiarch paths, alongside
`CUDAToolkit_ROOT`. A conventional SDK archive can use `TENSORRT_ROOT` alone.

### Qualified GPU execution on Asus

On 2026-10-26, the unchanged complete C++ demo compiled/linked directly with
G++ 15.2 and extracted OpenCV 4.10/TensorRT 8.6.1.6/CUDA 12.0/cuDNN 8.9.7 on
Ubuntu 26.04.1 (glibc 2.43). The RTX 3060 Laptop GPU, driver 595.91.07,
executed a fresh FP32 Nano engine at fixed batch one. This runner check did not
exercise CMake; the separate orb build above did.

**Default engine construction failed**: glibc rejected the executable stack
requested by `libnvinfer_builder_resource.so.8.6.1`. A separate builder-only
retry using `GLIBC_TUNABLES=glibc.rtld.execstack=2` succeeded. This is a
security-sensitive legacy compatibility opt-in, not a recommended global setting
or a stock-host SDK pass. No SDK binary or system loader configuration was
modified. Prefer a compatible, reviewed deployment environment rather than
silently enabling executable stacks. Subsequent demo and `trtexec` inference
succeeded **without that setting**.

The native demo retained dog/bicycle/truck with scores 0.89060/0.82008/0.69842.
A separate raw-output comparison of the same engine against CUDA PyTorch on
identical input passed all 714,000 values at `atol=rtol=0.001`, with maximum
absolute difference 0.000139 (TensorRT JSON text precision, not bitwise parity).
The publisher checkpoint digest is recorded in `docs/model_assets.md`.
INT64-conversion/clamping and legacy explicit-batch API warnings remain.
This is one FP32 Nano image case, not full-dataset accuracy, a benchmark, dynamic
batch support, FP16/INT8 validation, Python/torch2trt conversion, or TensorRT 10
support. Full commands and dependency provenance are recorded in `PLAN.md` and
the Asus validation report.

If you train your custom dataset, you may need to modify the value of `num_class`.

```c++
const int num_class = 80;
```

Install opencv with ```sudo apt-get install libopencv-dev``` (we don't need a higher version of opencv like v3.3+). 

build the demo:

```shell
cmake -S . -B build -DTENSORRT_ROOT=/path/to/TensorRT -DCUDAToolkit_ROOT=/path/to/cuda -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release --parallel
cd build
```

Then run the demo:

```shell
./yolox ../model_trt.engine -i ../../../../assets/dog.jpg
```

or

```shell
./yolox <path/to/your/engine_file> -i <path/to/image>
```

NOTE: for `trtexec` users, modify `INPUT_BLOB_NAME` and `OUTPUT_BLOB_NAME` as the following code.
```
const char* INPUT_BLOB_NAME = "images";
const char* OUTPUT_BLOB_NAME = "output";
```

Here is the command to convert the small onnx model to tensorrt engine file:
```
trtexec --onnx=yolox_s.onnx --saveEngine=yolox_s.trt
```
