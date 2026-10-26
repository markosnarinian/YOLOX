# Supported environments

## Repository validation scope

This repository owns working YOLOX code, packaging, export and portable ONNX
inference. The user's downstream device is Raspberry Pi Compute Module 5, but
physical-device integration/certification is **outside this repository's scope**.
The user accepts real ONNX Runtime inference on the Asus with the explicit
CPUExecutionProvider as sufficient deployment-path validation and will extrapolate
that result to Raspberry Pi. CM5 access, camera operation, target performance and
thermal measurements are not repository completion or shipping gates.

The [deployment guide](cm5_deployment.md) provides a source-only runtime bundle
and hash-locked headless dependencies without PyTorch. The 640×640 Nano export
was compared against CUDA PyTorch using actual CPU-only ORT inference on the Asus;
the exact 416×416 minimal bundle was separately executed on the orb's CPU with
no Torch installed/imported. These are codebase/runtime results, not claims of
measured CM5 compatibility, latency or accuracy. No physical CM5 tests were run.

TensorRT, OpenVINO, MegEngine, Android and other non-CM5 deployment examples are
retained as reference/legacy integrations outside this project's deployment
maintenance scope. Their recorded checks below are historical evidence, not
promises to keep every SDK/platform working. No integrations were removed.

## Development and training environment baseline

The modernization baseline is **CPython 3.12 and 3.13**, with Python 3.12 the
development default. Package metadata, dependency resolution and CI must agree.
Older Python versions are not supported by the modernized package.
The dependency baseline is PyTorch 2.13+/torchvision 0.28+ and NumPy 1.26–2.x;
the locked CUDA profile uses CUDA 13.0 binaries. Older PyTorch compatibility was
not retained at the expense of published security fixes. Linux binary wheels
require a compatible glibc (current PyTorch wheels use manylinux 2.28).

| Target | Contract |
| --- | --- |
| Linux x86-64 CPU | Installation, inference, export and automated tests |
| Windows x86-64 CPU | Installation, inference and export; CI validation required |
| macOS Apple Silicon CPU | Installation, inference and export; ARM64 CI validation required |
| Linux x86-64 NVIDIA CUDA | Primary training and evaluation backend; hardware validation required |
| Windows CUDA | Experimental single-device use; distributed and caching restrictions apply |
| Apple MPS | Not a supported training/evaluation backend; never silently substitute CPU |
| Intel macOS / Windows ARM64 | Not supported by the selected modern binary dependency baseline |
| Linux ARM64 / Jetson / AMD ROCm | Experimental; use hardware/vendor-compatible torch builds and validate separately |
| Android / native deployment demos | Separate optional integrations, not part of Python package support |

This table defines the intended support boundary, not a claim that every target
has already passed validation. Only executed checks are recorded in `PLAN.md`.
CPU inference does not imply CPU training. CUDA-named torch APIs also run on ROCm,
but NVIDIA-specific utilities and available binary distributions can still limit it.

On 2026-10-26, a MacBook Air M5 (macOS 27.0.1, arm64) passed the locked CPU/export
environment's 17 tests on Python 3.12 and 3.13. Python 3.12 also passed isolated
sdist/wheel builds, native COCO extension import and installed-wheel inference
outside the source tree. Pretrained ONNX CPU inference matched detection
classes/counts on four image/geometry cases, with box differences below 0.0002
original-image pixels; strict raw `atol=1e-5, rtol=1e-4` parity did not pass.
This is smoke/regression evidence, not full-dataset accuracy certification.

Availability of an ONNX execution provider is not an accuracy guarantee.
On that M5, default CoreML executed a hybrid CoreML/CPU graph but shifted retained
boxes by up to 3.02 input-coordinate pixels relative to PyTorch. CoreML CPU-only
options reduced those differences, but GPU/ANE placement and the cause were not
established. Default CoreML acceleration is not numerically certified here;
the maintained Apple Silicon path remains explicit CPU execution, not MPS.

The Asus RTX 3060 Laptop GPU (Ubuntu 26.04.1, driver 595.91.07) passed the locked
CUDA 13.0 profile's 17 tests on Python 3.12 after supplying missing OpenCV runtime
libraries and rebuilding the optional native evaluator with a compiler in
disposable user space. FP32 Nano inference, one real forward/backward/SGD step,
and CUDA export versus CPU ONNX Runtime comparison passed. This is not evidence
for long training, AMP, distributed training, GPU resume or dataset accuracy.

Set `YOLOX_DATADIR` to an explicit dataset root rather than depending on a writable
installation directory. Use save-only output on headless servers. Do not install
GUI and headless OpenCV distributions in the same environment.

## Checkpoint trust

Only load checkpoints from trusted sources. Training resume checkpoints contain
model/optimizer state and are distinct from exported inference models. Safe
weights-only loading is the default; loading arbitrary serialized Python objects
is not an implicit compatibility workaround.

## Deployment maintenance boundary

ONNX Runtime is the portable deployment path to modernize first. TensorRT remains
NVIDIA-specific and engines must match their target runtime/hardware. OpenVINO,
ncnn, MegEngine and Android examples require separate SDKs, assets and native or
device validation; installing the Python package does not certify those targets.
Legacy toolchains must not be advertised as compatible with current SDKs without
a corresponding build. Model downloads must be trusted and verified before use.

TensorRT/MegEngine build glue uses CMake 3.18+: TensorRT accepts `-DTENSORRT_ROOT=...` and
`-DCUDAToolkit_ROOT=...`, and its source still requires the TensorRT 7/8 binding
API. MegEngine accepts `-DMGE_INSTALL_PATH=...` and `-DOpenCV_DIR=...`; use
`-DCMAKE_TOOLCHAIN_FILE=...` for cross-compilation. These configurations remove
developer-specific paths/GNU runtime flags. TensorRT 8.6.1.6 with CUDA 12.0 and
OpenCV 4.6.0 compiled and linked on Linux x86-64. The same SDK later passed Nano
FP32 inference on the Asus GPU, compiled directly with G++ and OpenCV 4.10.
Engine construction on glibc 2.43 required a builder-only executable-stack
compatibility opt-in; default construction failed. This security-sensitive legacy
constraint is not a recommended global setting or stock-host SDK support.
Inference did not need that opt-in. See the TensorRT README for qualified evidence.
That native SDK is separate from the Python uv CUDA 13.0 profile. Do not assume
engines or SDK binaries are interchangeable across those environments.
MegEngine v1.13.3's CPU SDK and full demo passed Linux x86-64 FP32 Nano inference
with GCC 12/Haswell and the documented OpenBLAS `-O2 -fno-ipa-ra` workaround.
The fail-closed dependency check passed 25/25 suites at one and two threads;
the original build passed only 16/25 despite a misleading CTest 20/25 report.
A fresh, trusted-checkpoint export from isolated MegEngine 1.13.0/Python 3.10
passed eight native cases with exact preprocessed inputs and bounded raw/
decoded parity against FP64 PyTorch. This legacy exporter is not part of the
main Python package's support contract. See the MegEngine READMEs for numerical
budgets, build flags and provenance. Other models/options, non-Linux hosts,
ARM/CUDA and full-dataset accuracy remain unverified.
Android now pins Gradle 8.13/AGP 8.13.2/JDK 17, SDK 36, NDK 28.2 and ncnn
20250916. After explicit SDK-license approval, the debug APK compiled and linked
for ARM64 and ARMv7; signature and 16 KB APK ZIP alignment checks passed.
ARM64 ELF segments are 16 KB aligned; ARMv7 ELF segments retain 4 KB alignment.
Device inference and execution on a real 16 KB-page device remain unverified.
See the Android README
for exact versions, asset digests and the required matching param/bin pair.
OpenVINO uses the 2025.4.1 Runtime API in Python and C++ (CMake 3.16+/C++17).
Both Python and the full C++ demo passed CPU inference on Linux; C++ used the
native Ubuntu 22 SDK with ABI-compatible system OpenCV, not the Python wheel's
CMake package. Other plugins/platforms remain unverified. Optional integrations
are not included in the Python support promise.
