# YOLOX-CPP-MegEngine

Cpp file compile of YOLOX object detection base on [MegEngine](https://github.com/MegEngine/MegEngine).

## Pinned Linux CPU SDK build

Use MegEngine **v1.13.3**,
[commit 570d636](https://github.com/MegEngine/MegEngine/commit/570d63640e7a2662ce4ac06abfb38ceeb0612e96),
rather than an unpinned checkout.
This recipe uses source OpenBLAS, not Intel MKL, and does not install Python
bindings, CUDA, oneDNN, or training support. A Python 3 interpreter is still
needed for build-time code generation. Build prerequisites on Debian 12 are
GCC/G++, gfortran, make, Git, Python 3, CMake, Ninja, and `libopencv-dev`.
Use a temporary directory outside this repository and limit parallelism:

```shell
git clone --depth 1 --branch v1.13.3 https://github.com/MegEngine/MegEngine.git /tmp/MegEngine-v1.13.3
export megengine_root=/tmp/MegEngine-v1.13.3
git -C "$megengine_root" rev-parse HEAD
git -C "$megengine_root" submodule update --init --depth 1 \
    third_party/midout third_party/flatbuffers third_party/Json \
    third_party/cpuinfo third_party/OpenBLAS third_party/gflags

export EXTRA_CMAKE_ARGS='-DMGE_BLAS=OpenBLAS -DMGE_WITH_MKLDNN=OFF -DMGE_WITH_LITE=OFF -DMGE_BUILD_IMPERATIVE_RT=OFF -DMGE_WITH_JIT=OFF'
export MGE_INSTALL_PATH="$megengine_root/build_dir/host/MGE_WITH_CUDA_OFF/MGE_INFERENCE_ONLY_ON/Release/install"
export mge_build="$megengine_root/build_dir/host/MGE_WITH_CUDA_OFF/MGE_INFERENCE_ONLY_ON/Release/build"

# Seed the nested OpenBLAS cache. Its pinned, older CPU detector does not
# recognize every modern x86 CPU (getarch_2nd: SGEMM_DEFAULT_UNROLL_M undefined).
# HASWELL requires AVX2/FMA; choose a suitable OpenBLAS TARGET on other CPUs.
cmake -S "$megengine_root/third_party/OpenBLAS" \
    -B "$mge_build/third_party/OpenBLAS/src/openblas-build" -DTARGET=HASWELL \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_C_FLAGS_RELEASE='-O2 -DNDEBUG -fno-ipa-ra' \
    -DCMAKE_Fortran_FLAGS=-fallow-argument-mismatch
# The Fortran flag is needed by this older OpenBLAS's BLAS tests on GCC >= 10
# (sblat1.f passes both scalars and arrays to STEST1); tests remain enabled.
# This is a validated GCC 12/Haswell workaround, not a Clang/MSVC flag set.
# Old kernels modify registers declared input-only to inline assembly. Disable
# interprocedural register allocation; lowering optimization alone is not enough.
bash "$megengine_root/scripts/cmake-build/host_build.sh" -j 2
```

Run the fail-closed numerical gate before using the SDK (from the repository root):

```shell
for threads in 1 2; do
    python demo/MegEngine/cpp/validate_openblas.py \
        --source "$megengine_root/third_party/OpenBLAS" \
        --build "$mge_build/third_party/OpenBLAS/src/openblas-build" \
        --logs "/tmp/yolox-blas-check-$threads" --threads "$threads" || exit 1
done
```

Require **25/25 numerical suites passed** at both thread counts. The script
executes the suites directly in fresh directories and requires computational
coverage for all enabled routines, including both CBLAS layouts. Upstream CTest
alone is insufficient: its wrappers can hide crashes/zero-exit numerical errors.

Then, from this demo directory (Debian's OpenCV is discovered automatically):

```shell
./build.sh -DMGE_INSTALL_PATH="$MGE_INSTALL_PATH"
# For a non-system OpenCV, also pass -DOpenCV_DIR=/path/to/lib/cmake/opencv4.
# Override the wrapper's default two build jobs with CMAKE_BUILD_PARALLEL_LEVEL.
./build/yolox  # Prints usage and exits unsuccessfully without arguments.
```

The decoder requires one float32 output of shape **[1, 8400, 85]**: raw,
undecoded YOLOX predictions for 640x640 input, BGR pixels in the 0–255 range,
and 80 COCO classes. Use `../python/dump.py`, which disables the head's decoder,
with a matching checkpoint. Do not assume that the historical `0.0.1`
download below uses the current preprocessing/decoder contract; it is marked
for pre-0.1.0 code and is not an accuracy-validation fixture for this demo.

Run from a writable directory; the result is saved as `out.jpg` without a GUI:

```shell
LD_LIBRARY_PATH="$MGE_INSTALL_PATH/lib:${LD_LIBRARY_PATH:-}" \
    ./build/yolox /path/to/model.mge ../../../assets/dog.jpg cpu 1 1 0 0 0
# Device, warmup count, thread count, fast-run, weight preprocessing, fp16.
# For the CPU thread pool use: multithread 1 2 0 0 0
```

### Numerical validation on Debian 12 / GCC 12

The complete pinned CPU SDK finished building/installing, and the YOLOX C++
demo configured, compiled and linked successfully with GCC 12.2, CMake 3.25.1
and OpenCV 4.6.0 on 2026-10-25. The executable loaded its real SDK library and
returned exit 1 with usage when invoked without arguments. The shell wrapper
remains executable and limits build parallelism to two jobs by default.
Only OpenCV core/imgproc/imgcodecs are linked; no GUI session is needed.

On 2026-10-26 the same SDK was rebuilt/reinstalled with the validated BLAS profile
above, then the demo relinked and exercised with a freshly exported pretrained
Nano model. The original OpenBLAS revision remains pinned
(`5dde4e65d321076582a2fafe16949d2160551e81`). Its unmodified build really passed
only **16/25** numerical suites: CTest's misleading 20/25 count hid additional
crashes and numerical failures. `-O2` alone still failed (17/25); adding
`-fno-ipa-ra` produced **25/25** in both standalone and SDK builds, at one and
two threads. Independent scalar-reference real/complex GEMV probes also passed.
No failing test was disabled. This workaround is scoped to GCC 12/Haswell;
the old assembly still has invalid input-only operand declarations. OpenBLAS
0.3.6's upstream changelog documents later fixes for that defect class.

The fresh-model comparison found and fixed three separate integration bugs:
FPN bilinear rather than nearest-neighbor upsampling, selecting the legacy graph
loader for FBS_V2 exports, and premature float rounding of the resize ratio
(a 301-pixel width became 639 rather than 640). The ratio now stays double until
integer dimensions are computed. Input tensors now match Python byte-for-byte
on all four tested geometries.

Append a prefix to the ordinary CLI to save validation data:

```shell
./build/yolox /tmp/nano.mge ../../../assets/dog.jpg cpu 1 1 0 0 0 /tmp/check
# /tmp/check.input.f32: float32 [1,3,640,640], native endianness
# /tmp/check.output.f32: float32 [1,8400,85], raw predecode predictions
# /tmp/check.detections.csv: label,score,x,y,width,height (no header)
```

The [isolated Python exporter](../python/README.md) can save eager predictions.
For an independent higher-precision reference, load the same trusted checkpoint
in the modern environment, use `model.eval().double()`, disable
`model.head.decode_in_inference`, and run the identical saved float32 input cast
to float64. Compare all values, not just high-confidence predictions:

| Input geometry (width × height) | Threads | Detections | Maximum raw error vs FP64 |
| --- | --- | --- | --- |
| Dog image, 768 × 576 | 1 and 2 | 3 (dog, bicycle, truck) | 0.000151 |
| Resized portrait, 341 × 613 | 1 and 2 | 3 | 0.000215 |
| Resized small image, 301 × 187 | 1 and 2 | 4 | 0.000180 |
| Blank, 911 × 317 | 1 and 2 | 0 | 0.000475 |

All eight runs passed absolute budgets of **0.001 for every raw value** and
**0.0001 for every objectness/class probability**. A 0.001 center-coordinate
coefficient is 0.032 input pixels at stride 32; the same log-size coefficient
is approximately 0.1% relative box size. Independently decoded detections
matched classes/counts, with maximum box-coordinate error **0.000272 pixels**.
NMS comparisons used the C++ policy: every passing anchor/class pair,
class-agnostic suppression, threshold 0.45, confidence 0.25, no inclusive-pixel
area adjustment, and clipping after NMS. The Python demo's best-class-only
policy is different and must not be assumed to produce identical lists.

The initial mixed `atol=1e-4, rtol=1e-3` diagnostic was **not universally met**:
1/0/1/10 coefficients per image failed in native runs, all on near-zero,
background box channels. No outputs were excluded from the final budgets.
The residuals are bounded relative to measured FP32 backend variation:
default PyTorch FP32 differs from FP64 by up to 0.000820 on these inputs.
This is bounded numerical parity, not bit-identical execution or a COCO AP test.

Negative controls reject wrong input names/shapes, multiple outputs, wrong
output shapes/dtypes, nonfinite predictions, unreadable images and unwritable
dump paths. Both FBS and FBS_V2 loading were executed. Missing real checkpoint
weights are rejected instead of silently using random parameters.

The earlier exploratory historical pre-0.1.0 model produced implausible
detections. Its cause was not isolated, and it remains an unsuitable fixture;
do not attribute those detections to a proven BLAS or preprocessing root cause.
Only Linux x86-64 CPU FP32 Nano inference is validated here. Other model sizes,
Android/ARM/CUDA, FP16, fast-run/weight-preprocessing options, other compilers,
Windows/macOS and full-dataset accuracy remain unverified.

## Historical platform tutorial (not a current validation matrix)

The Android/CUDA instructions and benchmark numbers below are retained for
reference, not evidence that these targets were rebuilt with v1.13.3. Prefer
the pinned CPU recipe above for a host build.

### Step1: install toolchain

	* host: sudo apt install gcc/g++ (gcc/g++, which version >= 6) build-essential git git-lfs gfortran libgfortran-6-dev autoconf gnupg flex bison gperf curl zlib1g-dev gcc-multilib g++-multilib cmake
 * cross build android: download [NDK](https://developer.android.com/ndk/downloads)
   	* after unzip download NDK, then export NDK_ROOT="path of NDK"

### Step2: build MegEngine

```shell
git clone https://github.com/MegEngine/MegEngine.git

# then init third_party
 
export megengine_root="path of MegEngine"
cd $megengine_root && ./third_party/prepare.sh && ./third_party/install-mkl.sh

# build example:
# build host without cuda:   
./scripts/cmake-build/host_build.sh
# or build host with cuda:
./scripts/cmake-build/host_build.sh -c
# or cross build for android aarch64: 
./scripts/cmake-build/cross_build_android_arm_inference.sh
# or cross build for android aarch64(with V8.2+fp16): 
./scripts/cmake-build/cross_build_android_arm_inference.sh -f

# after build MegEngine, you need export the `MGE_INSTALL_PATH`
# host without cuda: 
export MGE_INSTALL_PATH=${megengine_root}/build_dir/host/MGE_WITH_CUDA_OFF/MGE_INFERENCE_ONLY_ON/Release/install
# or host with cuda: 
export MGE_INSTALL_PATH=${megengine_root}/build_dir/host/MGE_WITH_CUDA_ON/MGE_INFERENCE_ONLY_ON/Release/install
# or cross build for android aarch64: 
export MGE_INSTALL_PATH=${megengine_root}/build_dir/android/arm64-v8a/Release/install
```
* you can refs [build tutorial of MegEngine](https://github.com/MegEngine/MegEngine/blob/master/scripts/cmake-build/BUILD_README.md) to build other platform, eg, windows/macos/ etc!

### Step3: build OpenCV

```shell
git clone https://github.com/opencv/opencv.git

git checkout 3.4.15 (we test at 3.4.15, if test other version, may need modify some build)
```

- patch diff for android:

```
# ```
#     diff --git a/CMakeLists.txt b/CMakeLists.txt
#     index f6a2da5310..10354312c9 100644
#     --- a/CMakeLists.txt
#     +++ b/CMakeLists.txt
#     @@ -643,7 +643,7 @@ if(UNIX)
#        if(NOT APPLE)
#          CHECK_INCLUDE_FILE(pthread.h HAVE_PTHREAD)
#          if(ANDROID)
#     -      set(OPENCV_LINKER_LIBS ${OPENCV_LINKER_LIBS} dl m log)
#     +      set(OPENCV_LINKER_LIBS ${OPENCV_LINKER_LIBS} dl m log z)
#          elseif(CMAKE_SYSTEM_NAME MATCHES "FreeBSD|NetBSD|DragonFly|OpenBSD|Haiku")
#            set(OPENCV_LINKER_LIBS ${OPENCV_LINKER_LIBS} m pthread)
#          elseif(EMSCRIPTEN)
    
# ```
```

- build for host

```shell
cd root_dir_of_opencv
mkdir -p build/install
cd build
cmake -DBUILD_JAVA=OFF -DBUILD_SHARED_LIBS=ON -DCMAKE_INSTALL_PREFIX=$PWD/install 
make install -j32
```

* build for android-aarch64

```shell
cd root_dir_of_opencv
mkdir -p build_android/install
cd build_android

cmake -DCMAKE_TOOLCHAIN_FILE="$NDK_ROOT/build/cmake/android.toolchain.cmake" -DANDROID_NDK="$NDK_ROOT"  -DANDROID_ABI=arm64-v8a -DANDROID_NATIVE_API_LEVEL=21 -DBUILD_JAVA=OFF -DBUILD_ANDROID_PROJECTS=OFF -DBUILD_ANDROID_EXAMPLES=OFF -DBUILD_SHARED_LIBS=ON -DCMAKE_INSTALL_PREFIX=$PWD/install ..

make install -j32
```

* after build OpenCV, you need export  `OPENCV_INSTALL_INCLUDE_PATH ` and `OPENCV_INSTALL_LIB_PATH`

```shell
# host build: 
export OPENCV_INSTALL_INCLUDE_PATH=${path of opencv}/build/install/include
export OPENCV_INSTALL_LIB_PATH=${path of opencv}/build/install/lib
# or cross build for android aarch64:
export OPENCV_INSTALL_INCLUDE_PATH=${path of opencv}/build_android/install/sdk/native/jni/include
export OPENCV_INSTALL_LIB_PATH=${path of opencv}/build_android/install/sdk/native/libs/arm64-v8a
```

###  Step4: build test demo

```shell
cmake -S . -B build -DMGE_INSTALL_PATH=/path/to/megengine -DOpenCV_DIR=/path/to/opencv/lib/cmake/opencv4 -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release --parallel
# Or use ./build.sh with the same -D options. Cross builds must also pass
# -DCMAKE_TOOLCHAIN_FILE=... (and Android ABI/API options when applicable).
# The binary is in build/ (or build/Release/ on multi-config generators).

# if host:
export CXX=g++
./build.sh
# or cross android aarch64
export CXX=aarch64-linux-android21-clang++
./build.sh
```

### Step5: run demo

> **Note**: two ways to get `yolox_s.mge` model file
>
> * reference to python demo's `dump.py` script.
> * For users with code before 0.1.0 version, wget yolox-s weights [here](https://github.com/Megvii-BaseDetection/storage/releases/download/0.0.1/yolox_s.mge).
> * For users with code after 0.1.0 version, use [python code in megengine](../python) to generate mge file.

```shell
# if host:
LD_LIBRARY_PATH=$MGE_INSTALL_PATH/lib/:$OPENCV_INSTALL_LIB_PATH ./yolox yolox_s.mge ../../../assets/dog.jpg cuda/cpu/multithread <warmup_count> <thread_number>

# or cross android
adb push/scp $MGE_INSTALL_PATH/lib/libmegengine.so android_phone
adb push/scp $OPENCV_INSTALL_LIB_PATH/*.so android_phone
adb push/scp ./yolox yolox_s.mge android_phone
adb push/scp ../../../assets/dog.jpg android_phone

# login in android_phone by adb or ssh
# then run: 
LD_LIBRARY_PATH=. ./yolox yolox_s.mge dog.jpg cpu/multithread <warmup_count> <thread_number> <use_fast_run> <use_weight_preprocess>  <run_with_fp16>

# * <warmup_count> means warmup count, valid number >=0
# * <thread_number> means thread number, valid number >=1, only take effect `multithread` device
# * <use_fast_run> if >=1 , will use fastrun to choose best algo
# * <use_weight_preprocess> if >=1, will handle weight preprocess before exe
# * <run_with_fp16> if >=1, will run with fp16 mode
```

## Bechmark

* model info: yolox-s @ input(1,3,640,640)					

* test devices

```
  * x86_64  -- Intel(R) Xeon(R) CPU E5-2620 v4 @ 2.10GHz					
  * aarch64 -- xiamo phone mi9					
  * cuda    -- 1080TI @ cuda-10.1-cudnn-v7.6.3-TensorRT-6.0.1.5.sh @ Intel(R) Xeon(R) CPU E5-2620 v4 @ 2.10GHz
```

  | megengine @ tag1.4(fastrun + weight\_preprocess)/sec | 1 thread |
  | ---------------------------------------------------- | -------- |
  | x86\_64                                              | 0.516245 |
  | aarch64(fp32+chw44)                                  | 0.587857 |

  | CUDA @ 1080TI/sec   | 1 batch    | 2 batch   | 4 batch   | 8 batch   | 16 batch  | 32 batch | 64 batch |
  | ------------------- | ---------- | --------- | --------- | --------- | --------- | -------- | -------- |
  | megengine(fp32+chw) | 0.00813703 | 0.0132893 | 0.0236633 | 0.0444699 | 0.0864917 | 0.16895  | 0.334248 |

## Acknowledgement

* [MegEngine](https://github.com/MegEngine/MegEngine)
* [OpenCV](https://github.com/opencv/opencv)
* [NDK](https://developer.android.com/ndk)
* [CMAKE](https://cmake.org/)
