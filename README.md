# Fork documentation

This section documents the additions and changes in
[markosnarinian/YOLOX](https://github.com/markosnarinian/YOLOX) relative to
[upstream YOLOX at `6ddff48`](https://github.com/Megvii-BaseDetection/YOLOX/commit/6ddff4824372906469a7fae2dc3206c7aa4bbaee).
The fork retains upstream's detector architectures, pretrained models, datasets,
training recipes, assignment visualization, logging integrations and deployment
examples; those are **not new fork features**. The changes below modernize their
installation, correctness and portability. The original introduction, benchmarks
and tutorials follow this section; their historical benchmark figures are not
new measurements of this fork. These instructions take precedence over older
toolchain assumptions in the retained upstream material.

**Contents**

- [Installation and compatibility](#installation-and-compatibility)
- [Training, resume and distributed evaluation](#training-resume-and-distributed-evaluation)
- [Experiments, datasets and utility APIs](#experiments-datasets-and-utility-apis)
- [Interactive and headless demos](#interactive-and-headless-demos)
- [Export and portable ONNX inference](#export-and-portable-onnx-inference)
- [Reference deployment integrations](#reference-deployment-integrations)
- [Development, automation and documentation](#development-automation-and-documentation)
- [Validation evidence and deployment boundaries](#validation-evidence-and-deployment-boundaries)

## Installation and compatibility

Use **CPython 3.12 or 3.13**; `.python-version` selects 3.12. The package remains
version 0.3.0, but older Python versions are no longer supported. Install
[uv](https://docs.astral.sh/uv/getting-started/installation/), then run from the
repository root:

```sh
uv sync --locked --extra cpu --extra export
uv run --no-sync python tools/demo.py --help
```

- `cpu` selects PyTorch's CPU wheels. For NVIDIA development on Linux/Windows,
  replace it with `cuda`, which selects **CUDA 13.0** wheels and requires a
  compatible NVIDIA driver. These profiles are mutually exclusive.
- The declared baseline is torch 2.13+, torchvision 0.28+ and NumPy 1.26–2.x
  within the upper bounds in [pyproject.toml](pyproject.toml); [uv.lock](uv.lock)
  pins the resolved environment. Linux PyTorch wheels require compatible glibc
  (manylinux 2.28). Do not infer compatibility from a Python version alone.
- `export` installs ONNX, ONNX Runtime and the modern onnxsim dependency;
  `mlflow` and `wandb` enable the corresponding optional loggers. The default
  `dev` group supplies tests, build, lint and audit tools; add `--group docs`
  for Sphinx. A normal install still includes PyTorch and GUI OpenCV.
- Pip remains supported: `python -m pip install -e '.[export]'`.
  `requirements.txt` delegates to project metadata. Pip does **not** implement
  uv's index/profile selection: choose platform-compatible torch/torchvision
  binaries yourself. Do not mix GUI and headless OpenCV distributions.
- Isolated setuptools builds use static project metadata and pybind11 without
  requiring Torch at build time. The C++14 fast COCO extension is optional;
  unavailable native/JIT evaluation falls back to pycocotools with a warning.
  Wheels include default experiments and `yolox.tools`, so installed tools can
  also run as `python -m yolox.tools.export_onnx` outside the checkout.

CPU inference and export are supported; **training/evaluation CLIs require
CUDA**, and distributed execution is Linux-only. Windows CUDA is experimental
single-device use. The public model factory accepts CPU/CUDA devices, defaults
to CUDA when available and CPU otherwise, and rejects MPS/other backends rather
than silently substituting CPU. The PyTorch demo spells CUDA as `--device gpu`.
Intel macOS and Windows ARM64 are outside the selected binary baseline;
Linux ARM64, Jetson and ROCm need separately validated vendor-compatible builds.
See [platform support and checkpoint trust](docs/platform_support.md) for the
complete support policy, including Apple Silicon CPU and CoreML limitations.

## Training, resume and distributed evaluation

Use the existing [Quick Start training instructions](#quick-start) with a CUDA environment.
For example, resume the latest checkpoint in an experiment's output directory,
or select one explicitly:

```sh
uv run --no-sync python tools/train.py -n yolox-nano -d 1 -b 8 --resume
uv run --no-sync python tools/train.py -n yolox-nano -d 1 -b 8 --resume -c path/to/latest_ckpt.pth
```

- New checkpoints keep inference/EMA weights in `model`, live weights in
  `training_model`, and optimizer, AMP `scaler`, `ema_updates`, epoch and best-AP
  metadata separately. Resume restores these states instead of pairing EMA
  weights with the live optimizer. `latest` is saved after evaluation so its
  best score is current and a resumed run does not overwrite a better checkpoint.
- Non-EMA DDP saves unwrapped model keys; resume also accepts legacy uniformly
  `module.`-prefixed states. Legacy checkpoints without live weights remain
  loadable, but EMA resume warns that exact interrupted training state cannot
  be recovered. Missing legacy scaler/EMA-update metadata uses existing defaults.
  This is not a guarantee of bitwise-identical continuation or restored RNG state.
- Checkpoint loaders explicitly use `weights_only=True`, including model
  downloads, demos, evaluation, export and fine-tuning. Load **trusted** files
  only; arbitrary pickled Python objects are not an implicit compatibility path.
- Training and assignment use the modern `torch.amp` APIs. Distributed
  validation synchronizes the actual evaluated model's normalization buffers
  (including EMA), and COCO aggregation keeps one complete prediction set per
  image, including empty sets, rather than counting padded sampler duplicates.
- Rank-zero COCO JIT loading no longer requires an unmatched collective barrier.
  MLflow checkpoint artifacts are logged only on rank zero. An unset or blank
  `YOLOX_MLFLOW_RUN_NAME` now lets MLflow choose the name; other upstream
  [MLflow settings](docs/mlflow_integration.md) remain unchanged.
- CUDA discovery/memory reporting uses Torch rather than shelling out to
  `nvidia-smi`; memory occupation skips nonpositive allocations, CPU GPU-memory
  reporting returns zero, and CUDA memory queries require an available device.
  NCCL discovers interfaces/GIDs itself: existing user tuning is preserved and
  only an unset `NCCL_LAUNCH_MODE` defaults to `PARALLEL`.
- The launcher rejects absent/invalid GPU counts and non-Linux distributed
  launches. Distributed RAM caching (`--cache` or `--cache ram`) uses `fork`;
  `--cache disk` no longer unnecessarily selects that start method. These
  restrictions do not add CPU training or Windows distributed support.

## Experiments, datasets and utility APIs

- Trailing experiment overrides parse booleans explicitly: `ema False`,
  `enable_mixup 0`, `ema true` and `ema 1` work case-insensitively; other boolean
  strings raise an error. Existing tuple/list overrides remain available, e.g.
  `test_size '(416,416)'`. Put overrides **after all CLI options**.
- `get_exp(exp_file=...)` loads a custom experiment by absolute file identity,
  preventing collisions between same-named files in different directories. Its
  overridden state survives reconstruction in a fresh worker interpreter;
  the original file path must remain accessible. Import failures retain their
  cause and temporary import-path changes are restored. Experiment files are
  executable Python and must be trusted.
- Disk caching now initializes the shared cache state correctly. Use
  `--cache disk` with the existing [custom cache contract](docs/cache.md);
  caching is off unless requested. COCO boxes starting outside an image are
  clipped using their original endpoints, not enlarged by a clipped origin.
  VOC result writing handles empty detection arrays on NumPy 2.x.
- Assignment visualization now handles images with no labels and uses a valid
  periodic logging hook rather than expecting training-loss metrics. Its
  upstream [usage and options](docs/assignment_visualization.md) are unchanged.
- `adjust_status` restores each module's original training/evaluation mode even
  when the context raises; recursive `replace_module(..., replace_func=...)`
  forwards the custom callback to nested modules. Fine-tuning `load_ckpt` can
  remove a uniform legacy DDP prefix when no unprefixed keys match.

## Interactive and headless demos

The PyTorch image demo now actually displays results when saving is not
requested. Image/video/webcam **save-only mode avoids GUI calls**, and video
capture/writers are released. Run on a headless host with:

```sh
uv run --no-sync python tools/demo.py image -n yolox-nano -c yolox_nano.pth \
    --path assets/dog.jpg --device cpu --save_result
```

Without `--save_result`, OpenCV needs a working graphical session: images wait
for a key, video polls keys, and **Esc, q or Q** exits. These keys are retained
upstream shortcuts, not new fork bindings; save-only mode does not wait for them.
The PyTorch demo defaults to CPU, confidence 0.3 and NMS 0.3; `--conf`, `--nms`
and `--tsize` override these. CPU `--fp16` is rejected (except the separate TRT
path); TensorRT warm-up now matches the model's device and dtype. Results use
the existing experiment output/`vis_res` timestamp directories.
The MegEngine Python demo receives equivalent no-save/save-only and resource
cleanup repairs; it retains its own thresholds and legacy dependencies.

## Export and portable ONNX inference

The exporter now uses public `torch.onnx.export` with `dynamo=False`, defaults
to **opset 17** (formerly 11), simplifies unless `--no-onnxsim` is supplied,
and propagates failures as unsuccessful CLI exits. Batch defaults to 1;
`--dynamic` marks the **batch axis only**, not spatial dimensions. Select size
through the experiment, without editing exporter source:

```sh
uv run --no-sync python tools/export_onnx.py -n yolox-nano -c yolox_nano.pth \
    --output-name yolox_nano.onnx test_size '(416,416)'
uv run --no-sync python demo/ONNXRuntime/onnx_inference.py \
    -m yolox_nano.onnx -i assets/dog.jpg -o results \
    --providers CPUExecutionProvider --threads 1
```

Verify the trusted checkpoint using [model provenance and hashes](docs/model_assets.md)
before loading it. The runtime demo expects **raw, undecoded** predictions:
do not export with `--decode_in_inference` for this path.

- Static height/width is inferred from the graph, including Nano's default
  416×416; there is no longer an unconditional 640×640 default.
  `--input_shape H,W` must match static dimensions and is required for dynamic
  spatial dimensions. Sizes must be positive multiples of 32.
- The demo requires one float32 NCHW input, batch 1 with 3 channels, and one
  finite float32 output `[1,anchors,85]` for COCO-80, strides 8/16/32.
  P6, other class counts, embedded decode/NMS and arbitrary ONNX detectors are
  not supported by this runner. Shape validation cannot identify a decoded
  tensor that happens to have the same shape.
- Input remains BGR 0–255, aspect-preserving resize and top-left letterboxing
  with padding 114, then float32 CHW; no RGB conversion or normalization.
- Ordered `--providers` defaults to `CPUExecutionProvider`; every requested
  name must be installed or the demo fails. GPU providers require the matching
  ONNX Runtime distribution/libraries; this is independent of uv's Torch profile.
  Provider availability alone does not certify accelerator numerical accuracy.
  `--threads` sets positive intra-op CPU threads; omitted means ORT's default.
- `-s/--score_thr` defaults to 0.3 for rendering; candidate filtering stays 0.1
  and class-agnostic NMS IoU stays 0.45. Default model/image/output paths are
  `yolox.onnx`, `test_image.png`, `demo_output`. The result uses the input basename.
  Unreadable images, missing models, incompatible shapes/output types, nonfinite
  predictions and failed image writes exit unsuccessfully.
- `yolox.deploy` exposes `COCO_CLASSES`, `preproc`, `demo_postprocess`,
  `multiclass_nms`, `mkdir` and `vis` without importing Torch. Existing training
  import paths re-export the moved helpers for compatibility. This is **not**
  a separately published lightweight package; normal installation includes Torch.

For inference without the training stack, follow the
[standalone headless deployment guide](docs/cm5_deployment.md): export on the
development host, verify model/archive hashes, transfer the source-only bundle,
then install `demo/ONNXRuntime/requirements-cm5.txt` with
`uv pip sync --require-hashes --only-binary :all:` in a separate Python 3.13
environment. Its six-package lock includes headless OpenCV, NumPy and ORT, not
Torch. It was resolved for ARM64 manylinux 2.28; resolution is not ARM64 execution.
Do not run root `uv sync` to create this minimal environment. The guide includes
the exact bundle, installation and lock-regeneration commands.

## Reference deployment integrations

These upstream examples remain **reference/legacy integrations**, not additional
maintained deployment targets. Installing the Python package does not install
their SDKs or certify hardware. The fork's net changes are:

- **[OpenVINO](demo/OpenVINO/README.md):** Python/C++ use the 2025.4.1 Runtime
  `Core`, compiled-model and tensor APIs instead of removed Inference Engine
  APIs. Direct ONNX or IR loading and modern conversion are documented. Host
  tensors are explicitly FP32 even for internally FP16 graphs; input/output
  contracts and unreadable images are checked. Python infers static size and
  uses lightweight deployment helpers; C++ remains fixed Nano-416/COCO-80.
  [Python setup/options](demo/OpenVINO/python/README.md) and
  [C++ builds](demo/OpenVINO/cpp/README.md) cover CPU defaults, thresholds,
  CMake 3.16+/C++17, SDK/OpenCV discovery and native-versus-wheel C++ ABI limits.
  Python's `--labels`/`--number_top` are accepted but unused; do not rely on them.
- **[TensorRT C++](demo/TensorRT/cpp/README.md):** CMake 3.18+/C++14 discovers
  CUDA/TensorRT/OpenCV instead of developer-specific paths; configure
  `TENSORRT_ROOT` and `CUDAToolkit_ROOT` as needed. Array buffers are correctly
  freed. This remains the TensorRT **7/8 binding API**, not TensorRT 10 support;
  fixed 640×640, GPU 0 and default bindings `input_0`/`output_0` require a matching
  engine (or source edits). The tested native 8.6.1.6/CUDA 12 SDK is separate
  from Torch's CUDA 13 profile. The guide records the builder-only executable-stack
  compatibility limitation on newer glibc; it is not a recommended global setting.
- **MegEngine [Python](demo/MegEngine/python/README.md) and
  [C++](demo/MegEngine/cpp/README.md):** strict weight loading, unregistered
  derived grids and nearest-neighbor FPN upsampling align the source model with
  PyTorch. A pickle-free NPZ weight bridge separates modern Torch from the
  isolated Python 3.10/MegEngine 1.13.0 exporter. Deterministic float32 export
  supports `--input-npy`/`--reference-npy`. C++ detects graph format, preserves
  double-precision resize geometry, validates input/output and finite values,
  and optionally dumps input/output/detections via `validation_dump_prefix`.
  It retains fixed 640×640/COCO-80. CMake 3.18+ accepts `MGE_INSTALL_PATH`,
  `OpenCV_DIR` and toolchain files; `build.sh` defaults to two build jobs,
  overridable with `CMAKE_BUILD_PARALLEL_LEVEL`. The manual fail-closed
  `validate_openblas.py` gate checks all 25 numerical suites at `--threads 1`
  or `2`; it is not automatically run by CMake. The native 1.13.3/GCC12/Haswell
  recipe documents `-O2 -fno-ipa-ra`, not a universal BLAS repair. Follow the
  linked recipes for defaults, conversion, numerical budgets and untested modes.
- **[ncnn C++](demo/ncnn/cpp/README.md):** invocation now supports
  `yolox image [model.param model.bin]`, defaulting to `yolox.param`/`yolox.bin`.
  Loading, input/extraction failures and invalid packed/output dimensions fail
  instead of being decoded unchecked. It retains a graphical blocking demo,
  640×640 input, confidence 0.25 and IoU 0.45; this is not a new headless runner.
- **[ncnn Android](demo/ncnn/android/README.md):** pinned Gradle 8.13,
  AGP 8.13.2/JDK 17, compile/target SDK 36, Build Tools 35.0.0,
  NDK 28.2.13676358, CMake 3.22.1 and ncnn 20250916; minSdk remains 24,
  ABIs ARM64/ARMv7. Google/Maven Central replace JCenter and the wrapper has a
  distribution checksum. A case-colliding Java duplicate is removed. Separate
  CPU/Vulkan networks replace the removed extractor Vulkan setter; requesting
  GPU without a Vulkan device returns no result. Stage the documented SDK and
  **matching param/bin pair** yourself, replacing the historical param, not just
  adding a BIN. Build products/SDKs/local settings are ignored. Debug APK build,
  lint/signature/ZIP alignment evidence is not device inference certification;
  ARM64 ELF segments are 16 KB aligned, ARMv7 remains 4 KB.
- **[nebullvm](demo/nebullvm/README.md):** the implementation places model and
  synthetic inputs on CUDA when available, otherwise CPU, fixing the unreachable
  CPU fallback. The retained tutorial snippet/benchmark is legacy, not fork
  performance evidence or a pretrained-model accuracy check.

## Development, automation and documentation

Run the repository checks in the synced environment:

```sh
uv run --no-sync pytest -q
uv run --no-sync ruff check .
uv lock --check
git diff --check
uv build
uv sync --locked --extra cpu --extra export --group docs
uv run --no-sync sphinx-build -b html docs /tmp/yolox-docs
```

- Regression tests cover public model inference/training, experiment identity
  and overrides, checkpoint state/best-score preservation, cache/annotation
  correctness, native/fallback COCO, ONNX export/CLI failures, deployment import
  independence and legacy conversion/BLAS validation. Model utility tests now
  snapshot tensor values rather than aliased state dictionaries, and pytest's
  configured test root ensures the regression suite is collected.
- [CI](.github/workflows/ci.yaml) defines Linux/Windows/macOS CPU jobs on Python
  3.12/3.13, tests, isolated distribution builds and installed-wheel inference
  outside the checkout; separate jobs cover minimum Torch/NumPy versions,
  checksum-verified OpenVINO native CPU inference, Sphinx and dependency audits.
  Defined jobs are not evidence they passed. Audits normalize Torch's local
  CPU/CUDA version suffixes for an additional PyPI advisory check.
- [Pre-commit](.pre-commit-config.yaml) uses local uv/Ruff correctness checks,
  plus large-file, JSON/YAML and merge-marker checks, instead of whole-tree
  formatting. [Dependabot](.github/dependabot.yml) checks uv dependencies and
  Actions weekly, grouping Torch/torchvision updates.
- Documentation uses Sphinx 8/MyST and current theme dependencies; Read the
  Docs selects Ubuntu 24.04/Python 3.12. New platform, model-provenance and
  portable-runtime guides are included in the documentation index.
- [.agents/setup](.agents/setup) prepares Debian-style orbs with a compiler,
  CMake and OpenCV system libraries, syncs locked CPU/export/dev/docs dependencies
  and checks finite Nano CPU inference. It expects uv and permission to install
  missing system packages. Generated portals, builds and runtime artifacts stay
  out of version control.
- [AGENTS.md](AGENTS.md) records repository acceptance/toolchain guidance and
  requires documentation to change alongside user-visible behavior, while
  leaving implementation-only work free of unnecessary README edits.

## Validation evidence and deployment boundaries

[Platform support](docs/platform_support.md), [model assets](docs/model_assets.md)
and [PLAN.md](PLAN.md) retain exact runtime versions, hashes, comparisons and
limitations. PLAN contains historical iterations; its **final acceptance scope**
supersedes earlier physical-device gates. Recorded evidence includes Linux and
Apple Silicon CPU tests/builds, an Asus CUDA training-step smoke test, actual
Asus **CPUExecutionProvider** inference of a 640×640 Nano export (maximum raw
absolute difference 0.000183 versus CUDA Torch at `atol=rtol=0.001`), and separate
execution of the **416×416 standalone bundle on the orb CPU without Torch**.
These are distinct artifacts/hosts, not a bitwise-parity or full-dataset AP claim.

The downstream device is Raspberry Pi Compute Module 5. CPU-host inference is
accepted repository deployment-path evidence; **no physical CM5 execution,
camera, latency, thermal or accuracy measurements are claimed or required for
repository completion**. Hardware integration remains downstream work. Native
SDK smoke tests, Android builds and provider availability do not certify every
platform, accelerator, precision mode or long/distributed training run. Historical
Sphinx builds also reported warnings; do not confuse rendered HTML with a
warning-free documentation build. See the linked guides before selecting a
backend or reusing a legacy benchmark.

---

<div align="center"><img src="assets/logo.png" width="350"></div>
<img src="assets/demo.png" >

## Introduction
YOLOX is an anchor-free version of YOLO, with a simpler design but better performance! It aims to bridge the gap between research and industrial communities.
For more details, please refer to our [report on Arxiv](https://arxiv.org/abs/2107.08430).

This repo is an implementation of PyTorch version YOLOX, there is also a [MegEngine implementation](https://github.com/MegEngine/YOLOX).

<img src="assets/git_fig.png" width="1000" >

## Updates!!
* 【2023/02/28】 We support assignment visualization tool, see doc [here](./docs/assignment_visualization.md).
* 【2022/04/14】 We support jit compile op.
* 【2021/08/19】 We optimize the training process with **2x** faster training and **~1%** higher performance! See [notes](docs/updates_note.md) for more details.
* 【2021/08/05】 We release [MegEngine version YOLOX](https://github.com/MegEngine/YOLOX).
* 【2021/07/28】 We fix the fatal error of [memory leak](https://github.com/Megvii-BaseDetection/YOLOX/issues/103)
* 【2021/07/26】 We now support [MegEngine](https://github.com/Megvii-BaseDetection/YOLOX/tree/main/demo/MegEngine) deployment.
* 【2021/07/20】 We have released our technical report on [Arxiv](https://arxiv.org/abs/2107.08430).

## Benchmark

#### Standard Models.

|Model |size |mAP<sup>val<br>0.5:0.95 |mAP<sup>test<br>0.5:0.95 | Speed V100<br>(ms) | Params<br>(M) |FLOPs<br>(G)| weights |
| ------        |:---: | :---:    | :---:       |:---:     |:---:  | :---: | :----: |
|[YOLOX-s](./exps/default/yolox_s.py)    |640  |40.5 |40.5      |9.8      |9.0 | 26.8 | [github](https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_s.pth) |
|[YOLOX-m](./exps/default/yolox_m.py)    |640  |46.9 |47.2      |12.3     |25.3 |73.8| [github](https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_m.pth) |
|[YOLOX-l](./exps/default/yolox_l.py)    |640  |49.7 |50.1      |14.5     |54.2| 155.6 | [github](https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_l.pth) |
|[YOLOX-x](./exps/default/yolox_x.py)   |640   |51.1 |**51.5**  | 17.3    |99.1 |281.9 | [github](https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_x.pth) |
|[YOLOX-Darknet53](./exps/default/yolov3.py)   |640  | 47.7 | 48.0 | 11.1 |63.7 | 185.3 | [github](https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_darknet.pth) |

<details>
<summary>Legacy models</summary>

|Model |size |mAP<sup>test<br>0.5:0.95 | Speed V100<br>(ms) | Params<br>(M) |FLOPs<br>(G)| weights |
| ------        |:---: | :---:       |:---:     |:---:  | :---: | :----: |
|[YOLOX-s](./exps/default/yolox_s.py)    |640  |39.6      |9.8     |9.0 | 26.8 | [onedrive](https://megvii-my.sharepoint.cn/:u:/g/personal/gezheng_megvii_com/EW62gmO2vnNNs5npxjzunVwB9p307qqygaCkXdTO88BLUg?e=NMTQYw)/[github](https://github.com/Megvii-BaseDetection/storage/releases/download/0.0.1/yolox_s.pth) |
|[YOLOX-m](./exps/default/yolox_m.py)    |640  |46.4      |12.3     |25.3 |73.8| [onedrive](https://megvii-my.sharepoint.cn/:u:/g/personal/gezheng_megvii_com/ERMTP7VFqrVBrXKMU7Vl4TcBQs0SUeCT7kvc-JdIbej4tQ?e=1MDo9y)/[github](https://github.com/Megvii-BaseDetection/storage/releases/download/0.0.1/yolox_m.pth) |
|[YOLOX-l](./exps/default/yolox_l.py)    |640  |50.0  |14.5 |54.2| 155.6 | [onedrive](https://megvii-my.sharepoint.cn/:u:/g/personal/gezheng_megvii_com/EWA8w_IEOzBKvuueBqfaZh0BeoG5sVzR-XYbOJO4YlOkRw?e=wHWOBE)/[github](https://github.com/Megvii-BaseDetection/storage/releases/download/0.0.1/yolox_l.pth) |
|[YOLOX-x](./exps/default/yolox_x.py)   |640  |**51.2**      | 17.3 |99.1 |281.9 | [onedrive](https://megvii-my.sharepoint.cn/:u:/g/personal/gezheng_megvii_com/EdgVPHBziOVBtGAXHfeHI5kBza0q9yyueMGdT0wXZfI1rQ?e=tABO5u)/[github](https://github.com/Megvii-BaseDetection/storage/releases/download/0.0.1/yolox_x.pth) |
|[YOLOX-Darknet53](./exps/default/yolov3.py)   |640  | 47.4      | 11.1 |63.7 | 185.3 | [onedrive](https://megvii-my.sharepoint.cn/:u:/g/personal/gezheng_megvii_com/EZ-MV1r_fMFPkPrNjvbJEMoBLOLAnXH-XKEB77w8LhXL6Q?e=mf6wOc)/[github](https://github.com/Megvii-BaseDetection/storage/releases/download/0.0.1/yolox_darknet53.pth) |

</details>

#### Light Models.

|Model |size |mAP<sup>val<br>0.5:0.95 | Params<br>(M) |FLOPs<br>(G)| weights |
| ------        |:---:  |  :---:       |:---:     |:---:  | :---: |
|[YOLOX-Nano](./exps/default/yolox_nano.py) |416  |25.8  | 0.91 |1.08 | [github](https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_nano.pth) |
|[YOLOX-Tiny](./exps/default/yolox_tiny.py) |416  |32.8 | 5.06 |6.45 | [github](https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_tiny.pth) |


<details>
<summary>Legacy models</summary>

|Model |size |mAP<sup>val<br>0.5:0.95 | Params<br>(M) |FLOPs<br>(G)| weights |
| ------        |:---:  |  :---:       |:---:     |:---:  | :---: |
|[YOLOX-Nano](./exps/default/yolox_nano.py) |416  |25.3  | 0.91 |1.08 | [github](https://github.com/Megvii-BaseDetection/storage/releases/download/0.0.1/yolox_nano.pth) |
|[YOLOX-Tiny](./exps/default/yolox_tiny.py) |416  |32.8 | 5.06 |6.45 | [github](https://github.com/Megvii-BaseDetection/storage/releases/download/0.0.1/yolox_tiny_32dot8.pth) |

</details>

## Quick Start

<details>
<summary>Installation</summary>

Step1. Install YOLOX from source.
```shell
git clone https://github.com/markosnarinian/YOLOX.git
cd YOLOX
uv sync --locked --extra cpu --extra export
uv run --no-sync pytest -q
```

Use CPython 3.12 or 3.13; `.python-version` selects 3.12. Install
[uv](https://docs.astral.sh/uv/getting-started/installation/) first. The lockfile
pins development dependencies. `--extra cpu` selects PyTorch's CPU index; on
Linux/Windows NVIDIA systems use `--extra cuda` instead (CUDA 13.0 wheels).
Do not combine the CPU and CUDA profiles. A compatible NVIDIA driver is required.
For Jetson/ROCm, use vendor-compatible PyTorch separately; this lock is not a
validated configuration for those targets.

Pip remains supported: `python -m pip install -e '.[export]'`. Select compatible
torch/torchvision binaries for your platform first; pip does not interpret uv's
index profiles. Optional extras: `export`, `mlflow`, `wandb`. Development and docs
groups are available with `uv sync --extra cpu --extra export --group docs`.
Use `uv run --no-sync python ...` for the commands below after syncing a profile.

See [supported platforms and checkpoint trust](docs/platform_support.md).
Training/evaluation CLIs require CUDA; CPU inference/export is supported. MPS is
explicitly unsupported. Without `--save_result`, demos require a graphical
OpenCV session; save-only mode does not call GUI APIs.

</details>

<details>
<summary>Demo</summary>

Step1. Download a pretrained model from the benchmark table.

Step2. Use either -n or -f to specify your detector's config. For example:

```shell
python tools/demo.py image -n yolox-s -c /path/to/your/yolox_s.pth --path assets/dog.jpg --conf 0.25 --nms 0.45 --tsize 640 --save_result --device [cpu/gpu]
```
or
```shell
python tools/demo.py image -f exps/default/yolox_s.py -c /path/to/your/yolox_s.pth --path assets/dog.jpg --conf 0.25 --nms 0.45 --tsize 640 --save_result --device [cpu/gpu]
```
Demo for video:
```shell
python tools/demo.py video -n yolox-s -c /path/to/your/yolox_s.pth --path /path/to/your/video --conf 0.25 --nms 0.45 --tsize 640 --save_result --device [cpu/gpu]
```


</details>

<details>
<summary>Reproduce our results on COCO</summary>

Step1. Prepare COCO dataset
```shell
cd <YOLOX_HOME>
ln -s /path/to/your/COCO ./datasets/COCO
```

Step2. Reproduce our results on COCO by specifying -n:

```shell
python -m yolox.tools.train -n yolox-s -d 8 -b 64 --fp16 -o [--cache]
                               yolox-m
                               yolox-l
                               yolox-x
```
* -d: number of gpu devices
* -b: total batch size, the recommended number for -b is num-gpu * 8
* --fp16: mixed precision training
* --cache: caching imgs into RAM to accelarate training, which need large system RAM.



When using -f, the above commands are equivalent to:
```shell
python -m yolox.tools.train -f exps/default/yolox_s.py -d 8 -b 64 --fp16 -o [--cache]
                               exps/default/yolox_m.py
                               exps/default/yolox_l.py
                               exps/default/yolox_x.py
```

**Multi Machine Training**

We also support multi-nodes training. Just add the following args:
* --num\_machines: num of your total training nodes
* --machine\_rank: specify the rank of each node

Suppose you want to train YOLOX on 2 machines, and your master machines's IP is 123.123.123.123, use port 12312 and TCP.

On master machine, run
```shell
python tools/train.py -n yolox-s -b 128 --dist-url tcp://123.123.123.123:12312 --num_machines 2 --machine_rank 0
```
On the second machine, run
```shell
python tools/train.py -n yolox-s -b 128 --dist-url tcp://123.123.123.123:12312 --num_machines 2 --machine_rank 1
```

**Logging to Weights & Biases**

To log metrics, predictions and model checkpoints to [W&B](https://docs.wandb.ai/guides/integrations/other/yolox) use the command line argument `--logger wandb` and use the prefix "wandb-" to specify arguments for initializing the wandb run.

```shell
python tools/train.py -n yolox-s -d 8 -b 64 --fp16 -o [--cache] --logger wandb wandb-project <project name>
                         yolox-m
                         yolox-l
                         yolox-x
```

An example wandb dashboard is available [here](https://wandb.ai/manan-goel/yolox-nano/runs/3pzfeom0)

**Others**

See more information with the following command:
```shell
python -m yolox.tools.train --help
```

</details>


<details>
<summary>Evaluation</summary>

We support batch testing for fast evaluation:

```shell
python -m yolox.tools.eval -n  yolox-s -c yolox_s.pth -b 64 -d 8 --conf 0.001 [--fp16] [--fuse]
                               yolox-m
                               yolox-l
                               yolox-x
```
* --fuse: fuse conv and bn
* -d: number of GPUs used for evaluation. DEFAULT: All GPUs available will be used.
* -b: total batch size across on all GPUs

To reproduce speed test, we use the following command:
```shell
python -m yolox.tools.eval -n  yolox-s -c yolox_s.pth -b 1 -d 1 --conf 0.001 --fp16 --fuse
                               yolox-m
                               yolox-l
                               yolox-x
```

</details>


<details>
<summary>Tutorials</summary>

*  [Training on custom data](docs/train_custom_data.md)
*  [Caching for custom data](docs/cache.md)
*  [Manipulating training image size](docs/manipulate_training_image_size.md)
*  [Assignment visualization](docs/assignment_visualization.md)
*  [Freezing model](docs/freeze_module.md)

</details>

## Deployment


1. [MegEngine in C++ and Python](./demo/MegEngine)
2. [ONNX export and an ONNXRuntime](./demo/ONNXRuntime)
3. [TensorRT in C++ and Python](./demo/TensorRT)
4. [ncnn in C++ and Java](./demo/ncnn)
5. [OpenVINO in C++ and Python](./demo/OpenVINO)
6. [Accelerate YOLOX inference with nebullvm in Python](./demo/nebullvm)

## Third-party resources
* YOLOX for streaming perception: [StreamYOLO (CVPR 2022 Oral)](https://github.com/yancie-yjr/StreamYOLO)
* The YOLOX-s and YOLOX-nano are Integrated into [ModelScope](https://www.modelscope.cn/home). Try out the Online Demo at [YOLOX-s](https://www.modelscope.cn/models/damo/cv_cspnet_image-object-detection_yolox/summary) and [YOLOX-Nano](https://www.modelscope.cn/models/damo/cv_cspnet_image-object-detection_yolox_nano_coco/summary) respectively 🚀.
* Integrated into [Huggingface Spaces 🤗](https://huggingface.co/spaces) using [Gradio](https://github.com/gradio-app/gradio). Try out the Web Demo: [![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/Sultannn/YOLOX-Demo)
* The ncnn android app with video support: [ncnn-android-yolox](https://github.com/FeiGeChuanShu/ncnn-android-yolox) from [FeiGeChuanShu](https://github.com/FeiGeChuanShu)
* YOLOX with Tengine support: [Tengine](https://github.com/OAID/Tengine/blob/tengine-lite/examples/tm_yolox.cpp) from [BUG1989](https://github.com/BUG1989)
* YOLOX + ROS2 Foxy: [YOLOX-ROS](https://github.com/Ar-Ray-code/YOLOX-ROS) from [Ar-Ray](https://github.com/Ar-Ray-code)
* YOLOX Deploy DeepStream: [YOLOX-deepstream](https://github.com/nanmi/YOLOX-deepstream) from [nanmi](https://github.com/nanmi)
* YOLOX MNN/TNN/ONNXRuntime: [YOLOX-MNN](https://github.com/DefTruth/lite.ai.toolkit/blob/main/lite/mnn/cv/mnn_yolox.cpp)、[YOLOX-TNN](https://github.com/DefTruth/lite.ai.toolkit/blob/main/lite/tnn/cv/tnn_yolox.cpp) and [YOLOX-ONNXRuntime C++](https://github.com/DefTruth/lite.ai.toolkit/blob/main/lite/ort/cv/yolox.cpp) from [DefTruth](https://github.com/DefTruth)
* Converting darknet or yolov5 datasets to COCO format for YOLOX: [YOLO2COCO](https://github.com/RapidAI/YOLO2COCO) from [Daniel](https://github.com/znsoftm)

## Cite YOLOX
If you use YOLOX in your research, please cite our work by using the following BibTeX entry:

```latex
 @article{yolox2021,
  title={YOLOX: Exceeding YOLO Series in 2021},
  author={Ge, Zheng and Liu, Songtao and Wang, Feng and Li, Zeming and Sun, Jian},
  journal={arXiv preprint arXiv:2107.08430},
  year={2021}
}
```
## In memory of Dr. Jian Sun
Without the guidance of [Dr. Jian Sun](https://scholar.google.com/citations?user=ALVSZAYAAAAJ), YOLOX would not have been released and open sourced to the community.
The passing away of Dr. Sun is a huge loss to the Computer Vision field. We add this section here to express our remembrance and condolences to our captain Dr. Sun.
It is hoped that every AI practitioner in the world will stick to the belief of "continuous innovation to expand cognitive boundaries, and extraordinary technology to achieve product value" and move forward all the way.

<div align="center"><img src="assets/sunjian.png" width="200"></div>
没有孙剑博士的指导，YOLOX也不会问世并开源给社区使用。
孙剑博士的离去是CV领域的一大损失，我们在此特别添加了这个部分来表达对我们的“船长”孙老师的纪念和哀思。
希望世界上的每个AI从业者秉持着“持续创新拓展认知边界，非凡科技成就产品价值”的观念，一路向前。
