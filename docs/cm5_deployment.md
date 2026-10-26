# Portable ONNX runtime for downstream CM5 deployment

**Status: repository runtime validated; physical-device integration out of scope.**
The user accepts real CPU-only ONNX Runtime inference on the Asus as sufficient
deployment-path validation and will extrapolate to Raspberry Pi. The 640×640
Nano export passed that comparison; the exact 416×416 standalone bundle below
was also executed on the orb's CPU with no Torch installed/imported. Item 43 does
not require a CM5 connection or hardware certification. No CM5 timing, thermal,
camera or accuracy measurements are claimed.

The proposed downstream configuration is YOLOX-Nano, a static float32 ONNX model,
and ONNX Runtime's CPU provider on ARM64 Linux. Training/export stays on the
development host; inference does not need PyTorch, torchvision or CUDA. Physical
OS/power/cooling/camera integration belongs to the downstream application, not
this repository's shipping criteria.

## 1. Export on the development host

From the repository root, download the trusted Nano checkpoint from the URL in
`model_assets.md` and verify its digest **before loading it**:

```sh
python3 -c "import hashlib,pathlib; assert hashlib.sha256(pathlib.Path('yolox_nano.pth').read_bytes()).hexdigest() == 'cd28f55fbbc1829f99d9ac9b38a16d259a22889739c8728ea877610201feff7b'"
uv sync --locked --extra cpu --extra export
uv run --no-sync python tools/export_onnx.py -n yolox-nano \
    -c yolox_nano.pth --output-name yolox_nano.onnx test_size '(416,416)'
sha256sum yolox_nano.onnx > yolox_nano.onnx.sha256
```

The graph must have one float32 NCHW image input and one **raw, undecoded** COCO
output: `[1,3,416,416]` → `[1,3549,85]`. Do not pass
`--decode_in_inference` to the exporter. Input is BGR 0–255, top-left letterboxed
with value 114. Do not substitute RGB, normalization, or centered padding.
Retain the source checkpoint digest, ONNX digest, source revision/diff and exact
export dependency versions with the model. Generated graph bytes can differ
between export environments; the graph checksum accompanies that specific file.

## 2. Transfer the minimal runtime sources and model

The standalone runtime uses the source tree directly, not a training-package
installation. To make a small archive from the repository root:

```sh
tar --exclude=__pycache__ -czf /tmp/yolox-cm5-runtime.tgz \
    LICENSE yolox/__init__.py yolox/deploy \
    demo/ONNXRuntime/onnx_inference.py \
    demo/ONNXRuntime/requirements-cm5.txt \
    assets/dog.jpg yolox_nano.onnx yolox_nano.onnx.sha256
sha256sum /tmp/yolox-cm5-runtime.tgz
```

Transfer the archive through your trusted deployment channel and compare its
digest before extraction. Extract into a new directory, not over an existing
application or checkout. Model checksums detect corruption/substitution only
when obtained from a trusted source; they are not publisher signatures.

## 3. Install the target-only environment

Proposed target: 64-bit Raspberry Pi OS with a Python 3.13 interpreter. Check
`uname -m`, `getconf LONG_BIT`, `/etc/os-release` and the device-tree model first.
Do not install the training environment, use `uv sync` at the root, or install
both headless and GUI OpenCV into this environment. With uv available, run from
the extracted runtime directory:

```sh
uv venv --python 3.13 .venv
uv pip sync --python .venv/bin/python --require-hashes --only-binary :all: \
    demo/ONNXRuntime/requirements-cm5.txt
sha256sum -c yolox_nano.onnx.sha256
PYTHONPATH="$PWD" .venv/bin/python -c "import yolox.deploy, sys; from importlib.util import find_spec; assert find_spec('torch') is None; assert 'torch' not in sys.modules"
PYTHONPATH="$PWD" .venv/bin/python demo/ONNXRuntime/onnx_inference.py \
    -m yolox_nano.onnx -i assets/dog.jpg -o results \
    --providers CPUExecutionProvider --threads 1
```

The image is saved as `results/dog.jpg`; no display server is required. The CLI
infers static height/width from the model. `--input_shape H,W` is only needed for
dynamic spatial dimensions or an explicit check and must match any static model
dimensions. Missing images/models, incompatible shapes, invalid output tensors
and unsuccessful image writes must exit unsuccessfully. Only raw COCO models
with strides 8/16/32 are covered by this demo's decoder.

The hash-locked requirements contain six packages, including
`opencv-python-headless`, not PyTorch. They were resolved for CPython 3.13 /
ARM64 manylinux 2.28 and installed/executed on x86-64 Linux as a preparation
check. ARM64 execution and target glibc compatibility were not measured; any
physical-device certification is a separate downstream task, not a release gate.
For intentional dependency updates, regenerate with:

```sh
uv pip compile demo/ONNXRuntime/requirements-cm5.in \
    --python-version 3.13 --python-platform aarch64-manylinux_2_28 \
    --only-binary :all: --generate-hashes \
    -o demo/ONNXRuntime/requirements-cm5.txt
```

## 4. Repository acceptance and downstream boundary

Repository acceptance is based on repeatable installation/export, real CPU ONNX
inference, documented numerical comparisons, a standalone runtime without the
training stack, and CLI regression/error controls. These checks have executed;
their exact models, hosts, tolerances and limitations are recorded in `PLAN.md`.
Do not silently relax numerical assertions or label CPU-host results as CM5 data.

Actual target installation, camera capture, power/cooling integration, sustained
latency/RSS/thermal measurements and application accuracy targets are outside
this task. They may be useful to a downstream integrator but do not block shipping
this codebase. Do not request physical-device access unless the user separately
asks for that work. The legacy Android/TensorRT/MegEngine/OpenVINO examples and
additional ncnn/accelerator implementations are not required acceptance targets.
