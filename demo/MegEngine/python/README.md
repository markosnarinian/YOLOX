# YOLOX-Python-MegEngine

Python version of YOLOX object detection base on [MegEngine](https://github.com/MegEngine/MegEngine).

## Isolated Linux exporter

Do not install legacy MegEngine into the project's modern uv environment.
The tested export environment is CPython **3.10.21**, MegEngine **1.13.0**,
NumPy **1.24.4**, OpenCV **4.11.0.86**, and setuptools **80.9.0**. The native
runtime is separately built from MegEngine **v1.13.3**, as documented in the
[C++ recipe](../cpp/README.md). This exact exporter/runtime combination was
checked on Linux x86-64 CPU; it is not a general Python 3.10 package-support claim.

The 1.13.3 wheel is not on PyPI, and the legacy wheel index's TLS certificate
was expired during validation. Do not bypass certificate verification. Instead,
the tested 1.13.0 CPython 3.10 Linux wheel comes from PyPI's verified HTTPS index;
its SHA-256 is recorded in [model provenance](../../../docs/model_assets.md).
It can operate on CPU without a CUDA driver. Setuptools must remain below 81
because this old framework imports `pkg_resources`.

From the repository root:

```shell
uv python install 3.10.21
uv venv --python 3.10.21 /tmp/yolox-megengine-export
export export_python=/tmp/yolox-megengine-export/bin/python
uv pip install --python "$export_python" \
    'megengine==1.13.0' 'numpy==1.24.4' 'opencv-python==4.11.0.86' 'setuptools==80.9.0'
```

### Transfer trusted weights without sharing Python environments

Verify the publisher checkpoint's digest before converting. The NPZ bridge
contains plain arrays with PyTorch shapes; it does not import MegEngine in the
modern environment or Torch in the legacy environment. Pickle checkpoints
produced for MegEngine must also be treated as trusted code/data, not arbitrary
untrusted downloads.

```shell
uv run --locked --extra cpu python demo/MegEngine/python/convert_weights.py \
    -w /path/to/yolox_nano.pth -o /tmp/nano.npz
"$export_python" demo/MegEngine/python/convert_weights.py \
    -w /tmp/nano.npz -o /tmp/nano.pkl
"$export_python" demo/MegEngine/python/dump.py \
    -n yolox-nano -c /tmp/nano.pkl --dump_path /tmp/nano.mge
```

Loading requires all model weights, rather than silently leaving random
parameters. Derived grid caches are not checkpoint parameters. FPN upsampling
uses nearest-neighbor, matching the source PyTorch architectures.

Tracing uses a deterministic float32 input, fixed to **[1,3,640,640]**, with raw
**[1,8400,85]** predictions, `optimize_for_inference` and fused convolution/bias/
nonlinearity. The C++ loader detects both legacy FBS and modern FBS_V2 graph
formats. An old pre-0.1.0 model is not a substitute for this export.

To record eager raw predictions for numerical comparison, supply a saved,
preprocessed float32 input array (BGR 0–255, top-left padding 114):

```shell
"$export_python" demo/MegEngine/python/dump.py -n yolox-nano -c /tmp/nano.pkl \
    --input-npy /tmp/input.npy --reference-npy /tmp/eager.npy --dump_path /tmp/nano.mge
```

Compare the C++ validation dumps against these predictions and a higher-precision
PyTorch reference on **identical input tensors**, not just visual detections.
See the C++ README for executed checks, numerical budgets and limitations.

### Optional Python image demo

The source-only demo also needs Loguru and the repository on `PYTHONPATH`;
do not attempt to install the modern Python package into the legacy environment.

```shell
uv pip install --python "$export_python" 'loguru==0.7.3'
PYTHONPATH="$PWD" "$export_python" demo/MegEngine/python/demo.py image \
    -n yolox-nano -c /tmp/nano.pkl --path assets/dog.jpg \
    --conf 0.25 --nms 0.45 --tsize 640 --save_result
```
