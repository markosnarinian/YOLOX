# Model asset provenance

Do not treat a filename or a successful download as an integrity guarantee.
Download weights only from a trusted publisher, record the exact URL and digest,
and keep that record with the experiment/export. Safe weights-only loading is
used by YOLOX but does not replace source/integrity checks.

The following upstream asset was downloaded and used for CPU inference and ONNX
export validation on 2026-10-25:

| Asset | Publisher URL | SHA-256 |
| --- | --- | --- |
| YOLOX Nano PyTorch checkpoint | https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_nano.pth | `cd28f55fbbc1829f99d9ac9b38a16d259a22889739c8728ea877610201feff7b` |

Verify a downloaded file portably:

```shell
python -c "import hashlib,pathlib; p=pathlib.Path('yolox_nano.pth'); assert hashlib.sha256(p.read_bytes()).hexdigest() == 'cd28f55fbbc1829f99d9ac9b38a16d259a22889739c8728ea877610201feff7b'"
```

This digest records the bytes tested here, not a publisher signature. Other
legacy downloadable models/SDKs have not been integrity-certified. Regenerate
ONNX models with the locked export environment and retain their hashes along
with preprocessing size, decoder settings, and label order. TensorRT engines
are target-runtime/GPU-specific; do not distribute one as a portable model.

## MegEngine validation (2026-10-26)

The optional legacy exporter was isolated from the main uv environment:

| Asset | Publisher URL | SHA-256 |
| --- | --- | --- |
| MegEngine 1.13.0 CPython 3.10 Linux x86-64 wheel | https://files.pythonhosted.org/packages/03/0a/b33e59e56f670df402442d963ce66d72f393593b1addf36cb9d86cc3dd1d/MegEngine-1.13.0-cp310-cp310-manylinux2014_x86_64.whl | `1f93ed6eecc2729137b0b1677ec6b2e06707826c2161794b850c0ba931136dd4` |

The wheel digest comes from PyPI's HTTPS JSON/index metadata; uv validates
download hashes. The v1.13.3 wheel was unavailable there and the legacy index
had an expired TLS certificate; verification was not bypassed. The tested
combination is exporter 1.13.0 and separately built native SDK v1.13.3, not a
claim of arbitrary interchangeability. Python/NumPy/setuptools pins and commands
are in `demo/MegEngine/python/README.md`.

The original Nano checkpoint above was digest-verified again, converted through
plain NPZ arrays, strictly loaded, and freshly exported with the corrected
nearest-neighbor FPN. The tested graph was
`/tmp/yolox-megengine-validation/dog.mge`, SHA-256
`d0362ada70878898df62edf033bae7aa5f8abddb38fa4baca92867882c2e5970`.
Its input is float32 [1,3,640,640], BGR 0–255 with top-left letterboxing/pad 114;
its sole output is raw, undecoded float32 [1,8400,85], COCO class order. The
graph uses FBS_V2 with inference optimization/convolution-bias fusion enabled.
The same graph, not one traced per image, passed all four native input cases.
This records one tested export's bytes, not guaranteed binary reproducibility
across exporter/toolchain versions or a publisher signature for generated data.
