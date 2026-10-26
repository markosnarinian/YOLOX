# Repository guidance

## Scope and validation

- This repository owns working YOLOX code, packaging, training/export tooling
  and portable ONNX inference. Physical deployment-device integration is outside
  this task/repository's acceptance scope unless the user explicitly requests it.
- The user's downstream deployment is Raspberry Pi Compute Module 5, but do not
  require access to a CM5, camera, accelerator, or target thermal/performance
  measurements to complete codebase work or ship validated changes.
- The user accepts actual ONNX Runtime inference on the Asus with the explicit
  CPUExecutionProvider as sufficient deployment-path evidence for this repository
  and will extrapolate it to Raspberry Pi. Do not repeatedly reopen that decision
  or demand target-device certification. Preserve exact artifact/runtime details
  in evidence; do not claim that CPU-host results are measurements on a CM5.
- Validate code changes through relevant tests and real available runtime paths.
  Distinguish repository correctness from optional physical-device certification.
- TensorRT, OpenVINO, MegEngine, Android and other unrelated deployment examples
  are retained reference/legacy integrations, not required supported targets.

## Toolchain

- Use uv with `pyproject.toml`, `uv.lock`, and `.python-version` (Python 3.12;
  the package also supports Python 3.13).
- Development: `uv sync --locked --extra cpu --extra export`.
- NVIDIA development: use `--extra cuda` instead of `--extra cpu`; never combine
  them. This profile requires a compatible driver.
- Checks: `uv run --no-sync pytest -q`, `uv run --no-sync ruff check .`,
  `uv lock --check`, and `git diff --check`. Build distributions with `uv build`.
- The standalone headless ONNX runtime uses
  `demo/ONNXRuntime/requirements-cm5.txt` via `uv pip sync --require-hashes`;
  it deliberately does not install the training stack. See
  `docs/cm5_deployment.md` for bundle/export instructions.
