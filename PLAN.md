# Prioritized remediation plan

This is the consolidated, deduplicated fix order from the codebase, foundation,
and cross-platform reviews. Numbers indicate execution order; independent changes
can be validated together. Completion requires implementation and relevant checks,
not merely updating this document.

**Current acceptance scope:** working codebase/export/portable CPU ONNX inference.
The user accepts actual Asus CPUExecutionProvider validation as sufficient to
extrapolate to Raspberry Pi. Physical CM5 integration/certification is outside
this repository/task, not a shipping gate. Historical target-device requirements
below are superseded by the final scope decision recorded at the end.

- **P1 — Release blockers:** installation, lost training state, incorrect results,
  or broken established workflows.
- **P2 — Conditional defects:** configuration- or integration-dependent failures.
- **P3 — Modernization/support expansion:** planned improvements, not necessarily bugs.
- **No P0:** no critical security incident or universal catastrophic failure was established.

## P1 — Foundation and release blockers

- [x] 1. Define the supported Python/platform/accelerator contract; distinguish maintained and experimental targets.
- [x] 2. Repair isolated installation with explicit build requirements and static pyproject metadata, retaining setuptools.
- [x] 3. Establish tested Python/torch/torchvision/NumPy/export dependency ranges and a locked development environment.
- [x] 4. Make CI run tests, isolated builds, installed-wheel inference, extension/fallback and export checks; repair aliased snapshots/discovery and cover promised OS targets.
- [x] 5. Remove the case-colliding duplicate Android Java class.
- [x] 6. Prevent resumed runs from overwriting a better checkpoint through stale best_ap metadata.
- [x] 7. Preserve live model, EMA, optimizer and AMP scaler states separately for resume.
- [x] 8. Make non-EMA distributed checkpoints resumable without DDP key-prefix mismatch.
- [x] 9. Remove the unmatched distributed barrier during COCO JIT loading.
- [x] 10. Synchronize the EMA model actually used for distributed validation.
- [x] 11. Deduplicate distributed COCO predictions by image, not individual boxes.
- [x] 12. Repair modern-PyTorch ONNX export and propagate CLI failure exit status.
- [x] 13. Repair NumPy 2.x VOC detection-array emptiness checks.
- [x] 14. Repair disk-cache initialization.
- [x] 15. Repair MLflow setup with an unspecified run name.
- [x] 16. Restrict MLflow checkpoint logging to rank zero.

## P2 — Correctness and portability defects

- [x] 17. Parse boolean configuration overrides explicitly.
- [x] 18. Load custom experiment files by explicit file identity, including cross-process reconstruction.
- [x] 19. Calculate COCO annotation endpoints before clipping negative origins.
- [x] 20. Use reliable GPU discovery and guard NVIDIA-only memory helpers.
- [x] 21. Replace nonportable NCCL shell probing and preserve user configuration.
- [x] 22. Provide dependable platform-independent extension build/fallback behavior.
- [x] 23. Validate device requests instead of silently ignoring mps.
- [x] 24. Restore temporary model mode in finally.
- [x] 25. Match TensorRT warm-up input device/dtype to the model.
- [x] 26. Correct TensorRT C++ array deallocation.
- [x] 27. Check ncnn loading/extraction and output dimensions before decoding.
- [x] 28. Make default image-demo output visible or explicitly saved.
- [x] 29. Make save-only demo execution independent of GUI functionality.
- [x] 30. Repair MegEngine nonsaving image/video paths.
- [x] 31. Handle empty labels in assignment visualization.
- [x] 32. Give assignment visualization a valid logging hook.
- [x] 33. Make nebullvm CPU fallback reachable.
- [x] 34. Forward custom callbacks during recursive module replacement.

## P3 — Modernization and explicitly scoped support

- [x] 35. Adopt uv project workflows, a lockfile, Python pin and explicit CPU/CUDA profiles while retaining pip support.
- [x] 36. Separate export/logging/development/docs dependencies and modernize the simplifier pin without breaking profiling CLIs.
- [x] 37. Modernize AMP and make checkpoint trust/loading behavior explicit.
- [x] 38. Modernize docs, Read the Docs, CI actions and lint tooling; add dependency/security automation without whole-tree formatting.
- [x] 39. Resolve device-neutral training/evaluation scope: CLIs remain CUDA-only, explicitly documented and guarded; CPU inference/export is supported.
- [x] 40. Explicitly exclude MPS rather than silently ignoring the requested backend.
- [x] 41. Explicitly restrict distributed training/evaluation and its fork-dependent RAM caching to Linux; do not claim Windows distributed support.
- [x] 42. Establish separate policies for Intel Mac, Windows ARM64, Linux ARM64/Jetson and ROCm.
- [x] 43. Validate the repository's portable deployment path: reproducible export and minimal headless runtime, real ONNX CPU inference on Asus, documented numerical comparison and CLI error/regression checks. The user accepts extrapolation to Raspberry Pi; physical CM5 integration/performance/camera certification is outside this task and is not required for completion.
- [x] 44. Improve runtime independence/configuration: ONNX providers, unnecessary torch coupling, portable model paths and reproducible assets.

## Evidence and delivery

The initial reviews are local review artifacts, not proof of fixes. Each implemented
phase records its actual validation below. Hardware-specific claims require matching
hardware; Linux CPU checks cannot establish Windows, macOS, CUDA or mobile execution.
No push, release or deployment is authorized by this plan.

### Implementation and validation record — 2026-10-25

Checked items indicate implemented changes or an explicitly documented support
decision, not certification of untested platforms. Items were addressed in the
listed priority order; the final optional integrations are being validated last.

- **Foundation (1–5, 20–23, 35–42):** static PEP 517/621 packaging; optional
  pybind11 COCO extension without a torch build dependency; CPython 3.12/3.13;
  uv lock and mutually exclusive CPU/CUDA 13.0 profiles. Minimum tested versions
  are torch 2.13/torchvision 0.28 and NumPy 1.26.4. The locked CPU environment
  uses torch 2.14.1/torchvision 0.29.1 and NumPy 2.5.4. Older torch baselines were
  rejected following dependency-advisory checks. Platform limits and checkpoint
  trust are in `docs/platform_support.md`.
- **Regression checks (6–19, 24, 34):** `uv run --no-sync pytest -q`:
  **15 passed** on Python 3.12; the same 15 passed on Python 3.13 and in the
  minimum-dependency environment. Five regression cases also failed against the
  original source before passing with these changes. Custom experiments with
  identical basenames retain their identities and overrides when unpickled in
  a fresh interpreter. Native COCO and forced pycocotools fallback both pass.
- **Distributed controls (8–11, 16):** a real two-rank CPU/gloo exercise passed
  rank-zero-only JIT loading, DDP checkpoint resume, MLflow rank ownership and
  whole-image COCO deduplication, including an empty winning prediction set.
  Expected AP was 0.6633663366336634. The evaluator's CUDA tensor constructor
  was substituted with a CPU tensor constructor: this is not NCCL/GPU evidence.
- **Demo paths (25–33):** real pretrained Nano CPU inference and ONNX export/
  Runtime inference passed; rendered outputs matched exactly. Save-only
  inference ran with GUI functions forced to fail. Empty/positive assignment
  renders were inspected. MegEngine nonsaving image/video controls and the
  TensorRT warm-up dtype path passed with substitute backends, not native
  engine execution. VOC NumPy 2.x result writing and the tenth visualization
  logging interval passed. MLflow setup also ran against a local SQLite store
  with no run name configured.
- **Packaging/tooling (2–4, 35–38):** `uv lock --check`, Ruff, `git diff --check`
  and isolated sdist/wheel builds passed. The rebuilt wheel, installed outside
  the checkout, passed CPU inference, default experiment lookup, CLI and native
  extension imports. CI now defines Windows/macOS/Linux CPU jobs, but those
  remote jobs have not run. A final clean Sphinx build succeeded with 14 legacy
  heading/toctree/link warnings; this is not a warning-free docs build.
- **Security:** installed-environment `pip-audit` found no known advisories.
  Because its PyPI lookup skips torch's local `+cpu` wheel versions, torch and
  torchvision were also audited using normalized release versions; both locked
  and minimum versions returned no known advisories. This is a dated advisory
  check, not a guarantee of security or an audit of every optional SDK.
- **Native integrations (43):** TensorRT CMake discovery/configurable roots and
  MegEngine target-based CMake builds are implemented. OpenVINO Python/C++ use
  Runtime 2025.4.1; real Python CPU inference passed and produced an identical
  rendered image to the PyTorch result (mean pixel difference 0). Follow-up
  native validation is recorded below. Item 43 stays open until its remaining
  build/device requirements are resolved, not merely because of missing docs.
- **Portable runtime/assets (44):** `yolox.deploy` imports without torch installed;
  a lean ONNX Runtime environment ran the model. ONNX providers are explicit,
  ncnn accepts model paths, and `docs/model_assets.md` records the SHA-256 of
  the official Nano checkpoint used for validation. Other legacy assets are not
  silently claimed to be verified.

**Delivery state:** local changes only; not committed, pushed, merged or deployed.
Hardware and non-Linux validation remain external requirements even after source
changes are finished.

### Item 43 follow-up — native builds

- **OpenVINO C++: build and CPU inference passed.** The complete demo configured,
  compiled, linked and ran with the official Ubuntu 22 OpenVINO 2025.4.1 SDK,
  GCC 12.2, CMake 3.25.1 and OpenCV 4.6.0 on Debian 12 x86-64. The SDK archive
  SHA-256 matched its publisher's checksum. A fresh export from the verified
  official Nano checkpoint produced dog/bicycle/car detections at confidence
  0.85507/0.80264/0.79819. An unreadable-image control failed with exit 1 as
  expected. The Python wheel's ABI-0 native package failed to link with system
  OpenCV; using the ABI-compatible native SDK resolved that verified failure.
  CMake now requests only core/imgproc/imgcodecs. An Ubuntu 22 CI job repeats
  the checksum-verified export/build/inference path; remote CI has not run.
  A debug build also passed the same inference under UndefinedBehaviorSanitizer
  with recovery disabled. AddressSanitizer could not execute inference because
  oneTBB loads `libtbbbind_2_5.so.3` with `RTLD_DEEPBIND`, which ASan rejects;
  this is an explicit sanitizer-coverage limitation, not an ASan pass.
- **TensorRT C++: initial orb build/link passed; later GPU evidence below.** Built the complete
  executable against NVIDIA TensorRT 8.6.1.6/CUDA 12.0 (Toolkit 12.0.140,
  runtime 12.0.146) and OpenCV 4.6.0. Real SDK packages were extracted locally,
  verified against NVIDIA repository metadata; no inference implementation was
  substituted. The executable loads and returns its expected usage error.
  Legacy API warnings remain; this is not a TensorRT 10 port. There is no GPU
  or NVIDIA driver here to validate engine conversion/inference. Its native
  SDK is distinct from the Python uv CUDA 13.0 profile.
- **Android: both ARM builds and APK checks passed.** After the initial
  license-blocked attempt, the user explicitly accepted the Android SDK License
  Agreement and authorized installing the pinned components. Installed SDK 36,
  Build Tools 35.0.0, NDK 28.2 and CMake 3.22.1. With the matching verified model
  pair staged, `gradlew assembleDebug --no-daemon --console=plain` reported
  `BUILD SUCCESSFUL in 35s`, compiling/linking both ARM64 and ARMv7. `lintDebug`
  succeeded with 0 errors and 13 unsuppressed warnings. No additional source
  fixes were needed. The debug-signed test APK is saved under
  `.amp/in/artifacts/yolox-ncnn-debug.apk` (SHA-256
  `a1e0644becdd3872ff47946425148a53fb8f6ef86af9a78ad6245112e24f5278`).
  Signature v2 and `zipalign -c -P 16 4` checks passed independently; manifest
  reports minimum API 24/target API 36. ARM64 ELF LOAD segments are 16 KB
  aligned; ARMv7 segments are 4 KB aligned, although both APK library offsets
  satisfy 16 KB ZIP alignment. Packaged model hashes/JNI exports were checked.
  The tracked model param was restored and the temporary BIN removed. Future
  builds must stage the matching pair again. Device model loading/accuracy,
  image picking, CPU/Vulkan, lifecycle and real 16 KB-page execution remain
  untested. This is not a release-signed or runtime-certified APK.
- **MegEngine C++: Linux CPU Nano inference and bounded numerical parity passed.**
  Rebuilt/reinstalled the v1.13.3 SDK and relinked the demo with GCC 12.2,
  CMake 3.25.1 and OpenCV 4.6.0, using the documented validated BLAS profile.
  Both thread counts pass 25/25 numerical suites. Eight native image cases
  match exact inputs, raw-output error budgets and independently decoded
  detections against FP64 PyTorch. The historical-model run remains unsuitable
  as evidence. See the sequential follow-up below and the MegEngine READMEs
  for fixes, original failures, provenance, residuals and unverified scope.

### Item 43 follow-up: Oracle numerical-correctness plan

Implement in this order, without changing the main uv environment or treating
the pre-0.1.0 model as an accuracy fixture:

1. Add fail-closed BLAS validation; prove it rejects crashes, zero-exit numerical
   failures, empty/missing summaries and incomplete computational coverage.
2. Isolate library arithmetic from Fortran/CBLAS test interfaces using scalar
   reference probes; compare optimization/build settings one at a time.
3. Select the smallest demonstrated fix or pinned supported BLAS alternative,
   then rebuild/reinstall MegEngine and relink the demo.
4. Export a current, provenance-checked Nano model in an isolated compatible
   Python environment, with strict weight coverage and deterministic float32 input.
5. Compare identical raw input/output tensors across PyTorch, MegEngine eager,
   serialized graph and C++; compare detections under the same decoder policy.
6. Update the documented validated scope only after both numerical gates pass.
   Windows/macOS/ARM/CUDA and device validation remain independent requirements.

Execution evidence (2026-10-26, Linux x86-64/GCC 12.2):

- Added `demo/MegEngine/cpp/validate_openblas.py`, which invokes all 25 bundled
  suites directly in fresh working directories and checks computational passes,
  not just exit status. Original unmodified OpenBLAS build: **16/25 pass**, exit 1,
  single-thread. This supersedes the misleading upstream CTest 20/25 count:
  `dblat2`/`zblat2` crashes and `xscblat2`/`xccblat2` wrong results were hidden.
  Full direct logs: `/tmp/yolox-megengine-validation/baseline-direct/`.
- CLI regression check passed: `pytest tests/test_regressions.py::test_blas_validation_rejects_false_success -q`.
  It accepts complete output and rejects nonzero termination, zero-exit fatal
  output, a missing summary and a truncated computational report.
- Independent ctypes CBLAS GEMV probes with scalar-loop references reproduce
  wrong single/complex-single results and double/complex-double SIGSEGV without
  the upstream Fortran test wrappers. This establishes library/build failures,
  not their root cause. Baseline build/cache/logs retained under
  `/tmp/yolox-megengine-validation/baseline/` before new experiments.
- Fresh `openblas-o2` experiment changes only Release C optimization from `-O3`
  to `-O2`; all original targets/tests remain enabled. It still fails GEMV and
  passes only 17/25 numerical suites. Adding **only `-fno-ipa-ra`** to that
  profile gives **25/25 passes at both one and two threads**, plus zero errors
  in 192 scalar-reference CBLAS GEMV calls across real/complex, single/double,
  transpose, padded leading dimensions and signed/nonunit increments.
  Logs: `/tmp/yolox-megengine-validation/no-ipa-direct-{1,2}/`.
  Selected the smallest demonstrated build workaround rather than upgrading
  MegEngine or vendoring kernel patches. This is a GCC 12/Haswell profile,
  not a universal repair of the old assembly contracts. Upstream OpenBLAS
  0.3.6's changelog independently documents fixes for undeclared input-operand
  writes in the old x86 assembly; source inspection confirms the bundled
  GEMV counters are declared input-only. No failing suite was disabled.
- Fresh Nano export now uses strict checkpoint loading and deterministic
  float32 tracing. Strict loading exposed a dummy registered `head.grids.0`
  cache absent from PyTorch checkpoints; the derived cache now starts as None,
  not a parameter-like tensor. Deliberately removing a real convolution weight
  is rejected. A plain-array NPZ bridge separates modern Torch from the legacy
  exporter; the main uv environment/lock remains unchanged.
- The fresh-model comparison exposed a real architecture mismatch, independent
  of historical-model preprocessing: MegEngine's FPN used bilinear upsampling,
  while PyTorch uses nearest. On the four fixed inputs, changing only this mode
  reduced maximum raw discrepancies from **2.31–3.82** to **0.000193–0.000686**.
  Both FPN variants now match PyTorch's nearest mode. Original bilinear outputs
  were retained for comparison. No claim is made about the historical model.
- Investigated those residuals without relaxing the initial `atol=1e-4,
  rtol=1e-3` criterion: all four MegEngine eager outputs pass against an
  independent **FP64 PyTorch reference** (maximum absolute errors
  0.0000917/0.000141/0.0000931/0.000165 for dog/portrait/small/blank).
  PyTorch's default FP32 oneDNN outputs themselves differ from FP64 by up to
  0.000820. This eager check preceded the native verification below.
- Rebuilt/reinstalled the actual SDK's OpenBLAS dependency and MegEngine,
  then relinked the demo. The installed static BLAS archive matches the tested
  nested-build archive: SHA-256
  `c0035838f46590a22cd6ea4bb5180baaaf3cd7d36546871369a9b9e2bffc09c0`.
  That nested build separately passes **25/25 at one and two threads**;
  logs: `/tmp/yolox-megengine-validation/sdk-direct-{1,2}/`.
  `ldd` confirms the executable loads the rebuilt v1.13.3 SDK.
- The fresh graph initially failed with `wrong magic`: the C++ default loader
  selected legacy FBS while the exporter produces FBS_V2. Replaced the default
  with the SDK's existing format-identification API. Both formats subsequently
  executed successfully (legacy positive control plus real FBS_V2 Nano graph).
- A 301 × 187 input exposed a second native defect: the float resize ratio
  rounded the target width down to 639, changing 438,537 tensor values compared
  with Python. Keeping the ratio double fixes this; all four input tensors
  now match byte-for-byte. Preserved the failing tensor for comparison.
- Fresh, original-checkpoint-derived Nano graph (hash/provenance in
  `docs/model_assets.md`) passed **8 native cases**: dog 768×576, portrait
  341×613, small 301×187, blank 911×317; each on CPU and two-thread CPU pool.
  The **same graph** was reused across all inputs. Maximum native raw errors
  against FP64: 0.000151/0.000215/0.000180/0.000475; maximum probability error
  0.0000619. Exact class/count matches with independent C++-policy decoding;
  largest retained-box coordinate error 0.000272 pixels. Dog image detects
  dog/bicycle/truck, not the historical model's spurious traffic lights.
- Native diagnostic `atol=1e-4, rtol=1e-3` was **not universally met**:
  1/0/1/10 background box coefficients per case exceeded it. After localization
  and FP64/FP32-backend comparison, used explicit absolute budgets on **all**
  raw values: 0.001 for box coefficients, 0.0001 for probabilities. The former
  corresponds to 0.032 input pixels for center coefficients at stride 32, or
  approximately 0.1% size error for log-size coefficients. Decoded boxes also
  passed 0.01-pixel absolute/0.0001-relative comparison. No outputs were masked
  and classes/counts must agree exactly. This is bounded parity, not bitwise
  equality or dataset AP certification. Full metrics:
  `/tmp/yolox-megengine-validation/parity.{json,log}`.
- Native negative controls correctly reject wrong input names/shapes, multiple
  outputs, wrong output shapes/dtypes, nonfinite values, unreadable images and
  unwritable tensor dump paths. The executable now checks the contract before
  indexing/decoding and optionally dumps raw input/output and detections.
- Isolated exporter uses CPython 3.10.21/MegEngine 1.13.0/NumPy 1.24.4/
  setuptools 80.9.0 (PyPI's exact 1.13.3 wheel is unavailable; the old index has
  an expired TLS certificate, which was not bypassed). The 1.13.0→1.13.3
  graph combination is **executed evidence**, not assumed version compatibility.
  The main uv environment/lock was not changed. This deviates from the Oracle's
  source-build fallback to use an available publisher wheel, validated against
  the source model and target SDK rather than silently treating versions equal.
- Item 43 stays open for hardware/non-Linux coverage. Within MegEngine, other
  models, ARM/CUDA/FP16, fast-run/weight-preprocessing options and full-dataset
  accuracy remain unverified. No commits, pushes or deployments were made.
- Final repository checks: `.venv/bin/pytest -q` **17 passed**;
  `.venv/bin/ruff check .` **All checks passed**; `uv lock --check`
  resolved 201 packages without changing the lock; `git diff --check` passed.
  Strict documentation check (`sphinx-build -b html -W --keep-going`) rendered
  HTML but exited 1 with **16 warnings**, covering headings, omitted toctree
  entries and unresolved links (including demo-relative links in the generated
  documentation). This is not a warning-free docs build; log:
  `/tmp/yolox-docs-final.log`. Removed temporary investigation scripts and the
  generated repository-root image; numerical logs/tensors remain outside the
  repository for inspection.

### Cross-platform runner validation — 2026-10-26

The user authorized the MacBook Air M5 and Asus GPU runners. Both received the
same exact uncommitted source snapshot, not `origin/main`: archive SHA-256
`5e4e47990c2d54251bef311ce27ffdc0b95aabc81cbe83abe369611a3667bf9f`.
Both verified the archive before extraction and compared original files after
testing: no source/lock changes. Existing checkouts and shared/system dependencies
were not modified. Only this central plan/support documentation changed afterward.

#### MacBook Air M5

- macOS 27.0.1 (26A434), arm64, Apple M5; Xcode 27.1/Apple clang 21.0.0;
  uv 0.11.3. System Python 3.14 was not used: it is outside package support.
- `uv sync --locked --extra cpu --extra export` succeeded. Torch 2.14.1,
  torchvision 0.29.1, NumPy 2.5.4, ONNX 1.23.2, ORT 1.31.0.
- `pytest -q`: **17 passed in 29.62s on Python 3.12.13** and **17 passed in
  32.26s on Python 3.13.12**. Native and forced-fallback COCO checks included.
  Ruff passed. Isolated sdist and wheel-from-sdist builds succeeded on 3.12;
  outside-source `python -I` imported the installed arm64 extension and produced
  finite Nano output `[1,84,85]`. The 3.13 run used an editable environment.
- Trusted-checkpoint ONNX export/check/simplification and explicit ORT CPU
  inference executed. Default Nano export is 416; the initial 640-input harness
  was correctly rejected, then re-exported explicitly at 640 without source edits.
  Four geometry/blank cases had exact detection class/count agreement, maximum
  raw difference 0.000349, and maximum retained-box difference 0.000184
  original-image pixels. All four strict raw `atol=1e-5, rtol=1e-4` checks
  **failed**; decoded boxes passed `atol=0.01, rtol=1e-4`. No FP64 diagnosis
  or dataset AP evaluation was performed on this host.
- CoreML profiling proved one CoreML partition plus three CPU events, not pure
  accelerator execution. Default CoreML versus ORT CPU raw error reached 1.158;
  retained boxes shifted up to 3.019 input-coordinate pixels versus Torch.
  `MLComputeUnits=CPUOnly` reduced those figures to 0.000235 and 0.000550,
  respectively. GPU/ANE placement and the cause were not established. Default
  CoreML acceleration is **not numerically certified**. MPS was available but
  correctly rejected by the public model factory under the existing policy.
- Build deployment-target and legacy exporter deprecation warnings remain.
  Detailed commands/versions: `.amp/in/artifacts/yolox-macos-validation.md`;
  runner evidence: `/tmp/yolox-validation.3pTdpH`.

#### Asus RTX 3060 Laptop GPU

- Ubuntu 26.04.1, x86-64, glibc 2.43; GPU compute capability 8.6, 5806 MiB;
  NVIDIA driver 595.91.07, CUDA driver API 13.2. `nvidia-smi` was absent;
  procfs/PCI, successful `cuInit` and actual CUDA execution established hardware.
- Exact locked CUDA 13.0 profile executed with Python 3.12.14/uv 0.12.9,
  torch 2.14.1+cu130 and torchvision 0.29.1+cu130. No alternate Torch needed.
- Original host failed OpenCV imports (`libGL.so.1` missing); disposable library
  extraction and process-local `LD_LIBRARY_PATH` resolved it. Missing compiler
  left the optional extension absent: **16 passed/1 failed**. Rebuilding with
  disposable GCC 15.2 gave **17 passed in 14.25s**, extension import independently
  checked. Ruff and `uv lock --check` passed. `/tmp` quota failures were retained
  and superseded by clean reruns after relocating only disposable env/cache.
- Verified Nano FP32 CUDA inference produced finite `[1,8400,85]` predictions
  and dog/bicycle/truck detections. Real asymmetric-positive-label training on
  two 128×128 images executed assignment, backward and SGD momentum: total loss
  14.42475, 330 finite gradient tensors, 330 optimizer state entries, nonzero
  parameter updates. This is one training step, not a training-run certification.
- CUDA ONNX export/check and CPU ORT comparison passed over all 714,000 raw
  values at `atol=rtol=0.001`; maximum difference 0.000183. TF32 was disabled.
- Unchanged full TensorRT C++ demo compiled/linked with direct G++ invocation
  (not CMake on this runner), extracted OpenCV 4.10 and TensorRT 8.6.1.6/CUDA
  12.0/cuDNN 8.9.7. Native SDK libraries were separate from Torch CUDA 13.
  Publisher repository digests were checked before extraction.
- Default engine construction **failed**: glibc refused the executable stack
  requested by `libnvinfer_builder_resource.so.8.6.1`. Explicit builder-process
  `GLIBC_TUNABLES=glibc.rtld.execstack=2` allowed construction. This is a
  security-sensitive compatibility opt-in, **not stock-host support or a
  recommended global setting**. No SDK binaries or system configuration changed.
  Subsequent native demo/trtexec inference passed **without that tunable**.
- Native inference retained dog/bicycle/truck with scores 0.89060/0.82008/0.69842.
  Separate raw engine output comparison against CUDA Torch passed all 714,000
  values at `atol=rtol=0.001`, maximum difference 0.000139. JSON text precision,
  INT64 conversion and legacy explicit-batch API warnings remain; timings are
  not benchmarks. TensorRT 10 and torch2trt/Python conversion were not tested.
- Detailed commands/versions: `.amp/in/artifacts/yolox-asus-validation.md`;
  runner evidence: `/home/markos/.amp/in/artifacts/yolox-validation/`, larger
  assets `/tmp/yolox-validation.KhDTXbsX`. Both runners' evidence archives were
  retrieved and inspected in the orb at `/tmp/yolox-runner-evidence/`.

Item 43 remains open: Android device execution, other native OS/architecture
combinations, full-dataset AP, long training, AMP/FP16/INT8, NCCL/distributed,
GPU resume and dynamic TensorRT batches remain unverified. No code defect
requiring a central source fix was established by these runner checks. No commits,
pushes, system/driver installs, releases or deployments were performed.
After recording the runner results, `git diff --check` passed and Sphinx HTML
rendering succeeded with the same 16 warnings (not a strict documentation pass);
log: `/tmp/yolox-docs-runners.log`.

### Earlier deployment scope: CM5 device gates (superseded)

The user only requires deployment to CM5. This decision supersedes the earlier
open-ended native/mobile validation obligation in item 43; previous results above
remain historical evidence, not ongoing support commitments. Development,
training and export on the MacBook/Asus remain separate from target deployment.

Item 43 acceptance criteria, in execution order:

1. Identify the actual CM5, RAM, OS/version, ARM64 userspace, cooling, camera/input
   requirements and any attached accelerator. Do not infer CM5 hardware from a
   runner's name or treat another Raspberry Pi as target certification.
2. Start with YOLOX-Nano exported to ONNX on the development host. Use 64-bit
   Raspberry Pi OS and ONNX Runtime CPU as the proposed baseline, subject to
   actual target verification; neither the OS choice nor absence of an
   accelerator is confirmed yet. Keep target inference independent of PyTorch
   and the training environment, with pinned compatible runtime dependencies,
   verified model digests and repeatable installation/run instructions.
3. Execute image inference on the actual CM5, compare identical preprocessed
   tensors and decoded detections against the source model, and record all
   numerical discrepancies and tolerances. Exercise camera input if required
   and available. Include unreadable-input and model/shape-mismatch failures.
4. Measure end-to-end latency (including preprocessing/postprocessing), sustained
   throughput, memory and temperature/throttling under a documented workload,
   power/cooling setup, input size and thread count. Record results before
   setting any performance claim; no application FPS/latency target is agreed.
5. Consider ncnn or an accelerator-specific runtime only if target measurements
   or confirmed hardware justify it. Validate any selected alternative on CM5
   before promising support; backend availability alone is insufficient.

TensorRT, OpenVINO, MegEngine, Android and other non-CM5 deployment examples are
retained as reference/legacy integrations, not maintained deployment promises for
this project. ncnn is only a potential CM5 alternative, not a required backend.
No integration is deleted by this scope decision. Android device certification,
TensorRT modernization and unrelated native OS matrices no longer block item 43.
Item 43 stays unchecked until CM5 validation is executed. No CM5 tests or hardware
changes were performed as part of this documentation-only scope update.

### Item 43 execution: runtime preparation and former target-access blocker

Read-only runner discovery on 2026-10-26 did **not** find a Compute Module 5:

- `pi4`: device-tree model **Raspberry Pi 4 Model B Rev 1.1**, compatible
  `raspberrypi,4-model-b` / `brcm,bcm2711`; Debian 13.6, ARM64, Python 3.13.5,
  about 2 GB RAM. Sony IMX415 is registered, but no capture was started. Firmware
  `throttled=0x50005` and the hwmon voltage alarm indicate current/historical
  undervoltage and throttling. No stress/performance test was run.
- `raspberrypi`: device-tree model **Raspberry Pi 4 Model B Rev 1.5**, same
  BCM2711 compatibility; Debian 12, ARM64, Python 3.11.2, about 4 GB RAM.
  No camera sensor or AI accelerator was identified. Temperature was about
  73°C at the existing workload; historical throttling flags `0xe0000`, no
  current flags. Physical cooling/PSU details were not established on either Pi.
- Reports were saved/read back and retrieved into
  `.amp/in/artifacts/yolox-pi4-inventory.md` and
  `.amp/in/artifacts/yolox-raspberrypi-inventory.md`. A runner agent's subsequent
  empty-response error did not invalidate its completed command/report evidence.
  No system changes, camera streams, serial/MAC collection or stress tests.

Completed independently of target access:

- Added `docs/cm5_deployment.md`: trusted export/model provenance, minimal
  source-only bundle, headless target installation, explicit runtime contract
  and remaining correctness/performance/thermal/camera gates. Linked from docs.
- Added standalone `demo/ONNXRuntime/requirements-cm5.in` and hash-locked `.txt`.
  `uv pip compile ... --python-version 3.13 --python-platform aarch64-manylinux_2_28
  --only-binary :all: --generate-hashes` resolved **6 packages**. The target
  environment contains NumPy 2.5.4, ORT 1.31.0, headless OpenCV 4.14.0.94 and
  three transitive dependencies, not the PyTorch training stack. The main project
  uv lock/profile was unchanged. ARM64 resolution is not ARM64 execution.
- Reproduced an actual CLI failure before editing: default Nano ONNX input is
  `[1,3,416,416]`, but the demo supplied 640×640 and exited 1 with ORT
  `INVALID_ARGUMENT: Got: 640 Expected: 416`. The demo now infers static model
  dimensions, rejects incompatible overrides and bad input/output contracts,
  supports explicit CPU thread counts and checks that image saving succeeds.
  Both the original failing default invocation and explicit one-thread execution
  now save the expected result without requiring `--input_shape`.
- Extended the existing export-roundtrip regression through the public CLI:
  valid static shape inference; missing image/model; mismatched/zero shape;
  zero thread count; wrong output shape and nonfinite predictions. No relaxed
  numerical assertions or additional framework/mock layer.
- Created/extracted a minimal runtime archive and ran its **actual CLI outside
  the repository** in an isolated CPython 3.13.15 Linux x86-64 environment installed
  with `uv pip sync --require-hashes --only-binary :all:`. Trusted exported Nano
  graph digest `ba52fe62e78102f8f8c5d187fe59e55c36d7ee0235e33c3ed7a221e6646705be`
  verified. The 416×416 model produced a saved 768×576 dog image. `find_spec('torch')`
  was None and Torch was absent from `sys.modules` after the real run. This proves
  packaged source/runtime independence on the development host, **not CM5 support**.
- `.venv/bin/pytest -q`: **17 passed in 23.17s**; Ruff **All checks passed**;
  `uv lock --check` resolved 201 packages; `git diff --check` passed. Sphinx HTML
  build succeeded with the same 16 warnings, not a strict docs pass; log
  `/tmp/yolox-docs-cm5-runtime.log`. Scratch runtime/model fixtures remain under
  `/tmp/yolox-cm5-runtime` and `/tmp/yolox-cm5-package`, not tracked sources.

**Historical status, superseded below:** item 43 was held blocked/unchecked pending
ARM64 execution, numerical comparison, sustained latency/RSS/thermal measurements
and any required camera validation. No substitute Pi results were presented as
CM5 evidence. At that stage the conditional ship request was held pending and
no commit/push/merge/release was performed.

### Final acceptance scope and shipping authorization — 2026-10-26

The user clarified that physical deployment hardware is outside this task and
repository. The important outcome is working code. They explicitly accept real
CPU-only ONNX inference on the Asus as sufficient to extrapolate to Raspberry Pi,
asked that this decision be recorded in `AGENTS.md`, and authorized shipping.
This supersedes all earlier physical-CM5 gates and requests to connect the device.
No new physical-device validation is required or being requested.

Item 43 is complete under that agreed repository scope:

- [x] Actual Asus CPUExecutionProvider inference of the exported 640×640 Nano
  model, compared against PyTorch CUDA over all 714,000 raw values; max absolute
  error 0.000183 at `atol=rtol=0.001`. Exact host/runtime details above.
- [x] Reproducible standalone, hash-locked headless runtime, with the exact 416×416
  bundle separately executed on the orb CPU without Torch installed/imported.
  This artifact is distinct from the Asus 640×640 graph; no false equivalence.
- [x] Export/model provenance, static-shape inference, explicit CPU thread/provider
  configuration, output validation and public CLI regression/error controls.
- [x] Runtime/export instructions and support boundaries documented. Added root
  `AGENTS.md` so future work does not reopen physical-device certification as a
  requirement. CM5 cameras, OS/power/cooling integration, latency/thermal tests
  and hardware-specific accuracy remain downstream work, not blockers here.

The repository uses uv: `pyproject.toml`, `uv.lock`, `.python-version`, CPU/CUDA
profiles, `uv sync`/`uv run`/`uv build`, uv-based CI and a separately hash-locked
`uv pip sync` inference environment. No on-device result is implied by accepting
CPU-host validation. Final shipping checks/delivery are recorded below.

Final shipping verification:

- Fetched the complete origin history and rebased onto the existing orb-setup
  commit. Preserved its portal-ignore rule; the obsolete duplicate requirements
  list now delegates to pyproject metadata (including the modern simplifier).
- Updated the inherited `.agents/setup` to use locked uv CPU/export/dev/docs
  dependencies instead of global pip/obsolete lint installs. Kept system-package
  checks and added OpenCV runtime libraries. `bash -n` passed; two real setup
  runs passed in 17.84s and 3.80s, including finite CPU Nano inference.
- `uv build` produced the sdist and Linux CPython 3.12 wheel. Installed that exact
  wheel and tested outside the checkout with `python -I`: native extension,
  default experiment loading and finite CPU Nano output `[1,84,85]` passed.
  Installed deployment helpers imported without importing Torch. Restored the
  editable locked development environment afterward.
- Full post-rebase `uv run --no-sync pytest -q`: **17 passed**. Ruff, lockfile
  validation and whitespace checks passed. Sphinx HTML build succeeded with
  16 unsuppressed warnings; this is not a warning-free docs build.
- The user authorized shipping all changes to `origin/main`. Repository item 43
  is checked complete under the final agreed scope; no CM5 access or physical
  deployment is required or claimed. No package publication or manual deployment
  is part of this shipment.
