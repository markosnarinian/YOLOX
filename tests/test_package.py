"""Small public-contract checks, also run against supported OS environments."""

import importlib
from pathlib import Path
import subprocess
import sys

import cv2
import numpy as np
import onnx
import onnxruntime
import torch

from yolox.exp import get_exp
from yolox.models import yolox_nano


def test_model_inference():
    model = yolox_nano(pretrained=False, device="cpu").eval()
    with torch.no_grad():
        output = model(torch.zeros(1, 3, 64, 64))
    assert output.shape == (1, 84, 85)
    assert torch.isfinite(output).all()


def test_model_training_with_empty_and_positive_images():
    model = yolox_nano(pretrained=False, device="cpu").train()
    labels = torch.zeros(2, 4, 5)
    labels[1, 0] = torch.tensor([2, 24, 30, 16, 18])
    loss = model(torch.randn(2, 3, 64, 64), labels)["total_loss"]
    assert torch.isfinite(loss) and loss > 0
    loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)


def test_deploy_import_does_not_load_torch():
    subprocess.run(
        [sys.executable, "-c", "import yolox.deploy, sys; assert 'torch' not in sys.modules"],
        check=True,
    )


def test_native_evaluator_available():
    module = importlib.import_module("yolox.layers.fast_cocoeval")
    assert hasattr(module, "COCOevalEvaluateImages")


def test_export_roundtrip(tmp_path):
    exp = get_exp(exp_name="yolox-nano")
    model = exp.get_model().eval()
    model.head.decode_in_inference = False
    checkpoint = tmp_path / "nano.pth"
    output = tmp_path / "nano.onnx"
    torch.save({"model": model.state_dict()}, checkpoint)
    subprocess.run(
        [sys.executable, "-m", "yolox.tools.export_onnx", "-n", "yolox-nano",
         "-c", str(checkpoint), "--output-name", str(output), "test_size", "(64,64)"],
        check=True,
    )
    onnx.checker.check_model(onnx.load(output))
    sample = np.random.default_rng(42).normal(size=(1, 3, 64, 64)).astype(np.float32)
    session = onnxruntime.InferenceSession(str(output), providers=["CPUExecutionProvider"])
    actual = session.run(None, {"images": sample})[0]
    with torch.no_grad():
        expected = model(torch.from_numpy(sample)).numpy()
    np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-5)

    # The deployment CLI must use the exported shape, not assume 640x640.
    image = tmp_path / "input.png"
    cv2.imwrite(str(image), np.full((31, 20, 3), 114, dtype=np.uint8))
    script = Path(__file__).resolve().parents[1] / "demo/ONNXRuntime/onnx_inference.py"
    command = [sys.executable, str(script), "-m", str(output), "-i", str(image),
               "-o", str(tmp_path / "result"), "--threads", "1"]
    subprocess.run(command, check=True)
    assert cv2.imread(str(tmp_path / "result/input.png")).shape == (31, 20, 3)
    for options, message in (
        (["--input_shape", "96,64"], "does not match model"),
        (["--input_shape", "0,64"], "positive --input_shape"),
        (["-i", str(tmp_path / "missing.png")], "Cannot read input image"),
        (["-m", str(tmp_path / "missing.onnx")], "doesn't exist"),
        (["--threads", "0"], "--threads must be positive"),
    ):
        result = subprocess.run(command + options, capture_output=True, text=True)
        assert result.returncode != 0 and message in result.stderr, result.stderr

    for name, values in (("wrong-output", np.zeros((1, 84, 5), dtype=np.float32)),
                         ("nonfinite", np.full((1, 84, 85), np.nan, dtype=np.float32))):
        graph = onnx.helper.make_graph(
            [onnx.helper.make_node("Constant", [], ["output"],
                                   value=onnx.numpy_helper.from_array(values))],
            name,
            [onnx.helper.make_tensor_value_info("images", onnx.TensorProto.FLOAT, [1, 3, 64, 64])],
            [onnx.helper.make_tensor_value_info("output", onnx.TensorProto.FLOAT, values.shape)],
        )
        invalid = tmp_path / f"{name}.onnx"
        onnx.save(onnx.helper.make_model(graph, ir_version=9,
                                       opset_imports=[onnx.helper.make_opsetid("", 17)]), invalid)
        result = subprocess.run(command + ["-m", str(invalid)], capture_output=True, text=True)
        assert result.returncode != 0 and "Expected finite float32 raw COCO" in result.stderr
