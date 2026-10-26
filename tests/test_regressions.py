"""Observable contracts covered by the foundation/correctness remediation."""

import json
import pickle
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
import torch
from pycocotools.coco import COCO
from torch import nn

from yolox.core import Trainer
from yolox.data.datasets import COCODataset
from yolox.exp import get_exp
from yolox.layers.fast_coco_eval_api import COCOeval_opt
from yolox.utils import ModelEMA, adjust_status, replace_module


def test_experiment_overrides_and_file_identity(tmp_path):
    exp = get_exp(exp_name="yolox-nano")
    exp.merge(["ema", "False", "enable_mixup", "0", "test_size", "(96,64)"])
    assert exp.ema is False and exp.enable_mixup is False
    assert exp.test_size == (96, 64)
    with pytest.raises(ValueError):
        exp.merge(["ema", "not-a-boolean"])
    for number in (1, 2):
        directory = tmp_path / str(number)
        directory.mkdir()
        path = directory / "experiment.py"
        path.write_text("from yolox.exp import Exp as Base\n"
                        f"class Exp(Base):\n    identity = {number}\n")
        loaded = get_exp(exp_file=str(path))
        loaded.merge(["ema", "False"])
        serialized = pickle.dumps(loaded)
        restored = pickle.loads(serialized)
        assert restored.identity == number and restored.ema is False
        subprocess.run(
            [sys.executable, "-c",
             "import pickle, sys; exp = pickle.loads(sys.stdin.buffer.read()); "
             f"assert exp.identity == {number} and exp.ema is False"],
            input=serialized, check=True,
        )


def test_model_helpers_restore_and_replace_nested_modules():
    model = nn.Sequential(nn.Dropout(), nn.Sequential(nn.ReLU()))
    model[1].eval()
    with pytest.raises(RuntimeError), adjust_status(model, training=False):
        raise RuntimeError("failed inference")
    assert model.training and model[0].training and not model[1].training
    model = replace_module(model, nn.ReLU, nn.LeakyReLU,
                           lambda old, new: new(negative_slope=0.37))
    assert model[1][0].negative_slope == 0.37


def test_disk_cache_and_clipped_annotations(tmp_path):
    (tmp_path / "annotations").mkdir()
    (tmp_path / "train2017").mkdir()
    image = np.arange(12 * 20 * 3, dtype=np.uint8).reshape(12, 20, 3)
    cv2.imwrite(str(tmp_path / "train2017" / "a.png"), image)
    annotation = {
        "images": [{"id": 1, "file_name": "a.png", "height": 12, "width": 20}],
        "categories": [{"id": 4, "name": "object"}],
        "annotations": [{"id": 1, "image_id": 1, "category_id": 4,
                         "bbox": [-3, -2, 10, 7], "area": 70, "iscrowd": 0}],
    }
    (tmp_path / "annotations" / "instances_train2017.json").write_text(json.dumps(annotation))
    dataset = COCODataset(str(tmp_path), img_size=(12, 20), cache=True, cache_type="disk")
    actual, labels, _, _ = dataset.pull_item(0)
    np.testing.assert_array_equal(actual, image)
    np.testing.assert_array_equal(labels, [[0, 0, 7, 5, 0]])


@pytest.mark.parametrize("fallback", [False, True])
def test_coco_native_and_fallback_agree(fallback, monkeypatch):
    if fallback:
        def unavailable(*args, **kwargs):
            raise RuntimeError("compiler unavailable")
        monkeypatch.setattr("yolox.layers.fast_coco_eval_api.FastCOCOEvalOp.load", unavailable)
    gt = COCO()
    gt.dataset = {
        "images": [{"id": 1}], "categories": [{"id": 1}],
        "annotations": [{"id": 1, "image_id": 1, "category_id": 1,
                         "bbox": [1, 2, 10, 20], "area": 200, "iscrowd": 0}],
    }
    gt.createIndex()
    dt = gt.loadRes([{"image_id": 1, "category_id": 1,
                      "bbox": [1, 2, 10, 20], "score": 0.9}])
    evaluator = COCOeval_opt(gt, dt, "bbox")
    evaluator.evaluate()
    evaluator.accumulate()
    evaluator.summarize()
    assert evaluator.stats[0] == pytest.approx(1.0)


def test_checkpoint_resume_preserves_live_optimizer_and_ema(tmp_path):
    trainer = Trainer.__new__(Trainer)
    trainer.rank = 0
    trainer.device = "cpu"
    trainer.args = SimpleNamespace(logger="none", resume=True, ckpt=None, start_epoch=None)
    trainer.model = nn.Linear(2, 1)
    trainer.optimizer = torch.optim.SGD(trainer.model.parameters(), lr=0.1, momentum=0.9)
    trainer.scaler = torch.amp.GradScaler("cpu")
    trainer.use_model_ema = True
    trainer.ema_model = ModelEMA(trainer.model)
    trainer.ema_model.updates = 19
    trainer.file_name = str(tmp_path)
    trainer.epoch = 3
    trainer.best_ap = 0.6
    trainer.model(torch.tensor([[1., 3.]])).sum().backward()
    trainer.optimizer.step()
    live = {key: value.clone() for key, value in trainer.model.state_dict().items()}
    trainer.save_ckpt("latest")
    saved = torch.load(tmp_path / "latest_ckpt.pth", weights_only=True)
    assert not torch.equal(saved["model"]["weight"], saved["training_model"]["weight"])
    trainer.model = nn.Linear(2, 1)
    trainer.optimizer = torch.optim.SGD(trainer.model.parameters(), lr=0.9, momentum=0.1)
    trainer.scaler = torch.amp.GradScaler("cpu", init_scale=16)
    trainer.resume_train(trainer.model)
    for key, value in live.items():
        torch.testing.assert_close(trainer.model.state_dict()[key], value)
    assert trainer.optimizer.param_groups[0]["lr"] == 0.1
    momentum = next(iter(trainer.optimizer.state.values()))["momentum_buffer"]
    torch.testing.assert_close(momentum, torch.tensor([[1., 3.]]))
    assert trainer.resume_checkpoint["ema_updates"] == 19
    assert trainer.start_epoch == 4 and trainer.best_ap == 0.6
    assert trainer.scaler.get_scale() == 65536


def test_latest_keeps_updated_best_score(tmp_path):
    trainer = Trainer.__new__(Trainer)
    trainer.rank = 0
    trainer.args = SimpleNamespace(logger="none")
    trainer.model = nn.Linear(1, 1)
    trainer.optimizer = torch.optim.SGD(trainer.model.parameters(), lr=0.1)
    trainer.scaler = torch.amp.GradScaler("cpu", enabled=False)
    trainer.use_model_ema = False
    trainer.save_history_ckpt = False
    trainer.is_distributed = False
    trainer.file_name = str(tmp_path)
    trainer.epoch = 0
    trainer.best_ap = 0.4
    trainer.evaluator = None
    trainer.exp = SimpleNamespace(eval_interval=1, eval=lambda *a, **k: ((0.6, 0.7, ""), {}))
    trainer.after_epoch()
    assert torch.load(tmp_path / "latest_ckpt.pth", weights_only=True)["best_ap"] == 0.6
    trainer.exp.eval = lambda *a, **k: ((0.5, 0.6, ""), {})
    trainer.epoch = 1
    trainer.after_epoch()
    assert torch.load(tmp_path / "best_ckpt.pth", weights_only=True)["curr_ap"] == 0.6


@pytest.mark.skipif(sys.platform == "win32", reason="bundled OpenBLAS validation uses Unix executables")
def test_blas_validation_rejects_false_success(tmp_path):
    """Exercise the CLI with zero-exit failures, crashes and incomplete output."""
    source, build = tmp_path / "source", tmp_path / "build"
    fake = f"""#!{sys.executable}
import os, signal, sys
from pathlib import Path
name = Path(sys.argv[0]).name
mode = {str(tmp_path / 'mode')!r}
mode = Path(mode).read_text()
if name == 'openblas_utest':
    print('RESULTS: 2 tests (2 ok, 0 failed, 0 skipped)')
elif name.endswith('1'):
    print('----- PASS -----\\n' * 10)
else:
    cblas = name.startswith('x')
    text = ('SGEMV PASSED THE COLUMN-MAJOR COMPUTATIONAL TESTS ( 12 CALLS)\\n'
            'SGEMV PASSED THE ROW-MAJOR COMPUTATIONAL TESTS ( 12 CALLS)\\n'
            if cblas else 'SGEMV PASSED THE COMPUTATIONAL TESTS ( 12 CALLS)\\n')
    if name == 'xccblat2':
        text = 'COLUMN-MAJOR AND ROW-MAJOR DATA LAYOUTS ARE TESTED\\n'
        text += 'cblas_cgemv PASSED THE COMPUTATIONAL TESTS ( 12 CALLS)\\n' * (
            1 if mode == 'one-layout' else 2)
    text += 'END OF TESTS\\n'
    if name == 'sblat2':
        if mode == 'crash':
            os.kill(os.getpid(), signal.SIGTERM)
        if mode == 'fatal':
            text += 'FATAL ERROR - TESTS ABANDONED\\n'
        if mode == 'truncated':
            text = 'SGEMV PASSED THE TESTS OF ERROR-EXITS\\nEND OF TESTS\\n'
        if mode == 'missing':
            sys.exit(0)
    if cblas:
        print(text)
    else:
        Path(name.upper() + '.SUMM').write_text(text)
"""
    for directory in ("test", "ctest", "utest"):
        (source / directory).mkdir(parents=True)
        (build / directory).mkdir(parents=True)
    for dtype in "sdcz":
        for level in (1, 2, 3):
            for directory, name, data in (
                ("test", f"{dtype}blat{level}", f"{dtype}blat{level}.dat"),
                ("ctest", f"x{dtype}cblat{level}", f"{dtype}in{level}"),
            ):
                routine = "cblas_cgemv" if name == "xccblat2" else "SGEMV"
                (source / directory / data).write_text(f"{routine} T PUT F FOR NO TEST\n")
                executable = build / directory / name
                executable.write_text(fake)
                executable.chmod(0o755)
    executable = build / "utest/openblas_utest"
    executable.write_text(fake)
    executable.chmod(0o755)
    validator = Path(__file__).resolve().parents[1] / "demo/MegEngine/cpp/validate_openblas.py"
    for mode in ("valid", "fatal", "crash", "missing", "truncated", "one-layout"):
        (tmp_path / "mode").write_text(mode)
        result = subprocess.run(
            [sys.executable, str(validator), "--source", str(source), "--build", str(build),
             "--logs", str(tmp_path / mode)], capture_output=True, text=True,
        )
        assert result.returncode == (0 if mode == "valid" else 1), result.stdout
        assert ("25/25" if mode == "valid" else "24/25") in result.stdout


def test_megengine_weight_bridge_without_megengine(tmp_path):
    converter = Path(__file__).resolve().parents[1] / "demo/MegEngine/python/convert_weights.py"
    weights = {"conv.weight": torch.arange(6, dtype=torch.float32).reshape(2, 3),
               "bn.bias": torch.tensor([-1., 3.]), "bn.num_batches_tracked": torch.tensor(7)}
    checkpoint = tmp_path / "weights.pth"
    torch.save({"model": weights}, checkpoint)
    archive, copied = tmp_path / "weights.npz", tmp_path / "copied.npz"
    for source, destination in ((checkpoint, archive), (archive, copied)):
        subprocess.run([sys.executable, str(converter), "-w", str(source),
                        "-o", str(destination)], check=True)
    with np.load(copied, allow_pickle=False) as arrays:
        assert set(arrays.files) == set(weights)
        for name, expected in weights.items():
            np.testing.assert_array_equal(arrays[name], expected.numpy())
