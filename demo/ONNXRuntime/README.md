## YOLOX-ONNXRuntime in Python

This doc introduces how to convert your pytorch model into onnx, and how to run an onnxruntime demo to verify your convertion.

### Step1: Install onnxruntime

From the repository root, install the package and export dependencies:
```shell
uv sync --locked --extra cpu --extra export
```

The demo uses `yolox.deploy`, whose import path requires only NumPy/OpenCV, not
torch. The standard package installation still includes PyTorch for training.
An exported model can run in a separate deployment environment with NumPy,
OpenCV and ONNX Runtime plus a YOLOX wheel installed with `--no-deps`; this
advanced installation intentionally does not support the training/export CLIs.

For the proposed CM5 target, use the standalone, headless dependency lock and
source-only bundle described in `docs/cm5_deployment.md` at the repository root.
That path does not install the training package. Actual CPU-only ONNX inference
on the Asus is accepted repository validation; physical CM5 certification is
outside this task. CPU-host evidence is not presented as an on-device measurement.

### Step2: Get ONNX models

Users might download our pre-generated ONNX models or convert their own models to ONNX.

#### Download ONNX models.

| Model | Parameters | GFLOPs | Test Size | mAP | Weights |
|:------| :----: | :----: | :---: | :---: | :---: |
|  YOLOX-Nano |  0.91M  | 1.08 | 416x416 | 25.8 |[github](https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_nano.onnx) |
|  YOLOX-Tiny | 5.06M     | 6.45 | 416x416 |32.8 | [github](https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_tiny.onnx) |
|  YOLOX-S | 9.0M | 26.8 | 640x640 |40.5 | [github](https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_s.onnx) |
|  YOLOX-M | 25.3M | 73.8 | 640x640 |47.2 | [github](https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_m.onnx) |
|  YOLOX-L | 54.2M | 155.6 | 640x640 |50.1 | [github](https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_l.onnx) |
|  YOLOX-Darknet53| 63.72M | 185.3 | 640x640 |48.0 | [github](https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_darknet.onnx) |
|  YOLOX-X | 99.1M | 281.9 | 640x640 |51.5 | [github](https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_x.onnx) |

#### Convert Your Model to ONNX

First, you should move to <YOLOX_HOME> by:
```shell
cd <YOLOX_HOME>
```
Then, you can:

1. Convert a standard YOLOX model by -n:
```shell
python3 tools/export_onnx.py --output-name yolox_s.onnx -n yolox-s -c yolox_s.pth
```
Notes:
* -n: specify a model name. The model name must be one of the [yolox-s,m,l,x and yolox-nano, yolox-tiny, yolov3]
* -c: the model you have trained
* -o: opset version, default 17. Legacy OpenVINO conversion may require an older opset and a matching legacy SDK; it is not covered by the modern ONNX Runtime path.
* --no-onnxsim: disable onnxsim
* To customize an input shape for onnx model,  modify the following code in tools/export.py:

    ```python
    dummy_input = torch.randn(1, 3, exp.test_size[0], exp.test_size[1])
    ```

1. Convert a standard YOLOX model by -f. When using -f, the above command is equivalent to:

```shell
python3 tools/export_onnx.py --output-name yolox_s.onnx -f exps/default/yolox_s.py -c yolox_s.pth
```

3. To convert your customized model, please use -f:

```shell
python3 tools/export_onnx.py --output-name your_yolox.onnx -f exps/your_dir/your_yolox.py -c your_yolox.pth
```

### Step3: ONNXRuntime Demo

Step1.
```shell
cd <YOLOX_HOME>/demo/ONNXRuntime
```

Step2. 
```shell
python3 onnx_inference.py -m <ONNX_MODEL_PATH> -i <IMAGE_PATH> -o <OUTPUT_DIR> -s 0.3 --input_shape 640,640
```
Notes:
* -m: your converted onnx model
* -i: input_image
* -s: score threshold for visualization.
* --input_shape: optional height,width; inferred from static models by default.
  Overrides must match the model. Nano's default export is 416,416, not 640,640.
  The decoder expects raw COCO predictions with strides 8/16/32.
* --threads: optional positive CPU inference thread count. Measure on the target
  rather than assuming more threads improve sustained throughput.
* --providers: ordered providers, default `CPUExecutionProvider`. For example,
  `--providers CUDAExecutionProvider CPUExecutionProvider` requires the matching
  GPU-enabled ONNX Runtime package and CUDA libraries. Unavailable provider names
  fail explicitly instead of silently switching to CPU.
