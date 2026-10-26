#!/usr/bin/env python3
# Copyright (c) Megvii, Inc. and its affiliates.

import argparse
import os

import cv2
import numpy as np

import onnxruntime

from yolox.deploy import COCO_CLASSES, demo_postprocess, mkdir, multiclass_nms, vis
from yolox.deploy import preproc as preprocess


def make_parser():
    parser = argparse.ArgumentParser("onnxruntime inference sample")
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        default="yolox.onnx",
        help="Input your onnx model.",
    )
    parser.add_argument(
        "-i",
        "--image_path",
        type=str,
        default='test_image.png',
        help="Path to your input image.",
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        type=str,
        default='demo_output',
        help="Path to your output directory.",
    )
    parser.add_argument(
        "-s",
        "--score_thr",
        type=float,
        default=0.3,
        help="Score threshould to filter the result.",
    )
    parser.add_argument(
        "--input_shape",
        type=str,
        default=None,
        help="Height,width; inferred from a static model by default. Must match the model.",
    )
    parser.add_argument(
        "--providers", nargs="+", default=["CPUExecutionProvider"],
        help="Ordered ONNX Runtime execution providers; all requested providers must be installed.",
    )
    parser.add_argument("--threads", type=int, help="Positive CPU inference thread count.")
    return parser


def main(args):
    origin_img = cv2.imread(args.image_path)
    if origin_img is None:
        raise ValueError(f"Cannot read input image: {args.image_path}")

    unavailable = set(args.providers) - set(onnxruntime.get_available_providers())
    if unavailable:
        raise ValueError(f"Unavailable ONNX Runtime providers: {sorted(unavailable)}")
    options = onnxruntime.SessionOptions()
    if args.threads is not None:
        if args.threads < 1:
            raise ValueError("--threads must be positive")
        options.intra_op_num_threads = args.threads
    session = onnxruntime.InferenceSession(args.model, sess_options=options, providers=args.providers)
    inputs = session.get_inputs()
    if len(inputs) != 1 or len(session.get_outputs()) != 1:
        raise ValueError("Expected one image input and one raw YOLOX output")
    model_shape = inputs[0].shape
    if len(model_shape) != 4 or inputs[0].type != "tensor(float)":
        raise ValueError("Expected float32 NCHW input")
    input_shape = (tuple(map(int, args.input_shape.split(','))) if args.input_shape
                   else tuple(model_shape[2:]))
    if (len(input_shape) != 2
            or any(not isinstance(size, int) or size <= 0 or size % 32 for size in input_shape)):
        raise ValueError("Specify positive --input_shape height,width multiples of 32")
    expected_shape = (1, 3, *input_shape)
    if any(isinstance(size, int) and size != expected
           for size, expected in zip(model_shape, expected_shape)):
        raise ValueError(f"Input shape {expected_shape} does not match model {model_shape}")
    img, ratio = preprocess(origin_img, input_shape)

    ort_inputs = {inputs[0].name: img[None, :, :, :]}
    output = session.run(None, ort_inputs)
    anchors = sum((input_shape[0] // stride) * (input_shape[1] // stride)
                  for stride in (8, 16, 32))
    if (output[0].shape != (1, anchors, 85) or output[0].dtype != np.float32
            or not np.isfinite(output[0]).all()):
        raise ValueError("Expected finite float32 raw COCO predictions [1,anchors,85]")
    predictions = demo_postprocess(output[0], input_shape)[0]

    boxes = predictions[:, :4]
    scores = predictions[:, 4:5] * predictions[:, 5:]

    boxes_xyxy = np.ones_like(boxes)
    boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2]/2.
    boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3]/2.
    boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2]/2.
    boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3]/2.
    boxes_xyxy /= ratio
    dets = multiclass_nms(boxes_xyxy, scores, nms_thr=0.45, score_thr=0.1)
    if dets is not None:
        final_boxes, final_scores, final_cls_inds = dets[:, :4], dets[:, 4], dets[:, 5]
        origin_img = vis(origin_img, final_boxes, final_scores, final_cls_inds,
                         conf=args.score_thr, class_names=COCO_CLASSES)

    mkdir(args.output_dir)
    output_path = os.path.join(args.output_dir, os.path.basename(args.image_path))
    if not cv2.imwrite(output_path, origin_img):
        raise OSError(f"Cannot write output image: {output_path}")


if __name__ == '__main__':
    main(make_parser().parse_args())
