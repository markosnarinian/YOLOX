#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) Megvii, Inc. and its affiliates.

import argparse
import logging as log
import os
import sys

import cv2
import numpy as np

import openvino as ov
from openvino.preprocess import PrePostProcessor

from yolox.deploy import COCO_CLASSES, demo_postprocess, mkdir, multiclass_nms, vis
from yolox.deploy import preproc as preprocess


def parse_args() -> argparse.Namespace:
    """Parse and return command line arguments"""
    parser = argparse.ArgumentParser(add_help=False)
    args = parser.add_argument_group('Options')
    args.add_argument(
        '-h',
        '--help',
        action='help',
        help='Show this help message and exit.')
    args.add_argument(
        '-m',
        '--model',
        required=True,
        type=str,
        help='Required. Path to an .xml or .onnx file with a trained model.')
    args.add_argument(
        '-i',
        '--input',
        required=True,
        type=str,
        help='Required. Path to an image file.')
    args.add_argument(
        '-o',
        '--output_dir',
        type=str,
        default='demo_output',
        help='Path to your output dir.')
    args.add_argument(
        '-s',
        '--score_thr',
        type=float,
        default=0.3,
        help="Score threshould to visualize the result.")
    args.add_argument(
        '-d',
        '--device',
        default='CPU',
        type=str,
        help='Optional. OpenVINO device name (default: CPU). Other devices, '
             'such as GPU or AUTO, require a supported plugin and hardware.')
    args.add_argument(
        '--labels',
        default=None,
        type=str,
        help='Option:al. Path to a labels mapping file.')
    args.add_argument(
        '-nt',
        '--number_top',
        default=10,
        type=int,
        help='Optional. Number of top results.')
    return parser.parse_args()


def main():
    log.basicConfig(format='[ %(levelname)s ] %(message)s', level=log.INFO, stream=sys.stdout)
    args = parse_args()

    log.info('Creating OpenVINO Runtime Core')
    core = ov.Core()

    # Read an IR (.xml with companion .bin) or ONNX model.
    log.info(f'Reading the network: {args.model}')
    model = core.read_model(args.model)

    if len(model.inputs) != 1:
        log.error('Sample supports only single input topologies')
        return -1
    if len(model.outputs) != 1:
        log.error('Sample supports only single output topologies')
        return -1

    if model.input().partial_shape.is_dynamic:
        log.error('Sample requires a static [1, 3, H, W] input shape')
        return -1
    shape = list(model.input().shape)
    if len(shape) != 4 or shape[:2] != [1, 3]:
        log.error('Sample requires a static [1, 3, H, W] input shape')
        return -1
    _, _, h, w = shape

    # Keep BGR, unnormalized CHW preprocessing in Python. Explicit tensor types
    # also support models whose internal input/output element type is FP16.
    ppp = PrePostProcessor(model)
    ppp.input().tensor().set_element_type(ov.Type.f32)
    ppp.output().tensor().set_element_type(ov.Type.f32)
    model = ppp.build()

    log.info('Loading the model to the plugin')
    compiled_model = core.compile_model(model, args.device)
    infer_request = compiled_model.create_infer_request()

    origin_img = cv2.imread(args.input)
    if origin_img is None:
        log.error(f'Could not read image: {args.input}')
        return -1
    image, ratio = preprocess(origin_img, (h, w))

    log.info('Starting inference in synchronous mode')
    infer_request.infer({0: image[np.newaxis, ...]})
    # Copy before decoding, which modifies the array in-place.
    res = infer_request.get_output_tensor().data.copy()
    num_anchors = sum((h // stride) * (w // stride) for stride in (8, 16, 32))
    if res.shape != (1, num_anchors, len(COCO_CLASSES) + 5):
        log.error('Sample requires raw YOLOX output [1, anchors, 85] (strides 8, 16, 32)')
        return -1

    predictions = demo_postprocess(res, (h, w))[0]

    boxes = predictions[:, :4]
    scores = predictions[:, 4, None] * predictions[:, 5:]

    boxes_xyxy = np.ones_like(boxes)
    boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2]/2.
    boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3]/2.
    boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2]/2.
    boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3]/2.
    boxes_xyxy /= ratio
    dets = multiclass_nms(boxes_xyxy, scores, nms_thr=0.45, score_thr=0.1)

    if dets is not None:
        final_boxes = dets[:, :4]
        final_scores, final_cls_inds = dets[:, 4], dets[:, 5]
        origin_img = vis(origin_img, final_boxes, final_scores, final_cls_inds,
                         conf=args.score_thr, class_names=COCO_CLASSES)

    mkdir(args.output_dir)
    output_path = os.path.join(args.output_dir, os.path.basename(args.input))
    cv2.imwrite(output_path, origin_img)


if __name__ == '__main__':
    sys.exit(main())
