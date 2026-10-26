#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import argparse

import megengine as mge
import numpy as np
from megengine import jit

from build import build_and_load


def make_parser():
    parser = argparse.ArgumentParser("YOLOX Demo Dump")
    parser.add_argument("-n", "--name", type=str, default="yolox-s", help="model name")
    parser.add_argument("-c", "--ckpt", default=None, type=str, help="ckpt for eval")
    parser.add_argument(
        "--dump_path", default="model.mge", help="path to save the dumped model"
    )
    parser.add_argument("--input-npy", help="optional float32 [1,3,640,640] validation input")
    parser.add_argument("--reference-npy", help="save eager raw predictions for comparison")
    return parser


def dump_static_graph(model, graph_name="model.mge", input_array=None, reference_path=None):
    model.eval()
    model.head.decode_in_inference = False

    if input_array is None:
        input_array = np.zeros((1, 3, 640, 640), dtype=np.float32)
    if input_array.shape != (1, 3, 640, 640) or input_array.dtype != np.float32:
        raise ValueError("Expected float32 input [1,3,640,640]")
    data = mge.Tensor(input_array)
    if reference_path:
        np.save(reference_path, model(data).numpy(), allow_pickle=False)

    @jit.trace(capture_as_const=True)
    def pred_func(data):
        outputs = model(data)
        return outputs

    pred_func(data)
    pred_func.dump(
        graph_name,
        arg_names=["data"],
        optimize_for_inference=True,
        enable_fuse_conv_bias_nonlinearity=True,
    )


def main(args):
    model = build_and_load(args.ckpt, name=args.name)
    input_array = np.load(args.input_npy, allow_pickle=False) if args.input_npy else None
    dump_static_graph(model, args.dump_path, input_array, args.reference_npy)


if __name__ == "__main__":
    args = make_parser().parse_args()
    main(args)
