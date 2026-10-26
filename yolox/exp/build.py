#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii Inc. All rights reserved.

import importlib
import importlib.util
import hashlib
import os
import sys


def get_exp_by_file(exp_file):
    exp_file = os.path.abspath(exp_file)
    module_name = "_yolox_exp_" + hashlib.sha256(os.fsencode(exp_file)).hexdigest()
    old_path = sys.path[:]
    try:
        sys.path.insert(0, os.path.dirname(exp_file))
        current_exp = sys.modules.get(module_name)
        if current_exp is None:
            spec = importlib.util.spec_from_file_location(module_name, exp_file)
            current_exp = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = current_exp
            try:
                spec.loader.exec_module(current_exp)
            except Exception:
                sys.modules.pop(module_name, None)
                raise
        exp = current_exp.Exp()
        exp._exp_file = exp_file
        return exp
    except Exception as error:
        raise ImportError(f"Could not load experiment from {exp_file}") from error
    finally:
        sys.path[:] = old_path


def get_exp_by_name(exp_name):
    exp = exp_name.replace("-", "_")  # convert string like "yolox-s" to "yolox_s"
    module_name = ".".join(["yolox", "exp", "default", exp])
    exp_object = importlib.import_module(module_name).Exp()
    return exp_object


def get_exp(exp_file=None, exp_name=None):
    """
    get Exp object by file or name. If exp_file and exp_name
    are both provided, get Exp by exp_file.

    Args:
        exp_file (str): file path of experiment.
        exp_name (str): name of experiment. "yolo-s",
    """
    assert (
        exp_file is not None or exp_name is not None
    ), "plz provide exp file or exp name."
    if exp_file is not None:
        return get_exp_by_file(exp_file)
    else:
        return get_exp_by_name(exp_name)
