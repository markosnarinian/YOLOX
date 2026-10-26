#!/usr/bin/env python3
"""Native extension and package mapping; static metadata lives in pyproject.toml."""

from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import find_namespace_packages, setup


setup(
    package_dir={"yolox.tools": "tools", "yolox.exp.default": "exps/default"},
    packages=find_namespace_packages(include=("yolox", "yolox.*"))
    + ["yolox.tools", "yolox.exp.default"],
    ext_modules=[
        Pybind11Extension(
            "yolox.layers.fast_cocoeval",
            ["yolox/layers/cocoeval/cocoeval.cpp"],
            define_macros=[("TORCH_EXTENSION_NAME", "fast_cocoeval")],
            cxx_std=14,
            optional=True,
        )
    ],
    cmdclass={"build_ext": build_ext},
)
