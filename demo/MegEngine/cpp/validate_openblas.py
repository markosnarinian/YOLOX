#!/usr/bin/env python3
"""Fail-closed validation of the OpenBLAS suites bundled with MegEngine 1.13.3.

Run executables directly: their CTest wrappers can hide crashes and numerical
failures. Each suite runs in a fresh directory, so stale summaries cannot pass.
Only the Python standard library is needed; no MegEngine/Torch import occurs.
"""

import argparse
import os
from pathlib import Path
import re
import subprocess
import tempfile


FAILURE = re.compile(r"FATAL|FAILED ON|----- FAIL|ABANDONED|SIGSEGV|Segmentation fault", re.I)


def check_output(output, returncode, *, level=0, routines=(), cblas=False):
    if returncode != 0:
        raise ValueError(f"executable exited {returncode}")
    if FAILURE.search(output):
        raise ValueError("failure reported in numerical output")
    if level == 0:
        match = re.search(r"RESULTS: (\d+) tests \((\d+) ok, 0 failed, 0 skipped\)", output)
        if not match or int(match[1]) == 0 or match[1] != match[2]:
            raise ValueError("missing complete unit-test results")
    elif level == 1:
        if len(re.findall(r"----- PASS -----", output)) != 10:
            raise ValueError("missing level-1 computational passes")
    else:
        if not routines or "END OF TESTS" not in output:
            raise ValueError("missing computational coverage or end marker")
        layouts = ("COLUMN-MAJOR", "ROW-MAJOR") if cblas else ("",)
        for routine in routines:
            # The old complex level-2 suites print two identical, unlabelled
            # passes for column/row order. Require both, not just the first.
            if cblas and level == 2 and routine.lower().startswith(("cblas_c", "cblas_z")):
                pattern = (rf"\b{re.escape(routine)}\s+PASSED THE COMPUTATIONAL TESTS"
                           r"\s*\(\s*[1-9]\d* CALLS\)")
                if ("COLUMN-MAJOR AND ROW-MAJOR DATA LAYOUTS ARE TESTED" not in output
                        or len(re.findall(pattern, output, re.I)) != 2):
                    raise ValueError(f"missing two-layout computational passes: {routine}")
                continue
            for layout in layouts:
                pattern = (
                    rf"\b{re.escape(routine)}\s+PASSED THE\s+{layout}\s*"
                    r"COMPUTATIONAL TESTS\s*\(\s*[1-9]\d* CALLS\)"
                )
                if not re.search(pattern, output, re.I):
                    raise ValueError(f"missing computational pass: {routine} {layout}")


def validate(source, build, logs, threads):
    logs.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, OPENBLAS_NUM_THREADS=str(threads), OMP_NUM_THREADS=str(threads))
    suites = [("utest/openblas_utest", None, 0, False)]
    for directory, cblas in (("test", False), ("ctest", True)):
        for dtype in "sdcz":
            for level in (1, 2, 3):
                binary = f"x{dtype}cblat{level}" if cblas else f"{dtype}blat{level}"
                data = None if level == 1 else source / directory / (
                    f"{dtype}in{level}" if cblas else f"{dtype}blat{level}.dat"
                )
                suites.append((f"{directory}/{binary}", data, level, cblas))
    failed = 0
    for executable, data, level, cblas in suites:
        name = Path(executable).name
        output = ""
        try:
            input_text = data.read_text() if data else ""
            routines = re.findall(r"^(\w+)\s+T\s+PUT F FOR NO TEST", input_text, re.M)
            if data and re.search(r"^\w+\s+F\s+PUT F FOR NO TEST", input_text, re.M):
                raise ValueError("disabled computational routine in input")
            with tempfile.TemporaryDirectory(prefix="openblas-check-") as work:
                result = subprocess.run(
                    [str(build / executable)], input=input_text, text=True,
                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                    cwd=work, env=env, timeout=120,
                )
                output = result.stdout
                if data and not cblas:
                    summary = Path(work) / f"{name.upper()}.SUMM"
                    if not summary.is_file() or not summary.stat().st_size:
                        raise ValueError("missing or empty summary")
                    output += "\n" + summary.read_text()
                check_output(output, result.returncode, level=level,
                             routines=routines, cblas=cblas)
            print(f"PASS {name}")
        except (OSError, ValueError, subprocess.TimeoutExpired) as error:
            failed += 1
            output += f"\nVALIDATION FAILED: {error}\n"
            print(f"FAIL {name}: {error}")
        (logs / f"{name}.log").write_text(output)
    print(f"{len(suites) - failed}/{len(suites)} numerical suites passed ({threads} thread(s))")
    return int(failed != 0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="OpenBLAS source directory")
    parser.add_argument("--build", type=Path, required=True, help="OpenBLAS CMake build directory")
    parser.add_argument("--logs", type=Path, required=True, help="directory for complete suite logs")
    parser.add_argument("--threads", type=int, choices=(1, 2), default=1)
    args = parser.parse_args()
    raise SystemExit(validate(args.source.resolve(), args.build.resolve(), args.logs, args.threads))
