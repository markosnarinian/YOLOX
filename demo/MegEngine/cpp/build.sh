#!/usr/bin/env bash
set -euo pipefail

# CXX and CMAKE_TOOLCHAIN_FILE may select a cross compiler. CMake also works
# directly from PowerShell/MSVC; this wrapper is only a Unix convenience.
source_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cmake -S "$source_dir" -B "$source_dir/build" \
    -DCMAKE_BUILD_TYPE=Release -DCMAKE_EXPORT_COMPILE_COMMANDS=ON "$@"
cmake --build "$source_dir/build" --config Release \
    --parallel "${CMAKE_BUILD_PARALLEL_LEVEL:-2}"
