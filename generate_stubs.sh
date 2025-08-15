#!/usr/bin/env bash
set -euo pipefail

rm -rf build
cmake -B build -DUSE_PYBIND=ON lib/rai
cmake --build build -j 6
cp build/_robotic.cpython-3*-x86_64-linux-gnu.so robotic/
cd robotic
stubgen -p _robotic -o . --include-docstrings