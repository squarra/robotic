robotic is a python wrapper for rai

## Generating the stubs

```sh
cmake -B build -DUSE_PYBIND=ON lib/rai
cmake --build build -j 6
cp build/_robotic.cpython-3*-x86_64-linux-gnu.so robotic/
cd robotic
stubgen -p _robotic -o . --include-docstrings
```

## Setting up the environment

We use uv for best reproducibility. Here is how to get started if you want to run the examples.

```sh
uv sync --group examples
source .venv/bin/activate
uv pip install -e .
```