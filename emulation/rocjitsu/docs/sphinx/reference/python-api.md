# Python execution API

The `rocjitsu` package provides Python bindings for running workloads on
simulated AMD GPUs. Scripts, tests, interactive shells, and notebooks use the
same API:

- `run` and `Session` execute Python callables in subprocess workers.
- `enable` activates simulation in a prepared main interpreter.
- `config`, `cli`, and `diagnostics` expose installed presets, native commands,
  and installation information.

## Installation

The package requires Linux and a matching native Rocjitsu runtime. Bundled wheels
containing the Python package, native runtime, and JSON presets require follow-up
TheRock packaging support. Once a build containing that support is published,
install its wheel into your Python environment:

```sh
python -m pip install /path/to/rocjitsu-<version>-py3-none-<linux-platform>.whl
```

For a source build:

```sh
# From emulation/rocjitsu, with your Python virtual environment active:
python -m pip install ./python
cmake --install /path/to/native/build --prefix "$VIRTUAL_ENV"
```

The native executable and JSON presets come from the installation. Workers use
the calling interpreter's Python by default. Install workload dependencies in
the interpreter executing them: PyTorch needs a ROCm-enabled build, and Triton
needs a compiler supporting the simulated target.

## Subprocess execution

```python
import rocjitsu

def add_one():
    import torch
    return (torch.arange(8, device="cuda") + 1).cpu()

with rocjitsu.Session(rocjitsu.config.gfx1250_mi455x) as sim:
    result = sim.run(add_one)
print(result)
```

`run` starts a fresh worker for one call. `Session` retains imports and worker
state across calls; close it explicitly or use a context manager. Send CPU
values and return CPU results. GPU tensors and CPU tensors requiring gradients
cannot cross the process boundary.

Preset attributes match existing JSON filename stems. Use
`rocjitsu.config.names()` to list them. Paths and mappings using the native
config schema are also accepted.

If the caller uses a different GPU framework, such as CUDA PyTorch, install
`rocjitsu` and the ROCm workload dependencies into a separate environment. Both
environments need matching Python major/minor/patch and cloudpickle versions.
Select the worker interpreter with `python=`:

```python
with rocjitsu.Session(
    rocjitsu.config.gfx1250_mi455x,
    python="/path/to/rocm-venv/bin/python",
) as sim:
    result = sim.run(add_one)
```

## Main-process activation

Start the application with the interposer preloaded:

```sh
python -m rocjitsu app.py
# Equivalent native CLI invocation:
rocjitsu --preload-only -- python app.py
```

In `app.py`, activate simulation before GPU discovery or framework imports:

```python
import rocjitsu
rocjitsu.enable(rocjitsu.config.gfx1250_mi455x)

import torch
print((torch.arange(8, device="cuda") + 1).cpu())
```

`enable` keeps execution in the same interpreter. Activation is one-way;
changing configuration requires a new process. Importing the package alone
does not prepare an ordinary interpreter for activation. Use `run` or `Session`
when the calling interpreter is already running without the preload setup.

## Daemon execution

`enable()` and `--preload-only` configure local simulation in the current process.
For a simulator shared by multiple processes, use `Session(mode="daemon")` to
start a session-owned daemon, or `Session(mode="attach")` to connect to an
existing daemon. Closing an attached session leaves the daemon running for
other clients. Its devices and configuration are fixed at daemon startup;
the Python API does not request new devices or reconfigure it.

For daemon execution in the main interpreter, select the mode at launch:

```sh
python -m rocjitsu --daemon --config gpu.json -- app.py
# Or attach to an existing daemon:
python -m rocjitsu --attach --config gpu.json -- app.py
```

These launches already configure the runtime, so `app.py` must not call `enable()`.

## Notebook example

Jupyter and Colab can use the same subprocess APIs from an existing kernel.
The [example notebook](https://github.com/ROCm/rocm-systems/blob/develop/emulation/rocjitsu/python/examples/notebook.ipynb)
shows installation and execution with named presets. It requires no special
IPython integration; `%pip` is the notebook equivalent of the installation
command above.

See the [package documentation](https://github.com/ROCm/rocm-systems/blob/develop/emulation/rocjitsu/python/README.md)
for installation, callable transport, launch options, errors and activation
inheritance, and the [CLI reference](rocjitsu-cli.md) for native launch modes.
