# Python execution API

The `rocjitsu` package runs Python GPU workloads on simulated AMD GPUs from
scripts, tests, interactive shells, or notebooks. `run` executes a callable in
a fresh subprocess; `Session` reuses a worker across calls.

The package also exposes installed configuration presets, native CLI access,
and installation diagnostics. Linux and a matching native Rocjitsu runtime are
required. GPU frameworks are workload dependencies: PyTorch workloads need a
ROCm-enabled build, and Triton kernels need a compiler that supports the
simulated target.

## Installation

Bundling this package, the native launcher and libraries, and the existing JSON
presets in a wheel requires follow-up TheRock packaging changes. Once a build
containing that support is published, install its wheel into your Python
environment:

```sh
python -m pip install /path/to/rocjitsu-<version>-py3-none-<linux-platform>.whl
```

Use the actual wheel filename or the build's package index. Older TheRock
builds do not include this wheel. No source checkout, CMake install, or
`rocm-sdk init` is needed for a bundled wheel.

For a source build, run the shell examples below from `emulation/rocjitsu/`.
Install the Python package and native runtime into the same environment:

```sh
# With your Python virtual environment active:
python -m pip install ./python
cmake --install /path/to/native/build --prefix "$VIRTUAL_ENV"
```

A bundled wheel uses its own launcher; a missing bundled launcher is an error.
Source installs use the environment's `bin/rocjitsu`, then `rocjitsu` on `PATH`. Native libraries
and presets in `share/rocjitsu/configs` come from the selected installation.
Workers use the caller's Python executable by default. The PyTorch examples
below require ROCm-enabled PyTorch in the interpreter executing the workload;
a CUDA-only build cannot execute HIP code.

## Subprocess execution

Submit a callable with `run`, or use a `Session` for repeated calls:

```python
import rocjitsu

def workload(n):
    import torch
    x = torch.arange(n, device="cuda", dtype=torch.float32)
    return (x + 1).cpu()

result = rocjitsu.run(workload, 257, config=rocjitsu.config.gfx1201_r9700)

with rocjitsu.Session(rocjitsu.config.gfx1250_mi455x) as sim:
    first = sim.run(workload, 257, timeout=120)
    second = sim.run(workload, 129, timeout=120)
print(first, second)
```

`rocjitsu.config` exposes every installed JSON file under its existing filename
stem, with no aliases: `gfx1250_mi455x` selects `gfx1250_mi455x.json`.
Use `rocjitsu.config.names()` or tab completion to discover them. Strings such
as `"gfx1250_mi455x"` also select installed presets. Existing file paths and
mappings using the native config schema remain supported; an existing local
file takes precedence over a string preset name.

Each session starts a fresh Python interpreter, using the caller's Python
executable by default. The caller's first local import directory is forwarded;
other caller search paths and its stdlib are not. The worker keeps its own
installed packages, including `rocjitsu` and `cloudpickle`. The launcher does not
add the caller's package directory to `PYTHONPATH`; explicitly supplied
`PYTHONPATH` entries remain the caller's responsibility. The worker's imports,
compiler caches, and module state persist across calls. The caller's environment
is not modified. `env=` supplies overrides for the worker only. Calls on a session are
serialized; separate sessions can use separate targets.

Functions, closures, and arguments are serialized with cloudpickle. Importable
modules must remain accessible to the worker. Put Triton JIT kernels in an
importable Python module so the compiler can retrieve their source; see
`examples/toy_kernels.py`. Only trusted code should be submitted to a session.

Pass CPU tensors, NumPy arrays, or ordinary serializable Python values. Create
GPU tensors, models, and streams inside the worker, and copy returned tensors
to the CPU. GPU tensors are rejected during serialization; detach CPU tensors
that require gradients. Object identity, mutations, and autograd graphs do not
cross the process boundary. Imported module globals are the worker's own
globals, not a snapshot of an already initialized parent module.

PyTorch GPU work is synchronized before a successful reply, including all
initialized devices. Other frameworks must synchronize asynchronous work before
returning. Worker stdout and stderr are forwarded to the caller's Python streams.
A callable exception produces `RemoteError` with its traceback and leaves the
worker available. A crash produces `WorkerError`. A timeout terminates the worker and
its subprocess group and closes the session. The call timeout starts after
serialization and acquisition of the session lock; session startup has a
separate `startup_timeout` (30 seconds by default).

## Launch options and diagnostics

`Session` and `run` accept these keyword-only launch options:

| Option | Behavior |
| --- | --- |
| `cpu_thread_budget=N` | Native `--cpu-thread-budget` override; zero selects automatic sizing; source config is unchanged. |
| `mode="local"` | In-process simulation inside the Python worker (default). |
| `mode="daemon"` | Native `--daemon` with an owned simulator child. Closing the session terminates the whole private process group. |
| `mode="attach"` | Native `--attach` to an existing daemon; closing the session leaves that daemon running. The daemon owns the config; budget overrides are rejected. |
| `env={...}`, `cwd=...` | Worker-only environment overrides and working directory. Config paths resolve in the caller's directory. |
| `python=...` | Worker Python interpreter; defaults to the caller's interpreter. |
| `startup_timeout=30` | Deadline for both worker handshakes, in seconds. |
| `capture_output=True` | Keep output in memory without forwarding it to caller streams. |
| `log_limit=65536` | Maximum retained bytes per output stream (a rolling tail). |

Inspect `session.pid`, `session.command`, `session.closed`, `session.returncode`,
`session.stdout`, and `session.stderr`. Output collection is asynchronous; close
or a worker failure drains its output before reporting the final tails. A
`WorkerError` carries `command`, `returncode`, `stdout`, and `stderr`;
`returncode` is `None` until the worker exits or if its supervisor could not
report the status; its message includes the native diagnostic tail. A private
supervisor owns the worker status even when the caller ignores SIGCHLD or uses
an external child reaper. Startup failures and protocol/Python
version mismatches also raise `WorkerError`. `RemoteError` carries the remote
exception `kind`, `message`, and `remote_traceback`. A remote exception does not
roll back GPU work or worker state. Use a new session if the workload has left
its runtime unusable.

Calls on a session are serialized. `close()` from another thread cancels an
active call (which raises `WorkerError`), terminates the worker group, and frees
owned config files. The supervisor remains the private group leader through
cleanup, including delayed close after worker exit; the caller never signals a
stored numeric process group ID. Use a context manager or explicitly close sessions; live
sessions are otherwise retained until interpreter shutdown. Children that
intentionally leave the process group are outside this lifecycle. Plain fork
children cannot use a parent's session.

Use native config fields for runtime diagnostics and `env` for native environment
switches, for example `env={"RJ_VMEM_TRACE": "1"}` for simulated memory tracing.
Log groups controlled by the native `RJ_LOG_GROUPS` build setting still require
a binary built with those groups; Python cannot enable code compiled out of it.

```python
info = rocjitsu.diagnostics(config, cpu_thread_budget=2)
print(info["native_version"])
print(info["thread_budget_table"])

# Complete native CLI access, including flags for standalone/server modes:
help_text = rocjitsu.cli(["--help"]).stdout
capability = rocjitsu.cli(["--check-vfio-user"], check=False)
```

`diagnostics()` queries the version and VFIO capability without initializing a
GPU runtime; with a config it also queries `--thread-budget-table`. It returns
a dict with these supported fields; future versions may add fields:

| Key | Value |
| --- | --- |
| `executable` | Absolute path to the selected native launcher. |
| `python` | The caller's Python executable. |
| `python_version` | The caller's Python version string. |
| `native_version` | Native `--version` output, as text. |
| `vfio_user` | Whether the native VFIO capability check succeeded. |
| `vfio_diagnostic` | Text output from the capability check. |
| `thread_budget_table` | Native thread-budget table text; present only when `config` is supplied. |

`cli()` accepts argument tokens, an optional `config`, `env`, `cwd`,
`timeout` (30 seconds by default), and `check`. It returns a
`subprocess.CompletedProcess` with captured text. Failed commands raise
`CalledProcessError` containing stdout/stderr unless `check=False`. Long-running
server commands such as VFIO need an appropriate timeout (or `None`); they do
not run Python callables. Session modes preserve the native CLI's limitations,
including its DBT and attach restrictions. Native startup config paths used by
sessions must remain available and unchanged until the session closes; DBT
sibling config paths retain their original resolution rules. For mappings with
DBT config references, use absolute paths because the mapping is materialized
in a temporary directory.

`Session` and `run` expose the launch options in the table above. Other native
flags apply to whole programs launched with `cli()`.
All launch keywords in the convenience `run` function are reserved. To pass a
workload argument with the same name, wrap it in a closure or `functools.partial`.
In `Session.run`, only `timeout` is reserved. The private control protocol requires
matching package code, exact cloudpickle version and Python major/minor/patch
version on both ends. Pin those dependencies together when using `python=`.
Unpickling executes code; this API is for trusted local workloads, not sandboxing
or accepting requests from untrusted clients.

### A separate worker environment

Use `python=` when the calling application and simulated workload need different
GPU frameworks, such as CUDA PyTorch in the caller and ROCm PyTorch in the worker:

```python
with rocjitsu.Session(
    rocjitsu.config.gfx1250_mi455x,
    python="/path/to/rocm-venv/bin/python",
) as sim:
    output = sim.run(workload, 257)
```

Install `rocjitsu` and the workload's dependencies into the worker environment.
Both interpreters must have the same Python major/minor/patch and cloudpickle
versions, compatible serialization dependencies, and access to imported workload
modules. The calling process can continue using its physical GPU between calls.

## Daemon execution

Use `Session(mode="daemon")` to start a simulator owned by that session, or
`Session(mode="attach")` to reuse an existing daemon. Closing an attached
session leaves the daemon running for other clients. A daemon's devices and
configuration are fixed when it starts; Python does not request new devices
or reconfigure a running daemon.

The caller can continue using its physical GPU while submitting simulated work
to a session. Simulation runtime measurements are not hardware performance
measurements; avoid using simulator timings to tune GPU kernels.

## Notebook example

The same `run` and `Session` APIs work in Jupyter and hosted notebooks such as
Colab. They arrange preloading inside their workers, so the running kernel needs
no special startup command. See [notebook.ipynb](examples/notebook.ipynb) for a
complete example.

Install a bundled wheel in the kernel's environment using a notebook cell:

```python
%pip install /path/to/rocjitsu-<version>-py3-none-<linux-platform>.whl
```

Then submit the workload defined above with an installed preset:

```python
with rocjitsu.Session(rocjitsu.config.gfx1250_mi455x) as sim:
    output = sim.run(workload, 257)
output
```

For a kernel using CUDA PyTorch, select a [separate ROCm worker
environment](#a-separate-worker-environment) with `python=`. The kernel and worker
have the same dependency and serialization requirements as any other caller and
worker pair.

## Validation

Run the API integration tests after installing the package:

```sh
PATH=/path/to/build/tools/rocjitsu:$PATH \
  python python/tests/test_api.py -v
```

Use the matching native installation in the Python environment, or put a source
build on `PATH` when that environment has no native installation. CTest also
registers `PythonApiTest`, skipped if its Python lacks cloudpickle. The required
Release corpus CI lane installs the package and runs this script directly, so
missing dependencies fail that lane instead of skipping the suite.
The toy workloads check all 257 elements exactly, architecture identity, and
simulator allocation backing. With a compatible ROCm/PyTorch/Triton environment:

```sh
python python/examples/check_api.py --mode worker --arch gfx1201 \
  --config configs/gfx1201_r9700.json --hardware-index 0
python python/examples/check_api.py --mode worker --arch gfx1250 \
  --config configs/gfx1250_mi455x.json
```

Select the hardware index whose properties report `gfx1201`. The worker test
with that option alternates physical PyTorch/Triton execution in one parent
interpreter with simulated execution in the persistent worker.
