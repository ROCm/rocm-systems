# Python execution API

The `rocjitsu` package runs Python GPU workloads on simulated AMD GPUs. Use it
from scripts, tests, interactive shells, or notebooks. It provides two execution
styles:

- `run` executes a callable in a fresh subprocess; `Session` reuses a worker
  across calls.
- `enable` activates simulation in the current interpreter, started through
  `python -m rocjitsu` before GPU discovery.

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
flags apply to whole programs launched with `cli()` or `python -m rocjitsu --native`.
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

## Main-process activation

Start a fresh interpreter with the bootstrap:

```sh
python -m rocjitsu app.py
# Also supported: python -m rocjitsu -m your_module
# Equivalent native CLI invocation, using this environment's Python:
rocjitsu --preload-only -- python app.py
# Configure at launch for a script that does not call enable():
python -m rocjitsu --config gpu.json --cpu-thread-budget 2 -- preconfigured_app.py
# Every native flag is accessible without Python application wrapping:
python -m rocjitsu --native --help
```

Then `app.py` selects the configuration before importing GPU frameworks:

```python
import rocjitsu
rocjitsu.enable(rocjitsu.config.gfx1201_r9700)

import torch
x = torch.arange(257, device="cuda", dtype=torch.float32)
print((x + 1).cpu())
```

`enable` activates the existing interpreter: it does not restart the script,
change its PID, or move following statements into a subprocess. Bootstrap
preloads the native interposer before Python starts; activation requires the
matching native version with `--preload-only` support. Calling `enable` in an
ordinary, unprepared interpreter raises an actionable error. In an
interpreter whose startup command cannot be changed, use `run` or `Session`.

The configured launch form selects the simulation before Python starts.
`preconfigured_app.py` must not call `enable()`, and neither should callables
submitted to a `Session`: those workers already have their launch configuration.
`is_enabled()` specifically queries global activation, not configured launches.

Activation is one-way and must happen before any GPU discovery or workload.
Repeated calls with the same configuration are harmless; reconfiguration is
rejected. A library may initialize HIP during an import or availability check,
so activate before framework imports. GPU discovery before activation fails
without falling back to hardware. Global activation validates and snapshots the config into a private directory;
the original file can then be edited or removed. DBT configurations remain available
through the normal CLI and session launcher; global activation accepts local
simulation configs only.

New exec/spawn children inherit the activated configuration and preloading.
They may repeat `enable` with the same effective config and budget; this supports
scripts with top-level activation used by multiprocessing `spawn` and
`forkserver`. `is_enabled()` includes inherited global activation and returns
false for a fork that inherited an already initialized GPU context. It reports
global activation, not configured CLI/Session mode or runtime health.

The interpreter that calls `enable()` owns its config handoff. Keep that
interpreter alive until forked and exec children finish their GPU work. Children
that first initialize after it exits cannot inherit its configuration.
Forking after GPU
initialization does not create a usable GPU context in the child; use a fresh
interpreter, as with ordinary accelerator multiprocessing.

Each inherited child initializes its own local simulator. To share one
simulated GPU across processes, use native daemon/attach launch modes, also
available through `Session(mode="daemon")` and `Session(mode="attach")`.

`enable()` and `--preload-only` always select local simulation and cannot be
combined with daemon or attach mode. A daemon's devices and configuration are
fixed when it starts; Python does not request new devices or reconfigure a
running daemon. `mode="daemon"` starts a simulator owned by that session, while
`mode="attach"` lets multiple sessions reuse an existing simulator and leaves
it running when they close. To use a daemon in the main interpreter, launch
`python -m rocjitsu --daemon --config gpu.json -- app.py`, or use `--attach`
for an existing daemon. Such an application uses its launch configuration and
must not call `enable()`.

Switching an initialized HIP/PyTorch interpreter between hardware and simulation
is unsupported: device discovery, allocations, queues, and framework caches
outlive Python scopes. `Session` is the supported boundary for alternating
hardware and simulation. Simulation runtime measurements are not hardware
performance measurements; avoid using simulator timings to tune GPU kernels.

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
python -m rocjitsu python/examples/check_api.py --mode global --arch gfx1201 \
  --config configs/gfx1201_r9700.json
python python/examples/check_api.py --mode worker --arch gfx1250 \
  --config configs/gfx1250_mi455x.json
python -m rocjitsu python/examples/check_api.py --mode global --arch gfx1250 \
  --config configs/gfx1250_mi455x.json
```

Select the hardware index whose properties report `gfx1201`. The worker test
with that option alternates physical PyTorch/Triton execution in one parent
interpreter with simulated execution in the persistent worker, and verifies
that late global activation is rejected. The global test verifies that both
simulated frameworks execute in the PID that called `enable`.
