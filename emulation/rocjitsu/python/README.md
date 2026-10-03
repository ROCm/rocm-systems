# Python execution API

Run the shell examples below from `emulation/rocjitsu/` in the repository.
Install with `pip install ./python`. The package uses a
separately installed rocjitsu executable and its matching native libraries.
Put that executable on `PATH`, set `ROCJITSU_EXECUTABLE`, or pass `executable=`
to `run` or `Session`. Linux is required. The examples below also require
ROCm-enabled PyTorch and a compatible Triton compiler, including support for
the simulated target. A CUDA-only PyTorch installation cannot execute HIP code.

## One call or a persistent session

```python
import rocjitsu

def workload(n):
    import torch
    x = torch.arange(n, device="cuda", dtype=torch.float32)
    return (x + 1).cpu()

result = rocjitsu.run(workload, 257, config="gfx1201_r9700.json", timeout=120)

with rocjitsu.Session("gfx1250_mi455x.json") as sim:
    first = sim.run(workload, 257, timeout=120)
    second = sim.run(workload, 129, timeout=120)
```

`config` accepts a JSON file path or a mapping using the existing native config
schema. Each session starts a fresh Python interpreter, using the caller's Python
executable by default. User module search paths are forwarded; the worker keeps
its own installed packages, including rocjitsu and cloudpickle. The launcher does
not add the caller's package directory to `PYTHONPATH`; explicitly supplied
`PYTHONPATH` entries remain the caller's responsibility. The worker's imports, compiler
caches, and module state persist across calls. The caller's environment is not
modified. `env=` supplies overrides for the worker only. Calls on a session are
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
returning. Worker stdout and stderr are forwarded to the caller's Python streams,
including notebook output. A callable
exception produces `RemoteError` with its traceback and leaves the worker
available. A crash produces `WorkerError`. A timeout terminates the worker and
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
| `python=...`, `executable=...` | Worker Python and native rocjitsu executable. |
| `startup_timeout=30` | Deadline for both worker handshakes, in seconds. |
| `capture_output=True` | Keep output in memory without forwarding it to caller streams. |
| `log_limit=65536` | Maximum retained bytes per output stream (a rolling tail). |

Inspect `session.pid`, `session.command`, `session.closed`, `session.returncode`,
`session.stdout`, and `session.stderr`. Output collection is asynchronous; close
or a worker failure drains its output before reporting the final tails. A
`WorkerError` carries `command`, `returncode`, `stdout`, and `stderr`;
`returncode` is `None` if SIGCHLD handling or an external reaper lost the status; its message
includes the native diagnostic tail. Startup failures and protocol/Python
version mismatches also raise `WorkerError`. `RemoteError` carries the remote
exception `kind`, `message`, and `remote_traceback`. A remote exception does not
roll back GPU work or worker state. Use a new session if the workload has left
its runtime unusable.

Calls on a session are serialized. `close()` from another thread cancels an
active call (which raises `WorkerError`), terminates the worker group, and frees
owned config files. Use a context manager or explicitly close sessions; live
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
GPU runtime; with a config it also queries `--thread-budget-table`. `cli()`
accepts argument tokens, an optional `config`, `env`, `cwd`, `executable`,
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

All launch keywords in the convenience `run` function are reserved. To pass a
workload argument with the same name, wrap it in a closure or `functools.partial`.
In `Session.run`, only `timeout` is reserved. The private control protocol requires
matching package code, cloudpickle version and Python version on both ends.
Unpickling executes code; this API is for trusted local workloads, not sandboxing
or accepting requests from untrusted clients.

## Notebooks, including hosted notebooks

`run` and `Session` work from an already running notebook: they arrange
preloading inside their workers and require no changes to the notebook server
or kernel launcher. Configure the executable from a Python cell if necessary:

```python
sim = rocjitsu.Session(config, executable="/path/to/rocjitsu")
try:
    output = sim.run(workload, 257, timeout=120)
finally:
    sim.close()
```

The notebook can continue using a physical GPU between calls to `sim.run`.
Hardware execution stays in the notebook process, while simulated execution
uses the worker process. For a CUDA-based host notebook, install ROCm-enabled
PyTorch, Triton, and this package into a separate environment, then pass
`python="/path/to/rocm-venv/bin/python"` to `Session` or `run`. The worker must
have exactly the same Python version as the notebook, compatible serialization
dependencies, and access to any imported workload modules. This keeps the
notebook's installed GPU framework separate from the simulator's ROCm stack.

## Global activation

Start a fresh interpreter with the bootstrap:

```sh
python -m rocjitsu app.py
# Also supported: python -m rocjitsu -m your_module
# Configure directly at launch (application need not call enable):
python -m rocjitsu --config gpu.json --cpu-thread-budget 2 -- app.py
# Every native flag is accessible without Python application wrapping:
python -m rocjitsu --native --help
```

Then `app.py` selects the configuration before importing GPU frameworks:

```python
import rocjitsu
rocjitsu.enable("gfx1201_r9700.json", cpu_thread_budget=2)

import torch
x = torch.arange(257, device="cuda", dtype=torch.float32)
print((x + 1).cpu())
```

`enable` activates the existing interpreter: it does not restart the script,
change its PID, or move following statements into a subprocess. Bootstrap
preloads the native interposer before Python starts; activation requires the
matching native version with `--preload-only` support. Calling `enable` in an
ordinary, unprepared interpreter raises an actionable error. In a public
notebook whose kernel launcher cannot be changed, use `run` or `Session`.

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
Keep the enabling process alive until those children finish. Forking after GPU
initialization does not create a usable GPU context in the child; use a fresh
interpreter, as with ordinary accelerator multiprocessing.

Switching an initialized HIP/PyTorch interpreter between hardware and simulation
is unsupported: device discovery, allocations, queues, and framework caches
outlive Python scopes. `Session` is the supported boundary for alternating
hardware and simulation. Simulation runtime measurements are not hardware
performance measurements; avoid using simulator timings to tune GPU kernels.

## Validation

Run the API integration tests after installing the package:

```sh
ROCJITSU_EXECUTABLE=/path/to/build/tools/rocjitsu/rocjitsu \
  python python/tests/test_api.py -v
```

CTest also registers `PythonApiTest`, skipped if its Python lacks cloudpickle.
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
