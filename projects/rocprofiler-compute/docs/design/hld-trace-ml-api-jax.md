# JAX trace (`--trace-ml-api jax`) — High-Level Design

## System Context

For ML frameworks, rocprofiler-compute provides an ML API trace pipeline under
`--trace-ml-api`. It wraps each framework call in a ROCTX range, so the kernels
launched share correlation id with the marker. This is used to group the kernels under
the operation that launched them.

JAX compiles each `jax.jit` / `jax.pmap` function into an XLA module. The
function is lowered to HLO (High Level Optimizer IR), optimized, and emitted as
one module of fused kernels. The first call with a given input signature
compiles the module; later calls with the same signature reuse it.

The module is the compiled function that eventually runs. Fusion chooses the
kernel names and boundaries (`gemm_fusion_dot_general_0`, `Cijk_...`). Operators
inside the jit function code have no Python call and no marker. The only
identity available at the call, which is where a marker can be placed, is the
compiled function.

Without an outer `jit`, each primitive is compiled and executed on its own
when Python reaches it (`x @ x`, `jnp.sum`, …). That is eager dispatch.

## Problem statement

The ML API trace has no JAX backend, so JAX GPU time cannot be attributed to
the jit function or eager op that launched it.

## Requirements

Functional:

- Attribute kernels of a `jax.jit` / `jax.pmap` call to that compiled
  function, with no changes to the workload source.
- Attribute kernels of an eager JAX op to the primitive that dispatched
  them.
- Reuse the existing ML API trace `marker -> counter -> call-tree` pipeline.

## Considered and not chosen

**HLO operator and source.** Optimized HLO dumps can name the HLO operator
and source line of each instruction. We do not join that onto the tree. The
kernels come from compilation and fusion: several operators in a compiled
function can become one kernel, and one operator can become several kernels.
Putting that mapping into the call tree would turn a function-to-kernel
nesting into a many-to-many join.

**Native `Execute` hook or per-launch HIP markers.** A native hook on
`Execute` would mark one module (one input signature) rather than the
function, and would not use `inject_roctx`. Annotating each HIP launch would
name the kernel, not `jax.jit.<fn>`.

## Design

JAX is added as a backend to the existing `--trace-ml-api` pipeline, through
`inject_roctx`. `jax.jit` / `jax.pmap` open a range around each compiled
call. Eager ops (no outer `jit`) open a range at the eager dispatch hook,
named `jax.{primitive}`. In both cases `Execute` runs inside the range, so
kernels join to that marker.

### At a glance

The marker trace and the kernel trace are joined on `Correlation_ID` and
`GUID` in the existing pipeline (CSV `Correlation_Id` is renamed). The
marker and the kernels launched inside its range share one correlation id;
`GUID` separates processes.

```
   workload run (instrument)                     analysis
   ─────────────────────────                     ────────
 jax.jit / jax.pmap / eager dispatch ──ROCTX──►  marker trace ┐
                                                    │ join on
 XLA executable ────kernel launches►  kernel trace ┘ Correlation_ID + GUID
                                                    │
                                                    ▼
                                            call tree
```

### Where the range is marked

The launcher replaces `jax.jit` / `jax.pmap` at install. Compilation is left
untouched; the wrapper sits on the compiled callable. Each call opens a ROCTX
range named `jax.jit.<fn>` or `jax.pmap.<fn>` on that thread, runs the
executable, and closes the range when the callable returns.

A nested `jit` is inlined into the outer module, so it is not a second
`Execute`. The inner wrapper does not open a range; those kernels stay under
the outer `jax.jit.<fn>`. A direct call to the inner function is its own
range.

Eager ops do not go through those callables. `jnp`, `x @ x`, and `jax.lax`
meet at one eager dispatch. The launcher wraps that hook once. If a JAX range
is already open, or JAX is tracing a `jit`, the hook is unchanged. Otherwise
it opens `jax.{primitive}` around the dispatch, which compiles a one-op
module and runs `Execute`. We do not wrap `jnp.*`.

JAX dispatch is asynchronous: the callable returns before the GPU finishes.
The range encloses the host-side launch enqueue, not GPU completion. Those
enqueues inherit the marker's correlation id.

```
launching CPU thread (time →)

  ROCTX range open  "jax.jit.train_step"          [Correlation_Id = C]
  │
  └─ compiled callable                            (cached XLA executable)
       └─ PjRtLoadedExecutable::Execute
            ├─ launch  gemm_fusion_dot_general_0   ┐
            ├─ launch  input_reduce_fusion         │ launched within the range
            └─ launch  input_reduce_fusion_1       ┘ → inherit Correlation_Id = C
  │
  ROCTX range close

rocprofv3 marker trace :  "jax.jit.train_step"        Correlation_Id = C
rocprofv3 kernel trace :  gemm_fusion_dot_general_0   Correlation_Id = C
                          input_reduce_fusion         Correlation_Id = C
                          input_reduce_fusion_1       Correlation_Id = C
        join on Correlation_ID + GUID  ⇒  kernels attributed to jax.jit.train_step
```


### Difference from the Torch/Triton backends

Torch and Triton mark at operator or kernel granularity (Torch wraps each ATen
op, Triton each kernel launch), so the marker names the operation. After `jit`
the ops are fused into one executable with no per-op Python call, so that
marker names the function. Eager JAX is one primitive per `Execute`, so the
eager dispatch hook marks like an ATen op (`jax.dot_general`). Kernels are
direct children of that marker.

### Analyze display

The workload below is the reference for the tree. `train_step` calls
`matmul_relu` from inside its own range, so the inner call is not a separate
node. Its kernels are listed under `jax.jit.train_step`.

```python
# jax_workload.py
@jax.jit
def matmul_relu(a, b):
    return jnp.maximum(a @ b, 0.0)   # line 7

@jax.jit
def train_step(w, x):
    y = matmul_relu(x, w)            # line 12
    return jnp.sum(y * y)            # line 13
```

`--list-operators jax` prints the call tree the ML API trace already prints for
Torch and Triton: one node per compiled function or eager primitive, caller
`file:line` on that node, kernels underneath sorted by total GPU time. A
per-operator summary table follows the tree. A direct call to `matmul_relu`
is its own root, `jax.jit.matmul_relu`. An eager op in the same run is its
own root, for example `jax.dot_general`, with that op's kernels underneath.

```
================================================================================
JAX Operator Call Tree: <workload>
Sorted by total GPU kernel duration.
================================================================================

jax.jit.train_step jax_workload.py:26 (calls: N, dispatches: M, total: T)
├─ gemm_fusion_dot_general_0 (dispatches: M, total: T)
├─ Cijk_..._WGM6 (dispatches: M, total: T)
├─ input_reduce_fusion (dispatches: M, total: T)
└─ input_reduce_fusion_1 (dispatches: M, total: T)
```

`--operator jax` filters that tree by function name and prints the matched
nodes in the same form.

## Validation, security and debuggability

- **Marker wrapper.** A call to a `jax.jit` / `jax.pmap` function emits one
  range named `jax.jit.<fn>` or `jax.pmap.<fn>`. An eager op emits
  `jax.{primitive}`; the eager hook during `jit` tracing does not.
- **Call tree.** On a profiled workload, the kernels launched by a call appear
  under that marker, with the caller `file:line` and the call's dispatch
  count and GPU time.
- **Security.** The launcher is the same in-process injector used for other backends.
  Analysis reads the trace files the workload produced.


Coverage gaps:

- **`pjit` name.** Current sharding is `jax.jit`. Call sites that still use
  the deprecated `pjit` alias are unmarked unless that name is patched too.

Design trade-offs:

- A `jit` / `pmap` marker is the compiled function. Every kernel that call
  launches — fused kernels, library GEMM (`Cijk_...`), runtime copy/fill
  helpers, and autotuning kernels — is a direct child of that node. One
  function name covers every input signature.
