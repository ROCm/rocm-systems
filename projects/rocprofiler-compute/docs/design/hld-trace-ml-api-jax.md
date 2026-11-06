# JAX trace (`--trace-ml-api jax`) — High-Level Design

## System Context

- rocprofiler-compute provides an ML API trace pipeline under `--trace-ml-api`:
  - It wraps each framework call in a ROCTX range, so the kernels launched
    share `Correlation_ID` and `GUID` with the marker.
  - This is used to group the kernels under the operation that launched them.
- JAX compiles each `jax.jit` / `jax.pmap` / `jax.shard_map` function, and
  each eager primitive, into an XLA module:
  - The function or primitive is lowered to HLO (High Level Optimizer IR),
    optimized, and emitted as one module of fused kernels.
  - The first call with a given input signature compiles the module; later calls
    with the same signature reuse it.
  - `jax.jit` / `jax.pmap` / `jax.shard_map`: the module is the compiled
    function that eventually runs:
    - Fusion decides the kernel names and which operators' work goes into a kernel.
    - Operators inside the jit function code have no Python call and no marker.
    - The only identity available at the call, which is where a marker can be
      placed, is the compiled function.
  - Without an outer `jit`, each op is compiled and executed on its own when
    Python reaches it (`x @ x`, `jax.numpy.sum`, …). That is eager dispatch.
- Assumptions:
  - The HLD was planned by considering JAX 0.11.0, where `jax.jit` can be
    wrapped by intercepting `jaxlib._jax.PjitFunction`.
  - numpy is a jax dependency, and JAX 0.11.0 requires numpy >= 2.1.
  - ROCm-enabled JAX can be pip installed
    ([Install JAX for ROCm](https://rocm.docs.amd.com/projects/ai-ecosystem/en/latest/frameworks/jax/install.html)).

## Problem statement

- The ML API trace has no JAX backend, so JAX GPU time cannot be attributed to
  the compiled function or eager op that launched it.

## Requirements

- Attribute kernels of a `jax.jit` / `jax.pmap` / `jax.shard_map` call to
  that compiled function, and of an eager JAX op to the op that dispatched
  them, with no changes to the workload source.
- Reuse the existing ML API trace `marker -> counter -> call-tree` pipeline.

## Considered and not chosen

- **HLO operator and source.** Optimized HLO dumps can name the HLO operator
  and source line of each instruction. We do not join that onto the tree.
  - The kernels come from compilation and fusion: several operators in a
    compiled function can become one kernel, and one operator can become
    several kernels. Putting that mapping into the call tree would turn a function-to-kernel
    nesting into a many-to-many join.
- **Native `Execute` hook.** Hooking `Execute` would open a range around the
  compiled module that launches the kernels. We do not use it:
  - `Execute` does not know the Python function or caller `file:line`. It
    cannot link kernels to `jax.jit.<fn>` without another channel from the
    Python call.
  - When the Python wrapper already has a range open, kernels inherit that
    marker's `Correlation_ID`. A nested native range becomes current, so
    kernels inherit `Execute`'s id instead. The join then attaches to the
    module, not `jax.jit.<fn>`.

## Design

- JAX is added as a backend to the existing `--trace-ml-api` pipeline, through
  `inject_roctx`.
- `jax.jit` / `jax.pmap` / `jax.shard_map` open a range around each compiled
  call.
- Eager ops (no outer `jit`) open a range at primitive dispatch, named
  `jax.{primitive}` (for example `jax.dot_general`). There is no `jax.eager`.
  The primitive carries its own name at dispatch; we prefix `jax.`. We do not
  ship a list of primitive names.
- A nested or inner call does not open a second range; its kernels stay under
  the outer marker.
  - JAX compiles nested `jit` into the outer module, so the outer call is one
    `Execute`.
  - rocprofiler-compute emits one marker for that `Execute`. If an inner
    wrapper still runs while that range is open, it does not open another.
- `Execute` runs inside the range, so kernels join to that marker.

### At a glance

- The marker trace and the kernel trace are joined on `Correlation_ID` and
  `GUID` in the existing pipeline.


```
   workload run (instrument)                     analysis
   ─────────────────────────                     ────────
 jax.jit / jax.pmap / jax.shard_map / eager ──ROCTX──► marker ┐
                                                    │ join on
 XLA executable ────kernel launches►  kernel trace ┘ Correlation_ID + GUID
                                                    │
                                                    ▼
                                            call tree
```

### Where the range is marked

- The known JAX compute pathways this implementation plans to cover are
  compiled calls (`jit`, `pmap`, `shard_map`) and Primitive dispatch.
  `jax.numpy` is not a launch path of its own.
- `jax.jit` wraps how the compiled function is called, not `jax.jit` itself:
  - `jax.jit(fn)` returns a callable. It does not run `fn`.
  - On JAX 0.11.0 every such callable has the same type,
    `jaxlib._jax.PjitFunction`.
  - Calling it always goes through `PjitFunction.__call__`, so replacing
    that one method wraps every jitted function. The object JAX returns
    is still a `PjitFunction`.
  - The deprecated `pjit` alias returns the same type, so those calls
    show as `jax.jit.<fn>`.
  - Each outer call opens `jax.jit.<fn>`, runs the executable, and closes
    the range when the callable returns.
  - A direct call to an inner jitted function is its own outer call, so it
    gets its own range.
- `jax.pmap` and `jax.shard_map` wrap the factory, then the compiled call
  (same approach; on JAX 0.11.0 neither result is a `PjitFunction`):
  - `pmap` maps `fn` over devices. `shard_map` runs `fn` on a mesh.
  - Each outer call opens `jax.pmap.<fn>` or `jax.shard_map.<fn>`, runs the
    executable, and closes the range when the callable returns.
- Primitive wrap: `inject_roctx` wraps `jax.extend.core.Primitive`, not
  `jax.numpy`.
  - If a JAX range is already open, do not open another; run the original
    dispatch.
  - Otherwise read the primitive's name from that call, push `jax.{name}` with
    the caller `file:line`, run the original dispatch (the one-op `Execute`),
    then pop. `x @ x` therefore marks `jax.dot_general`; `jax.numpy.sum`
    marks `jax.reduce_sum`.
- JAX dispatch is asynchronous: the callable returns before the GPU finishes
  ([Asynchronous dispatch](https://docs.jax.dev/en/latest/async_dispatch.html)).
  - The range encloses the host-side launch enqueue, not GPU completion.
  - Those enqueues inherit the marker's correlation id.
  - `GUID` is the rocprofiler-sdk database/import identifier
    ([rocpd schema](https://github.com/ROCm/rocm-systems/blob/develop/projects/rocprofiler-sdk/source/lib/python/rocpd/README.md)).
    All marker and kernel rows from that instance share it.
  - The join is `Correlation_ID` + `GUID` so a colliding `Correlation_ID`
    from another instance does not match.

```
time →

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

- Torch nests markers (module, then each ATen op). Triton does not: one marker
  per kernel launch, and that kernel is a direct child.
- JAX also does not nest markers. After `jit` the ops are fused into one
  executable with no per-op Python call, so one marker names the function and
  several kernels sit directly under it. Eager JAX is one primitive per
  `Execute`, closer to Triton's one-launch marker.

### Analyze display

- The workload below is the reference for the tree.
- `train_step` calls `matmul_relu` from inside its own range, so the inner call
  is not a separate node. Its kernels are listed under `jax.jit.train_step`.

```python
# jax_workload.py
import jax.numpy as jnp

@jax.jit
def matmul_relu(a, b):
    return jnp.maximum(a @ b, 0.0)

@jax.jit
def train_step(w, x):
    y = matmul_relu(x, w)
    return jnp.sum(y * y)

w = jnp.ones((512, 512))
x = jnp.ones((512, 512))
train_step(w, x)                 # line 13
```

- `--list-operators jax` prints the call tree the ML API trace already prints
  for Torch and Triton:
  - One node per compiled function or eager primitive.
  - Caller `file:line` on that node.
  - Kernels underneath sorted by total GPU time.
  - A per-operator summary table follows the tree.
- A direct call to `matmul_relu` is its own root, `jax.jit.matmul_relu`.
- An eager op in the same run is its own root, for example `jax.dot_general`,
  with that op's kernels underneath.

```
================================================================================
JAX Operator Call Tree: <workload>
Sorted by total GPU kernel duration.
================================================================================

jax.jit.train_step jax_workload.py:13 (calls: N, dispatches: M, total: T)
├─ gemm_fusion_dot_general_0 (dispatches: M, total: T)
├─ Cijk_..._WGM6 (dispatches: M, total: T)
├─ input_reduce_fusion (dispatches: M, total: T)
└─ input_reduce_fusion_1 (dispatches: M, total: T)
```

- `--operator jax:<PATTERN>` filters that tree by function name, for example
  `--operator 'jax:*train_step*'`, and prints the matched nodes in the same
  form.

## Validation, security and debuggability

- Extend the existing `--trace-ml-api` integration tests (Torch / Triton) with
  a JAX backend. Same profile → analyze path; no separate harness.
- Those tests assert the following on a profiled workload:
  - An outer `jax.jit` / `jax.pmap` / `jax.shard_map` call emits one range
    `jax.jit.<fn>`, `jax.pmap.<fn>`, or `jax.shard_map.<fn>`. An outer eager
    op emits `jax.{primitive}`. Nested or inner calls do not emit a second
    range.
  - Kernels launched by that call appear under that marker, with the caller
    `file:line` and the call's dispatch count and GPU time.
- Unit tests cover marker naming and the "range already open" skip, without a
  GPU workload.
- **Security.** Same in-process `inject_roctx` injector as the other backends.
  Analysis reads the trace files the workload produced.
- **Debug.** The injector logs which entry points it wrapped during profile
  (`ml api trace` on the profile stdout). A wrap miss is an absent `jax.*`
  node in the `--list-operators jax` tree (analyze stdout).

Limitations:

- This wrap list is planned coverage, not a closed inventory of every GPU
  launch. Other JAX or XLA paths may still enqueue kernels without a
  `jax.*` marker; those cases need to be confirmed on real traces.

Design trade-offs:

- A `jit` / `pmap` / `shard_map` marker is the compiled function. Every
  kernel that call launches is a direct child of that node.
- One function name covers every input signature.
