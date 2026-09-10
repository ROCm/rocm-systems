# JAX trace (`--jax-trace`) — High-Level Design

## System Context

For ML frameworks rocprofiler-compute provides an ML API trace
pipeline (shared by `--torch-trace` and `--triton-trace`) that wraps each
framework call in a ROCTX range, so its kernels share one correlation id and
group under the operation that launched them.

JAX compiles each `jax.jit` / `jax.pmap` function into an XLA module: the
function is lowered to HLO (High Level Optimizer IR), optimized, and emitted as
a single module of fused kernels. The first call with a given input signature
compiles the module; later calls with the same signature reuse it. The module
is the level at which work is meaningful to the user — it corresponds to a
function they wrote — whereas the individual kernels are compiler artifacts
whose names and boundaries are chosen by fusion. The module is also the only
identity available at runtime, at the point where the compiled function is
invoked (where the markers are placed).

## Problem statement

The ML API trace has no JAX backend. XLA kernel dispatches carry only a
compiler-generated name (e.g. `gemm_fusion_dot_general_0`, `Cijk_...`) with no
link to the `jax.jit` / `jax.pmap` function, HLO operator, or source line that
produced them, so JAX GPU time cannot be attributed to user code the way
`--torch-trace` and `--triton-trace` already allow.

## Requirements

Functional:

- Attribute each collected kernel to the JAX compiled function that launched
  it, with no changes to the workload source.
- Resolve kernels further to the HLO operator and source location within the
  module where that information is available.
- Reuse the existing ML API trace marker -> counter -> call-tree pipeline
  rather than building a parallel one.
- Preserve per-invocation statistics (dispatch counts, timing) and the caller
  source location.

Non-functional:

- Degrade gracefully: with no HLO dump, still provide module-level attribution.

## Design

The module is the anchor: a kernel is first tied to the module that launched it,
then the module's HLO resolves the kernel to operators and source locations.

### At a glance

Three inputs are combined — two collected at runtime and one dumped by the
compiler — to attribute each kernel back to user code:

```
   workload run (instrument)                     analysis (offline)
   ─────────────────────────                     ──────────────────
 jax.jit / jax.pmap ──ROCTX range──►  marker trace ┐
                                                    │ join on
 XLA executable ────kernel launches►  kernel trace ┘ Correlation_Id
                                                    │
                                                    ▼
                                            per-function call tree
                                                    ▲
                                                    │ join on
 XLA_FLAGS ──────────dump──────────►  HLO dump ─────┘ (function, kernel)
                                                      → operator + source
```

The two runtime traces are joined on the rocprofiler `Correlation_Id` (the
marker and its kernels share one id); the HLO dump is a static, analysis-time
lookup joined on `(function, kernel_name)`.

Each level of the resulting tree comes from a different input:

```
 jax_workload.py:26          ← caller source (marker context)
 └─ jax.jit.train_step       ← function / module      (marker trace)
    └─ reduce_sum @ :13       ← HLO operator + source   (HLO dump)
       └─ input_reduce_fusion ← kernel + timing/counters (kernel trace)
```

### Approach

`--jax-trace` combines two mechanisms:

1. **Module attribution (ROCTX markers).** The workload's JAX transformation
   entry points are wrapped so that each call to a compiled function runs inside
   a ROCTX range. The range covers the executable's run on the launching thread,
   so the kernels it launches share a rocprofiler correlation id with the marker.
   This reuses the ML API trace pipeline shared with `--torch-trace` and
   `--triton-trace`.

2. **Operator and source attribution (HLO dump).** `XLA_FLAGS` is set for the
   workload to dump each module's optimized HLO as text. At analysis time the
   HLO is parsed into a `(function, kernel) -> (operator, source, shape)` map and
   joined onto the kernels in the call tree. The module supplied by the marker
   is the key that disambiguates kernel names, which are not unique across
   modules.

Marker injection requires no changes to the workload source. It is installed by
the `inject_roctx` launcher, the same component used by the Torch and Triton
backends. Command buffers are disabled through `XLA_FLAGS` so that kernels are
launched individually rather than batched into a single graph launch, which
keeps each launch correlated to its marker.

### Where the range is marked

At install the launcher replaces `jax.jit` / `jax.pmap` with a wrapper. The
wrapper leaves compilation untouched and wraps the *compiled callable*: it opens
a ROCTX range before the executable runs and closes it after the executable
returns, on the same thread (abbreviated):

```python
# jax.jit / jax.pmap are replaced at install:
#   jax.jit  = _wrap_transform(jax.jit,  "jit")
#   jax.pmap = _wrap_transform(jax.pmap, "pmap")

def _wrap_transform(transform, transform_name):
    def transform_wrapper(fn=None, *args, **kwargs):
        compiled = transform(fn, *args, **kwargs)          # the real jax.jit/pmap
        marker = f"jax.{transform_name}.{fn.__name__}"     # e.g. jax.jit.train_step

        def invocation_wrapper(*call_args, **call_kwargs):
            return _run_with_marker(                        # wrap EVERY call
                marker, lambda: compiled(*call_args, **call_kwargs)
            )
        return invocation_wrapper
    return transform_wrapper

def _run_with_marker(marker, thunk):
    if _in_call():                    # nested jit: reuse the outer range
        return thunk()
    _thread_local.in_call = True
    try:
        _push_scope(marker, f"#{index}@{location}", backend="jax")   # range OPEN
        return thunk()                # compiled(...) -> Execute -> kernel launches
    finally:
        _pop_scope()                                                 # range CLOSE
        _thread_local.in_call = False
```

JAX dispatches asynchronously, so `compiled(...)` returns before the GPU
finishes; the range encloses the host-side *launch enqueue*, not GPU completion.
The launches are enqueued on the calling thread inside the open range, so
rocprofv3 tags them with the marker's correlation id (launches on other threads
are the exception — see Open questions and trade-offs):

```
launching CPU thread (time →)

  _push_scope("jax.jit.train_step", ...)          ← ROCTX range begins   [Correlation_Id = C]
  │
  └─ compiled(w, x)                               (cached XLA executable)
       └─ PjRtLoadedExecutable::Execute
            ├─ launch  gemm_fusion_dot_general_0   ┐
            ├─ launch  input_reduce_fusion         │ launched within the range
            └─ launch  input_reduce_fusion_1       ┘ → inherit Correlation_Id = C
  │
  _pop_scope()                                    ← ROCTX range ends

rocprofv3 marker trace :  "jax.jit.train_step"        Correlation_Id = C
rocprofv3 kernel trace :  gemm_fusion_dot_general_0   Correlation_Id = C
                          input_reduce_fusion         Correlation_Id = C
                          input_reduce_fusion_1       Correlation_Id = C
        join on Correlation_Id  ⇒  kernels attributed to jax.jit.train_step
```

This linkage is what the generated `ml_api_trace/consolidated.csv` shows: every
kernel row launched by a `train_step` call carries the `jax.jit.train_step`
operator name, joined from the marker via the shared correlation id.

### Difference from the Torch/Triton backends

Torch and Triton mark at operator/kernel granularity (Triton wraps each kernel
launch, Torch each ATen op), so the marker itself names the operation. JAX
cannot: after `jit` the ops are fused into one executable with no per-op Python
call site, so the marker can only name the *function*. Kernel-to-operator
identity is therefore recovered offline from the HLO dump — the join the
Torch and Triton backends do not need.

### Injection point and alternatives

Module identity and the concrete kernel launches coincide at the executable
dispatch, `PjRtLoadedExecutable::Execute`: the Python callable knows the module
but not its kernels, while the HIP launch records know the kernels but not the
module. The backend wraps the Python `jax.jit` / `jax.pmap` callable rather than
`Execute` because the callable synchronously encloses `Execute` on the same
thread — a wider scope that covers the same launches without having to list the
kernels, and keeps the backend within the Python-only `inject_roctx` framework.

A native hook at `Execute` would be more precise (one dispatch is exactly one
module, whereas a Python callable maps to several modules — one per input
signature) but would not fix threading: correlation ids are thread-local either
way. Thread-robust attribution would require annotating each kernel launch
inside XLA.

### Operator and source resolution (HLO-dump join)

The markers attribute each kernel to a module. Resolving a kernel further to
the HLO operator it implements and the source line it came from is a separate
join against the dumped HLO, performed at analysis time.

XLA dumps one optimized-HLO text file per module
(`module_*.<name>.*after_optimizations.txt`). Each file carries:

- a `HloModule jit_<function>` header naming the function;
- `FileNames`, `FileLocations`, and `StackFrames` tables that resolve a
  `stack_frame_id` to a `file:line`;
- one line per instruction with `metadata={op_name="..." stack_frame_id=N}`.

The parser builds a `(function, kernel_name) -> (operator, source, shape)` map:

- The function is taken from the module header (the `jit_` prefix is dropped)
  and matched against the marker operator name (`jax.jit.<function>`).
- Instruction names are normalized to kernel symbols by replacing `.` with
  `_` (matching XLA's codegen), so the instruction `input_reduce_fusion.1`
  matches the kernel `input_reduce_fusion_1` and `gemm_fusion_dot_general.0`
  matches `gemm_fusion_dot_general_0`.
- When the emitted kernel symbol renumbers the instruction's numeric suffix,
  the join falls back to matching on the name stem, accepting it only when the
  operator and source for that stem are unambiguous.

The function is required as a key: the same kernel name can appear in several
modules with different operators, and the marker's module identity selects the
correct one.

Kernels with no HLO instruction are left unresolved and remain directly under
the module node — in this workload these are the library GEMM
(`Cijk_...`), the runtime copy/fill helpers (`__amd_rocclr_fillBufferAligned`,
`__amd_rocclr_copyBuffer`), and the autotuning kernels
(`RedzoneAllocatorKernelImpl`, `RepeatBufferKernelImpl`, `xla_fp_comparison`).

Each HLO `op_name` also carries its scope, so the resolved operator is placed
in the tree as a node (or nodes) between the module and the kernel. The leading
`jit(<function>)` scope is dropped as redundant with the module node, and any
remaining nested-transform scope (for example `jit(matmul_relu)` inside
`train_step`) becomes its own level. The operator node carries the operator's
result shape (as its args) and source location.

### Worked example

The excerpts below are from a run of this workload:

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

XLA dumps `module_0013.jit_train_step.gfx942_gpu_after_optimizations.txt`. Its
header names the function, the indexed tables resolve a `stack_frame_id` to a
`file:line`, and each `ENTRY` instruction carries `op_name` and
`stack_frame_id` (trimmed to the rows the join uses):

```
HloModule jit_train_step, is_scheduled=true, ...

FileNames
3 "jax_workload.py"

FileLocations
10 {file_name_id=3 function_name_id=9 line=13 ...}   # train_step, jnp.sum
15 {file_name_id=3 function_name_id=10 line=7 ...}   # matmul_relu, a @ b

StackFrames
10 {file_location_id=10 parent_frame_id=10}
26 {file_location_id=15 parent_frame_id=25}

ENTRY %main.3 (w.1: f32[512,512], x.1: f32[512,512]) -> f32[] {
  %gemm_fusion_dot_general.0 = f32[512,512]{1,0} fusion(%x.1, %w.1), kind=kCustom, ...
      metadata={op_name="jit(train_step)/jit(matmul_relu)/dot_general" stack_frame_id=26}
  %input_reduce_fusion = f32[512]{0} fusion(%gemm_fusion_dot_general.0), kind=kInput, ...
      metadata={op_name="jit(train_step)/reduce_sum" stack_frame_id=10}
  ROOT %input_reduce_fusion.1 = f32[] fusion(%input_reduce_fusion), kind=kInput, ...
      metadata={op_name="jit(train_step)/reduce_sum" stack_frame_id=10}
}
```

The join normalizes each instruction name, follows `stack_frame_id` through the
tables to a source line, and drops the leading `jit(train_step)/` scope. The
resolved rows are written to `ml_api_trace/kernel_source_map.csv`
(`Function` is the tree path; the result shape is kept on the tree node but not
in the CSV):

```
Function,Kernel_Name,Operator,Source
jax.jit.matmul_relu/max,loop_maximum_fusion,jit(matmul_relu)/max,jax_workload.py:7
jax.jit.train_step/jit(matmul_relu)/dot_general,gemm_fusion_dot_general_0,jit(train_step)/jit(matmul_relu)/dot_general,jax_workload.py:7
jax.jit.train_step/reduce_sum,input_reduce_fusion,jit(train_step)/reduce_sum,jax_workload.py:13
jax.jit.train_step/reduce_sum,input_reduce_fusion_1,jit(train_step)/reduce_sum,jax_workload.py:13
```

`analyze` places those resolutions into the call tree as operator nodes between
the module and its kernels:

```
jax_workload.py:26
└─ jax.jit.train_step (calls: 5, dispatches: 195, total: 1.28 ms)
   ├─ jit(matmul_relu)
   │  └─ dot_general args=(f32[512,512]) @ jax_workload.py:7
   │     └─ gemm_fusion_dot_general_0 (dispatches: 30, total: 0.54 ms)
   ├─ reduce_sum args=(f32[512]) @ jax_workload.py:13
   │  ├─ input_reduce_fusion (dispatches: 65, total: 0.21 ms)
   │  └─ input_reduce_fusion_1 (dispatches: 65, total: 0.18 ms)
   └─ Cijk_..._WGM6 (dispatches: 35, total: 0.35 ms)   # unresolved library GEMM
```

The same `a @ b` is lowered differently per module: in `train_step` it is a
Triton `gemm_fusion` instruction (resolved to `gemm_fusion_dot_general_0`),
while in the standalone `matmul_relu` module it is a `__cublas$gemm`
custom-call that runs as the `Cijk_...` library kernel and stays unresolved.

### Components

| Component | Responsibility |
| --- | --- |
| Argument parsing | Adds `--jax-trace`; adds the analyze filters `--jax-operator` and `--list-jax-operators` |
| Profiler backend | Maps `--jax-trace` to the `jax` framework and sets `XLA_FLAGS` for the HLO dump |
| JAX injection backend | Wraps `jax.jit` / `jax.pmap` with ROCTX ranges labelled by function name |
| ML API trace registry | Registers `jax` as a known ML API trace backend |
| HLO parser | Parses the HLO dump into the `(function, kernel) -> (operator, source, shape)` map |
| Analysis backend | Registers the `jax` analyze backend and joins the HLO map onto the call tree |

`XLA_FLAGS` set for the workload:

```
--xla_dump_to=<output>/hlo_dump --xla_dump_hlo_as_text --xla_dump_hlo_as_long_text --xla_gpu_enable_command_buffer=
```

Any existing `XLA_FLAGS` is replaced, with a warning.

## Validation, security and debuggability

- **Unit tests** cover the HLO parser: kernel-name normalization, the stem
  fallback, source-table resolution, nested operator paths, and CSV output.
- **Cross-check.** Each module's static `ENTRY` kernel set should appear in its
  function's observed kernel set. On the reference workload this holds exactly
  in steady state; the warm-up call is a superset whose extras are all
  autotuning/runtime kernels (unselected GEMM candidates, redzone and
  comparison harness) that have no HLO instruction.
- **Diagnostics.** Warnings are emitted when the HLO dump directory is missing
  or empty, when a dump file cannot be read, and when an existing `XLA_FLAGS`
  is replaced; debug logs report the parsed module and function counts.
- **Graceful degradation.** With no HLO dump the join is skipped and attribution
  falls back to module level.
- **Security.** Analysis reads only the local HLO dump files the workload
  produced; it needs no elevated privileges and no network access. Command
  buffers are disabled through `XLA_FLAGS` so kernels launch individually and
  stay correlated to their marker.

## Limitations and future work

JAX reaches the GPU through several paths. All of them ultimately dispatch
through XLA's `Execute` (except work launched outside XLA), but only the
compiled-function path passes through the wrapped `jax.jit` / `jax.pmap`
callable:

```
JAX program
│
├─ jax.jit / jax.pmap function ─► XLA module (fused) ─► Execute ─► kernels   [covered]
│     grad / vmap / shard_map composed under jit ─────────────────────────►  [covered]
│
├─ jax.pjit function ──────────► XLA module ─────────► Execute ─► kernels     [addable: wrap pjit]
│
├─ eager op (no jit) ─► per-op jit ─► XLA module (1 op) ─► Execute ─► kernel  [not covered]
│     grad / vmap used standalone run this way
│
└─ GPU work outside XLA (host callback, other library) ────────► kernel      [not covered]

any of the above launched on another thread ─► correlation id lost           [not covered]
```

Coverage gaps (uncovered work and how it could be added):

- **Eager and standalone transforms.** Only `jax.jit` / `jax.pmap` functions are
  instrumented. Transforms composed under `jit` (`jax.grad`, `jax.vmap`,
  `shard_map`) are already covered by the outer wrapper; eager (non-`jit`)
  execution is not. *To add:* wrap `jax.pjit` (a one-line addition, it behaves
  like `jax.jit`); for eager, hook `core.Primitive.bind` with a guard that fires
  only during eager execution (not tracing), giving per-primitive markers.
- **Non-XLA GPU work.** GPU work launched outside any XLA executable — e.g. from
  a host callback (`jax.pure_callback` / `io_callback`) or another library in the
  process — has no module, marker, or HLO to attribute it to. (Custom-call / FFI
  kernels invoked from *within* a compiled module are already captured at module
  level, unresolved, like the library GEMM above.) *To add:* wrap the callback
  entry points so each emits its own marker.
- **Cross-thread launches.** Correlation ids are thread-local, so kernels
  launched off the calling thread (background compilation, autotuning, `pmap`
  device workers) are unattributed. *To add:* annotate each kernel launch inside
  XLA rather than wrapping at the Python level.

Design trade-offs:

- Marker attribution is per compiled function, not per HLO operator: all kernels
  launched by one invocation share the function marker. The per-operator layer
  is recovered by the HLO-dump join; kernels with no HLO instruction (library
  GEMM, runtime copy/fill helpers, autotuning kernels) stay directly under the
  module node.
- The shape on an operator node is the operator's result shape, not its operand
  shapes, which would require resolving operand instructions.
- One function name may map to several modules (a distinct module per input
  signature), so the join resolves on `(function, kernel_name)` and relies on
  kernel names being distinct across a function's modules.
