# Triton recipes for `compare_correctness`

This directory holds Triton-authored kernels that the
`compare_correctness` harness runs through the native / legacy / salmon
three-mode comparison, alongside the hand-written HIP recipes in
`../`.

Each `<name>.py` here becomes **one recipe** in the harness:

- `<name>.py` is AOT-compiled to two code objects,
  `kernels/build/<name>.gfx942.co` and `kernels/build/<name>.gfx1250.co`.
- A JSON "sidecar" `kernels/build/<name>.sidecar.json` records the
  signature, constexprs, shape sweep, launch grid, inputs/outputs, the
  comparator tolerance, and — most importantly — the per-arch kernarg
  layout extracted from the `.co`'s AMDGPU metadata.
- At run time the harness scans `kernels/build/*.sidecar.json`, turns
  each into a `Recipe`, and the shared `tritonDispatch` packs kernargs
  at the exact offsets the code object expects.  **No C++ edits are
  needed per Triton kernel.**

## Why native-as-gold?

HIP recipes in this tool use a hand-written **CPU reference** as
ground truth, which makes the native GPU run part of the measurement
(a mismatch on native is a finding of its own).  That design doesn't
scale to Triton kernels: writing a correct CPU reference for flash
attention, mixed-precision GEMM, or fused softmax is its own small
research project, and we'd end up shipping bugs in the reference.

Instead, Triton recipes declare `goldSource = NativeExecution` and use
the output of the **native gfx942 run** as the gold.  The transpiled
legacy / salmon runs (loading the gfx1250 code object through the ROCR
hotswap hook) are judged against that.  This trades the CPU-grounded
self-check on the native column for the ability to throw arbitrary
Triton kernels at the harness; the trust boundary moves to "hipcc +
Triton on gfx942 compute the right answer".  If the native gfx942 run
fails to materialise, the harness marks the whole `(recipe, shape)`
row as `gold-missing` and skips the transpiled modes for that row —
nothing meaningful can be said about them without a gold.

## Prerequisites

Compilation is done with the in-tree Triton at `$HOME/rocm-systems/triton`.
The Makefile points to:

```
TRITON_VENV       = $(HOME)/rocm-systems/triton/.venv
TRITON_PYTHONPATH = $(HOME)/rocm-systems/triton/python:$(HOME)/rocm-systems/triton/third_party/amd/python
```

Override on the `make` command line if your Triton tree lives
elsewhere.  The venv needs Triton itself and `PyYAML` (shipped with
the repo's Triton venv; used by `aot_compile.py` to parse the
`llvm-readelf --notes` output).  An `llvm-readelf` binary is located
via `$LLVM_READELF`, `/opt/rocm*/lib/llvm/bin/llvm-readelf`, or
`$PATH`, in that order.

## Adding a recipe

1. Drop `<name>.py` here.  The file stem is the recipe name and must
   match the `name:` field of its recipe entry.
2. In `<name>.py`:
   ```python
   import triton
   import triton.language as tl

   @triton.jit
   def my_kernel(...):
       ...

   RECIPES = [{
       "name": "<name>",                    # must match file stem
       "kernel_fn": my_kernel,              # the @triton.jit object
       "kernel_symbol": "my_kernel",        # hipModuleGetFunction symbol
       "signature": { ... },                # Triton type strings
       "constexprs": { ... },
       "num_warps": 4,
       "shape_dim": "N",                    # single scalar shape param
       "default_shapes": [1024, 4096, ...], # values the harness sweeps
       "grid": {"x": "...", "y": "1", "z": "1"},
       "inputs":  [{"name": ..., "dtype": ..., "elems": ...}, ...],
       "outputs": [{"name": ..., "dtype": ..., "elems": ...}],
       "comparator": {"kind": "abs", "tol": 1e-2},
   }]
   ```
3. Add `<name>` to `$(TRITON_KERNELS)` in the Makefile.
4. `make ROCR_BUILD=$HOME/rocm-systems/projects/rocr-runtime/build`.

See `vecadd_f16.py` for a complete working example.

## Recipe schema

| field              | type                          | notes                                                                   |
|--------------------|-------------------------------|-------------------------------------------------------------------------|
| `name`             | `str`                         | unique; must equal the file stem                                        |
| `kernel_fn`        | Triton `@jit` object          | the actual kernel to compile                                            |
| `kernel_symbol`    | `str`                         | symbol looked up via `hipModuleGetFunction`                             |
| `signature`        | `OrderedDict[str, str]`       | Triton signature (`"*fp16"`, `"i32"`, …) in kernel argument order       |
| `constexprs`       | `dict[str, int]`              | Triton `tl.constexpr` values (e.g. `{"BLOCK_SIZE": 1024}`)              |
| `num_warps`        | `int`                         | threads/block = `num_warps * warp_size`                                 |
| `shape_dim`        | `str`                         | scalar sig arg the harness sweeps (Phase 1)                             |
| `default_shapes`   | `list[int]`                   | values of `shape_dim` to sweep                                          |
| `grid`             | `{"x", "y", "z": str}`        | expressions in `shape_dim` + `constexprs` + `ceil_div(a,b)`             |
| `inputs`           | `list[{name, dtype, elems}]`  | buffers filled with deterministic RNG, bound to pointer sig args        |
| `outputs`          | `list[{name, dtype, elems}]`  | buffers read back from the device and compared                          |
| `comparator`       | `{"kind": "abs"\|"rel", "tol": float}` | elementwise tolerance                                          |

`elems` and `grid` values are **expressions** evaluated at run time
against `shape_dim + constexprs`.  Supported: integer literals,
identifiers (`N`, `BLOCK_SIZE`, …), infix `+ - * /`, parens, and a
single built-in function `ceil_div(a, b)`.  Anything else triggers a
loud error in the harness — the evaluator intentionally refuses to
silently guess.

## Sidecar schema (produced by `aot_compile.py`)

```
{
  "name":          "<recipe name>",
  "kernel_symbol": "<ELF symbol>",
  "num_warps":     <int>,
  "signature":     [{"name":..., "type":...}, ...],    # order matters
  "constexprs":    { ... },
  "shape":         {"dim": "<name>", "values": [<int>, ...]},
  "grid":          {"x":"<expr>", "y":"<expr>", "z":"<expr>"},
  "inputs":        [{"name":..., "dtype":..., "elems":"<expr>"}, ...],
  "outputs":       [{"name":..., "dtype":..., "elems":"<expr>"}, ...],
  "comparator":    {"kind": "abs"|"rel", "tol": <float>},
  "metadata": {
    "gfx942":  { "kernarg_segment_size":       <int>,
                 "group_segment_fixed_size":   <int>,   # static LDS (always 0 for Triton)
                 "private_segment_fixed_size": <int>,   # scratch; harness refuses launch if != 0
                 "max_flat_workgroup_size":    <int>,
                 "shared_mem_bytes":           <int>,   # dynamic LDS (compiled.metadata.shared)
                 "args": [{"offset":..., "size":..., "value_kind":...}, ...] },
    "gfx1250": { ... }
  }
}
```

`metadata.<arch>.shared_mem_bytes` is sourced from
`compiled.metadata.shared` at AOT time — the per-block *dynamic* LDS
(shared memory) Triton's AMD backend reserves for reductions, softmax
scratch, and similar cross-wave accumulators.  It is **not** the same
thing as `group_segment_fixed_size` (static LDS baked into the kernel
descriptor, always 0 for Triton), and it is **not** discoverable from
the HSACO alone; the dispatch must pass it verbatim as
`hipModuleLaunchKernel`'s `dynamicSharedMemBytes` argument or any
reduction-bearing kernel silently returns zero output (the reduction
path stores into a non-existent LDS allocation).  Elementwise
kernels like `vecadd_f16` naturally have `shared_mem_bytes = 0` and
are unaffected; any future recipe with a cross-wave reduction
(layer-norm, softmax, block-sum, …) will have `shared_mem_bytes > 0`
and implicitly regression-test the plumbing — if dispatch ever
reverts to passing 0 its native lane goes from `gold` to
`WRONG`/`CRASH` on the first run.

The `metadata.<arch>.args` list is the kernarg layout for that arch,
copied verbatim from the code object's `.note.amdgpu_metadata`.  The
harness iterates the `signature` and the `args` list in lockstep —
pointer sig args must line up with `global_buffer` slots and scalar
sig args with `by_value` slots, or the dispatch dies with a clear
producer/consumer-drift error.  Any trailing metadata entries beyond
the last sig arg (Triton's implicit `global_scratch_base` etc.) are
left zero-initialised in the kernarg buffer; Triton expects that when
no scratch is required.

## Limitations (Phase 1)

- **One scalar shape dim per recipe.**  The harness sweeps a single
  named parameter (for vecadd, `N`).  Multi-dim shapes like matmul's
  `(M, N, K)` need a sidecar-schema extension (probably
  `"shape": {"dims": [...], "combinations": [...]}`) and a per-recipe
  shape-label column in the report.
- **No CLI filter for Triton shapes.**  `--shape=N=1024` isn't wired
  up yet.  `--recipe=<name>` narrows to a single recipe as usual, and
  the recipe's full shape sweep runs.
- **One recipe per `.py` file.**  The AOT script enforces this
  because the Makefile's `%` pattern is the file stem and producing N
  recipes from one file would conflict with the grouped target rule.
  Split files if you want multiple recipes.
