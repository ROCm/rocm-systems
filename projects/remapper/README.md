# remapper

Multi-SIMD thread trace (ATT) of one kernel dispatch on gfx1250.

ATT traces one SIMD of one WGP per run. To see every SIMD in the same execution, the remapper rewrites the
kernel's code object so that each workgroup picks its tile from a placement table keyed by where it landed,
then replays the dispatch with
[rocprofiler-sdk kernel replay](../rocprofiler-sdk/source/docs/how-to/using-kernel-replay.rst), so the traced WGP
runs the same work in every pass while each pass traces a different SIMD. The per-pass traces are then merged
into one view.

- `crates/remapper-rewrite`: code-object analysis and rewrite (remap prologue, workgroup-id operand
  patches), and the placement tables and their checks.
- `crates/remapper-runtime`: the pass schedule for one dispatch (record, table, checks, ATT sweeps, final
  application pass) and its device buffers, through the process's HSA runtime.
- `crates/remapper-ffi`: `libremapper.so`, the C ABI in [`include/remapper.h`](include/remapper.h).
- `crates/hsa-sys`: the HSA functions the runtime uses.
- `scripts/att_merge`: `layout.py` arranges rocprofv3's per-pass output; `run.py` merges and audits it.

`rocprofv3 --att-remap` drives the library. Each remapped dispatch writes `dispatch-<id>/`, which holds the
rewritten code object, `manifest.json`, `report.json` (stage results, or the reason a stage failed) and
`passes.json` (what each pass ran and traced).

## Building

```bash
cmake -S projects/remapper -B build/remapper -DROCM_PATH=/opt/rocm
cmake --build build/remapper
ctest --test-dir build/remapper
```

It requires Rust 1.85 or later, the ROCm HSA runtime, and ROCm LLVM's `llvm-mc` and `llvm-objdump` at run
time (`$REMAPPER_LLVM_BIN`, then `$ROCM_PATH/llvm/bin`). The merge scripts need Python 3 with NumPy.
