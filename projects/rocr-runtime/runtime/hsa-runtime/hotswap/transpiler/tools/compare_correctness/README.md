# `compare_correctness`

Head-to-head **numerical** comparison of the two hotswap transpilation
paths.  Where its sibling `tools/compare_transpilers` only asks "does the
code object load?", this tool asks the sharper question: "does the
translated kernel compute the right answer?".

## Why this tool exists

`compare_transpilers` answers "does the code object load under each
engine?".  That is necessary but weak: a code object can load cleanly
and still produce wrong numerical results, because the loader only
validates structural invariants (ELF format, ISA tags, kernel
descriptor fields) and not kernel semantics.  Either translation path
can in principle produce a kernel that loads, dispatches, writes
output without diagnostic, and still computes the wrong thing.

This tool closes that gap by running authored kernels end-to-end under
each engine and comparing the produced output buffers against a CPU
reference.

## How it works

For each `(recipe, shape N)` the parent computes a **CPU reference**
and spawns three children, one per engine:

| child mode | source ISA of .co | env                                 | path under test                              |
|------------|-------------------|-------------------------------------|----------------------------------------------|
| `native`   | gfx942            | *(none)*                            | direct HIP load, no transpiler               |
| `legacy`   | gfx1250           | `LD_PRELOAD=libsalmon_intercept.so` | ROCR hotswap hook → byte-level transpiler    |
| `salmon`   | gfx1250           | `LD_PRELOAD=libsalmon_intercept.so`, `HSA_HOTSWAP_IR_RAISER=1` | ROCR hotswap hook → Salmon IR raiser         |

Each child:

1. Reads the deterministic input buffer written by the parent.
2. `hipModuleLoadData` on the mode's `.co`, `hipModuleGetFunction`,
   `hipModuleLaunchKernel`.
3. Copies the device output back to host and writes it to the output
   tempfile.

The parent then compares each child's output to the CPU reference
elementwise and classifies each run as `match` / `WRONG k/N` / crash /
spawn-fail.

The three-process split is mandatory because ROCR reads
`HSA_HOTSWAP_IR_RAISER` into a `static const char*` on first use and
never re-reads it.  Per-`(recipe, shape, mode)` isolation also prevents
any ROCR-internal state leaking between runs.

## Prerequisites

- gfx942 hardware (MI300-class) — the target ISA is hard-coded.
- ROCm ≥ 7.2 for gfx1250 offload (override `ROCM=` if not at
  `/opt/rocm-7.2.1`).
- A **Salmon-enabled ROCR build tree**; pass its path as `ROCR_BUILD=`.
  A system ROCR will not work (see the transpiler `README.md` for how
  to produce one).

## Building

```sh
make ROCR_BUILD=$HOME/rocm-systems/projects/rocr-runtime/build
```

This produces:

- `compare_correctness` — the harness binary, `rpath`'d at the
  Salmon-enabled ROCR runtime in `$ROCR_BUILD/rocr/lib`.
- `libsalmon_intercept.so` — the same LD_PRELOAD shim that patches
  gfx1250 ELF `e_flags` so HIP accepts the binary.  `salmon_intercept`
  itself is built from `tests/salmon_intercept.cpp`.
- `kernels/build/<name>.<isa>.co` — per-kernel, per-ISA unbundled code
  objects via `hipcc --genco` + `clang-offload-bundler --unbundle`.

The harness binary links against `libhsa-runtime64.so` explicitly so
that `rocr_salmon_patch_elf` is visible to the shim's
`dlsym(RTLD_DEFAULT, ...)`; HIP otherwise dlopens the runtime with
local scope and the shim cannot find the symbol.

## Running

```sh
# Full default sweep: every recipe, cross-product of its default Ns and blocks
LD_PRELOAD=./libsalmon_intercept.so ./compare_correctness

# One recipe
LD_PRELOAD=./libsalmon_intercept.so ./compare_correctness --recipe=vecadd

# Restrict N (cross-produced with the recipe's default blocks)
LD_PRELOAD=./libsalmon_intercept.so ./compare_correctness --shape=256 --shape=1024

# Restrict block size (cross-produced with the recipe's default Ns)
LD_PRELOAD=./libsalmon_intercept.so ./compare_correctness --block=64 --block=128

# Pin a single (N, block)
LD_PRELOAD=./libsalmon_intercept.so ./compare_correctness --shape=1024 --block=128
```

`--shape` and `--block` are both repeatable.  If both are given, the
harness runs their cross-product.  If only one is given, the other
dimension uses the recipe's default list.

The parent sets `HSA_HOTSWAP_ISA_OVERRIDE=gfx942` and
`HSA_HOTSWAP_RULES=/dev/null` if not already set; these are inherited by
every child.  `HSA_HOTSWAP_IR_RAISER` is toggled by the child itself
(set to `1` for salmon, unset for legacy) before `hipInit`.

## Reading the output

The report has three sections:

1. **Grid** — one atomic row per `(recipe, N, block)` with a short
   status cell for each engine (`match` / `WRONG k/N` / `SIG…` /
   `EXIT=…` / `no-output`).  No detail lines are interleaved with the
   grid, so it scans vertically.
2. **Failures** — every non-`match` run reappears here with full
   detail (first mismatching index, reference vs. actual, max `|err|`,
   or the child's stderr tail for crashes), grouped by mode and then
   by recipe.  Each line is self-identified (`recipe N=… block=…`) so
   you always know which run it describes.
3. **Summary** — a 3×3 matrix of `{match, mismatch, crash} × {native,
   legacy, salmon}`.  This is the at-a-glance "which engine is how
   good" view.

`native` is the primary self-check of the harness: if it matches the
CPU reference, the kernel and the dispatch are self-consistent, and
any difference on the translated paths is attributable to the
translation itself.  If `native` does not match, every other row on
that shape should be treated as suspect.

## Adding a new kernel

1. Drop `<kernelname>.hip` into `kernels/`.  The kernel's top-level
   function must be `extern "C"` so the symbol name matches.
2. Add the recipe to `compare_correctness.cpp`: register an entry that
   provides deterministic inputs, a CPU reference, a `hipModule*`
   dispatch that packs the kernarg layout and launches, and an
   element-wise comparator.  If some `(N, block)` combinations would
   break your CPU reference (for example: block-reduction kernels with
   `block < 64`, or a pairwise-swap kernel with odd `N`), set the
   optional `validate` hook — the harness will skip those combos with a
   visible `[skip]` log line instead of producing a wrong comparison.
3. Add the kernel name to `$(KERNELS)` in the Makefile and `make`.

Kernels should be chosen so their **output shape is wave-size-agnostic**
(for example: one float per block, or an elementwise 1:1 output).
Internal wave-size-sensitive patterns (warp shuffles, ballot, ...) are
exactly what you want to probe — but if the output *count* depends on
`warpSize`, direct comparison across ISAs becomes apples-to-oranges and
you're comparing kernels that compute different quantities.

## Current probes

| recipe            | what it exercises                                             |
|-------------------|---------------------------------------------------------------|
| `vecadd`          | Baseline: pure VALU, no cross-lane ops.  Any failure here is structural, not intrinsic-related. |
| `block_sum_shfl`  | Two-phase block-sum reduction using `__shfl_xor` within each warp and cross-warp via shared memory.  One float per block. |
| `lane_swap`       | 1:1 output:  `out[tid] = in[tid ^ 1]`.  Crisp probe for whether cross-lane reads arrive from the expected partner. |

## Kernel provenance

The `.hip` files under `kernels/` are authored in-tree, not imported
from Triton/AITER/rocBLAS.  Each is compiled twice by hipcc
(`--offload-arch=gfx942` and `--offload-arch=gfx1250`) so both code
objects come from the same source.  That means per-arch hipcc codegen
differences are inside the measurement, not around it — fine for
probing the transpilation round-trip, not a bit-exact same-machine-
code-on-two-ISAs comparison.  The recipe interface is independent of
how the `.co` was produced, so swapping in Triton- or otherwise
externally-built code objects is a drop-in change.

## Portability

Hard-coded ISAs: target `gfx942`, source `gfx1250`.  Retargeting
requires editing `runChild` and `main` in `compare_correctness.cpp`
and `ISAS` + bundler triples in the `Makefile`.

Shapes and block sizes are swept by default (`defaultNs` and
`defaultBlocks` on each recipe) and can be restricted at the command
line via `--shape=` / `--block=` — see [Running](#running).

## Known limitations

- All buffers are built at process start in host memory and go through
  a file round-trip between parent and child.  Fine for per-shape
  correctness checks; not intended as a throughput benchmark.
- A single HIP device (device 0) is used by every child.  Multi-device
  support is a straightforward extension but not implemented.
- Kernel arguments are hand-packed inside each recipe's dispatch lambda.
  For programs with many argument-shape variants it may be worth
  pulling the layout from the `.co`'s MSGPACK metadata (as
  `cross_arch_gpu_test.cpp` does) — this tool keeps it manual because
  the kernels are authored here and their layouts are known.

## Exit code

Always `0` on a completed sweep — a mismatch against the CPU reference
is a *finding*, not a harness error.  A non-zero exit means the harness
itself failed (missing kernels, spawn failure, filter matched nothing).
The report body is the thing to read.
