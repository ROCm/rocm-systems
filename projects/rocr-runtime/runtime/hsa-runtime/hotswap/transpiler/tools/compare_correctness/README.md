# `compare_correctness`

Head-to-head **numerical** comparison of the two hotswap transpilation
paths.  Where its sibling `tools/smoke_test_compare_transpilers` only asks "does the
code object load?", this tool asks the sharper question: "does the
translated kernel compute the right answer?".

## Why this tool exists

`smoke_test_compare_transpilers` answers "does the code object load under each
engine?".  That is necessary but weak: a code object can load cleanly
and still produce wrong numerical results, because the loader only
validates structural invariants (ELF format, ISA tags, kernel
descriptor fields) and not kernel semantics.  Either translation path
can in principle produce a kernel that loads, dispatches, writes
output without diagnostic, and still computes the wrong thing.

This tool closes that gap by running authored kernels end-to-end under
each engine and comparing the produced output buffers against a gold
reference (see [How it works](#how-it-works)).

## How it works

Each recipe declares one of two **gold sources**:

- **CPU reference** — the parent runs a hand-written CPU implementation
  and treats that buffer as ground truth.  Every mode, including native,
  is judged against it.  This is the default and what every
  hand-written `.hip` recipe uses.
- **Native execution** — the parent runs the `native` child first and
  uses *its* output as the gold.  Subsequent `legacy` and `salmon` runs
  are judged against native.  Native's own status cell prints `gold`
  instead of `match`, because by definition it matches itself.  This
  gold source is used by **Triton recipes**, where authoring a CPU
  implementation per kernel (flash attention, mixed-precision GEMM,
  softmax+rope, …) would kill most of the harness's leverage; the trust
  boundary moves to "hipcc + Triton on gfx942 compute the right answer".

For each `(recipe, shape N)` the parent (after computing the gold when it
has one) spawns three children, one per engine:

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

The parent then compares each child's output to the gold buffer
elementwise and classifies each run as `match` / `WRONG k/N` / crash /
spawn-fail.  For NativeExecution recipes the native run itself is
classified as `gold` (since it *is* the gold), and legacy/salmon are
classified as `no-gold` when the native run failed to produce output.

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

1. **Grid** — one atomic row per `(recipe, shape)` with a short
   status cell for each engine.  Possible cells:
   - `match` / `WRONG k/N` — comparison against the gold (either a CPU
     reference or a prior native run).
   - `gold` — the native column under a NativeExecution recipe: this
     run produced the buffer every other mode is compared against.
   - `no-gold` — legacy / salmon were never spawned because the native
     gold failed to materialise (see `gold-missing` in the Summary).
   - `SIG…` / `EXIT=…` / `no-output` / `spawn-fail` — the child
     crashed, exited non-zero, produced no output, or failed to fork.
   No detail lines are interleaved with the grid, so it scans
   vertically.
2. **Failures** — every non-`match`/non-`gold` run reappears here with
   full detail (first mismatching index, reference vs. actual, max
   `|err|`, or the child's stderr tail for crashes), grouped by mode
   and then by recipe.  Each line is self-identified (`recipe shape`)
   so you always know which run it describes.
3. **Summary** — a matrix of categories × engines.  `match` /
   `mismatch` / `crash/no-exit` are always shown; `gold` and
   `gold-missing` appear only when at least one NativeExecution recipe
   ran.

For CpuReference recipes, `native` is the primary self-check of the
harness: if it matches the CPU reference, the kernel and the dispatch
are self-consistent, and any difference on the translated paths is
attributable to the translation itself.  If `native` does not match,
every other row on that shape should be treated as suspect.

For NativeExecution recipes, a `gold-missing` row means `native`
itself failed to produce output; there's no useful signal on legacy /
salmon for that shape and they are not spawned.  Fix the native run
(inspect the `native` entry under Failures) before reading anything
into the transpiled paths.

## Adding a new kernel

### Adding a HIP kernel (CPU-reference gold)

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

### Adding a Triton kernel (native-as-gold)

No C++ changes are needed — the harness registers Triton recipes
automatically by scanning `kernels/build/*.sidecar.json` at startup.

1. Drop `<recipename>.py` into `kernels/triton/`.  The file stem is
   the recipe name (and the `.co` filename root).  It must define at
   least one `@triton.jit` function and a module-level
   `RECIPES = [ ... ]` list with one entry.  See
   [`kernels/triton/README.md`](kernels/triton/README.md) for the
   schema and a worked example.
2. Add the recipe name to `$(TRITON_KERNELS)` in the Makefile.
3. `make ROCR_BUILD=…` — the Triton AOT step compiles for both
   gfx942 and gfx1250, extracts kernarg-layout metadata via
   `llvm-readelf`, and writes `<name>.sidecar.json` alongside the
   code objects.
4. `LD_PRELOAD=./libsalmon_intercept.so ./compare_correctness --recipe=<name>`.

The gold for Triton recipes is the `native` (gfx942) run itself —
legacy and salmon are judged against it.  The trade-off is explained
in the run-time [How it works](#how-it-works) section.

## Current probes

### HIP recipes (CPU-reference gold)

| recipe            | what it exercises                                             |
|-------------------|---------------------------------------------------------------|
| `vecadd`          | Baseline: pure VALU, no cross-lane ops.  Any failure here is structural, not intrinsic-related. |
| `block_sum_shfl`  | Two-phase block-sum reduction using `__shfl_xor` within each warp and cross-warp via shared memory.  One float per block. |
| `lane_swap`       | 1:1 output:  `out[tid] = in[tid ^ 1]`.  Crisp probe for whether cross-lane reads arrive from the expected partner. |
| `cvt_pkrtz`       | `V_CVT_PKRTZ_F16_F32` handler.  Packs two f32 into v2f16 with round-toward-zero (`__builtin_amdgcn_cvt_pkrtz`). |
| `cvt_pk_f16`      | `V_CVT_PK_F16_F32` handler.  Packs two f32 into v2f16 with round-to-nearest-even.  gfx942 native lowers without the packed opcode; the f16 bit pattern per lane matches, so the cross-engine compare is still meaningful. |
| `bfm_b32`         | `V_BFM_B32` handler.  Bit-field mask `((1<<w)-1)<<off`, forced via inline asm because hipcc lowers the same source to `v_lshlrev_b32` + `v_not_b32` on gfx1250. |
| `swap_b32`        | `V_SWAP_B32` handler.  Pairwise element exchange, forced via inline asm — this is one of the few VALU ops that writes both of its operands, so it's specifically worth stressing. |
| `mov_b64`         | `V_MOV_B64` handler (gfx11+).  Forced via inline asm under `#ifdef __gfx1250__`; gfx942 native uses a plain 64-bit copy because the opcode didn't exist yet.  Identity output. |
| `cvt_f32_bf16`    | `V_CVT_F32_BF16` handler (gfx950+).  Forced via inline asm under `#ifdef __gfx1250__`; gfx942 native uses the bit-level upcast `bf16 -> (u32 << 16) reinterpreted as f32`.  Both paths are bit-exact. |
| `v_add_lshl_u32`  | `V_ADD_LSHL_U32` handler.  Fused `(a+b) << (c & 31)` VOP3 op, forced via inline asm. |
| `v_bfe_i32`       | `V_BFE_I32` handler.  Signed bit-field extract from a vector src, with low-5-bit masking of offset/width.  Forced via inline asm. |
| `s_bfe_i32`       | `S_BFE_I32` handler (scalar signed BFE).  One output per block; per-block uniforms are pushed through `readfirstlane` to force SGPR operands. |
| `s_bitset0_b32`   | `S_BITSET0_B32` handler.  Scalar read-modify-write bit clear; exercises the tied `sdst_in` input path via an inline-asm `+s` constraint. |
| `s_bitset0_b64`   | `S_BITSET0_B64` handler.  64-bit sibling of `s_bitset0_b32`; dst/tied read are an SReg_64 pair, bit index is a 32-bit SGPR with low 6 bits consumed.  Covers `writeReg64` + tied-def plumbing for the 64-bit scalar path. |

### Triton recipes (native-as-gold)

| recipe        | what it exercises                                                                             |
|---------------|-----------------------------------------------------------------------------------------------|
| `vecadd_f16`  | End-to-end sanity check of the Triton integration: elementwise fp16 add, trivial Triton program, single scalar shape dim `N`, block size as constexpr.  Any failure here is framework plumbing (sidecar, kernarg packing, grid eval), not a Triton-codegen finding. |

## Handlers this harness does NOT cover

Three categories of handler cannot be exercised with today's
`gfx1250 -> gfx942` routing and are out of scope for this tool:

1. **gfx950-only instructions** — `v_cvt_scalef32_pk_fp4_f32` (scaled
   fp4 packing), `ds_read_b64_tr_b8` (LDS transpose load).  The
   gfx1250 assembler does not accept them (neither via inline asm nor
   spontaneous emission), so they cannot be placed in the source
   code object.
2. **gfx11+-only instructions with no gfx942 analogue** — covered
   above by `#ifdef __gfx1250__` gating, at the cost of only
   exercising the Salmon path (legacy translator has nothing gfx11
   to chew on either).
3. **Wave64-specific raiser paths** — the `EXEC_LO`/`EXEC_HI` partial
   write fix is only triggered when the source ISA is wave64, and
   gfx1250 is wave32.  Probing this requires either a gfx950/gfx942
   source ISA (a sibling harness change) or gfx950 hardware.

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
- HIP recipes hand-pack kernel arguments inside each recipe's dispatch
  lambda because their layouts are fixed and known at authoring time.
  Triton recipes, where the layout is Triton's choice and varies per
  signature, go through a metadata-driven packer that reads the
  per-arch kernarg layout out of the sidecar JSON (produced by
  `aot_compile.py` via `llvm-readelf --notes`).  `cross_arch_gpu_test.cpp`
  does the same against a code object's MSGPACK metadata directly.
- Triton recipes in Phase 1 have a single scalar shape dimension
  (e.g. `N`).  Multi-dim shapes (matmul's `M, N, K`, attention's
  `B, H, S, D`) need a sidecar-schema extension and a recipe shape
  sweep more expressive than a flat `default_shapes` list.

## Exit code

Always `0` on a completed sweep — a mismatch against the gold (CPU
reference or native execution) is a *finding*, not a harness error.
A non-zero exit means the harness itself failed (missing kernels,
spawn failure, filter matched nothing).  The report body is the thing
to read.
