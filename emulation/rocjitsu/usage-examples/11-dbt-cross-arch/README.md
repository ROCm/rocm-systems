# Example 11: Dynamic Binary Translation (DBT) — cross-arch

## Objective

Use rocjitsu's DBT pipeline to translate a kernel's machine code from one GPU
architecture to another and inspect the result — and see how the amount of
translation **scales with how different the two ISAs are**.

## What this example does

- Compiles a simple HIP kernel for **gfx950**, producing a bundled gfx950 code object.
- Uses **`rj_dbt_translate`** to translate that code object to other ISAs and print
  a per-instruction diff. **No GPU is required.**
- Compares a **close** target (gfx942, CDNA→CDNA) against a **divergent** one
  (gfx1200, CDNA→RDNA).

## Why cross-arch translation matters

In one line: **run a kernel that was compiled for one GPU on a different GPU
without rebuilding it from source.** DBT rewrites the kernel's already-compiled
**machine code** from one ISA to another.

**But if you have the source, just recompile** (`--offload-arch=<target>`) — that
is simpler and produces better code. DBT earns its keep precisely when
recompiling **isn't an option**:

- **You don't have the source.** A shipped binary, a distributed fat binary, a
  cached code object, or a **third-party / precompiled kernel** (e.g. a prebuilt
  Triton or library kernel). You only have machine code, so DBT translates the
  machine code directly. This is the canonical DBT use case.
- **You don't have the target toolchain or hardware.** Translate a gfx950 kernel
  and run/test it on a gfx942 GPU (or the emulator) with no gfx942 build setup.
- **You want it transparent.** The HSA hook translates code objects *as they load*
  (see [Running translated code](#running-translated-code-translate-and-run)), so
  an existing app runs on a different arch with no rebuild and no code change.
- **You want to inspect portability** — see, instruction by instruction, exactly
  what differs between two ISAs and what has no equivalent yet.

Caveats (DBT is **not** a universal "never recompile" button):

- It is **rule-driven per instruction**, so it can be **incomplete** — a kernel
  won't translate if it uses an instruction with no rule for that target (see
  [below](#why-these-translations-work--or-dont); CDNA→RDNA currently hits this).
- Translated code isn't necessarily as **optimal** as freshly compiled code.

> DBT (translating between real ISAs) is separate from rocjitsu's **emulator**
> (running code on a simulated GPU). They compose — translate a binary, then run
> it on a real or emulated target — but the "no recompile" benefit is the DBT part.

## Files

- `src/dbt_example.cpp` — a simple kernel compiled for gfx950 (the DBT input)
- `Makefile`

## Build and run

```bash
cd usage-examples/11-dbt-cross-arch
make                 # compile for gfx950
make list            # show the code objects bundled in the binary
make translate       # gfx950 -> gfx942 diff (MI350 -> MI300, down-level)
make translate-reverse # gfx942 -> gfx950 diff (MI300 -> MI350, up-level)
make translate-rdna  # gfx950 -> gfx1200 diff (CDNA -> RDNA)
make compare         # all three, side-by-side
make run-gfx950      # sanity: run the binary on its native arch under rocjitsu
```

## Translation scales with ISA distance — and direction

`make compare` — verbatim from `sharkmi300x-4` (ROCm 7.2.1):

```text
=== gfx942 -> gfx950  (MI300 -> MI350, up-level, CDNA3->CDNA4) ===
rj_dbt_translate: ok
  actions: copy_original=17 encode=0 identity=0 substitute=0 lower=0 expand=0 illegal=0 semantic=0
errors: 0

=== gfx950 -> gfx942  (MI350 -> MI300, down-level, CDNA4->CDNA3) ===
rj_dbt_translate: ok
  actions: copy_original=0 encode=0 identity=14 substitute=0 lower=3 expand=0 illegal=0 semantic=0
errors: 0

=== gfx950 -> gfx1200 (MI350 -> Radeon RX 9070, cross-family, CDNA->RDNA) ===
error: expand-missing .text+0x0020 v_add_u32_e32: legalization requires EXPAND, but no expansion rule is implemented
rj_dbt_translate: failed
  actions: copy_original=0 encode=0 identity=0 substitute=2 lower=3 expand=0 illegal=0 semantic=1
errors: 1
error: translation failed
```

| Translation | dominant actions | changes | result |
|---|---|---|---|
| **gfx942 → gfx950** (MI300→MI350, up-level) | `copy_original=17` | none | `ok` — pure copy |
| **gfx950 → gfx942** (MI350→MI300, down-level) | `identity=14 lower=3` | 1 | `ok` — minor lowering |
| **gfx950 → gfx1200** (CDNA→RDNA, cross-family) | `substitute=2 lower=3 semantic=1` | all | fails on a coverage gap |

Two things drive how much work the translator does:

- **ISA distance.** MI300↔MI350 (both CDNA) barely differ, so translation is
  trivial. MI350→RX 9070 (CDNA→RDNA) is a different architecture family, so every
  instruction is rewritten — and it hits the gap below.
- **Direction.** **Up-level** (MI300→MI350, older→newer) is a **pure pass-through**
  (`copy_original=17`, nothing changes) because the newer CDNA4 ISA is a
  near-superset of CDNA3 — gfx942 code is already valid on gfx950. **Down-level**
  (MI350→MI300, newer→older) needs a little **lowering** (3 instructions) because a
  few newer encodings have to be expressed in the older ISA.

### Real gfx950 → gfx1200 translations (`make translate-rdna`)

```text
0x0000 lower       s_load_dword s3, s[0:1], 0x1c   ->  s_load_b32 s3, s[0:1], 0x1c
0x0010 lower semantic  s_waitcnt vmcnt(63) expcnt(7) lgkmcnt(0)
                       ->  s_wait_storecnt_dscnt 240
                       ->  s_wait_kmcnt 0
0x0014 substitute  s_and_b32 s3, s3, 0xffff        ->  s_and_b32 s3, s3, 0xffff  (re-encoded)
0x001c substitute  s_mul_i32 s2, s2, s3            ->  s_mul_i32 s2, s2, s3      (re-encoded)
0x0020 EXPAND      v_add_u32_e32                    ->  (no expansion rule implemented)
```

The gfx950→gfx1200 run **fails on purpose here**: `v_add_u32_e32` (from the kernel's
index arithmetic) needs a legalization EXPAND that isn't implemented yet for RDNA
targets. Both RDNA targets (`gfx1200`, `gfx1201`) hit the same gap. This is a
**current coverage limit of the DBT translator for CDNA→RDNA**, not a bug in the
example — and it's exactly the kind of thing `rj_dbt_translate` is meant to surface.

`make run-gfx950` (native-arch sanity check):

```text
scale_kernel (gfx950): PASSED
```

## What the translation actions mean

Each instruction in the `diff` is tagged with the **action** the translator took.
From least to most invasive:

| Action | What it means | Seen here |
|---|---|---|
| `identity` | Encoding is already valid on the target — emitted unchanged (a literal may be re-canonicalized). | most gfx942 instructions |
| `substitute` | Same encoding format, **different opcode number** — a straight opcode swap. | `s_and_b32` (`86`→`8b`), `s_mul_i32` (`92`→`96`) for gfx1200 |
| `lower` | **Field remap** — bits are rearranged within the instruction to the target's encoding. | `s_load_dword`→`s_load_b32` for gfx1200 |
| `lower semantic` / `expand semantic` | **Multi-instruction lowering** — one source instruction becomes a *different sequence* of target instructions. | `s_waitcnt`→`s_wait_storecnt_dscnt` + `s_wait_kmcnt` for gfx1200 |
| `expand` | Multi-instruction lowering (e.g. matrix `MFMA`→`WMMA` with cross-lane shuffles). | not in this simple kernel |
| `copy_original` / `encode` | Copied verbatim / re-encoded with no legalization rule needed. | — |
| `illegal` | No legal representation on the target at all. | — |
| **`expand-missing`** (error) | The instruction *requires* a semantic expansion, but **no rule is implemented** — translation hard-fails with a diagnostic instead of emitting something wrong. | `v_add_u32_e32` for gfx1200 |

The `actions:` histogram in the output is just a count of these per translation.
Internally each is one of four rule kinds — `Identity`, `Substitute`,
`FieldRemap` (shown as `lower`), `Expand` (shown as `expand`/`semantic`) — looked
up per `(source encoding, opcode)` pair (`code/dbt/translation_rule.h`).

## Why these translations work — or don't

Translation is **table-driven**: for every `(source ISA, target ISA)` pair there
is a generated rule set (`code/dbt/generated/legalization_*.h`), and an
instruction translates only if that table has a rule for it.

- **gfx942 → gfx950 (up-level) is a pure copy** because CDNA4 is a near-superset of
  CDNA3: every gfx942 instruction is already valid on gfx950, so all are
  `copy_original` — nothing to translate, `errors: 0`.

- **gfx950 → gfx942 (down-level) works cleanly** with a little lowering: CDNA4 and
  CDNA3 are close relatives, so almost every encoding is byte-compatible
  (`identity`), with only a few field remaps for newer encodings that must be
  expressed in the older ISA. `errors: 0`.

- **gfx950 → gfx1200 does real work but currently fails** because CDNA and RDNA
  diverge substantially — scalar loads are re-encoded, the single unified
  `s_waitcnt` splits into RDNA's separate counter waits, opcodes are renumbered.
  The translator handles those, then reaches **`v_add_u32_e32`**: on RDNA that
  needs a semantic multi-instruction expansion, and that expansion rule isn't
  implemented yet. Rather than emit an incorrect encoding, the translator
  **hard-fails with a diagnostic** (`expand-missing … no expansion rule is
  implemented`). Since `v_add_u32_e32` comes from ordinary index arithmetic, most
  real kernels hit this — so **CDNA→RDNA translation is currently incomplete**.

The takeaway: a translation "works" exactly to the extent the target's rule table
covers the instructions your kernel uses. `rj_dbt_translate` is designed to make
those gaps **loud and precise** (a named instruction + offset) rather than
silently producing broken code.

## Running translated code (translate-and-run)

The translations above are static. rocjitsu can also **translate a code object at
load time and run it**, but this is an **HSA-level** feature, not a HIP one:

- Implemented as an HSA tools library (`librocjitsu_hooks.so`) that ROCr loads via
  `HSA_TOOLS_LIB`, intercepting `hsa_executable_load_agent_code_object()` and
  translating to `RJ_DBT_TARGET_ISA`.
- Verified on this host for the close pair (translate to gfx942 and run on the
  local gfx942 GPU, ~0.4 s) via the in-tree pure-HSA tests:
  `tests/dbt/hsa_hooks_test.cpp` (`HsaHooksTest.TranslateGfx950Mfma16x16ThroughToolsLib`)
  and `tests/dbt/cdna4_to_cdna3_dispatch_test.cpp` (`Cdna4ToCdna3DispatchTest.*`).

```bash
HSA_TOOLS_LIB=<build>/lib/rocjitsu/src/rocjitsu/hooks/librocjitsu_hooks.so \
RJ_DBT_TARGET_ISA=gfx942 RJ_DBT_SOURCE_ISA=gfx950 \
  ./a_pure_hsa_program
```

**HIP applications do not work with this hook** — HIP loads its code objects
through its own fat-binary machinery, which bypasses the intercepted HSA load
function, so the kernel is never translated and HIP crashes launching it on a
different arch. That's why this example demonstrates DBT with `rj_dbt_translate`
(no GPU, no HSA program needed) and points at the in-tree HSA tests for run.

## Environment variables (DBT HSA hook)

| Variable | Purpose |
|---|---|
| `RJ_DBT_TARGET_ISA` | **Required** — target arch (e.g. `gfx942`) |
| `RJ_DBT_SOURCE_ISA` | Optional source arch (e.g. `gfx950`) |
| `RJ_DBT_LOG` | DBT hook log level |

Supported targets: `gfx942`, `gfx950`, `gfx1200`, `gfx1201`.

## Key takeaways

- `rj_dbt_translate` inspects/translates a code object across ISAs **without a
  GPU** — ideal for checking cross-arch compatibility and seeing what the
  translator does per instruction.
- **Translation effort scales with ISA distance:** CDNA→CDNA (gfx950→gfx942) is
  mostly `identity`; CDNA→RDNA (gfx950→gfx1200) rewrites everything and currently
  hits an unimplemented `v_add_u32_e32` expansion.
- Translate-and-**run** is an HSA-level capability (`HSA_TOOLS_LIB` +
  `RJ_DBT_TARGET_ISA`), verified for pure-HSA programs; it is not wired for HIP apps.

## Related

- [`docs/rj_dbt_translate.md`](../../docs/rj_dbt_translate.md) — the translate tool
- [`docs/dbt-design.md`](../../docs/dbt-design.md) — DBT pipeline design
