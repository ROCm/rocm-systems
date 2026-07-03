This document describes the DBT/DBI text relocation algorithm.

The old layout used kernel-local code caves:

```text
.text
[kernel1 body]
[kernel1 cave]
[kernel2 body]
[kernel2 cave]
...
```

That does not scale when one kernel is larger than the signed 16-bit SOPP branch
range. An expanded instruction in the body may need to branch to a cave that is
too far away.

## Constraints

- Runtime complexity must be O(n log n) or better, with preference for O(n).
- Here n means the sum of blocks emitted for each kernel. If two kernels share a
  source helper block, that block is counted once per kernel because we emit a
  kernel-local copy.
- Translation must be fast enough for code objects larger than 200 MB with more
  than 800 kernels.
- Text relocation implementation must live in `code/patch/`.
- The binary translator in `code/dbt/` must build data structures exposed by
  `code/patch/`, but text emission and layout must live in `code/patch/`.
- Keep existing kernel descriptor and kernarg-preload behavior.
- Keep existing `CodeObjectPatcher` behavior.

## Main Idea

We stop treating a translated block as if it has the same byte size as the
source block. Instead, we emit translated words forward with a cursor.

For each kernel:

1. Build the kernel-local reachable block set.
2. Emit reachable blocks in source order.
3. Keep a map from each source block start to its translated block start.
4. Emit each translated instruction at the current cursor and advance the
   cursor by the number of emitted words.
5. Record branch and call sites that need a later fixup.
6. After all blocks for the kernel have been emitted, patch all recorded sites.

Blocks stay in source order. This preserves normal fallthrough behavior.

Shared source blocks are duplicated per kernel. This keeps branch targets and
call returns kernel-local and avoids sharing one emitted helper between kernels
with different return continuations.

## Branch Targets

Before emission starts, all source branch targets must be known. If the source
target cannot be recovered, translation fails.

"Known" here means known in the original source `.text`, not known in the final
translated `.text`. The translated target offset may not be known until later
because the output cursor moves as words are emitted.

There are two different cases.

### Direct Branches And Calls

Direct `s_branch`, `s_cbranch_*`, and `s_call_b64` instructions keep their
original encoding size. We patch their immediate after the target block has a
translated offset.

For now, we do not promote direct branches or direct calls to long sequences. If
the final translated target does not fit the original signed 16-bit dword
offset, translation fails closed.

### Recovered Indirect Branches And Calls

Recovered indirect branches and calls still use static branch recovery. We still
need the recovered source target and the SGPR pair used for the target PC.

The final emitted sequence rebuilds the translated target PC in the same SGPR
pair immediately before the control transfer. Because of that, the old
source-side address builder does not need to be the final relocation mechanism.

For a recovered jump, the canonical sequence is:

- `s_branch` if the translated target fits the signed 16-bit dword offset.
- Otherwise `s_getpc_b64`, scalar arithmetic, and `s_setpc_b64`.

For a recovered call, the canonical sequence is:

- `s_call_b64` if the translated target fits the signed 16-bit dword offset.
- Otherwise `s_getpc_b64`, scalar arithmetic, and `s_swappc_b64`.

If the translated target block has not been emitted yet, reserve enough words
for the largest canonical sequence. After the target offset is known, patch the
window and fill any unused words with `s_nop`.

## Invariants

These are design invariants. Violating them is an implementation bug, not a
normal user-facing failure mode.

- Every recovered source target is in the kernel-local reachable block set.
- Every recovered indirect branch or call site reserves the maximum canonical
  sequence size.
- Blocks are emitted in source order, so implicit fallthroughs are preserved.
- The translated target offset for every emitted source block is recorded before
  final fixup runs.

## Failure Modes

Translation fails closed in these cases:

- A source branch or call target cannot be recovered.
- A direct `s_branch`, `s_cbranch_*`, or `s_call_b64` final target is outside the
  original signed 16-bit dword range.
- Existing descriptor or kernarg-preload handling fails.
- `CodeObjectPatcher` cannot replace `.text` safely.

## Verification

Use these tests in order:

1. Translate `/home/kunwar/Work/runtime-evolution/dbt-benchmarks/resnet-f32`.
   This currently translates successfully and is the first regression check.
   Note that no file should take over 20 seconds to translate. If it does,
   we need to find and fix the regression. The largest file before this design
   took around 19 seconds to translate.

2. Translate `/home/kunwar/Work/runtime-evolution/dbt-benchmarks/resnet-f16`
   with continue-after-failure enabled. One HSACO currently fails with both
   semantic errors and branch target errors. After this layout fix, it should
   only report semantic errors.

   If branch target errors remain, either the layout implementation is wrong, or
   the input has direct `s_branch` / `s_call_b64` instructions whose final target
   is out of range. In that case, reduce the failing HSACO to a repro and fail
   closed.

3. Run the steps in `repro.md`.
