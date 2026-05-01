# CDNA4-to-RDNA3 Residual Coverage Closure

This note records the residual scan after the end-to-end
coverage stack. It complements `cdna4-to-rdna3-taxonomy.md` and keeps the
remaining unsupported paths as durable implementation categories.

## Current Passing Coverage

- `vector_add` translates from CDNA4/gfx950 to RDNA3/gfx1100, loads through
  HSA, dispatches on a real gfx1100 GPU, and matches CPU golden data.
- `memory_stream` now covers multiple pointer kernel arguments, two global
  loads, two output streams, and `D = 0.5*A + B`. This is the richer variant
  that previously produced roughly 1.3k mismatches on gfx1100.
- CPU translator tests assert that translated RDNA3 ELF output has gfx1100
  `e_flags`, emits no warnings for the supported representative kernels, and
  decodes cleanly with the RDNA3 decoder.
- CPU translator tests cover VGPR-backed MTBUF format-buffer translation,
  including D16 mnemonic-order legalization and RDNA3 decode of the emitted
  `tbuffer_*` instruction.
- Matrix/AccVGPR representative coverage remains honest: CDNA4 MFMA to RDNA3
  fails closed with diagnostics linked to the existing matrix follow-ups.

## Resolved Residual

The richer `memory_stream` mismatch was a small waitcnt correctness issue. The
translated code preserved CDNA4 partial `vmcnt` waits as RDNA3 partial waits.
With two outstanding global loads and an intervening store, `s_waitcnt vmcnt(1)`
could leave the second load outstanding before `v_fmac_f32` consumed it,
producing nondeterministic D mismatches on gfx1100.

The current conservative fix: active CDNA4/GFX9 `vmcnt` waits lower to
RDNA3 `vmcnt(0)`, while the all-relaxed no-wait encoding remains no-wait. This
over-waits vector memory for correctness. Precision/performance recovery is
left as a separate precision/performance recovery task.

## Remaining Gaps

| Bucket | Status |
| --- | --- |
| Dense MFMA to RDNA3 WMMA | Unsupported; requires proven RDNA3 WMMA lowering or a software fallback. |
| AccVGPR virtualization | Unsupported; requires supported AccVGPR remapping or matrix-idiom lowering, including MTBUF encodings with CDNA4 `acc=1`. |
| Sparse SMFMAC | Unsupported; requires sparse metadata semantics and RDNA3 fallback analysis. |
| Other non-matrix `Action::Expand` rows | Unsupported; requires per-family expansion lowering or an explicit unsupported diagnostic. |
| Precise partial VM waitcnt performance | Correctness is conservative; partial-wait precision recovery is separate follow-up work. |

## Fail-Closed Behavior

- Any unhandled `Action::Expand` row now returns an empty translated ELF with a
  fatal warning containing mnemonic, `.text` offset, `encoding_id`, opcode, and
  the relevant unsupported category.
- Any non-identity row whose generated encoder returns no instruction words now
  fails closed instead of copying source bytes. MTBUF now has an encoder; the
  CDNA4 `acc=1` form deliberately returns no words because RDNA3 has no MTBUF
  AccVGPR selector, producing a diagnostic tied to AccVGPR remapping.
- Identity rows may still source-copy when the table explicitly classifies the
  row as binary-compatible.

## Deliberate Exclusions

This change does not implement MFMA/AccVGPR, sparse matrix, or broad
software-emulation lowerings. Those require family-specific semantics and tests;
the translator rejects them with actionable diagnostics instead of emitting
invalid RDNA3 code.
