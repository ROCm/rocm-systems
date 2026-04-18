; RUN: %not %raise_cli %c2_dpp_quad_perm_co --isa=gfx1250 --target-isa=gfx942 \
; RUN:     --emit-ir=c2_dpp_quad_perm_kernel 2>&1 \
; RUN:   | %FileCheck %s --check-prefix=STDERR
;
; SPE_DESIGN.md §3 Class 2 (DPP sub-category) — the single largest
; outstanding correctness hazard for GPT-OSS per
; gpt-oss-derisking.md §7.3 (5/15 kernels affected). The classifier
; must refuse any kernel with DPP modifiers until
; CROSS_LANE_SURVEY.md P5 lands an update.dpp intrinsic lift.
;
; Detection note. The raiser canonicalises DPP modifiers away in
; opcode_map.cpp:buildDppToBaseMap BEFORE any SemOp-level handler
; sees the DPP variant, so the classifier cannot key on a
; SemOp. Instead it must match on the raw mnemonic containing
; `_dpp` (the disassembler's text output preserves the modifier
; even though the decoded MCInst has been canonicalised to the
; base opcode). This is a known syntactic limitation documented in
; wave_size_obstruction.cpp; the dataflow follow-up will extend the
; DecodedInst to retain the DPP modifier bits.
;
; MAINTENANCE. Same flip protocol as c2_permlane_swap.ll once P5
; lands.

; STDERR: transpiler: pre-translation abort:
; STDERR-SAME: cross-wave-shuffle-rewrite-pending

; STDERR: DppCrossLane
; STDERR-SAME: Class 2
; STDERR: rewrite: P5
; STDERR-SAME: pending
; STDERR: outcome: (c) refuse

; STDERR: raise_cli: kernel 'c2_dpp_quad_perm_kernel' failed to raise:
; STDERR-SAME: dpp
