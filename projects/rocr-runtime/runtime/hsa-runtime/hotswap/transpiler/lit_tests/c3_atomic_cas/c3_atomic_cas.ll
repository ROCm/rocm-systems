; RUN: %not %raise_cli %c3_atomic_cas_co --isa=gfx1250 --target-isa=gfx942 \
; RUN:     --emit-ir=c3_atomic_cas_kernel 2>&1 \
; RUN:   | %FileCheck %s --check-prefix=STDERR
;
; SPE_DESIGN.md §3 Class 3 "inter-replica race via shared state" —
; non-commutative atomics have no rewrite that preserves the source
; semantics on a wider target wave. The classifier must refuse.
;
; gpt-oss-derisking.md §4 reports 0/170 kernels use this pattern,
; so this test exists as a guard / regression fence, not because
; any corpus kernel trips it.

; STDERR: transpiler: pre-translation abort:
; STDERR-SAME: cross-wave-replica-race
; The mnemonic reported is whatever the LLVM disassembler names the
; atomic cmpxchg at decode time. On gfx1250 that is
; `global_atomic_cmpswap_b32` (or the `.._b64` variant depending on
; pointer width). Key on the stable `cmpswap` substring.
; STDERR-SAME: cmpswap

; STDERR: NonCommutativeAtomic
; STDERR-SAME: Class 3
; STDERR: outcome: (c) refuse

; STDERR: raise_cli: kernel 'c3_atomic_cas_kernel' failed to raise:
; STDERR-SAME: cmpswap
