; RUN: %raise_cli %ds_load_2addr_b32_co --isa=gfx1250 --target-isa=gfx942 \
; RUN:     --emit-ir=ds_load_2addr_b32_kernel 2>/dev/null \
; RUN:   | %FileCheck %s
;
; Lift test for the gfx11+ two-address LDS load family
; (DS_READ2_B32 / DS_READ2ST64_B32).  The handler dispatches through
; the dedicated `ds2Classify` block in transpiler/handle_ds.cpp;
; the companion `.hip` source documents WHY this fixture exists — in
; short, to pin the correct per-offset-independent lowering so we
; never regress back to the silent-miscompile `load <2 x i32>` shape
; the generic single-offset path produced before the rewrite.
;
; Lowering shape this fixture pins:
;
;   * DS_READ2_B32 offset0:4 offset1:6 emits
;       %ds2_ld0 = load i32, ptr addrspace(3) %..., align 4   ; byte 16
;       %ds2_ld1 = load i32, ptr addrspace(3) %..., align 4   ; byte 24
;     (two independent dword loads at raw-field * 4 byte offsets —
;     NOT a contiguous `<2 x i32>` vector load).
;
;   * DS_READ2ST64_B32 offset0:2 offset1:3 emits
;       %ds2_ld0 = load i32, ptr addrspace(3) %..., align 4   ; byte 512
;       %ds2_ld1 = load i32, ptr addrspace(3) %..., align 4   ; byte 768
;     (×256 byte stride per raw-field unit — the ST64 opcode's
;     distinguishing feature).
;
;   * Both variants route the byte offset through the chain
;       %ds2_addr = zext i32 %{{...}} to i64
;       %ds2_off  = add i64 %ds2_addr, <imm>
;       %ds2_p{0,1} = inttoptr i64 %ds2_off to ptr addrspace(3)
;     where `<imm>` is the per-access byte offset.  Pinning the
;     immediate values {16, 24, 512, 768} catches any regression to
;     the unscaled-raw-field shape the prior handler used.

; CHECK-LABEL: define amdgpu_kernel void @ds_load_2addr_b32_kernel(

; All assertions are CHECK-DAG so the order of the two inline-asm
; sites (DS_READ2_B32 first, DS_READ2ST64_B32 second) is irrelevant —
; the handler emits each site as an adjacent (add + inttoptr + load)
; triple, and either order within the function body is
; semantically valid.
;
; Per-access byte offsets — four distinct immediate values pinning
; both the non-ST64 (x4 scaling) and ST64 (x256 scaling) byte-offset
; arithmetic.  A regression to unscaled raw-field offsets (4, 6, 2,
; 3) would fail to match; a regression to contiguous single-offset
; lifting (only offset0 consumed) would also fail by missing half
; the adds.
; CHECK-DAG: add i64 %{{.*}}, 16
; CHECK-DAG: add i64 %{{.*}}, 24
; CHECK-DAG: add i64 %{{.*}}, 512
; CHECK-DAG: add i64 %{{.*}}, 768

; Four independent dword loads in address space 3 (LDS), each
; carrying the explicit `align 4` the `CreateAlignedLoad` path
; stamps.  Each load is pinned through its DISTINCT pointer SSA
; name — the first inline-asm site's pair is emitted as `%ds2_p0`
; / `%ds2_p1`; the second site re-enters the handler and LLVM's
; value-symbol-table auto-renames the repeated base names to
; `%ds2_p0<N>` / `%ds2_p1<N>` (the `[0-9]+` regex is the
; stability-safe version of that).  Four separate -DAG lines are
; used (rather than a single counted variant) because counted
; directives are order-sensitive under -DAG neighbourhood rules.
; CHECK-DAG: load i32, ptr addrspace(3) %ds2_p0, align 4
; CHECK-DAG: load i32, ptr addrspace(3) %ds2_p1, align 4
; CHECK-DAG: load i32, ptr addrspace(3) %ds2_p0{{[0-9]+}}, align 4
; CHECK-DAG: load i32, ptr addrspace(3) %ds2_p1{{[0-9]+}}, align 4

; Negative pins.  These would indicate a regression back to the
; generic single-offset DS_READ path (contiguous `<2 x i32>` vector
; load) or to the B64 / B128 widths (wrong dsClassify entry firing).
; CHECK-NOT: load <2 x i32>, ptr addrspace(3)
; CHECK-NOT: load <4 x i32>, ptr addrspace(3)
; CHECK-NOT: load i64, ptr addrspace(3)
; CHECK-NOT: call {{.*}}@llvm.amdgcn.ds.read
