; RUN: %raise_cli %global_load_ushort_saddr_co --isa=gfx1250 --target-isa=gfx942 \
; RUN:     --emit-ir=global_load_ushort_saddr_kernel 2>/dev/null \
; RUN:   | %FileCheck %s
;
; Sub-dword GLOBAL_LOAD SADDR + `scale_offset` lowering. The
; `handle_flat.cpp` handler must:
;
;   * Detect the decoded SADDR shape `(src[0]=SGPR64, src[1]=VGPR32)`
;     (MC-imposed order — differs from the assembler's written order)
;     and compute `addr = saddr + sext(vaddr) * elemBytes`.
;   * Consume the `scale_offset` bit from the decoded CPol operand
;     (`di.hasScaleOffset`), NOT from a `fullText` string search —
;     that was previously fragile and is the subject of this test.
;     For u16 loads `elemBytes = 2`.
;   * Emit an i16 load from `addrspace(1)` with `align 2`, zero-extend
;     to i32.
;
; If the `scale_offset` bit is misread (e.g. the `fullText` regression
; returns, or the decode path picks the wrong CPol bit) every lane
; would broadcast from `saddr + imm_offset`, which is how the original
; `cvt_f32_bf16` kernel silently produced all zeros. The `mul i64
; ..., 2` check below catches that class of regression.

; CHECK-LABEL: define amdgpu_kernel void @global_load_ushort_saddr_kernel(

; Address composition: sext(vaddr32) to i64, scale by elemBytes=2
; (u16 = 2 bytes), add to the SGPR64 base. The name bindings capture
; the per-step values so the CHECKs below can assert each one links
; to the next.
; CHECK:      %voff_sext = sext i32 %{{[^ ,]+}} to i64
; CHECK-NEXT: %scaled_voff = mul i64 %voff_sext, 2
; CHECK-NEXT: %saddr_vaddr = add i64 %{{[^ ,]+}}, %scaled_voff

; Pointer cast + load: the i64 sum is bitcast into a global-addrspace
; pointer and loaded as i16 with natural alignment. The key
; assertions are `addrspace(1)`, `i16`, and `align 2` — these are the
; three properties the pre-fix path corrupted.
; CHECK:      %{{[^ ]+}} = inttoptr i64 %saddr_vaddr to ptr addrspace(1)
; CHECK:      %gload_sub = load i16, ptr addrspace(1) %{{[^ ,]+}}, align 2

; Zero-extension to i32 (not sign-extension) because this is u16, not
; i16.
; CHECK:      %{{[^ ]+}} = zext i16 %gload_sub to i32

; Negative assertion: the pre-fix handler was using `elemBytes = 4`
; for the u16 case (copy-pasted from the dword path). A `mul i64
; ..., 4` here would mean the scaling regressed.
; CHECK-NOT:  mul i64 %voff_sext, 4
