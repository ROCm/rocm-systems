; RUN: %raise_cli %group_segment_fixed_size_attr_co --isa=gfx1250 --target-isa=gfx942 \
; RUN:     --emit-ir=group_segment_fixed_size_attr_kernel 2>/dev/null \
; RUN:   | %FileCheck %s --check-prefix=IR
;
; Regression fence for the static-LDS-size propagation in `raiser.cpp`.
; Without this attribute, every lifted kernel with static LDS would
; get `.group_segment_fixed_size: 0` in its HSACO KD, causing every
; LDS access to return zero (the segment is unallocated from the
; hardware's perspective).
;
; The fixture's source `__shared__ int lds[1024]` yields a source
; `.group_segment_fixed_size: 4096` in the gfx1250 HSACO's KD.  The
; raiser must propagate that into an `amdgpu-lds-size` function
; attribute on the lifted kernel so `AMDGPUMachineFunctionInfo`
; picks it up in the cross-targeted gfx942 codegen.
;
; We assert:
;   1) The lifted kernel function carries `"amdgpu-lds-size"="4096,4096"`
;      (min=max=4096 since the source's static size is known exactly).
;   2) The source's `__shared__ int lds[1024]` lowered to at least one
;      addrspace(3) ds_write / ds_load pair in the IR (i.e. the LDS is
;      actually used — otherwise the attribute would be a no-op and
;      the fixture would pass vacuously).

; IR-LABEL: define amdgpu_kernel void @group_segment_fixed_size_attr_kernel(
; IR-DAG: store {{.*}}, ptr addrspace(3)
; IR-DAG: load {{.*}}, ptr addrspace(3)
; IR: attributes #{{[0-9]+}} = { {{.*}}"amdgpu-lds-size"="4096,4096"{{.*}} }
