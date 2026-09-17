; Copyright (c) 2026 Advanced Micro Devices, Inc.
; SPDX-License-Identifier: MIT

; Two workgroups of 64 threads. The host checks every output element, including
; both waves on gfx1250. Feature module flags are appended by the test driver.
define amdgpu_kernel void @smoke(ptr addrspace(1) %input, ptr addrspace(1) %output) {
  %group = call i32 @llvm.amdgcn.workgroup.id.x()
  %lane = call i32 @llvm.amdgcn.workitem.id.x()
  %base = mul i32 %group, 64
  %index = add i32 %base, %lane
  %wide = zext i32 %index to i64
  %src = getelementptr i32, ptr addrspace(1) %input, i64 %wide
  %value = load i32, ptr addrspace(1) %src, align 4
  %result = add i32 %value, 7
  %dst = getelementptr i32, ptr addrspace(1) %output, i64 %wide
  store i32 %result, ptr addrspace(1) %dst, align 4
  ret void
}

declare i32 @llvm.amdgcn.workgroup.id.x()
declare i32 @llvm.amdgcn.workitem.id.x()
