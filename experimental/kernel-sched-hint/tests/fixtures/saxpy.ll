define amdgpu_kernel void @saxpy(ptr addrspace(1) %x, ptr addrspace(1) %y, float %a) {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %px = getelementptr inbounds float, ptr addrspace(1) %x, i32 %tid
  %py = getelementptr inbounds float, ptr addrspace(1) %y, i32 %tid
  %vx = load float, ptr addrspace(1) %px, align 4
  %vy = load float, ptr addrspace(1) %py, align 4
  %scaled = fmul float %vx, %a
  %sum = fadd float %scaled, %vy
  store float %sum, ptr addrspace(1) %py, align 4
  ret void
}

declare i32 @llvm.amdgcn.workitem.id.x()
