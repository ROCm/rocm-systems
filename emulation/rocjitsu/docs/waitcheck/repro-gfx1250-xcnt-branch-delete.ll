; Reduced from PyTorch's binary_cross_entropy_out_cuda<double> kernel.
; This is a code-generation reproducer and is not intended to be executed.

target datalayout = "e-m:e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32-p7:160:256:256:32-p8:128:128:128:48-p9:192:256:256:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64-S32-A5-G1-ni:7:8:9"
target triple = "amdgcn-amd-amdhsa"

declare void @llvm.trap() #0

define amdgpu_kernel void @repro(i8 %0, i1 %cond1, i8 %1, i32 %2) {
  switch i8 %0, label %4 [
    i8 0, label %common.ret1
    i8 26, label %5
  ]

common.ret1:
  ret void

4:
  tail call void @llvm.trap()
  unreachable

5:
  %6 = tail call i32 @llvm.amdgcn.workitem.id.x()
  %7 = zext i32 %6 to i64
  %8 = getelementptr i8, ptr addrspace(1) null, i64 %7
  %9 = load i8, ptr addrspace(1) %8, align 1
  %cond = icmp eq i8 %9, 0
  br i1 %cond, label %common.ret1, label %10

10:
  br i1 %cond1, label %12, label %11

11:
  tail call void @llvm.trap()
  unreachable

12:
  switch i8 %1, label %14 [
    i8 0, label %13
    i8 2, label %15
  ]

13:
  store i8 0, ptr addrspace(1) null, align 1
  br label %15

14:
  tail call void @llvm.trap()
  unreachable

15:
  %16 = sitofp i32 %2 to double
  %17 = call double @llvm.copysign.f64(double 0.000000e+00, double %16)
  %18 = fadd double %17, 0.000000e+00
  %19 = fsub double 0.000000e+00, %18
  %20 = fptosi double %19 to i16
  store i16 %20, ptr addrspace(1) addrspacecast (ptr null to ptr addrspace(1)), align 2
  br label %common.ret1
}

declare noundef range(i32 0, 1024) i32 @llvm.amdgcn.workitem.id.x() #1
declare double @llvm.copysign.f64(double, double) #2

attributes #0 = { cold noreturn nounwind memory(inaccessiblemem: write) }
attributes #1 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #2 = { nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none) }
