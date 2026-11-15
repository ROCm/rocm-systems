target triple = "amdgcn-amd-amdhsa"

define amdgpu_kernel void @k3(ptr addrspace(1) %out, i32 %a, i32 %b, i32 %c, i32 %d) {
entry:
  %a16 = trunc i32 %a to i16
  %b16 = trunc i32 %b to i16
  %c16 = trunc i32 %c to i16
  %s0 = add i16 %a16, %b16
  %s1 = shl i16 %s0, 3
  %s2 = xor i16 %s1, %c16
  %s3 = lshr i16 %s2, 2
  %s4 = mul i16 %s3, 5
  %cmp = icmp ult i16 %s4, %b16
  %s5 = select i1 %cmp, i16 %s4, i16 %a16
  %e = zext i16 %s5 to i32
  %f = add i32 %e, %d
  %g = and i32 %f, 4095
  %h = sub i32 %g, %c
  %p = getelementptr i32, ptr addrspace(1) %out, i64 1
  store i32 %h, ptr addrspace(1) %out
  store i32 %e, ptr addrspace(1) %p
  ret void
}
