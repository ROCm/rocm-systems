// ARCH: gfx1250
//
// Lift fixture for the gfx11+/gfx1250 96-bit scalar memory load
// (`s_load_b96`) over a kernarg buffer that contains a `by_value`
// aggregate larger than 8 bytes. See SemOp::S_LOAD_B96 in
// transpiler/semop.hpp; the dispatch in transpiler/handle_smem.cpp
// (the per-dword `extractKernargDword` loop in the
// `if (isKernarg && immOffset && byteOffset < ctx.kernargs.implicitArgsBase)`
// branch); and the `by_value` aggregate decomposition in
// transpiler/raiser.cpp (`paramTypes` + `kernargs.params` build with
// the `byValueSize > 8 && byValueSize % 4 == 0` branch). The new helper
// `extractKernargDword` lives in transpiler/kernarg_layout.{hpp,cpp};
// this fixture pins both halves of the contract bidirectionally per
// the transpiler test policy.
//
// WHY THIS FIXTURE EXISTS:
//
// Before this fixture / handler / raiser change landed, an
// `s_load_b96 s[N:N+2], s[kernarg_ptr_lo:hi], <imm>` whose <imm>
// landed strictly INSIDE a `by_value` struct (size > 8) would refuse
// the lift with `Cannot resolve kernarg at offset N`. Two coupled
// bugs: (a) raiser.cpp mapped the entire by_value to a single `i32`
// IR argument, which (a1) misaligned every following kernarg slot's
// runtime offset and (a2) made interior dword reads unrepresentable;
// (b) `KernargLayout::resolveLoad` only matched fully-contained
// arguments, so a sub-struct dword load would fall through to the
// "not resolved" branch even when (a) was fixed. The fix is the
// per-dword decomposition + `extractKernargDword` helper.
//
// 502 tensilelite + Triton attention kernels in the kerneldex sweep
// were blocked on this single missing-handler row before the fix.
// This fixture exists to regression-pin the IR shape that fix
// produces.
//
// HOW THE FIXTURE EXERCISES THE PATH:
//
// We declare a kernel signature with one 16-byte `by_value` struct
// `Big16` followed by an output pointer:
//
//     kernel(Big16 desc, unsigned int *out)
//
// The AMDGPU kernarg metadata for this kernel is (sizes per the
// AMDGPU code-object ABI):
//
//     by_value, offset 0,  size 16  (the Big16 aggregate)
//     global_buffer, offset 16, size 8 (the output pointer)
//
// raiser.cpp now expands the 16-byte by_value into FOUR i32 slots
// at offsets 0/4/8/12, producing an IR signature of:
//
//     define amdgpu_kernel void @s_load_b96_kernarg_kernel(
//         i32 %arg0, i32 %arg1, i32 %arg2, i32 %arg3,
//         ptr addrspace(1) %arg4)
//
// Inline asm then issues `s_load_b96 s[0:2], s[0:1], 0x4`, which
// loads kernarg bytes [4, 16) into s[0:2] — i.e. dwords 1/2/3 of the
// Big16 struct, which correspond to %arg1, %arg2, %arg3 in the IR.
// The handler emits three `storeSGPR32(s0,arg1)`,
// `storeSGPR32(s1,arg2)`, `storeSGPR32(s2,arg3)` calls.
//
// Side note: hipcc compiles the `out[3..6] = desc.a..d` tail into a
// `global_load_b128 v[0:3], v7, s[0:1]` — i.e. it assumes s[0:1] is
// still the kernarg pointer. The s_load_b96 above clobbers s[0:1]
// (the load destination s[0:2] overlaps the kernarg pointer pair),
// so this kernel is NOT runtime-correct. That is intentional: a
// FileCheck lift fixture only needs the disassembly to contain
// `s_load_b96` and the lift to be deterministic. The downstream
// `global_load_b128` consuming s[0:1] = (loaded_dword_1,
// loaded_dword_2) gives us a stable, IR-printer-stable pin: the
// lifted IR contains `zext i32 %arg1 to i64` and `zext i32 %arg2 to
// i64` feeding the address arithmetic of the `load <4 x i32>`,
// proving the s_load_b96 result was routed into s[0:1] = (%arg1,
// %arg2). See the .ll fixture for the exact pin.
//
// MAINTENANCE: any change to the by_value decomposition in
// raiser.cpp (the `byValueSize > 8 && % 4 == 0` branch), to
// `extractKernargDword` in kernarg_layout.cpp, or to the SMEM
// kernarg-load branch in handle_smem.cpp must keep the i32 slot
// signature this fixture pins. An alternative lowering (e.g. a
// single `[16 x i8]` arg, an `i128` arg, or an aggregate-typed
// `{i32, i32, i32, i32}` arg) would silently change the kernarg
// byte-offset → IR-arg map that the rest of the SMEM/FLAT/MUBUF
// kernarg paths depend on.

#include <hip/hip_runtime.h>

struct Big16 {
  unsigned int a, b, c, d;
};

extern "C" __global__ void
s_load_b96_kernarg_kernel(Big16 desc, unsigned int *out) {
  // Inline asm template: explicit `s[0:2]` destination, explicit
  // `s[0:1]` base. Using literal SGPR names (rather than constraint
  // placeholders) is what forces the disassembly to land on exactly
  // `s_load_b96 s[0:2], s[0:1], 0x4` — the canonical Triton /
  // tensilelite shape this fixture exists to pin.
  unsigned int v0, v1, v2;
  asm volatile("s_load_b96 s[%[lo]:%[hi]], s[0:1], 4\n\t"
               "s_wait_kmcnt 0\n\t"
               : "=s"(v0), "=s"(v1), "=s"(v2)
               : [lo] "i"(0), [hi] "i"(2));

  // Plant the loaded dwords at observable global offsets so DCE
  // can't collapse them.
  out[0] = v0;
  out[1] = v1;
  out[2] = v2;

  // Reference `desc` so the by_value struct is alive at the IR
  // level. Without this, hipcc may DCE the entire kernarg slot,
  // collapsing the by_value back into a no-op kernarg layout that
  // wouldn't exercise the decomposition path we're trying to pin.
  out[3] = desc.a;
  out[4] = desc.b;
  out[5] = desc.c;
  out[6] = desc.d;
}
