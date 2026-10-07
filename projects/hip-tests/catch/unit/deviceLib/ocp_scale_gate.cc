/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

/*
Coverage for the gfx1250-strict gating of the OCP scaled conversions.

Three preprocessor gates decide, per target, whether an entry point calls a
hardware builtin or the software fallback. They guard disjoint builtin sets,
measured on gfx1250-strict with __has_builtin:

                                       gfx1250  strict  gfx950  gfx942
  HIP_ENABLE_GFX1250_OCP_BUILTINS        40/40   40/40    0/40    0/40
  HIP_ENABLE_GFX1250_PK8_SCALE_BUILTINS    9/9     0/9     0/9     0/9
  HIP_ENABLE_GFX1250_PK16_SCALE_BUILTINS      6/6     0/6     0/6     0/6

The strict column for pk8 and pk16 is 0 only until the compiler carries
llvm/llvm-project#227426 and #227475, which move all fifteen cvt_scale_pk8_*
and cvt_scale_pk16_* opcodes off block16-cvt-scale-insts onto gfx1250-insts.
LCOMPILER-2841 established that all fifteen execute on gfx1250-strict; what
it lacks is the block16 half of the scale_sel range, not the opcodes.

A gate that disagrees with the compiler fails one of two ways, and they need
different checks:

  * gate 1, builtin unavailable -> the translation unit does not compile. A
    value comparison cannot see this: a TU that does not build produces
    nothing to compare, and a build failure drops the whole binary from the
    run rather than reporting a red test.
  * gate 0, builtin available   -> the software fallback is selected silently.
    It compiles, runs, and is merely slow. The __builtin_amdgcn_is_invocable
    sites with no else are worse still: they return zero-filled. Only a value
    check catches either.

So the expectation here is derived from the compiler, never hardcoded: probe
what the compiler offers for a target, then require the gate to agree. That
invariant holds before the LLVM changes, after them, and on every other
target, without this file needing an edit.

Compilation goes through hipRTC, which takes --offload-arch as an option, so
one binary checks both gfx1250 and gfx1250-strict regardless of which it runs
on. No strict hardware is required, and a compile failure is reported as a
test failure instead of a missing binary.

Two families of entry point are covered, and they reach hipRTC differently:

  __hip_cvt_*        34 sites in amd_hip_fp4.h, amd_hip_fp6.h, amd_hip_fp8.h.
                     hipRTC preloads these headers, so the source needs no
                     #include and no -I.
  __amd_cvt_*_scale  54 sites in amd_hip_ocp_fp.hpp. hipRTC does not preload
                     this header, so the source includes it and the compile
                     needs -I <hip include dir>, taken from ROCM_PATH. If that
                     is not set the case reports why instead of passing empty.

scale_sel is deliberately not varied. Every one of the 82 builtin call sites
in the headers passes a literal 0, which is block32 and valid on gfx1250-strict,
and no public entry point exposes the selector. The block16 values that
gfx1250-strict aliases are therefore unreachable through HIP. If a future
header plumbs scale_sel through to callers, that is when this file needs a range test.

Expected values are derived from the OCP field layout rather than tabulated,
so the reference is independent of the header. That matters because when a
gate selects the software path, the device and host fallbacks are the same
code - comparing them to each other would prove nothing.

The value cases cover fp4 and fp6 only. E2M1, E2M3 and E3M2 have no NaN or
Inf encoding, so every code is an exact comparison on every target. fp8 is
left out on purpose: its E5M2 software fallback swaps the largest finite
value with Inf (code 123 returns inf instead of 57344, 124 returns 65536
instead of Inf, and likewise negated). That defect is upstream, is shared
with the host path, and is unrelated to which path these gates select, so
asserting it here would make this file fail for someone else's bug.

The call lists below are generated from the declarations in the four headers.
Regenerate them when entry points are added or removed.
*/

#include <hip_test_common.hh>
#include <hip/hiprtc.h>
#include <hip/amd_detail/amd_hip_ocp_fp.hpp>

#include <cmath>
#include <cstdlib>
#include <string>
#include <vector>

#if !defined(HIP_ENABLE_GFX1250_PK8_SCALE_BUILTINS) || \
    !defined(HIP_ENABLE_GFX1250_PK16_SCALE_BUILTINS)
// Defaulting these to 0 would make the gate checks compare 0 == 0 and pass
// while testing nothing, so refuse to build instead.
#error "amd_hip_ocp_fp.hpp did not define the gfx1250 OCP gates"
#endif

namespace {

// ------------------------------------------------------- hipRTC compilation

struct Compile {
  hiprtcResult status;
  std::string log;
};

// Returns the HIP include directory, or an empty string if it cannot be
// determined. Only the __amd_cvt_*_scale case needs it.
std::string hipIncludeDir() {
  if (const char* p = std::getenv("ROCM_PATH")) return std::string(p) + "/include";
  if (const char* p = std::getenv("HIP_PATH")) return std::string(p) + "/include";
  return {};
}

Compile compileFor(const char* arch, const std::string& src, bool needsInclude) {
  Compile res{HIPRTC_ERROR_INTERNAL_ERROR, {}};
  hiprtcProgram prog{};
  if (hiprtcCreateProgram(&prog, src.c_str(), "ocp_scale.cu", 0, nullptr, nullptr) !=
      HIPRTC_SUCCESS) {
    return res;
  }
  const std::string archOpt = std::string("--offload-arch=") + arch;
  const std::string incOpt = std::string("-I") + hipIncludeDir();
  std::vector<const char*> opts{archOpt.c_str()};
  if (needsInclude) opts.push_back(incOpt.c_str());
  res.status = hiprtcCompileProgram(prog, static_cast<int>(opts.size()), opts.data());

  size_t logSize = 0;
  if (hiprtcGetProgramLogSize(prog, &logSize) == HIPRTC_SUCCESS && logSize > 1) {
    res.log.resize(logSize, '\0');
    hiprtcGetProgramLog(prog, &res.log[0]);
  }
  hiprtcDestroyProgram(&prog);
  return res;
}

// True when the compiler offers the scaled pk8/pk16 builtins for this target.
// Derived rather than assumed, so the gate checks stay correct across compiler
// versions.
bool compilerHasScaledConverts(const char* arch) {
  static const char* kProbe =
      "typedef unsigned int u32;\n"
      "typedef __attribute__((ext_vector_type(8))) _Float16 v8h;\n"
      "extern \"C\" __global__ void probe(v8h* o, u32 s, u32 sc) {\n"
      "  o[0] = __builtin_amdgcn_cvt_scale_pk8_f16_fp4(s, sc, 0);\n"
      "}\n";
  return compileFor(arch, kProbe, false).status == HIPRTC_SUCCESS;
}

// ------------------------------------------------- generated call lists

// 34 __hip_cvt_* entry points from amd_hip_fp4.h, amd_hip_fp6.h, amd_hip_fp8.h.
// hipRTC preloads those headers, so this needs no #include.
const char* kHipCvtEntryPoints =
    "extern \"C\" __global__ void touch_hip_cvt(unsigned* out) {\n"
    "  __hip_bfloat162_raw b2{}; __hip_bfloat16_raw b1{};\n"
    "  double2 d2{}; float2 f2{}; __half_raw h1{}; __half2_raw h2{};\n"
    "  __hip_fp8_storage_t s8{}; __hip_fp8x2_storage_t s16{};\n"
    "  unsigned acc = 0;\n"
    "  { auto r = __hip_cvt_bfloat16raw_to_fp4(b1, __HIP_E2M1, hipRoundNearest); acc += (unsigned)(*(const unsigned char*)&r); }\n"
    "  { auto r = __hip_cvt_bfloat16raw2_to_fp4x2(b2, __HIP_E2M1, hipRoundNearest); acc += (unsigned)(*(const unsigned char*)&r); }\n"
    "  { auto r = __hip_cvt_double_to_fp4(1.0, __HIP_E2M1, hipRoundNearest); acc += (unsigned)(*(const unsigned char*)&r); }\n"
    "  { auto r = __hip_cvt_double2_to_fp4x2(d2, __HIP_E2M1, hipRoundNearest); acc += (unsigned)(*(const unsigned char*)&r); }\n"
    "  { auto r = __hip_cvt_float_to_fp4(1.0f, __HIP_E2M1, hipRoundNearest); acc += (unsigned)(*(const unsigned char*)&r); }\n"
    "  { auto r = __hip_cvt_float2_to_fp4x2(f2, __HIP_E2M1, hipRoundNearest); acc += (unsigned)(*(const unsigned char*)&r); }\n"
    "  { auto r = __hip_cvt_fp4_to_halfraw(s8, __HIP_E2M1); acc += (unsigned)(*(const unsigned char*)&r); }\n"
    "  { auto r = __hip_cvt_fp4x2_to_halfraw2(s8, __HIP_E2M1); acc += (unsigned)(*(const unsigned char*)&r); }\n"
    "  { auto r = __hip_cvt_halfraw_to_fp4(h1, __HIP_E2M1, hipRoundNearest); acc += (unsigned)(*(const unsigned char*)&r); }\n"
    "  { auto r = __hip_cvt_halfraw2_to_fp4x2(h2, __HIP_E2M1, hipRoundNearest); acc += (unsigned)(*(const unsigned char*)&r); }\n"
    "  { auto r = __hip_cvt_bfloat16raw_to_fp6(b1, __HIP_E2M3, hipRoundNearest); acc += (unsigned)(*(const unsigned char*)&r); }\n"
    "  { auto r = __hip_cvt_bfloat16raw2_to_fp6x2(b2, __HIP_E2M3, hipRoundNearest); acc += (unsigned)(*(const unsigned char*)&r); }\n"
    "  { auto r = __hip_cvt_double_to_fp6(1.0, __HIP_E2M3, hipRoundNearest); acc += (unsigned)(*(const unsigned char*)&r); }\n"
    "  { auto r = __hip_cvt_double2_to_fp6x2(d2, __HIP_E2M3, hipRoundNearest); acc += (unsigned)(*(const unsigned char*)&r); }\n"
    "  { auto r = __hip_cvt_float_to_fp6(1.0f, __HIP_E2M3, hipRoundNearest); acc += (unsigned)(*(const unsigned char*)&r); }\n"
    "  { auto r = __hip_cvt_float2_to_fp6x2(f2, __HIP_E2M3, hipRoundNearest); acc += (unsigned)(*(const unsigned char*)&r); }\n"
    "  { auto r = __hip_cvt_fp6_to_halfraw(s8, __HIP_E2M3); acc += (unsigned)(*(const unsigned char*)&r); }\n"
    "  { auto r = __hip_cvt_fp6x2_to_halfraw2(s16, __HIP_E2M3); acc += (unsigned)(*(const unsigned char*)&r); }\n"
    "  { auto r = __hip_cvt_halfraw_to_fp6(h1, __HIP_E2M3, hipRoundNearest); acc += (unsigned)(*(const unsigned char*)&r); }\n"
    "  { auto r = __hip_cvt_halfraw2_to_fp6x2(h2, __HIP_E2M3, hipRoundNearest); acc += (unsigned)(*(const unsigned char*)&r); }\n"
    "  { auto r = __hip_cvt_float_to_fp8(1.0f, __HIP_SATFINITE, __HIP_E4M3); acc += (unsigned)(*(const unsigned char*)&r); }\n"
    "  { auto r = __hip_cvt_float2_to_fp8x2(f2, __HIP_SATFINITE, __HIP_E4M3); acc += (unsigned)(*(const unsigned char*)&r); }\n"
    "  { auto r = __hip_cvt_double_to_fp8(1.0, __HIP_SATFINITE, __HIP_E4M3); acc += (unsigned)(*(const unsigned char*)&r); }\n"
    "  { auto r = __hip_cvt_double2_to_fp8x2(d2, __HIP_SATFINITE, __HIP_E4M3); acc += (unsigned)(*(const unsigned char*)&r); }\n"
    "  { auto r = __hip_cvt_bfloat16raw_to_fp8(b1, __HIP_SATFINITE, __HIP_E4M3); acc += (unsigned)(*(const unsigned char*)&r); }\n"
    "  { auto r = __hip_cvt_bfloat16raw2_to_fp8x2(b2, __HIP_SATFINITE, __HIP_E4M3); acc += (unsigned)(*(const unsigned char*)&r); }\n"
    "  { auto r = __hip_cvt_fp8_to_halfraw(s8, __HIP_E4M3); acc += (unsigned)(*(const unsigned char*)&r); }\n"
    "  { auto r = __hip_cvt_fp8x2_to_halfraw2(s16, __HIP_E4M3); acc += (unsigned)(*(const unsigned char*)&r); }\n"
    "  { auto r = __hip_cvt_halfraw_to_fp8(h1, __HIP_SATFINITE, __HIP_E4M3); acc += (unsigned)(*(const unsigned char*)&r); }\n"
    "  { auto r = __hip_cvt_halfraw2_to_fp8x2(h2, __HIP_SATFINITE, __HIP_E4M3); acc += (unsigned)(*(const unsigned char*)&r); }\n"
    "  { auto r = __hip_cvt_double_to_e8m0(1.0, __HIP_SATFINITE, hipRoundNearest); acc += (unsigned)(*(const unsigned char*)&r); }\n"
    "  { auto r = __hip_cvt_float_to_e8m0(1.0f, __HIP_SATFINITE, hipRoundNearest); acc += (unsigned)(*(const unsigned char*)&r); }\n"
    "  { auto r = __hip_cvt_bfloat16raw_to_e8m0(b1, __HIP_SATFINITE, hipRoundNearest); acc += (unsigned)(*(const unsigned char*)&r); }\n"
    "  { auto r = __hip_cvt_e8m0_to_bf16raw(s8); acc += (unsigned)(*(const unsigned char*)&r); }\n"
    "  out[0] = acc;\n"
    "}\n";

// 54 __amd_cvt_*_scale entry points from amd_hip_ocp_fp.hpp, which hipRTC does
// not preload, so this includes it and the compile needs -I.
const char* kAmdScaleEntryPoints =
    "#include <hip/amd_detail/amd_hip_ocp_fp.hpp>\n"
    "extern \"C\" __global__ void touch_amd_scale(unsigned int* out) {\n"
    "  const __amd_scale_t s = static_cast<__amd_scale_t>(0);\n"
    "  unsigned int acc = 0;\n"
    "  { __amd_fp8_storage_t a0{}; auto r = __amd_cvt_fp8_to_float_scale(a0, __AMD_OCP_E4M3, s); acc += reinterpret_cast<const unsigned char*>(&r)[0]; }\n"
    "  { float a0{}; auto r = __amd_cvt_float_to_fp8_sr_scale(a0, __AMD_OCP_E4M3, 0u, s); acc += reinterpret_cast<const unsigned char*>(&r)[0]; }\n"
    "  { __amd_floatx2_storage_t a0{}; auto r = __amd_cvt_floatx2_to_fp4x2_sr_scale(a0, __AMD_OCP_E2M1, 0u, s); acc += reinterpret_cast<const unsigned char*>(&r)[0]; }\n"
    "  { __amd_fp4x2_storage_t a0{}; auto r = __amd_cvt_fp4x2_to_floatx2_scale(a0, __AMD_OCP_E2M1, s); acc += reinterpret_cast<const unsigned char*>(&r)[0]; }\n"
    "  { __amd_floatx2_storage_t a0{}; auto r = __amd_cvt_floatx2_to_fp4x2_scale(a0, __AMD_OCP_E2M1, s); acc += reinterpret_cast<const unsigned char*>(&r)[0]; }\n"
    "  { __amd_fp8x2_storage_t a0{}; auto r = __amd_cvt_fp8x2_to_floatx2_scale(a0, __AMD_OCP_E4M3, s); acc += reinterpret_cast<const unsigned char*>(&r)[0]; }\n"
    "  { __amd_floatx2_storage_t a0{}; auto r = __amd_cvt_floatx2_to_fp8x2_scale(a0, __AMD_OCP_E4M3, s); acc += reinterpret_cast<const unsigned char*>(&r)[0]; }\n"
    "  { __amd_bf16x32_storage_t a0{}; auto r = __amd_cvt_bf16x32_to_fp6x32_scale(a0, __AMD_OCP_E2M3, s); acc += reinterpret_cast<const unsigned char*>(&r)[0]; }\n"
    "  { __amd_bf16x16_storage_t a0{}; auto r = __amd_cvt_bf16x16_to_fp6x16_scale(a0, __AMD_OCP_E2M3, s); acc += reinterpret_cast<const unsigned char*>(&r)[0]; }\n"
    "  { __amd_fp16x32_storage_t a0{}; auto r = __amd_cvt_fp16x32_to_fp6x32_scale(a0, __AMD_OCP_E2M3, s); acc += reinterpret_cast<const unsigned char*>(&r)[0]; }\n"
    "  { __amd_fp16x16_storage_t a0{}; auto r = __amd_cvt_fp16x16_to_fp6x16_scale(a0, __AMD_OCP_E2M3, s); acc += reinterpret_cast<const unsigned char*>(&r)[0]; }\n"
    "  { __amd_fp8x2_storage_t a0{}; auto r = __amd_cvt_fp8x2_to_fp16x2_scale(a0, __AMD_OCP_E4M3, s); acc += reinterpret_cast<const unsigned char*>(&r)[0]; }\n"
    "  { __amd_fp8x8_storage_t a0{}; auto r = __amd_cvt_fp8x8_to_fp16x8_scale(a0, __AMD_OCP_E4M3, s); acc += reinterpret_cast<const unsigned char*>(&r)[0]; }\n"
    "  { __amd_fp8x2_storage_t a0{}; auto r = __amd_cvt_fp8x2_to_bf16x2_scale(a0, __AMD_OCP_E4M3, s); acc += reinterpret_cast<const unsigned char*>(&r)[0]; }\n"
    "  { __amd_fp8x8_storage_t a0{}; auto r = __amd_cvt_fp8x8_to_bf16x8_scale(a0, __AMD_OCP_E4M3, s); acc += reinterpret_cast<const unsigned char*>(&r)[0]; }\n"
    "  { __amd_fp6x32_storage_t a0{}; auto r = __amd_cvt_fp6x32_to_fp16x32_scale(a0, __AMD_OCP_E2M3, s); acc += reinterpret_cast<const unsigned char*>(&r)[0]; }\n"
    "  { __amd_fp6x32_storage_t a0{}; auto r = __amd_cvt_fp6x32_to_bf16x32_scale(a0, __AMD_OCP_E2M3, s); acc += reinterpret_cast<const unsigned char*>(&r)[0]; }\n"
    "  { __amd_fp6x32_storage_t a0{}; auto r = __amd_cvt_fp6x32_to_floatx32_scale(a0, __AMD_OCP_E2M3, s); acc += reinterpret_cast<const unsigned char*>(&r)[0]; }\n"
    "  { __amd_fp4x2_storage_t a0{}; auto r = __amd_cvt_fp4x2_to_fp16x2_scale(a0, __AMD_OCP_E2M1, s); acc += reinterpret_cast<const unsigned char*>(&r)[0]; }\n"
    "  { __amd_fp4x8_storage_t a0{}; auto r = __amd_cvt_fp4x8_to_fp16x8_scale(a0, __AMD_OCP_E2M1, s); acc += reinterpret_cast<const unsigned char*>(&r)[0]; }\n"
    "  { __amd_fp4x2_storage_t a0{}; auto r = __amd_cvt_fp4x2_to_bf16x2_scale(a0, __AMD_OCP_E2M1, s); acc += reinterpret_cast<const unsigned char*>(&r)[0]; }\n"
    "  { __amd_fp4x8_storage_t a0{}; auto r = __amd_cvt_fp4x8_to_bf16x8_scale(a0, __AMD_OCP_E2M1, s); acc += reinterpret_cast<const unsigned char*>(&r)[0]; }\n"
    "  { __amd_fp4x8_storage_t a0{}; auto r = __amd_cvt_fp4x8_to_floatx8_scale(a0, __AMD_OCP_E2M1, s); acc += reinterpret_cast<const unsigned char*>(&r)[0]; }\n"
    "  { __amd_floatx8_storage_t a0{}; auto r = __amd_cvt_floatx8_to_fp4x8_scale(a0, __AMD_OCP_E2M1, s); acc += reinterpret_cast<const unsigned char*>(&r)[0]; }\n"
    "  { __amd_fp16x2_storage_t a0{}; auto r = __amd_cvt_fp16x2_to_fp8x2_scale(a0, __AMD_OCP_E4M3, s); acc += reinterpret_cast<const unsigned char*>(&r)[0]; }\n"
    "  { __amd_bf16x2_storage_t a0{}; auto r = __amd_cvt_bf16x2_to_fp8x2_scale(a0, __AMD_OCP_E4M3, s); acc += reinterpret_cast<const unsigned char*>(&r)[0]; }\n"
    "  { __amd_bf16x8_storage_t a0{}; auto r = __amd_cvt_bf16x8_to_fp8x8_scale(a0, __AMD_OCP_E4M3, s); acc += reinterpret_cast<const unsigned char*>(&r)[0]; }\n"
    "  { __amd_fp8x8_storage_t a0{}; auto r = __amd_cvt_fp8x8_to_floatx8_scale(a0, __AMD_OCP_E4M3, s); acc += reinterpret_cast<const unsigned char*>(&r)[0]; }\n"
    "  { __amd_fp8_storage_t a0{}; auto r = __amd_cvt_fp8_to_fp16_scale(a0, __AMD_OCP_E4M3, s); acc += reinterpret_cast<const unsigned char*>(&r)[0]; }\n"
    "  { __amd_fp8_storage_t a0{}; auto r = __amd_cvt_fp8_to_bf16_scale(a0, __AMD_OCP_E4M3, s); acc += reinterpret_cast<const unsigned char*>(&r)[0]; }\n"
    "  { __amd_floatx16_storage_t a0{}; __amd_floatx16_storage_t a1{}; auto r = __amd_cvt_floatx16_floatx16_to_fp6x32_scale(a0, a1, __AMD_OCP_E2M3, s); acc += reinterpret_cast<const unsigned char*>(&r)[0]; }\n"
    "  { __amd_floatx32_storage_t a0{}; auto r = __amd_cvt_floatx32_to_fp6x32_scale(a0, __AMD_OCP_E2M3, s); acc += reinterpret_cast<const unsigned char*>(&r)[0]; }\n"
    "  { __amd_floatx16_storage_t a0{}; auto r = __amd_cvt_floatx16_to_fp6x16_scale(a0, __AMD_OCP_E2M3, s); acc += reinterpret_cast<const unsigned char*>(&r)[0]; }\n"
    "  { __amd_floatx16_storage_t a0{}; auto r = __amd_cvt_floatx16_to_fp6x16_sr_scale(a0, __AMD_OCP_E2M3, 0u, s); acc += reinterpret_cast<const unsigned char*>(&r)[0]; }\n"
    "  { __amd_fp6x16_storage_t a0{}; auto r = __amd_cvt_fp6x16_to_floatx16_scale(a0, __AMD_OCP_E2M3, s); acc += reinterpret_cast<const unsigned char*>(&r)[0]; }\n"
    "  { __amd_floatx32_storage_t a0{}; auto r = __amd_cvt_floatx32_to_fp6x32_sr_scale(a0, __AMD_OCP_E2M3, 0u, s); acc += reinterpret_cast<const unsigned char*>(&r)[0]; }\n"
    "  { __amd_fp16x32_storage_t a0{}; auto r = __amd_cvt_fp16x32_to_fp6x32_sr_scale(a0, __AMD_OCP_E2M3, 0u, s); acc += reinterpret_cast<const unsigned char*>(&r)[0]; }\n"
    "  { __amd_bf16x32_storage_t a0{}; auto r = __amd_cvt_bf16x32_to_fp6x32_sr_scale(a0, __AMD_OCP_E2M3, 0u, s); acc += reinterpret_cast<const unsigned char*>(&r)[0]; }\n"
    "  { __amd_bf16x2_storage_t a0{}; auto r = __amd_cvt_bf16x2_to_fp4x2_scale(a0, __AMD_OCP_E2M1, s); acc += reinterpret_cast<const unsigned char*>(&r)[0]; }\n"
    "  { __amd_bf16x8_storage_t a0{}; auto r = __amd_cvt_bf16x8_to_fp4x8_scale(a0, __AMD_OCP_E2M1, s); acc += reinterpret_cast<const unsigned char*>(&r)[0]; }\n"
    "  { __amd_fp16x2_storage_t a0{}; auto r = __amd_cvt_fp16x2_to_fp4x2_scale(a0, __AMD_OCP_E2M1, s); acc += reinterpret_cast<const unsigned char*>(&r)[0]; }\n"
    "  { __amd_fp16x8_storage_t a0{}; auto r = __amd_cvt_fp16x8_to_fp4x8_scale(a0, __AMD_OCP_E2M1, s); acc += reinterpret_cast<const unsigned char*>(&r)[0]; }\n"
    "  { __amd_floatx8_storage_t a0{}; auto r = __amd_cvt_floatx8_to_fp4x8_sr_scale(a0, __AMD_OCP_E2M1, 0u, s); acc += reinterpret_cast<const unsigned char*>(&r)[0]; }\n"
    "  { __amd_bf16x2_storage_t a0{}; auto r = __amd_cvt_bf16x2_to_fp4x2_sr_scale(a0, __AMD_OCP_E2M1, 0u, s); acc += reinterpret_cast<const unsigned char*>(&r)[0]; }\n"
    "  { __amd_bf16x8_storage_t a0{}; auto r = __amd_cvt_bf16x8_to_fp4x8_sr_scale(a0, __AMD_OCP_E2M1, 0u, s); acc += reinterpret_cast<const unsigned char*>(&r)[0]; }\n"
    "  { __amd_fp16x2_storage_t a0{}; auto r = __amd_cvt_fp16x2_to_fp4x2_sr_scale(a0, __AMD_OCP_E2M1, 0u, s); acc += reinterpret_cast<const unsigned char*>(&r)[0]; }\n"
    "  { __amd_fp16x8_storage_t a0{}; auto r = __amd_cvt_fp16x8_to_fp4x8_sr_scale(a0, __AMD_OCP_E2M1, 0u, s); acc += reinterpret_cast<const unsigned char*>(&r)[0]; }\n"
    "  { __amd_floatx8_storage_t a0{}; auto r = __amd_cvt_floatx8_to_fp8x8_sr_scale(a0, __AMD_OCP_E4M3, 0u, s); acc += reinterpret_cast<const unsigned char*>(&r)[0]; }\n"
    "  { __amd_fp16_storage_t a0{}; auto r = __amd_cvt_fp16_to_fp8_sr_scale(a0, __AMD_OCP_E4M3, 0u, s); acc += reinterpret_cast<const unsigned char*>(&r)[0]; }\n"
    "  { __amd_fp16x8_storage_t a0{}; auto r = __amd_cvt_fp16x8_to_fp8x8_sr_scale(a0, __AMD_OCP_E4M3, 0u, s); acc += reinterpret_cast<const unsigned char*>(&r)[0]; }\n"
    "  { __amd_bf16_storage_t a0{}; auto r = __amd_cvt_bf16_to_fp8_sr_scale(a0, __AMD_OCP_E4M3, 0u, s); acc += reinterpret_cast<const unsigned char*>(&r)[0]; }\n"
    "  { __amd_bf16x8_storage_t a0{}; auto r = __amd_cvt_bf16x8_to_fp8x8_sr_scale(a0, __AMD_OCP_E4M3, 0u, s); acc += reinterpret_cast<const unsigned char*>(&r)[0]; }\n"
    "  { __amd_fp16x8_storage_t a0{}; auto r = __amd_cvt_fp16x8_to_fp8x8_scale(a0, __AMD_OCP_E4M3, s); acc += reinterpret_cast<const unsigned char*>(&r)[0]; }\n"
    "  { __amd_floatx8_storage_t a0{}; auto r = __amd_cvt_floatx8_to_fp8x8_scale(a0, __AMD_OCP_E4M3, s); acc += reinterpret_cast<const unsigned char*>(&r)[0]; }\n"
    "  out[0] = acc;\n"
    "}\n";

// ---------------------------------------------------------- gate probe

constexpr int kPk8Count = 9;
constexpr int kPk16Count = 6;
constexpr int kPk8Gate = kPk8Count + kPk16Count;
constexpr int kPk16Gate = kPk8Gate + 1;
constexpr int kProbeSlots = kPk16Gate + 1;

const char* kPk8Names[kPk8Count] = {"pk8_f16_fp4",  "pk8_bf16_fp4", "pk8_f32_fp4",
                                    "pk8_f16_fp8",  "pk8_bf16_fp8", "pk8_f32_fp8",
                                    "pk8_f16_bf8",  "pk8_bf16_bf8", "pk8_f32_bf8"};
const char* kPk16Names[kPk16Count] = {"pk16_f16_fp6", "pk16_bf16_fp6", "pk16_f32_fp6",
                                      "pk16_f16_bf6", "pk16_bf16_bf6", "pk16_f32_bf6"};

#define INVOCABLE(b) (__builtin_amdgcn_is_invocable(b) ? 1 : 0)

__global__ void probeGates(int* out) {
  out[0] = INVOCABLE(__builtin_amdgcn_cvt_scale_pk8_f16_fp4);
  out[1] = INVOCABLE(__builtin_amdgcn_cvt_scale_pk8_bf16_fp4);
  out[2] = INVOCABLE(__builtin_amdgcn_cvt_scale_pk8_f32_fp4);
  out[3] = INVOCABLE(__builtin_amdgcn_cvt_scale_pk8_f16_fp8);
  out[4] = INVOCABLE(__builtin_amdgcn_cvt_scale_pk8_bf16_fp8);
  out[5] = INVOCABLE(__builtin_amdgcn_cvt_scale_pk8_f32_fp8);
  out[6] = INVOCABLE(__builtin_amdgcn_cvt_scale_pk8_f16_bf8);
  out[7] = INVOCABLE(__builtin_amdgcn_cvt_scale_pk8_bf16_bf8);
  out[8] = INVOCABLE(__builtin_amdgcn_cvt_scale_pk8_f32_bf8);
  out[9] = INVOCABLE(__builtin_amdgcn_cvt_scale_pk16_f16_fp6);
  out[10] = INVOCABLE(__builtin_amdgcn_cvt_scale_pk16_bf16_fp6);
  out[11] = INVOCABLE(__builtin_amdgcn_cvt_scale_pk16_f32_fp6);
  out[12] = INVOCABLE(__builtin_amdgcn_cvt_scale_pk16_f16_bf6);
  out[13] = INVOCABLE(__builtin_amdgcn_cvt_scale_pk16_bf16_bf6);
  out[14] = INVOCABLE(__builtin_amdgcn_cvt_scale_pk16_f32_bf6);
  out[kPk8Gate] = HIP_ENABLE_GFX1250_PK8_SCALE_BUILTINS;
  out[kPk16Gate] = HIP_ENABLE_GFX1250_PK16_SCALE_BUILTINS;
}

#undef INVOCABLE

// ----------------------------------------------------------- reference

// Decode an OCP code from its field layout. Covers E2M1, E2M3, E3M2 and the
// finite range of E4M3 / E5M2, which is every format this file exercises.
float decodeOcp(int code, int expBits, int mantBits, int bias) {
  const int mantMask = (1 << mantBits) - 1;
  const int expMask = (1 << expBits) - 1;
  const int mant = code & mantMask;
  const int exp = (code >> mantBits) & expMask;
  const float sign = (code >> (expBits + mantBits)) & 1 ? -1.0f : 1.0f;
  const float scale = static_cast<float>(mantMask + 1);
  if (exp == 0) return sign * (static_cast<float>(mant) / scale) * std::ldexp(1.0f, 1 - bias);
  return sign * (1.0f + static_cast<float>(mant) / scale) * std::ldexp(1.0f, exp - bias);
}

// __amd_scale_t is a raw power-of-two exponent, so 0 is a multiplier of 1.0.
constexpr __amd_scale_t kUnitScale = static_cast<__amd_scale_t>(0);

enum Dest { kToF32 = 0, kToF16, kToBf16, kDestCount };
const char* destName(int d) { return d == kToF32 ? "f32" : (d == kToF16 ? "f16" : "bf16"); }

__global__ void unpackFp4(float* out, int dest, int scaleExp) {
  const int code = static_cast<int>(threadIdx.x);
  const auto in = static_cast<__amd_fp4x2_storage_t>(code);
  const auto s = static_cast<__amd_scale_t>(scaleExp);
  if (dest == kToF32) {
    out[code] = __amd_cvt_fp4x2_to_floatx2_scale(in, __AMD_OCP_E2M1, s)[0];
  } else if (dest == kToF16) {
    out[code] = static_cast<float>(__amd_cvt_fp4x2_to_fp16x2_scale(in, __AMD_OCP_E2M1, s)[0]);
  } else {
    out[code] = static_cast<float>(__amd_cvt_fp4x2_to_bf16x2_scale(in, __AMD_OCP_E2M1, s)[0]);
  }
}

__global__ void unpackFp6(float* out, int dest, int interpret) {
  const int code = static_cast<int>(threadIdx.x);
  __amd_fp6x32_storage_t in{};
  // Element 0 occupies bits 0-5, so any code < 64 lands in byte 0.
  reinterpret_cast<unsigned char*>(&in)[0] = static_cast<unsigned char>(code);
  const auto how = interpret ? __AMD_OCP_E3M2 : __AMD_OCP_E2M3;
  if (dest == kToF32) {
    out[code] = __amd_cvt_fp6x32_to_floatx32_scale(in, how, kUnitScale)[0];
  } else if (dest == kToF16) {
    out[code] = static_cast<float>(__amd_cvt_fp6x32_to_fp16x32_scale(in, how, kUnitScale)[0]);
  } else {
    out[code] = static_cast<float>(__amd_cvt_fp6x32_to_bf16x32_scale(in, how, kUnitScale)[0]);
  }
}

template <typename Launch>
void collect(int codes, std::vector<float>& out, Launch launch) {
  float* d_out = nullptr;
  HIP_CHECK(hipMalloc(&d_out, codes * sizeof(float)));
  HIP_CHECK(hipMemset(d_out, 0, codes * sizeof(float)));
  launch(d_out);
  HIP_CHECK(hipGetLastError());
  HIP_CHECK(hipDeviceSynchronize());
  out.resize(codes);
  HIP_CHECK(hipMemcpy(out.data(), d_out, codes * sizeof(float), hipMemcpyDeviceToHost));
  HIP_CHECK(hipFree(d_out));
}

}  // namespace

/*
 * Every public entry point must build for both gfx1250 targets, or fail only
 * because the compiler genuinely lacks the builtins the gate selected.
 *
 * The __hip_cvt_* family needs no include: hipRTC preloads amd_hip_fp4.h,
 * amd_hip_fp6.h and amd_hip_fp8.h. It is therefore the case that runs
 * everywhere with no configuration.
 */
TEST_CASE("Unit_ocp_scale_hip_cvt_entry_points_compile") {
  for (const char* arch : {"gfx1250", "gfx1250-strict"}) {
    const auto res = compileFor(arch, kHipCvtEntryPoints, false);
    INFO("--offload-arch=" << arch << "\ncompile log:\n" << res.log);
    REQUIRE(res.status == HIPRTC_SUCCESS);
  }
}

/*
 * The __amd_cvt_*_scale family lives in amd_hip_ocp_fp.hpp, which hipRTC does
 * not preload, so this needs -I. Without ROCM_PATH the case cannot run, and
 * says so rather than passing on an empty include path.
 *
 * On a target where the gate selects hardware the compiler does not have, this
 * fails with "needs target feature block16-cvt-scale-insts". That is the gate
 * and the compiler disagreeing, which is what
 * Unit_ocp_scale_gates_match_compiler reports in detail.
 */
TEST_CASE("Unit_ocp_scale_amd_entry_points_compile") {
  const std::string inc = hipIncludeDir();
  if (inc.empty()) {
    WARN("Neither ROCM_PATH nor HIP_PATH is set, so amd_hip_ocp_fp.hpp cannot be "
         "located for hipRTC. Set ROCM_PATH to cover the __amd_cvt_*_scale entry points.");
    return;
  }
  for (const char* arch : {"gfx1250", "gfx1250-strict"}) {
    const auto res = compileFor(arch, kAmdScaleEntryPoints, true);
    INFO("--offload-arch=" << arch << "\n-I" << inc << "\ncompile log:\n" << res.log);
    REQUIRE(res.status == HIPRTC_SUCCESS);
  }
}

/*
 * Every one of the fifteen builtins must agree with the gate that guards it.
 *
 * Gate 1 with the builtin unavailable does not compile. Gate 0 with the
 * builtin available silently selects the software fallback, which is a
 * performance regression on the MX inference hot path and, for fp8 E5M2,
 * currently also a correctness one.
 */
TEST_CASE("Unit_ocp_scale_gates_match_compiler") {
  int* d_out = nullptr;
  HIP_CHECK(hipMalloc(&d_out, kProbeSlots * sizeof(int)));
  HIP_CHECK(hipMemset(d_out, 0xFF, kProbeSlots * sizeof(int)));

  probeGates<<<1, 1>>>(d_out);
  HIP_CHECK(hipGetLastError());
  HIP_CHECK(hipDeviceSynchronize());

  int v[kProbeSlots] = {};
  HIP_CHECK(hipMemcpy(v, d_out, sizeof(v), hipMemcpyDeviceToHost));
  HIP_CHECK(hipFree(d_out));

  const int pk8Gate = v[kPk8Gate];
  const int pk16Gate = v[kPk16Gate];
  INFO("HIP_ENABLE_GFX1250_PK8_SCALE_BUILTINS = " << pk8Gate);
  INFO("HIP_ENABLE_GFX1250_PK16_SCALE_BUILTINS   = " << pk16Gate);

  if (v[0] == 1 && pk8Gate == 0) {
    WARN("The compiler offers the pk8 scaled converts on this target but "
         "HIP_ENABLE_GFX1250_PK8_SCALE_BUILTINS still selects the software fallback. "
         "Enable it for this target in amd_hip_mx_common.h and amd_hip_ocp_fp.hpp. "
         "See LCOMPILER-2841 and llvm/llvm-project#227426.");
  }
  if (v[0] == 0 && pk8Gate == 1) {
    WARN("HIP_ENABLE_GFX1250_PK8_SCALE_BUILTINS selects the hardware path but the "
         "compiler does not offer the pk8 scaled converts for this target, so the "
         "headers will not build here. This needs a compiler carrying "
         "llvm/llvm-project#227426.");
  }
  if (v[kPk8Count] == 1 && pk16Gate == 0) {
    WARN("The compiler offers the pk16 scaled converts on this target but "
         "HIP_ENABLE_GFX1250_PK16_SCALE_BUILTINS still selects the software fallback. "
         "Enable it for this target in amd_hip_mx_common.h and amd_hip_ocp_fp.hpp. "
         "See LCOMPILER-2841 and llvm/llvm-project#227475.");
  }
  if (v[kPk8Count] == 0 && pk16Gate == 1) {
    WARN("HIP_ENABLE_GFX1250_PK16_SCALE_BUILTINS selects the hardware path but the "
         "compiler does not offer the pk16 scaled converts for this target, so the "
         "headers will not build here. This needs a compiler carrying "
         "llvm/llvm-project#227475.");
  }

  for (int i = 0; i < kPk8Count; ++i) {
    INFO(kPk8Names[i] << ": compiler " << v[i] << ", pk8 gate " << pk8Gate);
    REQUIRE(v[i] == pk8Gate);
  }
  for (int i = 0; i < kPk16Count; ++i) {
    INFO(kPk16Names[i] << ": compiler " << v[kPk8Count + i] << ", pk16 gate " << pk16Gate);
    REQUIRE(v[kPk8Count + i] == pk16Gate);
  }
}

/*
 * Whichever path each gate selected must compute correctly. E2M1, E2M3 and
 * E3M2 have no NaN or Inf encoding, so every code is an exact comparison, for
 * every destination type.
 */
TEST_CASE("Unit_ocp_scale_fp4_fp6_unpack_values") {
  SECTION("fp4 E2M1") {
    constexpr int kCodes = 16;
    for (int dest = 0; dest < kDestCount; ++dest) {
      std::vector<float> out;
      collect(kCodes, out, [&](float* d) { unpackFp4<<<1, kCodes>>>(d, dest, 0); });
      for (int code = 0; code < kCodes; ++code) {
        const float expected = decodeOcp(code, 2, 1, 1);
        INFO("fp4 E2M1 -> " << destName(dest) << " code " << code << ": expected " << expected
                            << ", got " << out[code]);
        REQUIRE(out[code] == expected);
      }
    }
  }

  SECTION("fp6 E2M3 and E3M2") {
    constexpr int kCodes = 64;
    for (int interpret = 0; interpret < 2; ++interpret) {
      for (int dest = 0; dest < kDestCount; ++dest) {
        std::vector<float> out;
        collect(kCodes, out, [&](float* d) { unpackFp6<<<1, kCodes>>>(d, dest, interpret); });
        for (int code = 0; code < kCodes; ++code) {
          // E2M3: 2 exp bits, 3 mantissa, bias 1.  E3M2: 3 exp, 2 mantissa, bias 3.
          const float expected = interpret ? decodeOcp(code, 3, 2, 3) : decodeOcp(code, 2, 3, 1);
          INFO("fp6 " << (interpret ? "E3M2" : "E2M3") << " -> " << destName(dest) << " code "
                      << code << ": expected " << expected << ", got " << out[code]);
          REQUIRE(out[code] == expected);
        }
      }
    }
  }
}

/*
 * The scale operand must actually be applied. Every other case here uses a
 * scale of 1.0, so a path that dropped the operand would pass them all.
 * fp4 E2M1 is exact in f16, bf16 and f32 for every exponent used here.
 */
TEST_CASE("Unit_ocp_scale_fp4_nonunit_scale") {
  constexpr int kCodes = 16;
  for (int scaleExp : {-4, -1, 1, 5}) {
    const float factor = std::ldexp(1.0f, scaleExp);
    for (int dest = 0; dest < kDestCount; ++dest) {
      std::vector<float> out;
      collect(kCodes, out, [&](float* d) { unpackFp4<<<1, kCodes>>>(d, dest, scaleExp); });
      for (int code = 0; code < kCodes; ++code) {
        const float expected = decodeOcp(code, 2, 1, 1) * factor;
        INFO("fp4 E2M1 -> " << destName(dest) << " code " << code << ", scale 2^" << scaleExp
                            << ": expected " << expected << ", got " << out[code]);
        REQUIRE(out[code] == expected);
      }
    }
  }
}
