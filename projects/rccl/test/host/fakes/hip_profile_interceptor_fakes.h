/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef RCCL_TEST_HOST_HIP_PROFILE_INTERCEPTOR_FAKES_H_
#define RCCL_TEST_HOST_HIP_PROFILE_INTERCEPTOR_FAKES_H_

// Single source of truth for the HIP entry points that ROCm's profile runtime
// defines as compiler-rt INTERCEPTORs, and that the host-only microtests must
// therefore define themselves.
//
// WHY THIS EXISTS
// ---------------
// The micro targets build with -fprofile-instr-generate, which puts
// libclang_rt.profile-<arch>.a on the link line. ROCm's clang fork adds
// InstrProfilingPlatformROCm.cpp.o to that archive; that member DEFINES all 16
// names below. Any host-only target that leaves one of them undefined makes lld
// extract the member to resolve it, which in turn drags in __interception::*,
// __sanitizer_internal_mem* and __prof_rocm::* -- none of which ship in any
// archive on a -no-hip-rt link. The result is 7 undefined-symbol errors that
// name symbols nobody in RCCL has ever heard of. Archive member selection
// happens during symbol resolution, so -Wl,--gc-sections cannot prevent it.
//
// Defining every name here means no reference can select that member. Do not
// trim this list to "the ones we currently call": the point is that a future
// unit reaching a new launch path must not resurrect the failure.
//
// Invisible on a stock ROCm SDK: its libclang_rt.profile archive has no such
// member, so this only ever fails in TheRock CI.
//
// AUTHORITATIVE SOURCE
// --------------------
// llvm/llvm-project, compiler-rt/lib/profile/InstrProfilingPlatformROCm.cpp,
// the INTERCEPTOR(...) definitions (byte-identical on ROCm/llvm-project
// amd-staging). Latest commit touching it: ae5e065a1128 "[PGO][HIP] Support
// hipModuleLoad in offload PGO" (#211875). Re-derive with:
//   grep -n 'INTERCEPTOR(' compiler-rt/lib/profile/InstrProfilingPlatformROCm.cpp
//
// SHAPE
// -----
// List here, definitions in hip_fakes.cc. Deliberately NOT
// inline definitions in this header: an unreferenced inline function need not be
// emitted, the symbol would stay undefined, and the archive member would be
// extracted again -- a silent regression visible only in CI.
//
// Every one of the 16 returns hipError_t, so the return type is not encoded in
// the list. If upstream ever adds one that does not, add a return-type column.
//
// Parameter lists mirror <hip/hip_runtime_api.h>, <hip/hip_ext.h> and
// <hip/amd_detail/amd_hip_runtime_pt_api.h>. Default arguments (__dparm) are
// omitted: repeating them on the definition is ill-formed. Any drift against
// the HIP headers is a compile error in the .cc, which is the intended alarm.

// X(name, (parameter types))
#define RCCL_HIP_PROFILE_INTERCEPTORS(X)                                          \
  /* launch */                                                                    \
  X(hipLaunchKernel, (const void*, dim3, dim3, void**, size_t, hipStream_t))       \
  X(hipLaunchKernel_spt, (const void*, dim3, dim3, void**, size_t, hipStream_t))   \
  X(hipExtLaunchKernel, (const void*, dim3, dim3, void**, size_t, hipStream_t,     \
                         hipEvent_t, hipEvent_t, int))                            \
  X(hipLaunchKernelExC, (const hipLaunchConfig_t*, const void*, void**))           \
  X(hipLaunchCooperativeKernel,                                                    \
    (const void*, dim3, dim3, void**, unsigned int, hipStream_t))                  \
  X(hipLaunchCooperativeKernel_spt,                                                \
    (const void*, dim3, dim3, void**, uint32_t, hipStream_t))                      \
  X(hipLaunchCooperativeKernelMultiDevice, (hipLaunchParams*, int, unsigned int))  \
  X(hipExtLaunchMultiKernelMultiDevice, (hipLaunchParams*, int, unsigned int))     \
  /* module launch -- hipModuleLaunchKernel is the one enqueue.cc actually        \
     reaches, via CUCHECKGOTO(cuLaunchKernel(...)) at src/enqueue.cc:2253/:2339 */ \
  X(hipModuleLaunchKernel,                                                         \
    (hipFunction_t, unsigned int, unsigned int, unsigned int, unsigned int,        \
     unsigned int, unsigned int, unsigned int, hipStream_t, void**, void**))       \
  X(hipExtModuleLaunchKernel,                                                      \
    (hipFunction_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t,    \
     size_t, hipStream_t, void**, void**, hipEvent_t, hipEvent_t, uint32_t))       \
  /* graph launch */                                                              \
  X(hipGraphLaunch, (hipGraphExec_t, hipStream_t))                                 \
  X(hipGraphLaunch_spt, (hipGraphExec_t, hipStream_t))                             \
  /* module load / unload */                                                      \
  X(hipModuleLoad, (hipModule_t*, const char*))                                    \
  X(hipModuleLoadData, (hipModule_t*, const void*))                                \
  X(hipModuleLoadDataEx,                                                           \
    (hipModule_t*, const void*, unsigned int, hipJitOption*, void**))              \
  X(hipModuleUnload, (hipModule_t))

#endif  // RCCL_TEST_HOST_HIP_PROFILE_INTERCEPTOR_FAKES_H_
