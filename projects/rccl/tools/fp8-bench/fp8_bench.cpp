/*
Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

/**
 * @file fp8_bench.cpp
 *
 * V1: execute FP8 hardware intrinsics on each GPU and report what works.
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>

#include <hip/hip_runtime.h>
#include <hip/hip_version.h>

#include "fp8_probe.h"

#define HIPCHECK(cmd)                                                          \
  do {                                                                         \
    hipError_t err = (cmd);                                                    \
    if (err != hipSuccess) {                                                   \
      std::cerr << "HIP error " << hipGetErrorString(err) << " at " << __FILE__ \
                << ":" << __LINE__ << "\n";                                    \
      std::exit(1);                                                            \
    }                                                                          \
  } while (0)

struct IntrinsicInfo {
  Fp8HwIntrinsic id;
  const char* name;
  const char* desc;
};

static void printUsage(const char* prog) {
  std::printf(
      "Usage: %s [-d DEVICE] [-h]\n\n"
      "Execute FP8 hardware intrinsics on each GPU and report results.\n"
      "Each intrinsic is run on the device; [yes] means it executed and passed\n"
      "a sanity check, [fail] means it ran but failed, [n/a] means not available\n"
      "for the compiled GPU target.\n\n"
      "Options:\n"
      "  -d DEVICE   Probe a single GPU (default: all GPUs)\n"
      "  -h          Show this help\n",
      prog);
}

static const char* testStatus(const Fp8HwTestResult& t) {
  if (!t.ran) return "n/a ";
  if (t.ok) return "yes";
  return "fail";
}

static void printTestLine(const Fp8ProbeResult& r, Fp8HwIntrinsic id,
                          const char* name, const char* desc) {
  const Fp8HwTestResult& t = r.tests[id];
  std::printf("  [%s] %-32s  %s\n", testStatus(t), name, desc);
}

static void printSection(const Fp8ProbeResult& r, const char* title,
                         const IntrinsicInfo* items, int n) {
  std::printf("%s\n", title);
  for (int i = 0; i < n; ++i)
    printTestLine(r, items[i].id, items[i].name, items[i].desc);
  std::printf("\n");
}

static void probeDevice(int device) {
  HIPCHECK(hipSetDevice(device));

  Fp8ProbeResult hResult{};
  Fp8ProbeResult* dResult = nullptr;
  HIPCHECK(hipMalloc(&dResult, sizeof(Fp8ProbeResult)));
  hipLaunchKernelGGL(fp8ProbeKernel, dim3(1), dim3(1), 0, 0, dResult);
  HIPCHECK(hipDeviceSynchronize());
  HIPCHECK(hipMemcpy(&hResult, dResult, sizeof(Fp8ProbeResult), hipMemcpyDeviceToHost));
  HIPCHECK(hipFree(dResult));

  hipDeviceProp_t prop;
  HIPCHECK(hipGetDeviceProperties(&prop, device));

  std::printf("================================================================\n");
  std::printf("GPU %d: %s (%s)\n\n", device, prop.name, prop.gcnArchName);

  static const IntrinsicInfo kTypes[] = {
    {HW_HIP_FP8_E4M3, "__hip_fp8_e4m3", "HIP FP8 e4m3 type + host/device convert"},
    {HW_HIP_FP8_E5M2, "__hip_fp8_e5m2", "HIP FP8 e5m2 (bf8) type + convert"},
    {HW_HIP_FP8_E4M3_FNUZ, "__hip_fp8_e4m3_fnuz", "e4m3 fnuz variant"},
    {HW_HIP_FP8_E5M2_FNUZ, "__hip_fp8_e5m2_fnuz", "e5m2 fnuz variant"},
  };

  static const IntrinsicInfo kUpcastE4m3[] = {
    {HW_CVT_PK_F32_FP8, "cvt_pk_f32_fp8", "e4m3/fp8x2 -> packed float32"},
    {HW_CVT_F32_FP8, "v_cvt_f32_fp8", "e4m3 scalar -> float32"},
    {HW_CVT_SCALEF32_PK_F16_FP8, "cvt_scalef32_pk_f16_fp8", "e4m3 -> packed f16 (scalef32)"},
  };

  static const IntrinsicInfo kUpcastE5m2[] = {
    {HW_CVT_PK_F32_BF8, "cvt_pk_f32_bf8", "e5m2/bf8x2 -> packed float32"},
    {HW_CVT_F32_BF8, "v_cvt_f32_bf8", "e5m2 scalar -> float32"},
    {HW_CVT_SCALEF32_PK_F16_BF8, "cvt_scalef32_pk_f16_bf8", "e5m2 -> packed f16 (scalef32)"},
  };

  static const IntrinsicInfo kDowncastE4m3[] = {
    {HW_CVT_PK_FP8_F32, "cvt_pk_fp8_f32", "packed float32 -> e4m3/fp8x2 (RNE)"},
    {HW_CVT_SR_FP8_F32, "cvt_sr_fp8_f32", "float32 -> e4m3 (stochastic rounding)"},
    {HW_CVT_SCALEF32_PK_FP8_F16, "cvt_scalef32_pk_fp8_f16", "packed f16 -> e4m3 (scalef32)"},
  };

  static const IntrinsicInfo kDowncastE5m2[] = {
    {HW_CVT_PK_BF8_F32, "cvt_pk_bf8_f32", "packed float32 -> e5m2/bf8x2 (RNE)"},
    {HW_CVT_SR_BF8_F32, "cvt_sr_bf8_f32", "float32 -> e5m2 (stochastic rounding)"},
    {HW_CVT_SCALEF32_PK_BF8_F16, "cvt_scalef32_pk_bf8_f16", "packed f16 -> e5m2 (scalef32)"},
  };

  static const IntrinsicInfo kArith[] = {
    {HW_V_PK_ADD_F32, "v_pk_add_f32", "packed float32 add"},
    {HW_V_PK_ADD_F16, "v_pk_add_f16", "packed float16 add"},
    {HW_FMED3F, "fmed3f", "float clip / median-of-3"},
  };

  static const IntrinsicInfo kFp8Add[] = {
    {HW_FP8_E4M3_ADD_PK1, "fp8_e4m3_add_pk1", "device add 1+2 via HW path (scalar)"},
    {HW_FP8_E4M3_ADD_PK2, "fp8_e4m3_add_pk2", "device add via HW path (packed fp8x2)"},
    {HW_FP8_E5M2_ADD_PK1, "fp8_e5m2_add_pk1", "device bf8 add 1+2 (scalar)"},
    {HW_FP8_E5M2_ADD_PK2, "fp8_e5m2_add_pk2", "device bf8 add (packed)"},
  };

  static const IntrinsicInfo kGfx12F16Pk2[] = {
    {HW_CVT_PK_F16_FP8, "cvt_pk_f16_fp8", "fp8x2 -> packed f16 (gfx1250)"},
    {HW_CVT_PK_F16_BF8, "cvt_pk_f16_bf8", "bf8x2 -> packed f16 (gfx1250)"},
    {HW_CVT_PK_FP8_F16, "cvt_pk_fp8_f16", "packed f16 -> fp8x2 (gfx1250)"},
    {HW_CVT_PK_BF8_F16, "cvt_pk_bf8_f16", "packed f16 -> bf8x2 (gfx1250)"},
    {HW_CVT_F16_FP8, "cvt_f16_fp8", "scalar fp8 -> f16 (gfx1250)"},
    {HW_CVT_F16_BF8, "cvt_f16_bf8", "scalar bf8 -> f16 (gfx1250)"},
    {HW_CVT_SR_FP8_F16, "cvt_sr_fp8_f16", "f16 -> fp8 stochastic (gfx1250)"},
    {HW_CVT_SR_BF8_F16, "cvt_sr_bf8_f16", "f16 -> bf8 stochastic (gfx1250)"},
  };

  static const IntrinsicInfo kPk8Scalef32DownE4m3[] = {
    {HW_CVT_SCALEF32_PK8_FP8_F16, "cvt_scalef32_pk8_fp8_f16", "f16x8 -> fp8x8 (scalef32)"},
    {HW_CVT_SCALEF32_PK8_FP8_F32, "cvt_scalef32_pk8_fp8_f32", "f32x8 -> fp8x8 (scalef32)"},
    {HW_CVT_SCALEF32_PK8_FP8_BF16, "cvt_scalef32_pk8_fp8_bf16", "bf16x8 -> fp8x8 (scalef32)"},
  };

  static const IntrinsicInfo kPk8Scalef32DownE5m2[] = {
    {HW_CVT_SCALEF32_PK8_BF8_F16, "cvt_scalef32_pk8_bf8_f16", "f16x8 -> bf8x8 (scalef32)"},
    {HW_CVT_SCALEF32_PK8_BF8_F32, "cvt_scalef32_pk8_bf8_f32", "f32x8 -> bf8x8 (scalef32)"},
    {HW_CVT_SCALEF32_PK8_BF8_BF16, "cvt_scalef32_pk8_bf8_bf16", "bf16x8 -> bf8x8 (scalef32)"},
  };

  static const IntrinsicInfo kPk8Scalef32SrDownE4m3[] = {
    {HW_CVT_SCALEF32_SR_PK8_FP8_F16, "cvt_scalef32_sr_pk8_fp8_f16", "f16x8 -> fp8x8 (scalef32+sr)"},
    {HW_CVT_SCALEF32_SR_PK8_FP8_F32, "cvt_scalef32_sr_pk8_fp8_f32", "f32x8 -> fp8x8 (scalef32+sr)"},
    {HW_CVT_SCALEF32_SR_PK8_FP8_BF16, "cvt_scalef32_sr_pk8_fp8_bf16", "bf16x8 -> fp8x8 (scalef32+sr)"},
  };

  static const IntrinsicInfo kPk8Scalef32SrDownE5m2[] = {
    {HW_CVT_SCALEF32_SR_PK8_BF8_F16, "cvt_scalef32_sr_pk8_bf8_f16", "f16x8 -> bf8x8 (scalef32+sr)"},
    {HW_CVT_SCALEF32_SR_PK8_BF8_F32, "cvt_scalef32_sr_pk8_bf8_f32", "f32x8 -> bf8x8 (scalef32+sr)"},
    {HW_CVT_SCALEF32_SR_PK8_BF8_BF16, "cvt_scalef32_sr_pk8_bf8_bf16", "bf16x8 -> bf8x8 (scalef32+sr)"},
  };

  static const IntrinsicInfo kPk8ScaleUpE4m3[] = {
    {HW_CVT_SCALE_PK8_F16_FP8, "cvt_scale_pk8_f16_fp8", "fp8x8 -> f16x8 (block scale)"},
    {HW_CVT_SCALE_PK8_F32_FP8, "cvt_scale_pk8_f32_fp8", "fp8x8 -> f32x8 (block scale)"},
    {HW_CVT_SCALE_PK8_BF16_FP8, "cvt_scale_pk8_bf16_fp8", "fp8x8 -> bf16x8 (block scale)"},
  };

  static const IntrinsicInfo kPk8ScaleUpE5m2[] = {
    {HW_CVT_SCALE_PK8_F16_BF8, "cvt_scale_pk8_f16_bf8", "bf8x8 -> f16x8 (block scale)"},
    {HW_CVT_SCALE_PK8_F32_BF8, "cvt_scale_pk8_f32_bf8", "bf8x8 -> f32x8 (block scale)"},
    {HW_CVT_SCALE_PK8_BF16_BF8, "cvt_scale_pk8_bf16_bf8", "bf8x8 -> bf16x8 (block scale)"},
  };

  printSection(hResult, "FP8 types (HIP)", kTypes, sizeof(kTypes) / sizeof(kTypes[0]));
  printSection(hResult, "Upcast — e4m3", kUpcastE4m3, sizeof(kUpcastE4m3) / sizeof(kUpcastE4m3[0]));
  printSection(hResult, "Upcast — e5m2", kUpcastE5m2, sizeof(kUpcastE5m2) / sizeof(kUpcastE5m2[0]));
  printSection(hResult, "Downcast — e4m3", kDowncastE4m3, sizeof(kDowncastE4m3) / sizeof(kDowncastE4m3[0]));
  printSection(hResult, "Downcast — e5m2", kDowncastE5m2, sizeof(kDowncastE5m2) / sizeof(kDowncastE5m2[0]));
  printSection(hResult, "gfx1250 f16 <-> fp8 (pk2/scalar)", kGfx12F16Pk2,
               sizeof(kGfx12F16Pk2) / sizeof(kGfx12F16Pk2[0]));
  printSection(hResult, "pk8 scalef32 downcast — e4m3", kPk8Scalef32DownE4m3,
               sizeof(kPk8Scalef32DownE4m3) / sizeof(kPk8Scalef32DownE4m3[0]));
  printSection(hResult, "pk8 scalef32 downcast — e5m2", kPk8Scalef32DownE5m2,
               sizeof(kPk8Scalef32DownE5m2) / sizeof(kPk8Scalef32DownE5m2[0]));
  printSection(hResult, "pk8 scalef32 sr downcast — e4m3", kPk8Scalef32SrDownE4m3,
               sizeof(kPk8Scalef32SrDownE4m3) / sizeof(kPk8Scalef32SrDownE4m3[0]));
  printSection(hResult, "pk8 scalef32 sr downcast — e5m2", kPk8Scalef32SrDownE5m2,
               sizeof(kPk8Scalef32SrDownE5m2) / sizeof(kPk8Scalef32SrDownE5m2[0]));
  printSection(hResult, "pk8 scale upcast — e4m3", kPk8ScaleUpE4m3,
               sizeof(kPk8ScaleUpE4m3) / sizeof(kPk8ScaleUpE4m3[0]));
  printSection(hResult, "pk8 scale upcast — e5m2", kPk8ScaleUpE5m2,
               sizeof(kPk8ScaleUpE5m2) / sizeof(kPk8ScaleUpE5m2[0]));
  printSection(hResult, "Arithmetic", kArith, sizeof(kArith) / sizeof(kArith[0]));
  printSection(hResult, "End-to-end FP8 add on GPU", kFp8Add, sizeof(kFp8Add) / sizeof(kFp8Add[0]));
}

int main(int argc, char** argv) {
  int device = -1;

  for (int i = 1; i < argc; ++i) {
    if (!std::strcmp(argv[i], "-h") || !std::strcmp(argv[i], "--help")) {
      printUsage(argv[0]);
      return 0;
    }
    if (!std::strcmp(argv[i], "-d") && i + 1 < argc) {
      device = std::atoi(argv[++i]);
    } else {
      std::cerr << "Unknown argument: " << argv[i] << "\n";
      printUsage(argv[0]);
      return 1;
    }
  }

  int nGpus = 0;
  HIPCHECK(hipGetDeviceCount(&nGpus));
  if (nGpus < 1) {
    std::cerr << "No HIP devices found\n";
    return 1;
  }

  std::printf("fp8-bench GPU hardware probe (HIP %d, %d GPU(s))\n", HIP_VERSION, nGpus);
  std::printf("Note: intrinsics are compiled for --offload-arch=native; rebuild if GPU arch differs.\n\n");

  if (device >= 0) {
    if (device >= nGpus) {
      std::cerr << "Invalid device " << device << " (have " << nGpus << " GPU(s))\n";
      return 1;
    }
    probeDevice(device);
  } else {
    for (int i = 0; i < nGpus; ++i)
      probeDevice(i);
  }

  return 0;
}
