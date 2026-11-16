// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Standalone gfx942 hardware capture. Build with hipcc --offload-arch=gfx942 -O2
// and run with HIP_VISIBLE_DEVICES=0. Each JSON line retains every output word.
// A controls are swept with BLGP=0; B controls are swept with CBSZ=ABID=0.
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <hip/hip_runtime.h>

using Half4 = _Float16 __attribute__((ext_vector_type(4)));
using Short4 = short __attribute__((ext_vector_type(4)));
using Float4 = float __attribute__((ext_vector_type(4)));
using Float16 = float __attribute__((ext_vector_type(16)));
using Float32 = float __attribute__((ext_vector_type(32)));
using Int4 = int __attribute__((ext_vector_type(4)));
using Int16 = int __attribute__((ext_vector_type(16)));
using Int32 = int __attribute__((ext_vector_type(32)));

enum class Format { F32, F16, BF16, I8 };

void check(hipError_t result, const char *operation) {
  if (result != hipSuccess) {
    std::fprintf(stderr, "%s: %s\n", operation, hipGetErrorString(result));
    std::exit(1);
  }
}

template <Format F, int M, int Cbsz, int Abid, int Blgp, bool TagA>
__global__ void capture(float *out) {
  const unsigned lane = threadIdx.x;
  const float a = TagA ? float(lane + 1) : 1.0f;
  const float b = TagA ? 1.0f : float(lane + 1);
  if constexpr (F == Format::F32) {
    const float left = a, right = b;
    if constexpr (M == 4) {
      const auto result =
          __builtin_amdgcn_mfma_f32_4x4x1f32(left, right, Float4{}, Cbsz, Abid, Blgp);
      for (int reg = 0; reg < 4; ++reg)
        out[lane * 4 + reg] = float(result[reg]);
    }
    if constexpr (M == 16) {
      const auto result =
          __builtin_amdgcn_mfma_f32_16x16x1f32(left, right, Float16{}, Cbsz, Abid, Blgp);
      for (int reg = 0; reg < 16; ++reg)
        out[lane * 16 + reg] = float(result[reg]);
    }
    if constexpr (M == 32) {
      const auto result =
          __builtin_amdgcn_mfma_f32_32x32x1f32(left, right, Float32{}, Cbsz, Abid, Blgp);
      for (int reg = 0; reg < 32; ++reg)
        out[lane * 32 + reg] = float(result[reg]);
    }
  }
  if constexpr (F == Format::F16) {
    const Half4 left = {(_Float16)a, (_Float16)a, (_Float16)a, (_Float16)a};
    const Half4 right = {(_Float16)b, (_Float16)b, (_Float16)b, (_Float16)b};
    if constexpr (M == 4) {
      const auto result =
          __builtin_amdgcn_mfma_f32_4x4x4f16(left, right, Float4{}, Cbsz, Abid, Blgp);
      for (int reg = 0; reg < 4; ++reg)
        out[lane * 4 + reg] = float(result[reg]);
    }
    if constexpr (M == 16) {
      const auto result =
          __builtin_amdgcn_mfma_f32_16x16x4f16(left, right, Float16{}, Cbsz, Abid, Blgp);
      for (int reg = 0; reg < 16; ++reg)
        out[lane * 16 + reg] = float(result[reg]);
    }
    if constexpr (M == 32) {
      const auto result =
          __builtin_amdgcn_mfma_f32_32x32x4f16(left, right, Float32{}, Cbsz, Abid, Blgp);
      for (int reg = 0; reg < 32; ++reg)
        out[lane * 32 + reg] = float(result[reg]);
    }
  }
  if constexpr (F == Format::BF16) {
    const short av = short(__float_as_uint(a) >> 16), bv = short(__float_as_uint(b) >> 16);
    const Short4 left = {av, av, av, av}, right = {bv, bv, bv, bv};
    if constexpr (M == 4) {
      const auto result =
          __builtin_amdgcn_mfma_f32_4x4x4bf16_1k(left, right, Float4{}, Cbsz, Abid, Blgp);
      for (int reg = 0; reg < 4; ++reg)
        out[lane * 4 + reg] = float(result[reg]);
    }
    if constexpr (M == 16) {
      const auto result =
          __builtin_amdgcn_mfma_f32_16x16x4bf16_1k(left, right, Float16{}, Cbsz, Abid, Blgp);
      for (int reg = 0; reg < 16; ++reg)
        out[lane * 16 + reg] = float(result[reg]);
    }
    if constexpr (M == 32) {
      const auto result =
          __builtin_amdgcn_mfma_f32_32x32x4bf16_1k(left, right, Float32{}, Cbsz, Abid, Blgp);
      for (int reg = 0; reg < 32; ++reg)
        out[lane * 32 + reg] = float(result[reg]);
    }
  }
  if constexpr (F == Format::I8) {
    const int left = int(a) * 0x01010101, right = int(b) * 0x01010101;
    if constexpr (M == 4) {
      const auto result = __builtin_amdgcn_mfma_i32_4x4x4i8(left, right, Int4{}, Cbsz, Abid, Blgp);
      for (int reg = 0; reg < 4; ++reg)
        out[lane * 4 + reg] = float(result[reg]);
    }
    if constexpr (M == 16) {
      const auto result =
          __builtin_amdgcn_mfma_i32_16x16x4i8(left, right, Int16{}, Cbsz, Abid, Blgp);
      for (int reg = 0; reg < 16; ++reg)
        out[lane * 16 + reg] = float(result[reg]);
    }
    if constexpr (M == 32) {
      const auto result =
          __builtin_amdgcn_mfma_i32_32x32x4i8(left, right, Int32{}, Cbsz, Abid, Blgp);
      for (int reg = 0; reg < 32; ++reg)
        out[lane * 32 + reg] = float(result[reg]);
    }
  }
}

template <Format F, int M, int Cbsz, int Abid, int Blgp, bool TagA> void run(float *device) {
  constexpr int regs = M;
  constexpr int k = F == Format::F32 ? 1 : 4;
  const char *format = F == Format::F32    ? "F32"
                       : F == Format::F16  ? "F16"
                       : F == Format::BF16 ? "BF16"
                                           : "I8";
  for (int repeat = 0; repeat < 3; ++repeat) {
    check(hipMemset(device, 0xff, 64 * regs * sizeof(float)), "sentinel");
    capture<F, M, Cbsz, Abid, Blgp, TagA><<<1, 64>>>(device);
    check(hipGetLastError(), "launch");
    check(hipDeviceSynchronize(), "synchronize");
    float host[64 * regs];
    check(hipMemcpy(host, device, sizeof(host), hipMemcpyDeviceToHost), "readback");
    std::printf("{\"format\":\"%s\",\"m\":%d,\"tag\":\"%s\",\"cbsz\":%d,\"abid\":%d,\"blgp\":%d,"
                "\"repeat\":%d,\"values\":[",
                format, M, TagA ? "A" : "B", Cbsz, Abid, Blgp, repeat);
    for (int lane = 0; lane < 64; ++lane) {
      for (int reg = 0; reg < regs; ++reg) {
        // Invert the batched output layout without emulator layout helpers.
        const int batch = M == 4 ? lane / 4 : reg / (M * M / 64);
        const int row =
            M == 4 ? reg : (reg % (M * M / 64) / 4) * (256 / M) + 4 * (lane / M) + reg % 4;
        const int group = batch / (1 << Cbsz) * (1 << Cbsz);
        const int a_lane = (group + Abid) * M + row;
        const int b_lane = batch * M + lane % M;
        const int mapped_b = Blgp == 0   ? b_lane
                             : Blgp <= 2 ? b_lane % 32 + 32 * (Blgp - 1)
                             : Blgp == 3 ? (b_lane + 16) % 64
                                         : b_lane % 16 + 16 * (Blgp - 4);
        const float expected = float(k * (1 + (TagA ? a_lane : mapped_b)));
        const int index = lane * regs + reg;
        if (host[index] != expected) {
          std::fprintf(
              stderr,
              "mismatch %s M=%d A=%d CBSZ=%d ABID=%d BLGP=%d lane=%d reg=%d got=%g expected=%g\n",
              format, M, TagA, Cbsz, Abid, Blgp, lane, reg, double(host[index]), double(expected));
          std::exit(1);
        }
        std::printf("%s%.9g", index ? "," : "", double(host[index]));
      }
    }
    std::puts("]}");
  }
}

template <Format F, int M, int Cbsz, int Abid = 0> void broadcast_cases(float *device) {
  run<F, M, Cbsz, Abid, 0, true>(device);
  if constexpr (Abid + 1 < (1 << Cbsz))
    broadcast_cases<F, M, Cbsz, Abid + 1>(device);
}

template <Format F, int M, int Blgp = 0> void permutation_cases(float *device) {
  run<F, M, 0, 0, Blgp, false>(device);
  if constexpr (Blgp < 7)
    permutation_cases<F, M, Blgp + 1>(device);
}

template <Format F, int M> void shape(float *device) {
  broadcast_cases<F, M, 0>(device);
  broadcast_cases<F, M, 1>(device);
  if constexpr (M <= 16)
    broadcast_cases<F, M, 2>(device);
  if constexpr (M == 4) {
    broadcast_cases<F, M, 3>(device);
    broadcast_cases<F, M, 4>(device);
  }
  permutation_cases<F, M>(device);
}

template <Format F> void shapes(float *device) {
  shape<F, 4>(device);
  shape<F, 16>(device);
  shape<F, 32>(device);
}

int main() {
  int count = 0;
  check(hipGetDeviceCount(&count), "device count");
  if (count != 1)
    return 1;
  hipDeviceProp_t properties{};
  check(hipGetDeviceProperties(&properties, 0), "device properties");
  std::fprintf(stderr, "device=%s arch=%s warpSize=%d\n", properties.name, properties.gcnArchName,
               properties.warpSize);
  if (std::strncmp(properties.gcnArchName, "gfx942", 6) || properties.warpSize != 64)
    return 1;
  float *device = nullptr;
  check(hipMalloc(&device, 64 * 32 * sizeof(float)), "allocate");
  shapes<Format::F32>(device);
  shapes<Format::F16>(device);
  shapes<Format::BF16>(device);
  shapes<Format::I8>(device);
  check(hipFree(device), "free");
}
