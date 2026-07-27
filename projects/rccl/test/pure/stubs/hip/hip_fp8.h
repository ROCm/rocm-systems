// Stub hip_fp8.h — minimal FP8 types for CPU-only RCCL tests.
// Types must be trivially constructible for union placement in rccl_float8.h.
#pragma once
#include <cstdint>

struct __hip_fp8_e4m3_fnuz {
  uint8_t __x;
  explicit operator float() const { return 0.0f; }
};
struct __hip_fp8_e5m2_fnuz {
  uint8_t __x;
  explicit operator float() const { return 0.0f; }
};
struct __hip_fp8_e4m3 {
  uint8_t __x;
  __hip_fp8_e4m3() = default;
  explicit __hip_fp8_e4m3(float) : __x(0) {}
  explicit operator float() const { return 0.0f; }
};
struct __hip_fp8_e5m2 {
  uint8_t __x;
  __hip_fp8_e5m2() = default;
  explicit __hip_fp8_e5m2(float) : __x(0) {}
  operator float() const { return 0.0f; }
};
