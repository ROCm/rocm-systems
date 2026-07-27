// Minimal stub for amd_hip_bf16.h — CPU-only RCCL unit tests.
#pragma once
#include <cstdint>

struct __hip_bfloat16 {
    uint16_t data;

    __hip_bfloat16 operator+(const __hip_bfloat16& rhs) const {
        return {static_cast<uint16_t>(data + rhs.data)};
    }
};
