// Minimal stub for hip_fp16.h — CPU-only RCCL unit tests.
#pragma once
#include <cstdint>

struct __half {
    uint16_t __x;
};
using half = __half;

struct __half2 {
    __half x, y;
};

inline __half2 __halves2half2(__half a, __half b) {
    return {a, b};
}

inline __half2 __hadd2(__half2 a, __half2 b) {
    return {{static_cast<uint16_t>(a.x.__x + b.x.__x)},
            {static_cast<uint16_t>(a.y.__x + b.y.__x)}};
}
