#pragma once

#define HIP_VERSION_MAJOR 1
#define HIP_VERSION_MINOR 0
#define HIP_VERSION_PATCH 0

#include <cstdint>

extern "C" {
// fake hip function
void
hip_init(void);
}

namespace hip
{
struct HipApiTable
{
    uint64_t size     = 0;
    void (*functor)() = nullptr;
};

// populates hip api table with function pointers
inline void
initialize_hip_api_table(HipApiTable* dst)
{
    dst->size    = sizeof(HipApiTable);
    dst->functor = &hip_init;
}

// copies the api table from src to dst
inline void
copy_hip_api_table(HipApiTable* dst, const HipApiTable* src)
{
    *dst = *src;
}
}  // namespace hip
