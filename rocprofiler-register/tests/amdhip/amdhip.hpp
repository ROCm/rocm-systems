#pragma once

#define HIP_VERSION_MAJOR 6
#define HIP_VERSION_MINOR 0
#define HIP_VERSION_PATCH 1

#include <cstdint>

extern "C" {
// fake hip function
void
hip_init(void) __attribute__((visibility("default")));
}

namespace hip
{
struct HipApiTable
{
    uint64_t              size        = 0;
    decltype(::hip_init)* hip_init_fn = nullptr;
};

void
hip_init();

// populates hip api table with function pointers
inline void
initialize_hip_api_table(HipApiTable* dst)
{
    dst->size        = sizeof(HipApiTable);
    dst->hip_init_fn = &::hip::hip_init;
}

// copies the api table from src to dst
inline void
copy_hip_api_table(HipApiTable* dst, const HipApiTable* src)
{
    *dst = *src;
}
}  // namespace hip
