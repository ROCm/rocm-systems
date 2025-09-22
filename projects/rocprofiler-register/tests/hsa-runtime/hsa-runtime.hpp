#pragma once

#define HSA_VERSION_MAJOR 2
#define HSA_VERSION_MINOR 1
#define HSA_VERSION_PATCH 0

#include <cstdint>

extern "C" {
// fake hsa function
void
hsa_init(void) __attribute__((visibility("default")));
}

namespace hsa
{
struct HsaApiTable
{
    uint64_t              size        = 0;
    decltype(::hsa_init)* hsa_init_fn = nullptr;
};

void
hsa_init();

// populates hsa api table with function pointers
inline void
initialize_hsa_api_table(HsaApiTable* dst)
{
    dst->size        = sizeof(HsaApiTable);
    dst->hsa_init_fn = &::hsa::hsa_init;
}

// copies the api table from src to dst
inline void
copy_hsa_api_table(HsaApiTable* dst, const HsaApiTable* src)
{
    *dst = *src;
}
}  // namespace hsa
