#pragma once

#define ROCTX_VERSION_MAJOR 4
#define ROCTX_VERSION_MINOR 6
#define ROCTX_VERSION_PATCH 1

#include <cstdint>

extern "C" {
void
roctxRangePush(const char*) __attribute__((visibility("default")));
void
roctxRangePop(const char*) __attribute__((visibility("default")));
}

namespace roctx
{
struct ROCTxApiTable
{
    uint64_t                    size              = 0;
    decltype(::roctxRangePush)* roctxRangePush_fn = nullptr;
    decltype(::roctxRangePop)*  roctxRangePop_fn  = nullptr;
};

void
roctx_range_push(const char*);

void
roctx_range_pop(const char*);

// populates roctx api table with function pointers
inline void
initialize_roctx_api_table(ROCTxApiTable* dst)
{
    dst->size              = sizeof(ROCTxApiTable);
    dst->roctxRangePush_fn = &::roctx::roctx_range_push;
    dst->roctxRangePop_fn  = &::roctx::roctx_range_pop;
}

// copies the api table from src to dst
inline void
copy_roctx_api_table(ROCTxApiTable* dst, const ROCTxApiTable* src)
{
    *dst = *src;
}
}  // namespace roctx
