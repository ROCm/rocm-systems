#pragma once

#define RCCL_API_TRACE_VERSION_MAJOR 0
#define RCCL_API_TRACE_VERSION_PATCH 0

#include <cstddef>
#include <cstdint>

extern "C" {
// fake rccl function
typedef int ncclResult_t;

enum ncclDataType_t
{
};

ncclResult_t
ncclGetVersion(int* version) __attribute__((visibility("default")));
}

namespace rccl
{
struct rcclApiFuncTable
{
    uint64_t                    size              = 0;
    decltype(::ncclGetVersion)* ncclGetVersion_fn = nullptr;
};

ncclResult_t
ncclGetVersion(int* version);

// populates rccl api table with function pointers
inline void
initialize_rccl_api_table(rcclApiFuncTable* dst)
{
    dst->size              = sizeof(rcclApiFuncTable);
    dst->ncclGetVersion_fn = &::rccl::ncclGetVersion;
}

// copies the api table from src to dst
inline void
copy_rccl_api_table(rcclApiFuncTable* dst, const rcclApiFuncTable* src)
{
    *dst = *src;
}
}  // namespace rccl
