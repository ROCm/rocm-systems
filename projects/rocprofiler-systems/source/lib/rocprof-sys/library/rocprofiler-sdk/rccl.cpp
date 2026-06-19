// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// All rccl template implementations are defined in rccl.hpp.
// This explicit instantiation stamps out the production specialisation.

#include "library/rocprofiler-sdk/rccl.hpp"
#include <cstdint>

#if ROCPROFILER_VERSION >= 600

namespace rocprofsys::rocprofiler_sdk
{

template void
tool_tracing_callback_rccl<backend>(std::uint32_t, backend::rccl_api_data*, std::uint64_t,
                                    std::uint64_t);
template std::uint32_t rccl_get_device_id<backend>(backend::nccl_comm_t);

}  // namespace rocprofsys::rocprofiler_sdk

#endif
