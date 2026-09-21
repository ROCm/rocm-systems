// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "avail/records.hpp"

namespace rocprofsys::avail
{
std::string_view
to_string(source_id source) noexcept
{
    switch(source)
    {
        case source_id::rocprofiler_sdk: return "rocprofiler-sdk";
        case source_id::amd_smi: return "amd-smi";
    }
    return "unknown";
}
}  // namespace rocprofsys::avail
