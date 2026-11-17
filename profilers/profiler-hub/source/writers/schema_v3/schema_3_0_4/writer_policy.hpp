// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

// 3.0.4 adds rocpd_hip_event. Kernel dispatch, memory copy, and PMC event are unchanged.

#include "writers/schema_v3/schema_3_0_3/writer_policy.hpp"
#include "writers/schema_v3/schema_3_0_4/hip_event_writer.hpp"

namespace profiler_hub
{

struct writer_policy_3_0_4 : writer_policy_3_0_3
{
    using hip_event_writer_t = hip_event_writer<data_storage::schema_3_0_4_tag>;
};

}  // namespace profiler_hub
