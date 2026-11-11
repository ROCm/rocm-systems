// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

// 3.0.2 adds graph ids on kernel dispatch and memory copy, and rocpd_graph_launch.

#include "writers/schema_v3/schema_3_0_1/writer_policy.hpp"
#include "writers/schema_v3/schema_3_0_2/graph_launch_writer.hpp"
#include "writers/schema_v3/schema_3_0_2/kernel_dispatch_writer.hpp"
#include "writers/schema_v3/schema_3_0_2/memory_copy_writer.hpp"

namespace profiler_hub
{

struct writer_policy_3_0_2 : writer_policy_3_0_1
{
    using kernel_dispatch_writer_t =
        kernel_dispatch_writer<data_storage::schema_3_0_2_tag>;
    using memory_copy_writer_t  = memory_copy_writer<data_storage::schema_3_0_2_tag>;
    using graph_launch_writer_t = graph_launch_writer<data_storage::schema_3_0_2_tag>;
};

}  // namespace profiler_hub
