// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

// 3.0.1 allows NIC. No table gained a column.

#include "writers/schema_v3/schema_3_0_1/info_registration_writer.hpp"
#include "writers/schema_v3/writer_policy.hpp"

namespace profiler_hub
{

struct writer_policy_3_0_1 : writer_policy_3_0_0
{
    using info_writer_t = info_registration_writer<data_storage::schema_3_0_1_tag>;
};

}  // namespace profiler_hub
