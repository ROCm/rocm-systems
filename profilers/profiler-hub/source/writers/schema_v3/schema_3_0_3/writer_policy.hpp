// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

// 3.0.3 adds pmc_info.spm_support and PMC event sample_id plus location.

#include "writers/schema_v3/schema_3_0_2/writer_policy.hpp"
#include "writers/schema_v3/schema_3_0_3/info_registration_writer.hpp"
#include "writers/schema_v3/schema_3_0_3/pmc_event_writer.hpp"

namespace profiler_hub
{

struct writer_policy_3_0_3 : writer_policy_3_0_2
{
    using info_writer_t      = info_registration_writer<data_storage::schema_3_0_3_tag>;
    using pmc_event_writer_t = pmc_event_writer<data_storage::schema_3_0_3_tag>;
};

}  // namespace profiler_hub
