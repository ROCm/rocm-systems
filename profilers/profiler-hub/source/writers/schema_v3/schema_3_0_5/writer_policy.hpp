// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

// 3.0.5 moves the PMC location from rocpd_pmc_event onto rocpd_info_pmc.

#include "writers/schema_v3/schema_3_0_4/writer_policy.hpp"
#include "writers/schema_v3/schema_3_0_5/info_registration_writer.hpp"
#include "writers/schema_v3/schema_3_0_5/pmc_event_writer.hpp"

namespace profiler_hub
{

struct writer_policy_3_0_5 : writer_policy_3_0_4
{
    using info_writer_t      = info_registration_writer<data_storage::schema_3_0_5_tag>;
    using pmc_event_writer_t = pmc_event_writer<data_storage::schema_3_0_5_tag>;
};

}  // namespace profiler_hub
