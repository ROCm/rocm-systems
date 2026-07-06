// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

namespace profiler_hub::data_storage
{

struct schema_v3_tag
{};

// v4.0 rocpd schema: rocpd_track is the universal identity anchor and timestamps
// are normalized through the rocpd_timestamp spine. Detection keys on the presence
// of the rocpd_timestamp_{uuid} table (introduced in v4.0, absent in v3).
struct schema_v4_tag
{};

}  // namespace profiler_hub::data_storage
