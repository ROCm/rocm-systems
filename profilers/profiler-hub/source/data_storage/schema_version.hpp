// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "profiler-hub/version.hpp"

namespace profiler_hub::data_storage
{

// rocpd_metadata.sql is stamped with this version; hardcoded to the baseline schema
// version for now until schema files are selected by version rather than always
// taking the latest cloned/bundled copy.
inline constexpr profiler_hub::version_t rocpd_schema_version{ 3, 0, 1 };

struct schema_v3_tag
{};

}  // namespace profiler_hub::data_storage
