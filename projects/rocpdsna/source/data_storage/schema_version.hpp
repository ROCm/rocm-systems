// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT

#pragma once

namespace rocpdsna::data_storage
{

// =============================================================================
// Schema Version Tags (compile-time selection)
// =============================================================================

/// @brief Tag type for schema v3
struct schema_v3_tag
{};

/// @brief Tag type for schema v4 (latest schema)
struct schema_v4_tag
{};

}  // namespace rocpdsna::data_storage
