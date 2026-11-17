// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

namespace profiler_hub::data_storage
{

struct schema_v3_tag
{};

// One tag per schema whose writer differs from the previous schema.
// A version with no tag reuses the previous writer.
struct schema_3_0_1_tag
{};
struct schema_3_0_2_tag
{};
struct schema_3_0_3_tag
{};
struct schema_3_0_4_tag
{};
struct schema_3_0_5_tag
{};

}  // namespace profiler_hub::data_storage
