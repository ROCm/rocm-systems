// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

namespace profiler_hub::data_storage
{

struct schema_v3_tag
{};

// Tag type identifying the v4.0 rocpd schema (see read_statements_v4.hpp for the
// schema layout and reader_impl.cpp for version detection).
struct schema_v4_tag
{};

}  // namespace profiler_hub::data_storage
