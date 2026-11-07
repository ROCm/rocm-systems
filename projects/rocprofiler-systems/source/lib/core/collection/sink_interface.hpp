// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "core/collection/record.hpp"
#include "core/collection/schema.hpp"

namespace rocprofsys::collection
{
class sink_interface
{
public:
    virtual ~sink_interface() = default;

    virtual schema_id register_schema(source_id source, const schema& stream) = 0;

    virtual void write_batch(const sample_batch& batch) noexcept = 0;
    virtual void write_gap(const gap_marker& gap) noexcept       = 0;
};
}  // namespace rocprofsys::collection
