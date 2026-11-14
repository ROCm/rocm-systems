// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "core/collection/sample_batch.hpp"
#include "core/collection/schema.hpp"

#include <cstdint>

namespace rocprofsys::collection
{
/** Where sources write. All methods may be called from any thread, concurrently. */
class sink_interface
{
public:
    sink_interface()                                 = default;
    sink_interface(const sink_interface&)            = delete;
    sink_interface& operator=(const sink_interface&) = delete;
    sink_interface(sink_interface&&)                 = delete;
    sink_interface& operator=(sink_interface&&)      = delete;
    virtual ~sink_interface()                        = default;

    /**
     * Registers a schema. Equal schemas get the same id (also across re-setup).
     * @throws std::invalid_argument for a schema without columns
     */
    [[nodiscard]] virtual schema_id register_schema(const schema& description) = 0;

    /** Never throws: a record the storage rejects is dropped and counted. */
    virtual void write_batch(const sample_batch& batch) noexcept = 0;

    /** A pause of all sources from begin_ns to end_ns (timeline ns). Never throws. */
    virtual void write_gap(std::uint64_t begin_ns, std::uint64_t end_ns) noexcept = 0;
};
}  // namespace rocprofsys::collection
