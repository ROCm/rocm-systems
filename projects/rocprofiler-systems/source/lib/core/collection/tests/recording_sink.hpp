// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "core/collection/sample_batch.hpp"
#include "core/collection/schema.hpp"
#include "core/collection/sink_interface.hpp"

#include <algorithm>
#include <cstdint>
#include <iterator>
#include <mutex>
#include <stdexcept>
#include <vector>

namespace rocprofsys::collection::testing
{
/** A batch copied out of the writer's row buffer. */
struct recorded_batch
{
    schema_id           schema    = 0;
    std::uint64_t       timestamp = 0;
    std::vector<double> values;
};

struct recorded_gap
{
    std::uint64_t begin_ns = 0;
    std::uint64_t end_ns   = 0;
};

/** Fake sink: keeps every schema, batch and gap. Equal schemas share an id, a schema
 * without columns throws, as the production sink does. Thread-safe. */
class recording_sink final : public sink_interface
{
public:
    [[nodiscard]] schema_id register_schema(const schema& description) override
    {
        if(description.columns.empty())
        {
            throw std::invalid_argument("recording_sink: schema without columns");
        }
        const std::lock_guard lock(m_mutex);
        const auto            found = std::ranges::find(m_schemas, description);
        if(found != m_schemas.end())
        {
            return static_cast<schema_id>(std::distance(m_schemas.begin(), found));
        }
        m_schemas.push_back(description);
        return static_cast<schema_id>(m_schemas.size() - 1);
    }

    void write_batch(const sample_batch& batch) noexcept override
    {
        const std::lock_guard lock(m_mutex);
        m_batches.push_back({ .schema    = batch.schema,
                              .timestamp = batch.timestamp,
                              .values = { batch.values.begin(), batch.values.end() } });
    }

    void write_gap(std::uint64_t begin_ns, std::uint64_t end_ns) noexcept override
    {
        const std::lock_guard lock(m_mutex);
        m_gaps.push_back({ .begin_ns = begin_ns, .end_ns = end_ns });
    }

    [[nodiscard]] std::vector<schema> schemas() const
    {
        const std::lock_guard lock(m_mutex);
        return m_schemas;
    }

    [[nodiscard]] std::vector<recorded_batch> batches() const
    {
        const std::lock_guard lock(m_mutex);
        return m_batches;
    }

    [[nodiscard]] std::vector<recorded_gap> gaps() const
    {
        const std::lock_guard lock(m_mutex);
        return m_gaps;
    }

    /** Batches written for one schema, in write order. */
    [[nodiscard]] std::vector<recorded_batch> batches_of(schema_id wanted) const
    {
        const std::lock_guard       lock(m_mutex);
        std::vector<recorded_batch> result;
        std::ranges::copy_if(
            m_batches, std::back_inserter(result),
            [wanted](const recorded_batch& item) { return item.schema == wanted; });
        return result;
    }

private:
    mutable std::mutex          m_mutex;
    std::vector<schema>         m_schemas;
    std::vector<recorded_batch> m_batches;
    std::vector<recorded_gap>   m_gaps;
};
}  // namespace rocprofsys::collection::testing
