// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "core/collection/schema.hpp"
#include "core/trace_cache/counter_samples.hpp"
#include "logger/debug.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <span>
#include <type_traits>
#include <unordered_set>
#include <utility>
#include <vector>

namespace rocprofsys::trace_cache
{
/**
 * Per-schema column slots, built once from the registry schemas before processing.
 * Single-threaded (one processor instance per consumer thread).
 */
template <typename Slot>
class counter_schema_table
{
public:
    counter_schema_table()  = default;
    ~counter_schema_table() = default;

    // Slots may view into m_schemas: a copy or move would leave them dangling
    counter_schema_table(const counter_schema_table&)            = delete;
    counter_schema_table& operator=(const counter_schema_table&) = delete;
    counter_schema_table(counter_schema_table&&)                 = delete;
    counter_schema_table& operator=(counter_schema_table&&)      = delete;

    /**
     * Replaces the content. `make_slot(schema, column)` is called once per column, after
     * the schemas are stored, so a slot may keep views into them.
     */
    template <typename MakeSlot>
        requires std::is_invocable_r_v<Slot, MakeSlot&, const collection::schema&,
                                       const collection::column_spec&>
    void build(std::vector<collection::schema> schemas, MakeSlot&& make_slot)
    {
        m_schemas = std::move(schemas);
        m_slots.clear();
        m_slots.reserve(m_schemas.size());
        m_warned.clear();
        for(const auto& description : m_schemas)
        {
            auto& slots = m_slots.emplace_back();
            slots.reserve(description.columns.size());
            for(const auto& column : description.columns)
            {
                slots.push_back(make_slot(description, column));
            }
        }
    }

    /**
     * Slots of the batch's schema; empty when the id is unknown or the value count
     * differs from the column count. Each problem is warned once per schema id.
     */
    [[nodiscard]] std::span<Slot> find(const counter_batch& batch)
    {
        if(batch.schema_id >= m_slots.size())
        {
            if(first_problem(batch.schema_id))
            {
                LOG_WARNING("Counter batch of unknown schema {} dropped",
                            batch.schema_id);
            }
            return {};
        }
        auto& slots = m_slots[batch.schema_id];
        if(batch.values.size() != slots.size())
        {
            if(first_problem(batch.schema_id))
            {
                LOG_WARNING(
                    "Counter batch of schema {} dropped: {} values for {} columns",
                    batch.schema_id, batch.values.size(), slots.size());
            }
            return {};
        }
        return slots;
    }

    /** Calls visit(slot, value) for each column of the batch that has a value (not
     * NaN); nothing when find() drops the batch. */
    template <typename VisitFn>
    void for_each_value(const counter_batch& batch, VisitFn&& visit)
    {
        const auto slots = find(batch);
        for(std::size_t col = 0; col < slots.size(); ++col)
        {
            if(!std::isnan(batch.values[col]))
            {
                visit(slots[col], batch.values[col]);
            }
        }
    }

    /** Calls visit(slot) for every slot of every schema. */
    template <typename VisitFn>
    void for_each_slot(VisitFn&& visit)
    {
        for(auto& slots : m_slots)
        {
            for(auto& slot : slots)
            {
                visit(slot);
            }
        }
    }

private:
    [[nodiscard]] bool first_problem(std::uint32_t schema_id)
    {
        return m_warned.insert(schema_id).second;
    }

    // never modified after build: slots may view it
    std::vector<collection::schema>   m_schemas;
    std::vector<std::vector<Slot>>    m_slots;
    std::unordered_set<std::uint32_t> m_warned;
};
}  // namespace rocprofsys::trace_cache
