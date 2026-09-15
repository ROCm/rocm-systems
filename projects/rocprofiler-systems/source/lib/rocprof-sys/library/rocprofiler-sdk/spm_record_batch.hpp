// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "library/rocprofiler-sdk/spm_sample.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <limits>
#include <optional>
#include <span>
#include <stdexcept>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

namespace rocprofsys::rocprofiler_sdk::spm::detail
{
/// Decoded SPM records grouped by counter instance and hardware timestamp.
struct decoded_record_batch
{
    std::vector<counter_info>     counters;
    std::vector<timestamp_sample> samples;
};

namespace impl
{
template <typename RecordT, typename CounterDecoderT, std::uint32_t MaxSerializedCount>
class record_batch_builder
{
public:
    static_assert(std::is_invocable_r_v<std::optional<counter_info>, CounterDecoderT&,
                                        std::uint64_t>,
                  "CounterDecoderT must return optional counter metadata for an ID");
    static_assert(MaxSerializedCount > 0, "The serialized count limit must be positive");

    explicit record_batch_builder(CounterDecoderT decoder)
    : m_decode_counter{ std::move(decoder) }
    {}

    [[nodiscard]] decoded_record_batch build(std::span<const RecordT* const> records) &&
    {
        resolve_records(records);
        // Materialization reads builder state, so finish it before moving counters.
        auto samples = materialize_samples();
        return { .counters = std::move(m_counters), .samples = std::move(samples) };
    }

private:
    struct resolved_record
    {
        std::uint32_t sample_index       = 0;
        std::uint32_t counter_info_index = 0;
        double        value              = 0.0;
    };

    struct timestamp_slot
    {
        std::uint64_t timestamp   = 0;
        std::size_t   value_count = 0;
    };

    [[nodiscard]] std::optional<std::uint32_t> intern_counter_info(
        std::uint64_t instance_id)
    {
        auto [counter_itr, inserted] = m_counter_info_indices.try_emplace(instance_id, 0);
        if(!inserted)
        {
            return counter_itr->second;
        }

        const std::optional<counter_info> info = m_decode_counter(instance_id);
        if(!info)
        {
            m_counter_info_indices.erase(counter_itr);
            return std::nullopt;
        }
        if(m_counters.size() >= MaxSerializedCount)
        {
            throw std::length_error{ "SPM counter count exceeds the serialized limit" };
        }

        const auto counter_info_index = static_cast<std::uint32_t>(m_counters.size());
        counter_itr->second           = counter_info_index;
        m_counters.emplace_back(*info);
        return counter_info_index;
    }

    [[nodiscard]] std::uint32_t intern_timestamp(std::uint64_t timestamp)
    {
        auto [sample_itr, inserted] = m_sample_indices.try_emplace(timestamp, 0);
        if(!inserted)
        {
            return sample_itr->second;
        }
        if(m_timestamp_slots.size() >= MaxSerializedCount)
        {
            throw std::length_error{ "SPM sample count exceeds the serialized limit" };
        }

        const auto sample_index = static_cast<std::uint32_t>(m_timestamp_slots.size());
        sample_itr->second      = sample_index;
        m_timestamp_slots.push_back(timestamp_slot{ .timestamp = timestamp });
        return sample_index;
    }

    void count_sample_value(std::uint32_t sample_index)
    {
        // Counting first enables exact per-sample reservation during materialization.
        auto& value_count = m_timestamp_slots[sample_index].value_count;
        if(value_count >= MaxSerializedCount)
        {
            throw std::length_error{
                "SPM values per sample exceed the serialized limit"
            };
        }
        ++value_count;
    }

    void resolve_record(const RecordT& record)
    {
        const auto counter_info_index = intern_counter_info(record.id);
        if(!counter_info_index)
        {
            return;
        }

        const auto sample_index = intern_timestamp(record.timestamp);
        count_sample_value(sample_index);
        m_resolved_records.push_back(
            resolved_record{ .sample_index       = sample_index,
                             .counter_info_index = *counter_info_index,
                             .value              = record.value });
    }

    void resolve_records(std::span<const RecordT* const> records)
    {
        m_resolved_records.reserve(records.size());
        for(const auto* record : records)
        {
            if(record != nullptr)
            {
                resolve_record(*record);
            }
        }
    }

    [[nodiscard]] std::vector<timestamp_sample> materialize_samples() const
    {
        auto samples = std::vector<timestamp_sample>{};
        samples.reserve(m_timestamp_slots.size());
        std::ranges::transform(
            m_timestamp_slots, std::back_inserter(samples), [](const auto& slot) {
                auto sample =
                    timestamp_sample{ .timestamp = slot.timestamp, .values = {} };
                sample.values.reserve(slot.value_count);
                return sample;
            });

        for(const auto& record : m_resolved_records)
        {
            samples[record.sample_index].values.emplace_back(counter_value{
                .counter_info_index = record.counter_info_index, .value = record.value });
        }
        return samples;
    }

    CounterDecoderT                                  m_decode_counter;
    std::vector<counter_info>                        m_counters;
    std::vector<timestamp_slot>                      m_timestamp_slots;
    std::vector<resolved_record>                     m_resolved_records;
    std::unordered_map<std::uint64_t, std::uint32_t> m_counter_info_indices;
    std::unordered_map<std::uint64_t, std::uint32_t> m_sample_indices;
};
}  // namespace impl

/**
 * Groups records by timestamp while preserving first-success counter order,
 * first-valid timestamp order, and record order within each sample. The decoder is
 * invoked synchronously and may be retried for a later record after a failed decode.
 *
 * @param records Callback-owned record pointers consumed synchronously; null entries
 * are skipped.
 * @param decoder Maps a counter-instance ID to counter metadata.
 * @return Decoded counters and timestamp-grouped values with capacity requested from
 * each timestamp's exact decoded value count.
 * @throws std::length_error if the configured element-count limit is exceeded.
 * Exceptions raised by the decoder are propagated unchanged.
 * @tparam MaxSerializedCount Maximum representable count. It defaults to the wire
 * format's 32-bit limit and can be reduced to exercise limit handling in tests.
 * @tparam RecordT Record type providing `id`, `timestamp`, and `value` fields.
 * @tparam CounterDecoderT Callable returning optional counter metadata for an ID.
 */
template <std::uint32_t MaxSerializedCount = std::numeric_limits<std::uint32_t>::max(),
          typename RecordT, typename CounterDecoderT>
[[nodiscard]] decoded_record_batch
build_record_batch(std::span<const RecordT* const> records, CounterDecoderT decoder)
{
    return impl::record_batch_builder<RecordT, CounterDecoderT, MaxSerializedCount>{
        std::move(decoder)
    }
        .build(records);
}
}  // namespace rocprofsys::rocprofiler_sdk::spm::detail
