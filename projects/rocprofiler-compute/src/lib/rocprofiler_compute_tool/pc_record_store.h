// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT
#pragma once

#include <cstdint>
#include <iterator>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

namespace rocprofiler_compute_tool
{
// A single decoded PC sample. Fields mirror the analyzer-read JSON plus parity
// fields carried for completeness. Enum-valued fields (stall_reason, inst_type)
// are stored as their SDK string names so the store stays free of SDK headers.
struct pc_sample_record_t
{
    // Program counter attribution.
    uint64_t code_object_id     = 0;
    uint64_t code_object_offset = 0;

    // Decoded snapshot / classification.
    std::string stall_reason{};  ///< SDK name; empty for host_trap samples
    int         wave_issued = 0;
    std::string inst_type{};  ///< SDK name

    uint64_t dispatch_id = 0;

    // Index into the instruction/comment string table, resolved by the writer
    // from code_object_id + code_object_offset. Defaults to -1 (unresolved).
    int inst_index = -1;
};

// Thread-safe collection of decoded PC samples, partitioned by sampling method.
class pc_record_store_t
{
public:
    void add_stochastic(const pc_sample_record_t& record)
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_stochastic.push_back(record);
    }

    void add_host_trap(const pc_sample_record_t& record)
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_host_trap.push_back(record);
    }

    // Batch inserts: a buffer callback delivers many samples at once, so append
    // a whole delivery under a single lock instead of one lock per record.
    void add_stochastic(std::vector<pc_sample_record_t>&& records)
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_stochastic.insert(m_stochastic.end(),
                            std::make_move_iterator(records.begin()),
                            std::make_move_iterator(records.end()));
    }

    void add_host_trap(std::vector<pc_sample_record_t>&& records)
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_host_trap.insert(m_host_trap.end(),
                           std::make_move_iterator(records.begin()),
                           std::make_move_iterator(records.end()));
    }

    const std::vector<pc_sample_record_t>& stochastic() const { return m_stochastic; }

    const std::vector<pc_sample_record_t>& host_trap() const { return m_host_trap; }

private:
    mutable std::mutex              m_mutex{};
    std::vector<pc_sample_record_t> m_stochastic{};
    std::vector<pc_sample_record_t> m_host_trap{};
};
}  // namespace rocprofiler_compute_tool
