// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "core/perfetto/sinks/trace_sink.hpp"

#include <utility>
#include <vector>

namespace rocprofsys::core
{
// Test-only sink: captures (source_id, bytes) tuples in arrival order so
// unit tests can assert the engine's drain contract without touching disk.
class recording_sink : public trace_sink
{
public:
    using record_t = std::pair<int, std::vector<char>>;

    void on_source_drained(int source_id, std::vector<char> bytes) override
    {
        m_records.emplace_back(source_id, std::move(bytes));
    }
    void finalize() override { m_finalized = true; }

    const std::vector<record_t>& records() const noexcept { return m_records; }
    bool                         finalized() const noexcept { return m_finalized; }

private:
    std::vector<record_t> m_records{};
    bool                  m_finalized{ false };
};
}  // namespace rocprofsys::core
