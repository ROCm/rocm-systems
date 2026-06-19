// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "autoincrementer.hpp"

#include <cstddef>

namespace profiler_hub
{

using primary_key_t = size_t;

struct primary_key_providers
{
    [[nodiscard]] auto& process_info() { return m_process_info; }
    [[nodiscard]] auto& agent_info() { return m_agent_info; }
    [[nodiscard]] auto& pmc_info() { return m_pmc_info; }
    [[nodiscard]] auto& thread_info() { return m_thread_info; }
    [[nodiscard]] auto& stream_info() { return m_stream_info; }
    [[nodiscard]] auto& queue_info() { return m_queue_info; }
    [[nodiscard]] auto& track_info() { return m_track_info; }
    [[nodiscard]] auto& string_info() { return m_string_info; }
    [[nodiscard]] auto& event_data() { return m_event_data; }
    [[nodiscard]] auto& sample_data() { return m_sample_data; }
    [[nodiscard]] auto& region_data() { return m_region_data; }
    [[nodiscard]] auto& arg() { return m_arg; }
    [[nodiscard]] auto& pmc_event_data() { return m_pmc_event_data; }
    [[nodiscard]] auto& kernel_dispatch_data() { return m_kernel_dispatch_data; }
    [[nodiscard]] auto& memory_copy_data() { return m_memory_copy_data; }
    [[nodiscard]] auto& memory_alloc_data() { return m_memory_alloc_data; }

private:
    autoincrementer<primary_key_t> m_process_info;
    autoincrementer<primary_key_t> m_agent_info;
    autoincrementer<primary_key_t> m_pmc_info;
    autoincrementer<primary_key_t> m_thread_info;
    autoincrementer<primary_key_t> m_stream_info;
    autoincrementer<primary_key_t> m_queue_info;
    autoincrementer<primary_key_t> m_track_info;
    autoincrementer<primary_key_t> m_string_info;
    autoincrementer<primary_key_t> m_event_data;
    autoincrementer<primary_key_t> m_sample_data;
    autoincrementer<primary_key_t> m_region_data;
    autoincrementer<primary_key_t> m_arg;
    autoincrementer<primary_key_t> m_pmc_event_data;
    autoincrementer<primary_key_t> m_kernel_dispatch_data;
    autoincrementer<primary_key_t> m_memory_copy_data;
    autoincrementer<primary_key_t> m_memory_alloc_data;
};

}  // namespace profiler_hub
