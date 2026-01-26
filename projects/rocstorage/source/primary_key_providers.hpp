// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT

#pragma once

#include "autoincrementer.hpp"

namespace rocstorage
{

using primary_key = size_t;

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
    autoincrementer<primary_key> m_process_info{ "process_info" };
    autoincrementer<primary_key> m_agent_info{ "agent_info" };
    autoincrementer<primary_key> m_pmc_info{ "pmc_info" };
    autoincrementer<primary_key> m_thread_info{ "thread_info" };
    autoincrementer<primary_key> m_stream_info{ "stream_info" };
    autoincrementer<primary_key> m_queue_info{ "queue_info" };
    autoincrementer<primary_key> m_track_info{ "track_info" };
    autoincrementer<primary_key> m_string_info{ "string_info" };
    autoincrementer<primary_key> m_event_data{ "event_data" };
    autoincrementer<primary_key> m_sample_data{ "sample_data" };
    autoincrementer<primary_key> m_region_data{ "region_data" };
    autoincrementer<primary_key> m_arg{ "arg" };
    autoincrementer<primary_key> m_pmc_event_data{ "pmc_event_data" };
    autoincrementer<primary_key> m_kernel_dispatch_data{ "kernel_dispatch_data" };
    autoincrementer<primary_key> m_memory_copy_data{ "memory_copy_data" };
    autoincrementer<primary_key> m_memory_alloc_data{ "memory_alloc_data" };
};

}  // namespace rocstorage
