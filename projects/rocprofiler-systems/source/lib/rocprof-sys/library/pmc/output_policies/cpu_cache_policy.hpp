// Copyright (c) 2018-2025 Advanced Micro Devices, Inc. All Rights Reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// with the Software without restriction, including without limitation the
// rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
// sell copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// * Redistributions of source code must retain the above copyright notice,
// this list of conditions and the following disclaimers.
//
// * Redistributions in binary form must reproduce the above copyright
// notice, this list of conditions and the following disclaimers in the
// documentation and/or other materials provided with the distribution.
//
// * Neither the names of Advanced Micro Devices, Inc. nor the names of its
// contributors may be used to endorse or promote products derived from
// this Software without specific prior written permission.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS WITH
// THE SOFTWARE.

#pragma once

#include "core/config.hpp"
#include "core/trace_cache/cache_manager.hpp"
#include "core/trace_cache/metadata_registry.hpp"
#include "library/pmc/cpu/types.hpp"
#include "library/pmc/cpu/sample.hpp"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <optional>
#include <set>
#include <vector>

namespace rocprofsys
{
namespace pmc
{
namespace output_policies
{

struct cpu_cache_policy
{
    static void initialize_category_metadata()
    {
        if(!get_use_cache_output()) return;
        trace_cache::get_metadata_registry().add_string(
            trait::name<category::cpu_freq>::value);
        trace_cache::get_metadata_registry().add_string(
            trait::name<category::cpu_load>::value);
    }

    static void initialize_cpu_tracks_metadata(const std::set<size_t>& cpu_ids)
    {
        if(!get_use_cache_output()) return;

        const auto thread_id = std::nullopt;

        for(const auto& cpu_id : cpu_ids)
        {
            trace_cache::get_metadata_registry().add_track(
                {trace_cache::info::format_track_name<category::cpu_freq>(cpu_id).c_str(),
                 thread_id, "{}"});
        }

        for(const auto& cpu_id : cpu_ids)
        {
            trace_cache::get_metadata_registry().add_track(
                {trace_cache::info::format_track_name<category::cpu_load>(cpu_id).c_str(),
                 thread_id, "{}"});
        }

        // Process-level tracks (not per-CPU)
        trace_cache::get_metadata_registry().add_track(
            {trait::name<category::process_page>::value, thread_id, "{}"});
        trace_cache::get_metadata_registry().add_track(
            {trait::name<category::process_virt>::value, thread_id, "{}"});
        trace_cache::get_metadata_registry().add_track(
            {trait::name<category::process_peak>::value, thread_id, "{}"});
        trace_cache::get_metadata_registry().add_track(
            {trait::name<category::process_context_switch>::value, thread_id, "{}"});
        trace_cache::get_metadata_registry().add_track(
            {trait::name<category::process_page_fault>::value, thread_id, "{}"});
        trace_cache::get_metadata_registry().add_track(
            {trait::name<category::process_user_mode_time>::value, thread_id, "{}"});
        trace_cache::get_metadata_registry().add_track(
            {trait::name<category::process_kernel_mode_time>::value, thread_id, "{}"});
    }

    static void initialize_cpu_pmc_metadata(const std::set<size_t>& cpu_ids, size_t dev_id)
    {
        if(!get_use_cache_output()) return;

        constexpr size_t      EVENT_CODE       = 0;
        constexpr size_t      INSTANCE_ID      = 0;
        constexpr const char* LONG_DESCRIPTION = "";
        constexpr const char* COMPONENT        = "";
        constexpr const char* BLOCK            = "";
        constexpr const char* EXPRESSION       = "";
        constexpr const char* MEMORY           = "MB";
        constexpr const char* TIME             = "sec";
        constexpr const char* TARGET_ARCH      = "CPU";

        for(const auto& cpu_id : cpu_ids)
        {
            trace_cache::get_metadata_registry().add_pmc_info(
                {agent_type::CPU, dev_id, TARGET_ARCH, EVENT_CODE, INSTANCE_ID,
                 trace_cache::info::format_track_name<category::cpu_freq>(cpu_id).c_str(),
                 "Frequency", trait::name<category::cpu_freq>::description,
                 LONG_DESCRIPTION, COMPONENT, "MHz",
                 rocprofsys::trace_cache::ABSOLUTE, BLOCK, EXPRESSION, 0, 0});

            trace_cache::get_metadata_registry().add_pmc_info(
                {agent_type::CPU, dev_id, TARGET_ARCH, EVENT_CODE, INSTANCE_ID,
                 trace_cache::info::format_track_name<category::cpu_load>(cpu_id).c_str(),
                 "Load", trait::name<category::cpu_load>::description,
                 LONG_DESCRIPTION, COMPONENT, "%",
                 rocprofsys::trace_cache::ABSOLUTE, BLOCK, EXPRESSION, 0, 0});
        }

        trace_cache::get_metadata_registry().add_pmc_info(
            {agent_type::CPU, dev_id, TARGET_ARCH, EVENT_CODE, INSTANCE_ID,
             trait::name<category::process_page>::value, "Memory Usage",
             trait::name<category::process_page>::description, LONG_DESCRIPTION,
             COMPONENT, MEMORY, rocprofsys::trace_cache::ABSOLUTE, BLOCK, EXPRESSION,
             0, 0});

        trace_cache::get_metadata_registry().add_pmc_info(
            {agent_type::CPU, dev_id, TARGET_ARCH, EVENT_CODE, INSTANCE_ID,
             trait::name<category::process_virt>::value, "Virtual Memory Usage",
             trait::name<category::process_virt>::description, LONG_DESCRIPTION,
             COMPONENT, MEMORY, rocprofsys::trace_cache::ABSOLUTE, BLOCK, EXPRESSION,
             0, 0});

        trace_cache::get_metadata_registry().add_pmc_info(
            {agent_type::CPU, dev_id, TARGET_ARCH, EVENT_CODE, INSTANCE_ID,
             trait::name<category::process_peak>::value, "Peak Memory",
             trait::name<category::process_peak>::description, LONG_DESCRIPTION,
             COMPONENT, MEMORY, rocprofsys::trace_cache::ABSOLUTE, BLOCK, EXPRESSION,
             0, 0});

        trace_cache::get_metadata_registry().add_pmc_info(
            {agent_type::CPU, dev_id, TARGET_ARCH, EVENT_CODE, INSTANCE_ID,
             trait::name<category::process_context_switch>::value, "Context Switches",
             trait::name<category::process_context_switch>::description,
             LONG_DESCRIPTION, COMPONENT, "",
             rocprofsys::trace_cache::ABSOLUTE, BLOCK, EXPRESSION, 0, 0});

        trace_cache::get_metadata_registry().add_pmc_info(
            {agent_type::CPU, dev_id, TARGET_ARCH, EVENT_CODE, INSTANCE_ID,
             trait::name<category::process_page_fault>::value, "Page Faults",
             trait::name<category::process_page_fault>::description,
             LONG_DESCRIPTION, COMPONENT, "",
             rocprofsys::trace_cache::ABSOLUTE, BLOCK, EXPRESSION, 0, 0});

        trace_cache::get_metadata_registry().add_pmc_info(
            {agent_type::CPU, dev_id, TARGET_ARCH, EVENT_CODE, INSTANCE_ID,
             trait::name<category::process_user_mode_time>::value, "User Time",
             trait::name<category::process_user_mode_time>::description,
             LONG_DESCRIPTION, COMPONENT, TIME,
             rocprofsys::trace_cache::ABSOLUTE, BLOCK, EXPRESSION, 0, 0});

        trace_cache::get_metadata_registry().add_pmc_info(
            {agent_type::CPU, dev_id, TARGET_ARCH, EVENT_CODE, INSTANCE_ID,
             trait::name<category::process_kernel_mode_time>::value, "Kernel Time",
             trait::name<category::process_kernel_mode_time>::description,
             LONG_DESCRIPTION, COMPONENT, TIME,
             rocprofsys::trace_cache::ABSOLUTE, BLOCK, EXPRESSION, 0, 0});
    }

    static void store_sample(const cpu::enabled_metrics& enabled_metrics,
                             const cpu::metrics& metrics,
                             uint64_t timestamp,
                             const std::vector<uint8_t>& serialized_freqs,
                             const std::vector<uint8_t>& serialized_loads)
    {
        if(!get_use_cache_output()) return;

        trace_cache::get_buffer_storage().store(trace_cache::cpu_pmc_sample{
            enabled_metrics, timestamp, metrics.process_data,
            serialized_freqs, serialized_loads});
    }
};

}  // namespace output_policies
}  // namespace pmc
}  // namespace rocprofsys
