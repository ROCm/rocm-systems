// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "core/trace_cache/cpu_snap_decoder.hpp"
#include "core/trace_cache/sample_type.hpp"
#include "logger/debug.hpp"
#include "rocprofiler-systems/categories.h"

#include <cpu_snap/core/module_map.hpp>
#include <cpu_snap/core/reader.hpp>
#include <cpu_snap/core/records.hpp>
#include <cpu_snap/processing/symbols/symbolizer.hpp>

#include <fmt/format.h>
#include <nlohmann/json.hpp>
#include <nlohmann/json_fwd.hpp>

#include <algorithm>
#include <cstdint>
#include <memory>
#include <ranges>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

namespace rocprofsys::trace_cache
{
namespace
{
// Name of category::timer_sampling (core/categories.hpp), which the perfetto processor
// matches to pick the track category.
constexpr std::string_view k_timer_sampling_category = "timer_sampling";

// A frame's slice name and its call_stack JSON, as the old sampler writes them
// (library/sampling.cpp, generate_call_stack_json).
struct frame_strings
{
    std::string name;
    std::string call_stack;
};
}  // namespace

struct cpu_snap_decoder::impl
{
    explicit impl(emit_fn emit_in)
    : emit{ std::move(emit_in) }
    {
        reader.on_event([this](const ::cpu_snap::core::record_variant& record) {
            on_record(record);
        });
    }

    void on_record(const ::cpu_snap::core::record_variant& record)
    {
        using ::cpu_snap::core::record_type;

        const auto type = ::cpu_snap::core::type_of(record);
        switch(type)
        {
            case record_type::sample:
                on_sample(std::get<::cpu_snap::core::sample_record>(record));
                return;
            case record_type::lost:
                lost_samples +=
                    std::get<::cpu_snap::core::lost_record>(record).lost_count_delta;
                return;
            case record_type::maps_init:
                modules.apply(std::get<::cpu_snap::core::maps_init_record>(record));
                return;
            case record_type::maps_delta:
                modules.apply(std::get<::cpu_snap::core::maps_delta_record>(record));
                return;
            // The reader keeps the frame trie itself; the rest carries no slice.
            case record_type::new_node:
            case record_type::checkpoint:
            case record_type::eof: return;
            default:
                throw std::logic_error(
                    fmt::format("cpu_snap::core::record_type {} not handled",
                                static_cast<int>(type)));
        }
    }

    void on_sample(const ::cpu_snap::core::sample_record& sample)
    {
        auto const [previous, is_first] =
            previous_timestamp.try_emplace(sample.tid, sample.timestamp_ns);
        auto const start = std::exchange(previous->second, sample.timestamp_ns);
        // Leaf 0 is an intentionally empty stack.
        if(is_first || sample.leaf_node_id == 0)
        {
            return;
        }

        auto const frames = reader.resolve_stack(sample.leaf_node_id);
        if(frames.empty())
        {
            ++unresolved_samples;
            return;
        }

        symbolize(frames, sample.timestamp_ns);

        auto const track_name = fmt::format("Thread {}", sample.tid);
        for(auto const address : frames | std::views::reverse)
        {
            auto const& frame = frame_cache.at(address);
            emit(backtrace_region_sample{
                static_cast<std::uint32_t>(ROCPROFSYS_CATEGORY_TIMER_SAMPLING),
                sample.tid, track_name, frame.name, start, sample.timestamp_ns,
                k_timer_sampling_category, frame.call_stack, std::string_view{}, "{}" });
        }
    }

    // Names the addresses not seen before, each against the modules loaded when this
    // sample was taken. Later samples reuse the cached strings.
    void symbolize(const std::vector<std::uint64_t>& frames, std::uint64_t timestamp_ns)
    {
        pending.clear();
        for(auto const address : frames)
        {
            if(!frame_cache.contains(address) &&
               std::ranges::find(pending, address, &address_at::first) == pending.end())
            {
                pending.emplace_back(address, timestamp_ns);
            }
        }
        if(pending.empty())
        {
            return;
        }

        symbols.preload_at(pending);
        for(auto const& [address, timestamp] : pending)
        {
            auto name = std::string{ symbols.resolve(address) };
            if(name.empty())
            {
                name = fmt::format("0x{:x}", address);
            }
            auto const     module = modules.resolve(address, timestamp);
            nlohmann::json call_stack;
            call_stack["name"] = name;
            call_stack["pc"]   = fmt::format("{:X}", address);
            call_stack["file"] = module ? module->path : std::string{};
            frame_cache.emplace(address,
                                frame_strings{ .name       = std::move(name),
                                               .call_stack = call_stack.dump() });
        }
    }

    using address_at = std::pair<std::uint64_t, std::uint64_t>;

    emit_fn                  emit;
    ::cpu_snap::core::reader reader;
    // Modules loaded over time, from the stream's maps records; read by the symbolizer.
    ::cpu_snap::core::module_map                modules;
    ::cpu_snap::processing::symbols::symbolizer symbols{
        modules,
        ::cpu_snap::processing::symbols::symbolizer_config{ .include_binary_name = false }
    };
    std::unordered_map<std::uint64_t, frame_strings> frame_cache;
    // Addresses of the current sample that still need a name, with its timestamp.
    std::vector<address_at> pending;
    // Timestamp of each thread's previous sample, by tid.
    std::unordered_map<std::uint32_t, std::uint64_t> previous_timestamp;
    std::uint64_t                                    lost_samples       = 0;
    std::uint64_t                                    unresolved_samples = 0;
};

cpu_snap_decoder::cpu_snap_decoder(emit_fn emit)
: m_impl{ std::make_unique<impl>(std::move(emit)) }
{}

cpu_snap_decoder::~cpu_snap_decoder() = default;

void
cpu_snap_decoder::feed(const cpu_snap_sample& chunk)
{
    m_impl->reader.feed(std::as_bytes(std::span{ chunk.bytes }));
}

void
cpu_snap_decoder::report_losses() const
{
    if(m_impl->lost_samples > 0)
    {
        LOG_WARNING("{}: {} samples lost by the kernel", k_cpu_snap_display_name,
                    m_impl->lost_samples);
    }
    if(m_impl->unresolved_samples > 0)
    {
        LOG_WARNING("{}: {} samples dropped, their call stack is unknown",
                    k_cpu_snap_display_name, m_impl->unresolved_samples);
    }

    if(m_impl->reader.is_fatal())
    {
        LOG_WARNING("{}: the stream header is unreadable, nothing was decoded",
                    k_cpu_snap_display_name);
    }
    const auto& counters = m_impl->reader.counters();
    if(counters.parse_errors > 0 || counters.gap_count > 0 ||
       counters.chunks_rejected > 0)
    {
        LOG_WARNING("{}: damaged stream: {} parse errors, {} chunk gaps, {} rejected "
                    "chunks",
                    k_cpu_snap_display_name, counters.parse_errors, counters.gap_count,
                    counters.chunks_rejected);
    }
}
}  // namespace rocprofsys::trace_cache
