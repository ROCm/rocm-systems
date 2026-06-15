// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "core/trace_cache/cached_perfetto_session.hpp"

#include "core/config.hpp"
#include "core/mpi.hpp"
#include "core/output_file_registry.hpp"
#include "core/perfetto/engine.hpp"
#include "core/trace_cache/post_processor.hpp"
#include "core/track_registry.hpp"
#include "logger/debug.hpp"

#include <cstdint>
#include <exception>
#include <filesystem>
#include <memory>
#include <vector>

namespace rocprofsys::trace_cache
{
namespace
{
core::single_file_sink
make_merged_append_sink(output_file_registry& registry, std::size_t source_count)
{
    const auto base_filename = config::get_perfetto_output_filename();
    const auto merged_path =
        (std::filesystem::path{ base_filename }.parent_path() / "merged.proto").string();
    const auto env_rank_raw = mpi::rank_from_env();
    auto       sink         = core::single_file_sink{ registry, merged_path };

    if(env_rank_raw < 0)
    {
        LOG_ERROR("cached Perfetto merged output skipped: launcher rank {} is invalid",
                  env_rank_raw);
        sink.set_append_mode(core::append_mode_config{ .source_count = 0 });
        return sink;
    }

    const auto env_rank    = static_cast<std::uint32_t>(env_rank_raw);
    const auto seq_id_base = core::append_seq_id_base_for_rank(env_rank);
    if(!seq_id_base)
    {
        LOG_ERROR("cached Perfetto merged output skipped: launcher rank {} exceeds "
                  "the trusted_packet_sequence_id merge window",
                  env_rank);
        sink.set_append_mode(core::append_mode_config{ .source_count = 0 });
        return sink;
    }

    sink.set_append_mode(core::append_mode_config{ .seq_id_base  = *seq_id_base,
                                                   .source_count = source_count });
    return sink;
}

std::unique_ptr<core::trace_sink>
make_sink(output_file_registry& registry, pid_t root_pid,
          core::perfetto_output_layout layout, std::size_t source_count)
{
    if(layout == core::perfetto_output_layout::single_file_only)
    {
        return std::make_unique<core::trace_sink>(
            make_merged_append_sink(registry, source_count));
    }

    if(layout == core::perfetto_output_layout::full)
    {
        return std::make_unique<core::trace_sink>(
            core::tee_sink{ core::per_pid_file_sink{ root_pid, registry },
                            make_merged_append_sink(registry, source_count) });
    }

    return std::make_unique<core::trace_sink>(
        core::per_pid_file_sink{ root_pid, registry });
}
}  // namespace

cached_perfetto_session::cached_perfetto_session(output_file_registry&        registry,
                                                 pid_t                        root_pid,
                                                 core::perfetto_output_layout layout,
                                                 const std::vector<int>&      source_pids,
                                                 post_processor&              processor)
: m_engine{ std::make_unique<core::perfetto_engine>(
      core::build_engine_config_from_settings()) }
, m_sink{ make_sink(registry, root_pid, layout, source_pids.size()) }
, m_tracks{ std::make_unique<track_registry>() }
{
    m_engine->init_sdk();
    m_engine->start(core::perfetto_engine::mode::cached_interceptor, *m_sink);
    m_engine->preregister_pids(source_pids);
    processor.set_perfetto_engine(m_engine.get(), m_tracks.get());
    m_started = true;
}

cached_perfetto_session::~cached_perfetto_session() noexcept
{
    if(m_started)
    {
        try
        {
            m_engine->stop();
        } catch(const std::exception& exp)
        {
            LOG_ERROR("Perfetto engine stop/drain raised: {}", exp.what());
        } catch(...)
        {
            LOG_ERROR("Perfetto engine stop/drain raised a non-std::exception");
        }
    }

    // The engine drains into the sink during stop(); keep explicit teardown
    // order so future member reordering cannot silently invert that lifetime.
    m_engine.reset();
    m_sink.reset();
    m_tracks.reset();
}
}  // namespace rocprofsys::trace_cache
