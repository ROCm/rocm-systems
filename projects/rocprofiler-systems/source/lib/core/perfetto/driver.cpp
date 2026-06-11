// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "core/perfetto/driver.hpp"
#include "core/perfetto/category_registry.hpp"

#include "core/config.hpp"
#include "core/mpi.hpp"
#include "core/output_file_registry.hpp"
#include "core/perfetto/engine.hpp"
#include "core/perfetto/fwd.hpp"
#include "core/perfetto/sinks.hpp"
#include "core/timemory.hpp"
#include "core/utility.hpp"
#include "library/runtime.hpp"
#include "logger/debug.hpp"

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <fcntl.h>
#include <fstream>
#include <ios>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace rocprofsys
{
namespace perfetto
{
namespace
{
std::vector<char>
read_tmp_file_bytes(tmp_file& tf)
{
    tf.close();

    std::ifstream ifs{ tf.filename, std::ios::binary | std::ios::ate };
    if(!ifs)
    {
        LOG_ERROR("perfetto temp trace file '{}' could not be opened for read",
                  tf.filename);
        return {};
    }

    // ate + tellg gives a signed file size; a negative value (overflow on
    // huge files or a stream error) would otherwise wrap into size_t and
    // request a SIZE_MAX-byte vector. Treat any negative size as an open
    // failure rather than crashing the post-process pipeline.
    const auto end = ifs.tellg();
    if(end < 0)
    {
        LOG_ERROR("perfetto temp trace file '{}' size query failed", tf.filename);
        return {};
    }

    std::vector<char> data(static_cast<std::size_t>(end));
    ifs.seekg(0, std::ios::beg);
    if(!data.empty()) ifs.read(data.data(), static_cast<std::streamsize>(data.size()));

    if(!ifs)
    {
        throw std::runtime_error(
            fmt::format("short read on perfetto trace file '{}' (expected {} bytes)",
                        tf.filename, data.size()));
    }
    return data;
}
}  // namespace

namespace
{
std::atomic<live_perfetto_driver*> g_active_driver{ nullptr };

// 0600 (-rw-------) keeps the live-mode tmp trace readable only by the
// owning process, matching the threat model that profiling data may carry
// callstacks the user does not want world-readable on shared workstations.
constexpr ::mode_t TMP_FILE_PERMS = 0600;

// Stride between per-process seq_id ranges in the cross-process merged
// output. Each rocprof-sys instance (one per launcher rank) gets a
// disjoint slice [rank*STRIDE+1, (rank+1)*STRIDE] of the
// trusted_packet_sequence_id space when writing to the shared merged file
// under append-with-flock mode. 1<<20 leaves room for ~1M unique
// sources per rank, well past anything a single trace would produce.
constexpr std::uint32_t MERGED_SEQ_ID_RANK_STRIDE = 1u << 20;

}  // namespace

live_perfetto_driver::live_perfetto_driver() noexcept
{
    // Single-instance invariant: callers that construct this hold the
    // unique_ptr; the atomic publishes it so other TUs can reach it.
    g_active_driver.store(this, std::memory_order_release);
}

live_perfetto_driver::~live_perfetto_driver() noexcept
{
    auto*      expected = this;
    const bool cleared  = g_active_driver.compare_exchange_strong(
        expected, nullptr, std::memory_order_acq_rel, std::memory_order_acquire);
    if(!cleared && expected != nullptr)
    {
        // Another driver took over g_active_driver while this one was still
        // alive; loud-warn so the lifecycle violation is visible instead of
        // silently leaving the global pointing at a soon-to-be-destroyed
        // object.
        LOG_WARNING("live_perfetto_driver destructor saw g_active_driver pointing at a "
                    "different instance ({} vs this={}) — overlapping driver "
                    "lifetimes are not supported",
                    static_cast<const void*>(expected), static_cast<const void*>(this));
    }
}

void
live_perfetto_driver::setup()
{
    m_engine = std::make_unique<core::perfetto_engine>(
        core::build_engine_config_from_settings());
    m_engine->init_sdk();
}

void
live_perfetto_driver::start()
{
    if(!m_engine) return;

    int fd = -1;
    if(config::get_use_tmp_files())
    {
        if(!m_tmp_file)
        {
            m_tmp_file = config::get_tmp_file("perfetto-trace", "proto");
            m_tmp_file->open(O_RDWR | O_CREAT | O_TRUNC, TMP_FILE_PERMS);
        }
        fd = m_tmp_file->fd;
    }

    LOG_DEBUG("Setup perfetto...");
    m_engine->start(core::perfetto_engine::mode::live_fd, fd);
}

void
live_perfetto_driver::stop()
{
    if(!m_engine) return;
    m_engine->stop();
}

void
live_perfetto_driver::post_process(bool&                 perfetto_output_error,
                                   output_file_registry& registry)
{
    if(!m_engine) return;

    m_engine->stop();

    const auto pid = static_cast<pid_t>(::getpid());

    std::vector<char> bytes;
    if(m_tmp_file && *m_tmp_file)
    {
        bytes = read_tmp_file_bytes(*m_tmp_file);
    }
    else
    {
        bytes = m_engine->read_trace(pid);
    }

    m_engine->destroy_session(pid);

    const auto layout        = config::get_perfetto_output_layout();
    const bool want_per_rank = core::writes_per_process_files(layout);
    const bool want_merged   = core::writes_merged_file(layout);

    if(want_per_rank)
    {
        auto sink         = core::live_fd_sink{ &perfetto_output_error, registry };
        auto bytes_for_pr = want_merged ? bytes : std::move(bytes);
        sink.on_source_drained(static_cast<int>(pid), std::move(bytes_for_pr));
        sink.finalize();
    }

    if(want_merged)
    {
        // Each process writes via single_file_sink in append-with-flock
        // mode. The per-source seq_id rewrite keeps each rank's
        // interned-data namespace intact across the concatenated bytes, so
        // the merge stays correct even when rocprof-sys is not linked
        // against MPI. The shared filename is a literal so every rank
        // targets the same path.
        const auto base_filename = config::get_perfetto_output_filename();
        const auto merged_path   = filepath::dirname(base_filename) + "/merged.proto";
        const auto env_rank      = static_cast<std::uint32_t>(mpi::rank_from_env());
        auto       sink          = core::single_file_sink{ registry, merged_path };
        sink.set_append_mode(env_rank * MERGED_SEQ_ID_RANK_STRIDE);
        sink.on_source_drained(static_cast<int>(env_rank), std::move(bytes));
        sink.finalize();
    }

    if(m_tmp_file)
    {
        m_tmp_file->remove();
        m_tmp_file.reset();
    }

    m_engine.reset();
}

void
live_perfetto_driver::detach_inherited_session(pid_t parent_pid)
{
    if(!m_engine) return;
    m_engine->forget_session(parent_pid);
}

live_perfetto_driver*
active_driver() noexcept
{
    return g_active_driver.load(std::memory_order_acquire);
}

}  // namespace perfetto

void
detach_inherited_perfetto_session(pid_t parent_pid)
{
    if(auto* drv = perfetto::active_driver()) drv->detach_inherited_session(parent_pid);
}
}  // namespace rocprofsys

PERFETTO_TRACK_EVENT_STATIC_STORAGE();
