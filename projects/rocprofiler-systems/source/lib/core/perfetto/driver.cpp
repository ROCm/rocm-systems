// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "core/perfetto/driver.hpp"

#include "core/config.hpp"
#include "core/output_file_registry.hpp"
#include "core/perfetto/engine.hpp"
#include "core/perfetto/fwd.hpp"
#include "core/perfetto/merge_script.hpp"
#include "core/perfetto/sinks.hpp"
#include "core/timemory.hpp"
#include "core/utility.hpp"
#include "library/runtime.hpp"
#include "logger/debug.hpp"

#include <chrono>
#include <cstdio>
#include <fcntl.h>
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
    std::vector<char> data{};
    tf.close();

    FILE* fdata = ::fopen(tf.filename.c_str(), "rb");
    if(!fdata)
    {
        LOG_ERROR("perfetto temp trace file '{}' could not be read", tf.filename);
        return data;
    }

    ::fseek(fdata, 0, SEEK_END);
    size_t fnum_elem = ::ftell(fdata);
    ::fseek(fdata, 0, SEEK_SET);

    data.resize(fnum_elem, '\0');
    auto fnum_read = ::fread(data.data(), sizeof(char), fnum_elem, fdata);
    ::fclose(fdata);

    if(fnum_read != fnum_elem)
    {
        throw std::runtime_error(
            fmt::format("read {} elements from perfetto trace file '{}'. Expected {}",
                        fnum_read, tf.filename, fnum_elem));
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

// When ROCPROFSYS_USE_MPI is enabled and ROCPROFSYS_PERFETTO_COMBINED_TRACES
// is set, gather every rank's trace bytes into a single buffer on the
// collecting rank so the live sink writes one combined .proto instead of one
// per rank. Returns the input unchanged when MPI is disabled at build time or
// the combined-traces setting is off.
std::vector<char>
gather_combined_trace_bytes(std::vector<char> local_bytes)
{
#if defined(ROCPROFSYS_USE_MPI) && ROCPROFSYS_USE_MPI > 0
    if(!config::get_perfetto_combined_traces()) return local_bytes;

    using char_vec_t = std::vector<char>;
    // Held pending a project-local mpi gather helper that handles
    // variable-size byte buffers and the collapse_processes setting.
    using perfetto_mpi_get_t = tim::operation::finalize::mpi_get<char_vec_t, true>;

    auto rank_data = std::vector<char_vec_t>{};
    auto combine   = [](char_vec_t& dst, const char_vec_t& src) -> char_vec_t& {
        dst.insert(dst.end(), src.begin(), src.end());
        return dst;
    };

    perfetto_mpi_get_t{ config::get_perfetto_combined_traces(),
                        settings::node_count() }(rank_data, local_bytes, combine);

    char_vec_t combined{};
    for(auto& itr : rank_data)
        combined = (combined.empty()) ? std::move(itr) : combine(combined, itr);
    return combined;
#else
    return local_bytes;
#endif
}
}  // namespace

live_perfetto_driver::live_perfetto_driver() noexcept
{
    // Single-instance invariant: callers that construct this hold the
    // unique_ptr; the atomic publishes it so other TUs can reach it.
    g_active_driver.store(this, std::memory_order_release);
}

live_perfetto_driver::~live_perfetto_driver()
{
    auto* expected = this;
    g_active_driver.compare_exchange_strong(expected, nullptr, std::memory_order_acq_rel,
                                            std::memory_order_acquire);
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

    bytes = gather_combined_trace_bytes(std::move(bytes));

    auto sink = core::live_fd_sink{ &perfetto_output_error, registry };
    sink.on_source_drained(static_cast<int>(pid), std::move(bytes));
    sink.finalize();

    core::perfetto::run_merge_script(
        filepath::dirname(config::get_perfetto_output_filename()));

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
