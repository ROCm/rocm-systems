// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "core/perfetto/driver.hpp"

#include "core/config.hpp"
#include "core/output_file_registry.hpp"
#include "core/perfetto/engine.hpp"
#include "core/perfetto/fwd.hpp"
#include "core/perfetto/sinks.hpp"
#include "core/timemory.hpp"
#include "core/utility.hpp"
#include "library/runtime.hpp"
#include "logger/debug.hpp"

#include <cerrno>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <fcntl.h>
#include <memory>
#include <string>
#include <sys/wait.h>
#include <unistd.h>
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

    auto _rank_data = std::vector<char_vec_t>{};
    auto _combine   = [](char_vec_t& _dst, const char_vec_t& _src) -> char_vec_t& {
        _dst.reserve(_dst.size() + _src.size());
        for(auto&& itr : _src)
            _dst.emplace_back(itr);
        return _dst;
    };

    perfetto_mpi_get_t{ config::get_perfetto_combined_traces(),
                        settings::node_count() }(_rank_data, local_bytes, _combine);

    char_vec_t combined{};
    for(auto& itr : _rank_data)
        combined = (combined.empty()) ? std::move(itr) : _combine(combined, itr);
    return combined;
#else
    return local_bytes;
#endif
}

// Spawns rocprof-sys-merge-output.sh on the directory containing the
// freshly-written perfetto trace so multi-rank or multi-pid outputs end up in
// one merged.proto. Rank 0 only — other ranks return without forking. Uses
// fork+execlp instead of std::system to avoid the shell-injection footprint
// when _output_folder contains characters that would terminate a
// single-quoted argument (a single quote, in particular).
void
run_merge_output_script(const std::string& output_filename)
{
    if(dmp::rank() != 0) return;

    auto _output_folder = filepath::dirname(output_filename);
    auto _script_path   = std::string{ "rocprof-sys-merge-output.sh" };
    auto _script_dir    = get_env("ROCPROFSYS_SCRIPT_PATH", std::string{}, false);

    if(!_script_dir.empty())
        _script_path = fmt::format("{}/{}", _script_dir, _script_path);

    if(!filepath::exists(_script_path))
    {
        LOG_WARNING("Script not found: {}", _script_path);
        return;
    }

    pid_t pid = ::fork();
    if(pid < 0)
    {
        LOG_ERROR("fork failed for merge script {}: errno={}", _script_path, errno);
        return;
    }
    if(pid == 0)
    {
        ::execlp(_script_path.c_str(), _script_path.c_str(), _output_folder.c_str(),
                 nullptr);
        // execlp only returns on failure
        ::_exit(127);
    }

    int status = 0;
    while(::waitpid(pid, &status, 0) < 0 && errno == EINTR)
    {
    }
    if(WIFEXITED(status) && WEXITSTATUS(status) == 0)
        LOG_INFO("Successfully executed: {} {}", _script_path, _output_folder);
    else
        LOG_ERROR("Failed to execute: {} {} (status={})", _script_path, _output_folder,
                  status);
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
            m_tmp_file->open(O_RDWR | O_CREAT | O_TRUNC, 0600);
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

    run_merge_output_script(config::get_perfetto_output_filename());

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
