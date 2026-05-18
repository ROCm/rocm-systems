// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "core/perfetto/driver.hpp"

#include "core/config.hpp"
#include "core/output_file_registry.hpp"
#include "core/perfetto/engine.hpp"
#include "core/perfetto/fwd.hpp"
#include "core/perfetto/sinks.hpp"
#include "core/utility.hpp"
#include "library/runtime.hpp"
#include "logger/debug.hpp"

#include <chrono>
#include <cstdio>
#include <fcntl.h>
#include <memory>

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
}  // namespace

live_perfetto_driver::live_perfetto_driver() noexcept
{
    // Single-instance invariant is enforced by the owning unique_ptr in the
    // composition root; exchange publishes the pointer for cross-TU access.
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
live_perfetto_driver::post_process(tim::manager*         timemory_manager,
                                   bool&                 perfetto_output_error,
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

    auto sink = core::live_fd_sink{ timemory_manager, &perfetto_output_error, registry };
    sink.on_source_drained(static_cast<int>(pid), std::move(bytes));
    sink.finalize();

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
