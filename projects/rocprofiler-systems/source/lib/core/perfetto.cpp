// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "perfetto.hpp"
#include "config.hpp"
#include "library/runtime.hpp"
#include "logger/debug.hpp"
#include "output_file_registry.hpp"
#include "perfetto_engine.hpp"
#include "perfetto_fwd.hpp"
#include "perfetto_sinks.hpp"
#include "utility.hpp"

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
// Live-mode driver is a TU-scoped perfetto_engine instance. The paired
// live_fd_sink is constructed on the stack inside post_process() at the only
// callsite that needs it. Per-pid sessions are tracked inside the engine;
// fork_gotcha calls detach_inherited_perfetto_session() in the child to drop
// the inherited session pointer without destroying it (the parent owns the
// underlying session).
std::unique_ptr<core::perfetto_engine>      g_engine{};
std::shared_ptr<tmp_file>                   g_tmp_file{};

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

void
setup()
{
    g_engine = std::make_unique<core::perfetto_engine>(
        core::build_engine_config_from_settings());
    g_engine->init_sdk();
}

void
start()
{
    if(!g_engine) return;

    // Optional on-disk capture via a tmp file (legacy ROCPROFSYS_USE_TMP_FILES
    // path). When set the SDK streams output to this fd alongside its
    // internal buffer; post_process reads the file back rather than calling
    // ReadTraceBlocking, preserving the pre-refactor byte source.
    int fd = -1;
    if(config::get_use_tmp_files())
    {
        if(!g_tmp_file)
        {
            g_tmp_file = config::get_tmp_file("perfetto-trace", "proto");
            g_tmp_file->open(O_RDWR | O_CREAT | O_TRUNC, 0600);
        }
        fd = g_tmp_file->fd;
    }

    LOG_DEBUG("Setup perfetto...");
    g_engine->start(core::perfetto_engine::mode::live_fd, fd);
}

void
stop()
{
    if(!g_engine) return;
    g_engine->stop();
}

void
post_process(tim::manager* _timemory_manager, bool& _perfetto_output_error,
             output_file_registry& _output_registry)
{
    if(!g_engine) return;

    g_engine->stop();

    const auto pid = static_cast<pid_t>(::getpid());

    std::vector<char> bytes;
    if(g_tmp_file && *g_tmp_file)
    {
        bytes = read_tmp_file_bytes(*g_tmp_file);
    }
    else
    {
        bytes = g_engine->read_trace(pid);
    }

    g_engine->release_session(pid);

    auto sink = core::live_fd_sink{ _timemory_manager, &_perfetto_output_error,
                                    _output_registry };
    sink.on_source_drained(static_cast<int>(pid), std::move(bytes));
    sink.finalize();

    if(g_tmp_file)
    {
        g_tmp_file->remove();
        g_tmp_file.reset();
    }

    g_engine.reset();
}

}  // namespace perfetto

void
detach_inherited_perfetto_session(pid_t parent_pid)
{
    if(!::rocprofsys::perfetto::g_engine) return;
    ::rocprofsys::perfetto::g_engine->forget_session(parent_pid);
}
}  // namespace rocprofsys

PERFETTO_TRACK_EVENT_STATIC_STORAGE();
