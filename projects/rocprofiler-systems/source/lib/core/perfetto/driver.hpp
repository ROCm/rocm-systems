// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "core/categories.hpp"
#include "core/common.hpp"
#include "core/config.hpp"
#include "core/perfetto/counter_track.hpp"

#include <atomic>
#include <memory>

#include "logger/debug.hpp"

#if defined(TIMEMORY_USE_PERFETTO)
#    include <timemory/components/perfetto/backends.hpp>
#else
#    include <perfetto.h>
PERFETTO_DEFINE_CATEGORIES(ROCPROFSYS_PERFETTO_CATEGORIES);
#endif

#include <timemory/process/process.hpp>

namespace rocprofsys
{
namespace core
{
class perfetto_engine;
}
namespace config
{
struct tmp_file;
}
class output_file_registry;

namespace perfetto
{
class live_perfetto_driver
{
public:
    live_perfetto_driver() noexcept;
    ~live_perfetto_driver();

    live_perfetto_driver(const live_perfetto_driver&)            = delete;
    live_perfetto_driver& operator=(const live_perfetto_driver&) = delete;
    live_perfetto_driver(live_perfetto_driver&&)                 = delete;
    live_perfetto_driver& operator=(live_perfetto_driver&&)      = delete;

    void setup();
    void start();
    void stop();
    void post_process(bool& perfetto_output_error, output_file_registry& registry);

    // Drop the inherited TracingSession for parent_pid without destroying
    // the underlying session (the parent process owns it; calling .reset()
    // in a fork()ed child would double-free).
    void detach_inherited_session(pid_t parent_pid);

private:
    std::unique_ptr<core::perfetto_engine> m_engine;
    std::shared_ptr<config::tmp_file>      m_tmp_file;
};

live_perfetto_driver*
active_driver() noexcept;
}  // namespace perfetto

void
detach_inherited_perfetto_session(pid_t parent_pid);
}  // namespace rocprofsys
