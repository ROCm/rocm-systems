// MIT License
//
// Copyright (c) 2023-2025 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include "tmp_file_buffer.hpp"
#include "domain_type.hpp"
#include "lib/common/filesystem.hpp"

#include <fmt/format.h>

#include <array>
#include <atomic>
#include <mutex>
#include <utility>
#include <vector>

namespace rocprofiler
{
namespace tool
{
namespace fs = common::filesystem;

namespace
{
// Constant-initialized: querying this from a pthread_atfork prepare handler must not run a TLS
// initializer while the file-buffer registry is locked.
thread_local bool fork_output_suspended = false;
}  // namespace

std::string
compose_tmp_file_name(const output_config& cfg, domain_type buffer_type)
{
    auto tmp_directory = fs::path{rocprofiler::tool::format_path(cfg.tmp_directory)};
    auto filename =
        tmp_directory / ".rocprofv3" /
        fmt::format("{}-{}.dat", "%ppid%-%pid%", get_domain_trace_file_name(buffer_type));

    return rocprofiler::tool::format_path(filename.string());
}

bool
is_fork_output_suspended() noexcept
{
    return fork_output_suspended;
}

void
suspend_fork_output() noexcept
{
    fork_output_suspended = true;
}

void
resume_fork_output() noexcept
{
    fork_output_suspended = false;
}

namespace
{
auto&
get_file_buffer_registry_mutex()
{
    static auto* val = new std::mutex{};
    return *val;
}

auto&
get_file_buffer_registry()
{
    static auto* val = new std::vector<file_buffer_fork_handlers>{};
    return *val;
}

std::atomic<uint64_t> registry_epoch{0};
static_assert(std::atomic<uint64_t>::is_always_lock_free);
struct fork_snapshot
{
    static constexpr size_t                             max_handlers = 64;
    std::array<file_buffer_fork_handlers, max_handlers> handlers{};
    size_t                                              size  = 0;
    uint64_t                                            epoch = 0;
    bool                                                valid = false;
};
thread_local fork_snapshot snapshot{};

}  // namespace

void
register_file_buffer(file_buffer_fork_handlers handlers)
{
    ROCP_FATAL_IF(!handlers.flush || !handlers.reset) << "incomplete file-buffer fork handlers";
    auto _lk = std::lock_guard<std::mutex>{get_file_buffer_registry_mutex()};
    registry_epoch.fetch_add(1, std::memory_order_acq_rel);
    get_file_buffer_registry().emplace_back(handlers);
}

void
initialize_fork_file_buffers()
{
    // Resolve every lazy parent-side accessor before this thread can enter an atfork prepare.
    (void) get_file_buffer_registry_mutex();
    (void) get_file_buffer_registry();
    snapshot = {};
}

void
prepare_fork_file_buffers()
{
    auto  lock     = std::lock_guard<std::mutex>{get_file_buffer_registry_mutex()};
    auto& registry = get_file_buffer_registry();
    snapshot.valid = false;
    snapshot.size  = 0;
    if(registry.size() > snapshot.handlers.size()) return;
    for(const auto& handler : registry)
        snapshot.handlers.at(snapshot.size++) = handler;
    for(size_t i = 0; i < snapshot.size; ++i)
        snapshot.handlers.at(i).flush();
    snapshot.epoch = registry_epoch.load(std::memory_order_acquire);
    snapshot.valid = true;
}

void
parent_fork_file_buffers()
{
    snapshot.size  = 0;
    snapshot.valid = false;
}

bool
fork_file_buffers_unchanged() noexcept
{
    return snapshot.valid && snapshot.epoch == registry_epoch.load(std::memory_order_acquire);
}

void
child_fork_file_buffers()
{
    // No registry traversal or inherited lock acquisition. The coordinator validates both the
    // registry epoch and the tool lifecycle epoch before permitting this reconstruction.
    for(size_t i = 0; i < snapshot.size; ++i)
        snapshot.handlers.at(i).reset();
    snapshot.size  = 0;
    snapshot.valid = false;
}

tmp_file_name_callback_t&
get_tmp_file_name_callback()
{
    static tmp_file_name_callback_t val = [](domain_type type) -> std::string {
        ROCP_CI_LOG(WARNING) << "rocprofv3 does not have a tmp file name callback defined for "
                             << get_domain_column_name(type) << ".";
        auto _cfg = output_config::load_from_env();
        return compose_tmp_file_name(_cfg, type);
    };
    return val;
}
}  // namespace tool
}  // namespace rocprofiler
