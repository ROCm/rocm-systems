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
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

#include "lib/rocprofiler-sdk/page_migration/page_migration.hpp"
#include "lib/common/logging.hpp"
#include "lib/common/mpl.hpp"
#include "lib/common/static_object.hpp"
#include "lib/common/utility.hpp"
#include "lib/rocprofiler-sdk/agent.hpp"
#include "lib/rocprofiler-sdk/buffer.hpp"
#include "lib/rocprofiler-sdk/context/context.hpp"
#include "lib/rocprofiler-sdk/details/kfd_ioctl.h"
#include "lib/rocprofiler-sdk/internal_threading.hpp"
#include "lib/rocprofiler-sdk/page_migration/utils.hpp"

#include <rocprofiler-sdk/agent.h>
#include <rocprofiler-sdk/buffer_tracing.h>
#include <rocprofiler-sdk/fwd.h>
#include <rocprofiler-sdk/hsa/api_id.h>
#include <rocprofiler-sdk/hsa/table_id.h>

#include <fmt/core.h>
#include <hsa/amd_hsa_signal.h>
#include <hsa/hsa.h>

#include <sys/poll.h>
#include <unistd.h>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <memory>
#include <ratio>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <type_traits>
#include <unordered_map>
#include <utility>

#include <fcntl.h>
#include <poll.h>
#include <sys/eventfd.h>
#include <sys/ioctl.h>

#define ROCPROFILER_LIB_ROCPROFILER_SDK_PAGE_MIGRATION_PAGE_MIGRATION_CPP_IMPL 1
#include "page_migration.def.cpp"
#undef ROCPROFILER_LIB_ROCPROFILER_SDK_PAGE_MIGRATION_PAGE_MIGRATION_CPP_IMPL

namespace rocprofiler
{
namespace page_migration
{
template <typename T>
using small_vector = common::container::small_vector<T>;

using context_t       = context::context;
using context_array_t = common::container::small_vector<const context_t*>;

template <size_t>
struct page_migration_info;

template <size_t>
struct kfd_event_info;

template <typename EnumT, int ValueE>
struct page_migration_enum_info;

template <typename EnumT>
struct page_migration_bounds;

// Parsing and utilities
namespace
{
constexpr auto
page_to_bytes(size_t val)
{
    // each page is 4KB = 4096 bytes
    return val << 12;
}

template <size_t>
page_migration_record_t parse_event(std::string_view)
{
    ROCP_FATAL_IF(false) << page_migration_info<ROCPROFILER_PAGE_MIGRATION_NONE>::format_str;
    return {};
}

auto
get_node_agent_id(uint32_t _node_id)
{
    using agent_id_map_t = std::unordered_map<uint64_t, rocprofiler_agent_id_t>;
    static auto*& _data  = static_object<agent_id_map_t>::construct([]() {
        auto _v = std::unordered_map<uint64_t, rocprofiler_agent_id_t>{};
        for(const auto* agent : agent::get_agents())
            _v.emplace(agent->gpu_id, agent->id);
        return _v;
    }());

    CHECK(_data != nullptr);
    ROCP_FATAL_IF(_data->count(_node_id) == 0) << "page_migration: unknown node id: " << _node_id;
    return _data->at(_node_id);
}

template <>
page_migration_record_t
parse_event<ROCPROFILER_PAGE_MIGRATION_PAGE_FAULT_START>(std::string_view str)
{
    auto     rec = page_migration_record_t{};
    auto&    e   = rec.args.page_fault_start;
    uint32_t kind{};
    uint32_t _node_id = 0;

    char fault;
    std::sscanf(str.data(),
                page_migration_info<ROCPROFILER_PAGE_MIGRATION_PAGE_FAULT_START>::format_str.data(),
                &kind,
                &rec.timestamp,
                &rec.pid,
                &e.address,
                &_node_id,
                &fault);

    e.read_fault = (fault == 'R');
    e.address    = page_to_bytes(e.address);
    e.agent_id   = get_node_agent_id(_node_id);

    ROCP_TRACE << fmt::format("Page fault start [ ts: {} pid: {} addr: 0x{:X} node: {} ] \n",
                              rec.timestamp,
                              rec.pid,
                              e.address,
                              e.agent_id.handle);

    return rec;
}

template <>
page_migration_record_t
parse_event<ROCPROFILER_PAGE_MIGRATION_PAGE_FAULT_END>(std::string_view str)
{
    auto     rec = page_migration_record_t{};
    auto&    e   = rec.args.page_fault_end;
    uint32_t kind{};
    uint32_t _node_id = 0;

    char migrated;
    std::sscanf(str.data(),
                page_migration_info<ROCPROFILER_PAGE_MIGRATION_PAGE_FAULT_END>::format_str.data(),
                &kind,
                &rec.timestamp,
                &rec.pid,
                &e.address,
                &_node_id,
                &migrated);

    // M or U -> migrated / unmigrated?
    if(migrated == 'M')
        e.migrated = true;
    else if(migrated == 'U')
        e.migrated = false;
    else
        ROCP_WARNING << "Unknown PAGE_FAULT_END migrated/unmigrated state";

    e.address  = page_to_bytes(e.address);
    e.agent_id = get_node_agent_id(_node_id);

    ROCP_TRACE << fmt::format(
        "Page fault end [ ts: {} pid: {} addr: 0x{:X} node: {} migrated: {} ] \n",
        rec.timestamp,
        rec.pid,
        e.address,
        e.agent_id.handle,
        migrated);

    return rec;
}

template <>
page_migration_record_t
parse_event<ROCPROFILER_PAGE_MIGRATION_PAGE_MIGRATE_START>(std::string_view str)
{
    auto     rec = page_migration_record_t{};
    auto&    e   = rec.args.page_migrate_start;
    uint32_t kind{};
    uint32_t trigger{};
    uint32_t _from_node      = 0;
    uint32_t _to_node        = 0;
    uint32_t _prefetch_node  = 0;
    uint32_t _preferred_node = 0;

    std::sscanf(
        str.data(),
        page_migration_info<ROCPROFILER_PAGE_MIGRATION_PAGE_MIGRATE_START>::format_str.data(),
        &kind,
        &rec.timestamp,
        &rec.pid,
        &e.start_addr,
        &e.end_addr,
        &_from_node,
        &_to_node,
        &_prefetch_node,
        &_preferred_node,
        &trigger);

    e.end_addr += e.start_addr;
    e.trigger         = static_cast<migrate_trigger_t>(trigger);
    e.start_addr      = page_to_bytes(e.start_addr);
    e.end_addr        = page_to_bytes(e.end_addr) - 1;
    e.from_agent      = get_node_agent_id(_from_node);
    e.to_agent        = get_node_agent_id(_to_node);
    e.prefetch_agent  = get_node_agent_id(_prefetch_node);
    e.preferred_agent = get_node_agent_id(_preferred_node);

    ROCP_TRACE << fmt::format(
        "Page migrate start [ ts: {} pid: {} addr s: 0x{:X} addr "
        "e: 0x{:X} size: {}B from node: {} to node: {} prefetch node: {} preferred node: {} "
        "trigger: {} ] \n",
        rec.timestamp,
        rec.pid,
        e.start_addr,
        e.end_addr,
        (e.end_addr - e.start_addr),
        e.from_agent.handle,
        e.to_agent.handle,
        e.prefetch_agent.handle,
        e.preferred_agent.handle,
        trigger);

    return rec;
}

template <>
page_migration_record_t
parse_event<ROCPROFILER_PAGE_MIGRATION_PAGE_MIGRATE_END>(std::string_view str)
{
    auto     rec = page_migration_record_t{};
    auto&    e   = rec.args.page_migrate_end;
    uint32_t kind{};
    uint32_t trigger{};
    uint32_t _from_node  = 0;
    uint32_t _to_node    = 0;
    int32_t  _error_code = 0;

    std::sscanf(str.data(),
                page_migration_info<ROCPROFILER_PAGE_MIGRATION_PAGE_MIGRATE_END>::format_str.data(),
                &kind,
                &rec.timestamp,
                &rec.pid,
                &e.start_addr,
                &e.end_addr,
                &_from_node,
                &_to_node,
                &trigger,
                &_error_code);

    e.end_addr += e.start_addr;
    e.trigger    = static_cast<migrate_trigger_t>(trigger);
    e.start_addr = page_to_bytes(e.start_addr);
    e.end_addr   = page_to_bytes(e.end_addr) - 1;
    e.from_agent = get_node_agent_id(_from_node);
    e.to_agent   = get_node_agent_id(_to_node);

    // For older kernel versions, no event is generated in the error case,
    // so the default value of 0 is correct for any given end event.
    e.error_code = _error_code;

    ROCP_TRACE << fmt::format("Page migrate end [ ts: {} pid: {} addr s: 0x{:X} addr e: "
                              "0x{:X} from node: {} to node: {} trigger: {} error code: {}] \n",
                              rec.timestamp,
                              rec.pid,
                              e.start_addr,
                              e.end_addr,
                              e.from_agent.handle,
                              e.to_agent.handle,
                              trigger,
                              _error_code);

    return rec;
}

template <>
page_migration_record_t
parse_event<ROCPROFILER_PAGE_MIGRATION_QUEUE_EVICTION>(std::string_view str)
{
    auto     rec = page_migration_record_t{};
    auto&    e   = rec.args.queue_eviction;
    uint32_t kind{};
    uint32_t trigger{};
    uint32_t _node_id = 0;

    std::sscanf(str.data(),
                page_migration_info<ROCPROFILER_PAGE_MIGRATION_QUEUE_EVICTION>::format_str.data(),
                &kind,
                &rec.timestamp,
                &rec.pid,
                &_node_id,
                &trigger);

    e.trigger  = static_cast<queue_suspend_trigger_t>(trigger);
    e.agent_id = get_node_agent_id(_node_id);

    ROCP_TRACE << fmt::format("Queue evict [ ts: {} pid: {} node: {} trigger: {} ] \n",
                              rec.timestamp,
                              rec.pid,
                              e.agent_id.handle,
                              trigger);

    return rec;
}

template <>
page_migration_record_t
parse_event<ROCPROFILER_PAGE_MIGRATION_QUEUE_RESTORE>(std::string_view str)
{
    auto     rec = page_migration_record_t{};
    auto&    e   = rec.args.queue_restore;
    uint32_t kind{};
    uint32_t _node_id = 0;

    std::sscanf(str.data(),
                page_migration_info<ROCPROFILER_PAGE_MIGRATION_QUEUE_RESTORE>::format_str.data(),
                &kind,
                &rec.timestamp,
                &rec.pid,
                &_node_id);
    // check if we have a valid char at the end. -1 has \0
    if(str[str.size() - 2] == 'R')
        e.rescheduled = true;
    else
        e.rescheduled = false;
    e.agent_id = get_node_agent_id(_node_id);

    ROCP_TRACE << fmt::format(
        "Queue restore [ ts: {} pid: {} node: {} ] \n", rec.timestamp, rec.pid, e.agent_id.handle);

    return rec;
}

template <>
page_migration_record_t
parse_event<ROCPROFILER_PAGE_MIGRATION_UNMAP_FROM_GPU>(std::string_view str)
{
    auto     rec = page_migration_record_t{};
    auto&    e   = rec.args.unmap_from_gpu;
    uint32_t kind{};
    uint32_t trigger{};
    uint32_t _node_id = 0;

    std::sscanf(str.data(),
                page_migration_info<ROCPROFILER_PAGE_MIGRATION_UNMAP_FROM_GPU>::format_str.data(),
                &kind,
                &rec.timestamp,
                &rec.pid,
                &e.start_addr,
                &e.end_addr,
                &_node_id,
                &trigger);

    e.end_addr += e.start_addr;
    e.trigger    = static_cast<unmap_from_gpu_trigger_t>(trigger);
    e.start_addr = page_to_bytes(e.start_addr);
    e.end_addr   = page_to_bytes(e.end_addr);
    e.agent_id   = get_node_agent_id(_node_id);

    ROCP_TRACE << fmt::format(
        "Unmap from GPU [ ts: {} pid: {} start addr: 0x{:X} end addr: 0x{:X}  "
        "node: {} trigger {} ] \n",
        rec.timestamp,
        rec.pid,
        e.start_addr,
        e.end_addr,
        e.agent_id.handle,
        trigger);

    return rec;
}

template <>
page_migration_record_t
parse_event<ROCPROFILER_PAGE_MIGRATION_DROPPED_EVENT>(std::string_view str)
{
    auto     rec = page_migration_record_t{};
    auto&    e   = rec.args.dropped_event;
    uint32_t kind{};

    std::sscanf(str.data(),
                page_migration_info<ROCPROFILER_PAGE_MIGRATION_DROPPED_EVENT>::format_str.data(),
                &kind,
                &rec.timestamp,
                &rec.pid,
                &e.dropped_events_count);

    ROCP_TRACE << fmt::format("Dropped events [ ts: {} pid: {} dropped count: {} ] \n",
                              rec.timestamp,
                              rec.pid,
                              e.dropped_events_count);

    return rec;
}

template <>
page_migration_record_t parse_event<ROCPROFILER_PAGE_MIGRATION_NONE>(std::string_view)
{
    ROCP_CI_LOG(WARNING)
        << "ROCPROFILER_PAGE_MIGRATION_NONE for parsing page migration events should not happen";
    return common::init_public_api_struct(page_migration_record_t{});
}

template <size_t OpInx, size_t... OpInxs>
page_migration_record_t
parse_event(size_t event_id, std::string_view strn, std::index_sequence<OpInx, OpInxs...>)
{
    if(OpInx == static_cast<uint32_t>(event_id))
    {
        auto rec      = parse_event<OpInx>(strn);
        rec.size      = sizeof(page_migration_record_t);
        rec.kind      = ROCPROFILER_BUFFER_TRACING_PAGE_MIGRATION;
        rec.operation = static_cast<rocprofiler_page_migration_operation_t>(OpInx);
        return rec;
    }

    if constexpr(sizeof...(OpInxs) > 0)
        return parse_event(event_id, strn, std::index_sequence<OpInxs...>{});

    return page_migration_record_t{};
}

/* -----------------------------------------------------------------------------------*/

}  // namespace

size_t
get_rocprof_op(const std::string_view event_data)
{
    uint32_t kfd_event_id{};
    std::sscanf(event_data.data(), "%x ", &kfd_event_id);

    auto rocprof_id =
        kfd_to_rocprof_op(static_cast<kfd_event_id_t>(kfd_event_id),
                          std::make_index_sequence<ROCPROFILER_PAGE_MIGRATION_LAST>{});

    ROCP_CI_LOG_IF(WARNING, rocprof_id == 0)
        << fmt::format("Failed to parse KFD event ID {}. Parsed ID: {}, SDK ID: {}\n",
                       event_data[0],
                       kfd_event_id,
                       rocprof_id);

    return rocprof_id;
}

void
kfd_readlines(const std::string_view str, void(handler)(std::string_view))
{
    const auto  find_newline = [&](auto b) { return std::find(b, str.cend(), '\n'); };
    const auto* cursor       = str.cbegin();

    for(const auto* pos = find_newline(cursor); pos != str.cend(); pos = find_newline(cursor))
    {
        size_t char_count = pos - cursor;
        assert(char_count > 0);
        std::string_view event_str{cursor, char_count};

        ROCP_INFO << fmt::format("KFD event: [{}]", event_str);
        handler(event_str);

        cursor = pos + 1;
    }
}

// Event capture and reporting
namespace
{
constexpr auto kfd_ioctl_version = (1000 * KFD_IOCTL_MAJOR_VERSION) + KFD_IOCTL_MINOR_VERSION;
// Support has been added in kfdv >= 1.10+
static_assert(kfd_ioctl_version >= 1010, "KFD SMI support missing in kfd_ioctl.h");

auto
get_contexts(int operation)
{
    auto active_contexts = context::get_active_contexts([](const auto* ctx) {
        return (ctx->buffered_tracer &&
                ctx->buffered_tracer->domains(ROCPROFILER_BUFFER_TRACING_PAGE_MIGRATION));
    });
    auto operation_ctxs  = context::context_array_t{};

    for(const auto* itr : active_contexts)
    {
        // if the given domain + op is not enabled, skip this context
        if(itr->buffered_tracer->domains(ROCPROFILER_BUFFER_TRACING_PAGE_MIGRATION, operation))
        {
            operation_ctxs.emplace_back(itr);
        }
    }

    return operation_ctxs;
}

void
handle_reporting(std::string_view event_data)
{
    const auto op_inx            = get_rocprof_op(event_data);
    auto       buffered_contexts = get_contexts(op_inx);
    if(buffered_contexts.empty()) return;

    // Parse and process the event
    auto record = parse_event(
        op_inx, event_data, std::make_index_sequence<ROCPROFILER_PAGE_MIGRATION_LAST>{});

    for(const auto& itr : buffered_contexts)
    {
        auto* buffer = buffer::get_buffer(
            itr->buffered_tracer->buffer_data.at(ROCPROFILER_BUFFER_TRACING_PAGE_MIGRATION));
        CHECK_NOTNULL(buffer)->emplace(
            ROCPROFILER_BUFFER_CATEGORY_TRACING, ROCPROFILER_BUFFER_TRACING_PAGE_MIGRATION, record);
    }
}

}  // namespace

namespace
{
void poll_events(small_vector<pollfd>);
}

// KFD utils
namespace kfd
{
using fd_flags_t = decltype(EFD_NONBLOCK);
using fd_t       = decltype(pollfd::fd);
constexpr auto KFD_DEVICE_PATH{"/dev/kfd"};

SPECIALIZE_KFD_IOC_IOCTL(kfd_ioctl_get_version_args, AMDKFD_IOC_GET_VERSION);
SPECIALIZE_KFD_IOC_IOCTL(kfd_ioctl_smi_events_args, AMDKFD_IOC_SMI_EVENTS);

namespace
{
template <typename T>
auto
ioctl(int kfd_fd, T& args)
{
    // from hsaKmt library (hsakmt/src/libhsakmt.c)
    int exit_code{};

    do
    {
        exit_code = ::ioctl(kfd_fd, IOC_event<T>::value, static_cast<void*>(&args));
    } while(exit_code == -1 && (errno == EINTR || errno == EAGAIN));

    if(exit_code == -1 && errno == EBADF)
    {
        /* In case pthread_atfork didn't catch it, this will
         * make any subsequent hsaKmt calls fail in CHECK_KFD_OPEN.
         */
        CHECK(true && "KFD file descriptor not valid in this process\n");
    }
    return exit_code;
}

struct kfd_device_fd
{
    fd_t fd{-1};

    kfd_device_fd()
    {
        fd = ::open(KFD_DEVICE_PATH, O_RDWR | O_CLOEXEC);
        ROCP_FATAL_IF(fd == -1) << "Error opening KFD handle @ " << KFD_DEVICE_PATH;
    }

    ~kfd_device_fd()
    {
        if(fd >= 0) close(fd);
    }
};

const kfd_ioctl_get_version_args
get_version()
{
    static kfd_ioctl_get_version_args version = [&]() {
        auto          args = kfd_ioctl_get_version_args{0, 0};
        kfd_device_fd kfd_fd{};

        if(ioctl(kfd_fd.fd, args) != -1)
            ROCP_INFO << fmt::format("KFD v{}.{}", args.major_version, args.minor_version);
        else
            ROCP_ERROR << fmt::format("Could not determine KFD version");
        return args;
    }();

    return version;
}

struct poll_kfd_t
{
    static constexpr auto DEFAULT_FLAGS{EFD_CLOEXEC};

    struct gpu_fd_t
    {
        unsigned int               node_id = 0;
        fd_t                       fd      = {};
        const rocprofiler_agent_t* agent   = nullptr;
    };

    kfd_device_fd kfd_fd        = {};
    pollfd        thread_notify = {};
    std::thread   bg_thread     = {};
    bool          active        = {false};

    poll_kfd_t() = default;

    poll_kfd_t(const small_vector<size_t>& rprof_ev)
    : kfd_fd{kfd_device_fd{}}
    {
        small_vector<pollfd> file_handles = {};

        const auto kfd_flags =
            kfd_bitmask(rprof_ev, std::make_index_sequence<ROCPROFILER_PAGE_MIGRATION_LAST>{});

        ROCP_TRACE << fmt::format("Setting KFD flags to [0b{:b}] \n", kfd_flags);

        // Create fd for notifying thread when we want to wake it up, and an eventfd for any events
        // to this thread
        file_handles.emplace_back(
            pollfd{.fd = eventfd(0, DEFAULT_FLAGS), .events = 0, .revents = 0});

        fd_t thread_pipes[2]{};

        [&]() {
            const auto retcode = pipe2(&thread_pipes[0], DEFAULT_FLAGS);
            const auto _err    = errno;
            ROCP_FATAL_IF(retcode != 0)
                << fmt::format("Pipe creation for page-migration thread notify returned {} :: {}\n",
                               retcode,
                               strerror(_err));
        }();

        thread_notify = pollfd{
            .fd      = thread_pipes[1],
            .events  = POLLIN,
            .revents = 0,
        };

        // add pipe listening end to fds to watch
        file_handles.emplace_back(pollfd{thread_pipes[0], POLLIN, 0});

        // get FD, start thread, and then enable events
        for(const auto& agent : agent::get_agents())
        {
            if(agent->type == ROCPROFILER_AGENT_TYPE_GPU)
            {
                auto gpu_event_fd = get_node_fd(agent->gpu_id);
                file_handles.emplace_back(pollfd{gpu_event_fd, POLLIN, 0});
                ROCP_TRACE << fmt::format(
                    "GPU node {} with fd {} added\n", agent->gpu_id, gpu_event_fd);
            }
        }

        // Enable KFD masked events by writing flags to kfd fd
        for(size_t i = 2; i < file_handles.size(); ++i)
        {
            auto& fd         = file_handles[i];
            auto  write_size = write(fd.fd, &kfd_flags, sizeof(kfd_flags));
            ROCP_TRACE << fmt::format(
                "Writing {} to GPU fd {} ({} bytes)\n", kfd_flags, fd.fd, write_size);
            CHECK(write_size == sizeof(kfd_flags));
        }

        // start bg thread
        internal_threading::notify_pre_internal_thread_create(ROCPROFILER_LIBRARY);
        bg_thread = std::thread{poll_events, file_handles};
        internal_threading::notify_post_internal_thread_create(ROCPROFILER_LIBRARY);

        active = true;
    }

    static auto get_event_id(const std::string_view& strn)
    {
        uint32_t event_id{std::numeric_limits<uint32_t>::max()};
        std::sscanf(strn.data(), "%x ", &event_id);

        CHECK(event_id <= KFD_SMI_EVENT_ALL_PROCESS);
    }

    poll_kfd_t(const poll_kfd_t&) = delete;
    poll_kfd_t& operator=(const poll_kfd_t&) = delete;

    poll_kfd_t(poll_kfd_t&&) noexcept = default;
    poll_kfd_t& operator=(poll_kfd_t&&) noexcept = default;

    ~poll_kfd_t()
    {
        ROCP_TRACE << fmt::format("Terminating poll_kfd\n");
        if(!active) return;

        // wake thread up
        auto bytes_written{-1};
        do
        {
            bytes_written = write(thread_notify.fd, "E", 1);
        } while(bytes_written == -1 && (errno == EINTR || errno == EAGAIN));

        bg_thread.join();

        close(thread_notify.fd);

        ROCP_TRACE << fmt::format("Background thread signalled\n");
    }

    node_fd_t get_node_fd(int gpu_node_id) const
    {
        kfd_ioctl_smi_events_args args{};
        args.gpuid = gpu_node_id;

        if(auto ret = ioctl(kfd_fd.fd, args); ret == -1)
            ROCP_ERROR << fmt::format(
                "Could not get GPU node {} file descriptor (exit code: {})", gpu_node_id, ret);
        return args.anon_fd;
    }
};
}  // namespace
}  // namespace kfd

// for all contexts
struct config
{
private:
    kfd::poll_kfd_t kfd_handle{};

    static inline config* _config{nullptr};

    config(const small_vector<size_t>& _event_ids)
    : kfd_handle{_event_ids}
    {}

public:
    static void init(const small_vector<size_t>& event_ids) { _config = new config{event_ids}; }

    static void reset()
    {
        config* ptr = nullptr;
        std::swap(ptr, _config);
        delete ptr;
    }

    static void reset_on_fork() { _config = nullptr; }
};

namespace
{
void
poll_events(small_vector<pollfd> file_handles)
{
    // storage to write records to, 1MB
    constexpr size_t PREALLOCATE_ELEMENT_COUNT{1024 * 128};
    std::string      scratch_buffer(PREALLOCATE_ELEMENT_COUNT, '\0');
    auto&            exitfd = file_handles[1];

    // Wait or spin on events.
    //  0 -> return immediately even if no events
    // -1 -> wait indefinitely

    pthread_setname_np(pthread_self(), "bg:pagemigr");

    for(auto& fd : file_handles)
    {
        ROCP_TRACE << fmt::format(
            "Handle = {}, events = {}, revents = {}\n", fd.fd, fd.events, fd.revents);
    }

    while(true)
    {
        auto poll_ret = poll(file_handles.data(), file_handles.size(), -1);

        if(poll_ret == -1)
        {
            ROCP_CI_LOG(WARNING)
                << "Background thread file descriptors for page-migration are invalid";
            return;
        }

        if((exitfd.revents & POLLIN) != 0)
        {
            for(const auto& f : file_handles)
            {
                close(f.fd);
            }
            ROCP_INFO << "Terminating background thread\n";
            return;
        }

        using namespace std::chrono_literals;

        // 0 and 1 are for generic and pipe-notify handles
        for(size_t i = 2; i < file_handles.size(); ++i)
        {
            auto& fd = file_handles[i];

            // We have data to read, perhaps multiple events
            if((fd.revents & POLLIN) != 0)
            {
                size_t status_size   = read(fd.fd, scratch_buffer.data(), scratch_buffer.size());
                auto   event_strings = std::string_view{scratch_buffer.data(), status_size};
                kfd_readlines(event_strings, handle_reporting);
            }
            fd.revents = 0;
        }
    }
}

template <size_t Idx, size_t... IdxTail>
const char*
name_by_id(const uint32_t id, std::index_sequence<Idx, IdxTail...>)
{
    if(Idx == id) return page_migration_info<Idx>::name;
    if constexpr(sizeof...(IdxTail) > 0)
        return name_by_id(id, std::index_sequence<IdxTail...>{});
    else
        return nullptr;
}

template <size_t... Idx>
void
get_ids(std::vector<uint32_t>& _id_list, std::index_sequence<Idx...>)
{
    auto _emplace = [](auto& _vec, uint32_t _v) {
        if(_v < static_cast<uint32_t>(ROCPROFILER_HSA_AMD_EXT_API_ID_LAST)) _vec.emplace_back(_v);
    };

    (_emplace(_id_list, page_migration_info<Idx>::operation), ...);
}

bool
context_filter(const context::context* ctx)
{
    return (ctx->buffered_tracer &&
            (ctx->buffered_tracer->domains(ROCPROFILER_BUFFER_TRACING_PAGE_MIGRATION)));
}

template <size_t... Idx>
void
to_bitmask(small_vector<size_t>& _id_list, std::index_sequence<Idx...>)
{
    auto _emplace = [](auto& _vec, uint32_t _v) {
        if(_v < static_cast<uint32_t>(ROCPROFILER_HSA_AMD_EXT_API_ID_LAST)) _vec.emplace_back(_v);
    };

    (_emplace(_id_list, page_migration_info<Idx>::operation), ...);
}

template <size_t... Inxs>
rocprofiler_status_t init(std::index_sequence<Inxs...>)
{
    static const small_vector<size_t> event_ids{Inxs...};
    // Check if version is more than 1.11
    auto ver = kfd::get_version();
    if(ver.major_version * 1000 + ver.minor_version > 1011)
    {
        if(!context::get_registered_contexts(context_filter).empty())
        {
            config::init(event_ids);
        }
        return ROCPROFILER_STATUS_SUCCESS;
    }
    else
    {
        // Add a buffer record with this info
        ROCP_ERROR << fmt::format(
            "KFD does not support SVM event reporting in v{}.{} (requires v1.11)",
            ver.major_version,
            ver.minor_version);
        return ROCPROFILER_STATUS_ERROR_INCOMPATIBLE_KERNEL;
    }
}
}  // namespace

}  // namespace page_migration
}  // namespace rocprofiler

namespace rocprofiler::page_migration
{
rocprofiler_status_t
init()
{
    pthread_atfork(nullptr, nullptr, []() {
        // null out child's copy on fork and reinitialize
        // otherwise all children wait on the same thread to join
        config::reset_on_fork();
        init(std::make_index_sequence<ROCPROFILER_PAGE_MIGRATION_LAST>{});
    });

    return init(std::make_index_sequence<ROCPROFILER_PAGE_MIGRATION_LAST>{});
}

void
finalize()
{
    config::reset();
}

const char*
name_by_id(uint32_t id)
{
    return name_by_id(id, std::make_index_sequence<ROCPROFILER_PAGE_MIGRATION_LAST>{});
}

std::vector<uint32_t>
get_ids()
{
    auto _data = std::vector<uint32_t>{};
    _data.reserve(ROCPROFILER_PAGE_MIGRATION_LAST);
    get_ids(_data, std::make_index_sequence<ROCPROFILER_PAGE_MIGRATION_LAST>{});
    return _data;
}
}  // namespace rocprofiler::page_migration
