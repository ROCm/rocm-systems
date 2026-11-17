// MIT License
//
// Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
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
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

#include "att_remap.hpp"

#include "lib/common/logging.hpp"
#include "lib/common/synchronized.hpp"

#include <rocprofiler-sdk/context.h>
#include <rocprofiler-sdk/experimental/thread_trace.h>

#include <fmt/format.h>

#include <dlfcn.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>
#include <algorithm>
#include <array>
#include <cctype>
#include <cerrno>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string_view>
#include <unordered_map>

// The declarations of projects/remapper/include/remapper.h this tool uses. The library reports its
// REMAPPER_ABI_VERSION, which must match.
extern "C" {
struct remapper_t;
struct remapper_session_t;

struct remapper_dim3_t
{
    uint32_t x;
    uint32_t y;
    uint32_t z;
};

struct remapper_dispatch_t
{
    uint64_t        kernel_object;
    uint64_t        kernarg_address;
    remapper_dim3_t workgroup_size;
    remapper_dim3_t grid_size;
    remapper_dim3_t cluster_size;
    uint32_t        group_segment_size;
    uint32_t        private_segment_size;
};

struct remapper_kernel_t
{
    uint64_t            size;
    uint64_t            dispatch_id;
    const char*         kernel_name;
    const void*         code_object;
    uint64_t            code_object_size;
    uint32_t            agent_node_id;
    uint32_t            reserved;
    remapper_dispatch_t dispatch;
};

struct remapper_pass_plan_t
{
    uint64_t            size;
    int32_t             replace_dispatch;
    int32_t             trace;
    remapper_dispatch_t dispatch;
    uint32_t            target_cu;
    uint32_t            simd;
    uint64_t            shader_engine_mask;
};
}

namespace rocprofiler
{
namespace tool
{
namespace att_remap
{
namespace
{
constexpr uint32_t remapper_abi_version = 1;

static_assert(sizeof(remapper_dispatch_t) == 64, "remapper.h layout");
static_assert(offsetof(remapper_kernel_t, dispatch) == 48, "remapper.h layout");
static_assert(sizeof(remapper_pass_plan_t) == 96, "remapper.h layout");

struct library_t
{
    uint32_t (*abi_version)()                                                    = nullptr;
    remapper_t* (*create)(const char*, const char*)                              = nullptr;
    void (*destroy)(remapper_t*)                                                 = nullptr;
    int (*wants_dispatch)(remapper_t*)                                           = nullptr;
    remapper_session_t* (*begin_dispatch)(remapper_t*, const remapper_kernel_t*) = nullptr;
    int (*pass_enter)(remapper_session_t*, uint64_t, remapper_pass_plan_t*)      = nullptr;
    int (*pass_exit)(remapper_session_t*, uint64_t)                              = nullptr;
    const char* (*session_dir)(remapper_session_t*)                              = nullptr;
    int (*end_dispatch)(remapper_session_t*)                                     = nullptr;
    const char* (*last_error)()                                                  = nullptr;

    remapper_t* remapper = nullptr;
};

library_t&
library()
{
    static auto* _v = new library_t{};
    return *_v;
}

/// One replay loop the remapper runs.
struct session_t
{
    remapper_session_t*      handle        = nullptr;
    uint64_t                 dispatch_id   = 0;
    rocprofiler_agent_id_t   agent         = {};
    int                      last_exit     = 0;
    bool                     traced        = false;
    rocprofiler_context_id_t trace_context = {};
};

thread_local session_t* tl_session = nullptr;

/// The device thread trace contexts of one agent
struct agent_trace_t
{
    uint64_t                                        target_cu          = 0;
    uint64_t                                        shader_engine_mask = 0;
    std::array<rocprofiler_context_id_t, num_simds> contexts           = {};
};

// agent handle -> its thread trace contexts. Written during tool initialization only.
auto agent_traces = common::Synchronized<std::unordered_map<uint64_t, agent_trace_t>>{};
// agent handle -> the pass it is tracing
auto traced_passes = common::Synchronized<std::unordered_map<uint64_t, traced_pass>>{};

auto code_objects = common::Synchronized<std::unordered_map<uint64_t, std::vector<uint8_t>>>{};
// dispatch directories of the finished sessions
auto finished = common::Synchronized<std::vector<std::string>>{};

std::string
last_error()
{
    const auto* msg = library().last_error ? library().last_error() : nullptr;
    return (msg != nullptr) ? std::string{msg} : std::string{};
}

uint64_t
pass_count(rocprofiler_kernel_dispatch_info_t /*dispatch_info*/, rocprofiler_user_data_t user_data)
{
    return (user_data.ptr != nullptr) ? 0 : 1;
}

int
continue_loop(rocprofiler_kernel_dispatch_info_t /*dispatch_info*/,
              uint64_t /*current_pass*/,
              uint64_t /*total_passes*/,
              rocprofiler_user_data_t user_data)
{
    const auto* s = static_cast<const session_t*>(user_data.ptr);
    return (s != nullptr && s->last_exit == 1) ? 1 : 0;
}

remapper_dim3_t
to_dim3(rocprofiler_dim3_t v)
{
    return {v.x, v.y, v.z};
}

rocprofiler_dim3_t
from_dim3(remapper_dim3_t v)
{
    return {v.x, v.y, v.z};
}

std::string
json_string(const std::string& s)
{
    auto out = std::string{"\""};
    for(char c : s)
    {
        if(c == '"' || c == '\\')
            out += fmt::format("\\{}", c);
        else if(static_cast<unsigned char>(c) < 0x20)
            out += fmt::format("\\u{:04x}", static_cast<unsigned>(c));
        else
            out += c;
    }
    return out + "\"";
}

std::string
percent_decode(const std::string& s)
{
    auto out = std::string{};
    for(size_t i = 0; i < s.size(); ++i)
    {
        if(s[i] == '%' && i + 2 < s.size() && std::isxdigit(s[i + 1]) != 0 &&
           std::isxdigit(s[i + 2]) != 0)
        {
            out += static_cast<char>(std::stoi(s.substr(i + 1, 2), nullptr, 16));
            i += 2;
        }
        else
            out += s[i];
    }
    return out;
}

// Stop the thread trace context `s` started for its current pass. The pass has completed, and the
// stop delivers its trace to the shader data callback before returning.
void
stop_trace(session_t& s, uint64_t pass)
{
    if(!s.traced) return;
    s.traced          = false;
    const auto status = rocprofiler_stop_context(s.trace_context);
    ROCP_ERROR_IF(status != ROCPROFILER_STATUS_SUCCESS)
        << "--att-remap: dispatch " << s.dispatch_id << " pass " << pass
        << ": stopping thread trace failed: " << rocprofiler_get_status_string(status);
    traced_passes.wlock([](auto& _passes, uint64_t _handle) { _passes.erase(_handle); },
                        s.agent.handle);
}

template <typename FuncT>
bool
resolve(void* handle, const char* name, FuncT& fn)
{
    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
    fn = reinterpret_cast<FuncT>(dlsym(handle, name));
    return fn != nullptr;
}
}  // namespace

bool
enabled()
{
    return get_config().att_remap;
}

void
init(const std::string& output_path)
{
    const auto& cfg    = get_config();
    auto&       lib    = library();
    auto*       handle = dlopen(cfg.att_remap_library.c_str(), RTLD_NOW | RTLD_LOCAL);
    ROCP_FATAL_IF(handle == nullptr)
        << "--att-remap: cannot load " << cfg.att_remap_library << ": " << dlerror();

    const bool resolved = resolve(handle, "remapper_abi_version", lib.abi_version) &&
                          resolve(handle, "remapper_create", lib.create) &&
                          resolve(handle, "remapper_destroy", lib.destroy) &&
                          resolve(handle, "remapper_wants_dispatch", lib.wants_dispatch) &&
                          resolve(handle, "remapper_begin_dispatch", lib.begin_dispatch) &&
                          resolve(handle, "remapper_pass_enter", lib.pass_enter) &&
                          resolve(handle, "remapper_pass_exit", lib.pass_exit) &&
                          resolve(handle, "remapper_session_dir", lib.session_dir) &&
                          resolve(handle, "remapper_end_dispatch", lib.end_dispatch) &&
                          resolve(handle, "remapper_last_error", lib.last_error);
    ROCP_FATAL_IF(!resolved) << "--att-remap: " << cfg.att_remap_library
                             << " is not a remapper library: " << dlerror();
    ROCP_FATAL_IF(lib.abi_version() != remapper_abi_version)
        << "--att-remap: " << cfg.att_remap_library << " has remapper ABI version "
        << lib.abi_version() << "; this rocprofv3 needs version " << remapper_abi_version;

    // Processes sharing an output path (MPI ranks, repeated runs) each get their own directory.
    auto output  = cfg.att_remap_output.empty()
                       ? fmt::format("{}/att_remap/{}", output_path, getpid())
                       : cfg.att_remap_output;
    lib.remapper = lib.create(output.c_str(), cfg.att_remap_options.c_str());
    ROCP_FATAL_IF(lib.remapper == nullptr) << "--att-remap: " << last_error();
    ROCP_INFO << "--att-remap: remapping into " << output;
}

void
fini(const std::string& trace_path)
{
    auto& lib = library();
    if(lib.remapper == nullptr) return;

    const auto& index = get_config().att_remap_index;
    finished.rlock([&index, &trace_path](const auto& dirs) {
        if(index.empty() || dirs.empty()) return;
        // Processes of one run may share the index, so each line is a single append.
        auto* file = std::fopen(index.c_str(), "a");
        if(file == nullptr)
        {
            ROCP_ERROR << "--att-remap: cannot append to " << index << ": " << std::strerror(errno);
            return;
        }
        for(const auto& dir : dirs)
        {
            auto line = fmt::format("{{\"dispatch_dir\": {}, \"trace_dir\": {}}}\n",
                                    json_string(dir),
                                    json_string(trace_path));
            std::fwrite(line.data(), 1, line.size(), file);
            std::fflush(file);
        }
        std::fclose(file);
    });

    lib.destroy(lib.remapper);
    lib.remapper = nullptr;
}

void
configure_agent(rocprofiler_agent_id_t                                   agent,
                uint64_t                                                 num_shader_engines,
                const std::vector<rocprofiler_thread_trace_parameter_t>& parameters,
                rocprofiler_thread_trace_shader_data_callback_t          shader_callback)
{
    auto trace = agent_trace_t{};
    trace.shader_engine_mask =
        (num_shader_engines >= 64) ? ~uint64_t{0} : (uint64_t{1} << num_shader_engines) - 1;

    auto base = std::vector<rocprofiler_thread_trace_parameter_t>{};
    for(const auto& itr : parameters)
    {
        if(itr.type == ROCPROFILER_THREAD_TRACE_PARAMETER_TARGET_CU) trace.target_cu = itr.value;
        if(itr.type != ROCPROFILER_THREAD_TRACE_PARAMETER_SIMD_SELECT &&
           itr.type != ROCPROFILER_THREAD_TRACE_PARAMETER_SHADER_ENGINE_MASK)
            base.emplace_back(itr);
    }
    base.push_back(
        {ROCPROFILER_THREAD_TRACE_PARAMETER_SHADER_ENGINE_MASK, {trace.shader_engine_mask}});

    for(uint64_t simd = 0; simd < num_simds; ++simd)
    {
        auto params = base;
        params.push_back({ROCPROFILER_THREAD_TRACE_PARAMETER_SIMD_SELECT, {simd}});

        auto& context = trace.contexts.at(simd);
        auto  status  = rocprofiler_create_context(&context);
        if(status == ROCPROFILER_STATUS_SUCCESS)
            status = rocprofiler_configure_device_thread_trace_service(
                context,
                agent,
                params.data(),
                params.size(),
                shader_callback,
                rocprofiler_user_data_t{.value = 0});
        ROCP_FATAL_IF(status != ROCPROFILER_STATUS_SUCCESS)
            << "--att-remap: configuring thread trace of SIMD " << simd << " on agent "
            << agent.handle << " failed: " << rocprofiler_get_status_string(status);
    }

    agent_traces.wlock(
        [](auto& _traces, uint64_t _handle, const agent_trace_t& _trace) {
            _traces[_handle] = _trace;
        },
        agent.handle,
        trace);
}

bool
traces_agent(rocprofiler_agent_id_t agent)
{
    return agent_traces.rlock(
        [](const auto& _traces, uint64_t _handle) { return _traces.count(_handle) > 0; },
        agent.handle);
}

void
add_code_object(uint64_t code_object_id, std::vector<uint8_t> bytes)
{
    code_objects.wlock(
        [code_object_id](auto& _objs, std::vector<uint8_t>&& _bytes) {
            _objs[code_object_id] = std::move(_bytes);
        },
        std::move(bytes));
}

bool
begin(rocprofiler_callback_tracing_kernel_replay_data_t* payload,
      rocprofiler_user_data_t*                           user_data,
      const dispatch_source&                             source)
{
    auto& lib = library();
    if(lib.remapper == nullptr || lib.wants_dispatch(lib.remapper) == 0) return false;

    constexpr auto setter_end =
        offsetof(rocprofiler_callback_tracing_kernel_replay_data_t, replay_set_dispatch) +
        sizeof(rocprofiler_kernel_replay_dispatch_cb_t);
    constexpr auto app_dispatch_end =
        offsetof(rocprofiler_kernel_replay_dispatch_t, private_segment_size) + sizeof(uint32_t);
    const auto& info = payload->dispatch_info;
    if(payload->size < setter_end || payload->app_dispatch == nullptr ||
       payload->app_dispatch->size < app_dispatch_end)
    {
        ROCP_ERROR << "--att-remap: dispatch " << info.dispatch_id
                   << " not remapped: this rocprofiler-sdk cannot replace a replay pass's dispatch";
        return false;
    }
    if(source.kernel_name == nullptr)
    {
        ROCP_ERROR << "--att-remap: dispatch " << info.dispatch_id
                   << " not remapped: unknown kernel " << info.kernel_id;
        return false;
    }

    // The code object outlives the call: its bytes are only removed when the map is.
    const auto* code = code_objects.rlock(
        [](const auto& _objs, uint64_t _id) -> const std::vector<uint8_t>* {
            auto itr = _objs.find(_id);
            return (itr != _objs.end()) ? &itr->second : nullptr;
        },
        source.code_object_id);
    if(code == nullptr || code->empty())
    {
        ROCP_ERROR << "--att-remap: dispatch " << info.dispatch_id << " (" << source.kernel_name
                   << ") not remapped: the bytes of code object " << source.code_object_id
                   << " were not captured";
        return false;
    }

    const auto& app         = *payload->app_dispatch;
    auto        kernel      = remapper_kernel_t{};
    kernel.size             = sizeof(kernel);
    kernel.dispatch_id      = info.dispatch_id;
    kernel.kernel_name      = source.kernel_name;
    kernel.code_object      = code->data();
    kernel.code_object_size = code->size();
    kernel.agent_node_id    = source.agent_node_id;
    kernel.dispatch =
        remapper_dispatch_t{app.kernel_object,
                            // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
                            reinterpret_cast<uint64_t>(app.kernarg_address),
                            to_dim3(app.workgroup_size),
                            to_dim3(app.grid_size),
                            to_dim3(app.cluster_size),
                            app.group_segment_size,
                            app.private_segment_size};

    auto* handle = lib.begin_dispatch(lib.remapper, &kernel);
    if(handle == nullptr)
    {
        ROCP_WARNING << "--att-remap: dispatch " << info.dispatch_id << " (" << source.kernel_name
                     << ") not remapped: " << last_error();
        return false;
    }

    auto* s        = new session_t{};
    s->handle      = handle;
    s->dispatch_id = info.dispatch_id;
    s->agent       = info.agent_id;
    user_data->ptr = s;
    tl_session     = s;

    payload->replay_pass_count = pass_count;
    payload->replay_continue   = continue_loop;
    ROCP_INFO << "--att-remap: remapping dispatch " << info.dispatch_id << " ("
              << source.kernel_name << ") into " << lib.session_dir(handle);
    return true;
}

bool
active()
{
    return tl_session != nullptr;
}

void
pass_enter(rocprofiler_callback_tracing_kernel_replay_data_t* payload,
           rocprofiler_user_data_t                            user_data)
{
    auto* s = static_cast<session_t*>(user_data.ptr);
    if(s == nullptr) return;

    const auto pass = payload->current_pass;
    auto       plan = remapper_pass_plan_t{};
    plan.size       = sizeof(plan);
    if(library().pass_enter(s->handle, pass, &plan) != 0)
        ROCP_ERROR << "--att-remap: dispatch " << s->dispatch_id << " pass " << pass << ": "
                   << last_error();

    s->traced = false;
    if(plan.replace_dispatch != 0)
    {
        auto d          = rocprofiler_kernel_replay_dispatch_t{};
        d.size          = sizeof(d);
        d.kernel_object = plan.dispatch.kernel_object;
        // NOLINTNEXTLINE(performance-no-int-to-ptr)
        d.kernarg_address      = reinterpret_cast<void*>(plan.dispatch.kernarg_address);
        d.workgroup_size       = from_dim3(plan.dispatch.workgroup_size);
        d.grid_size            = from_dim3(plan.dispatch.grid_size);
        d.cluster_size         = from_dim3(plan.dispatch.cluster_size);
        d.group_segment_size   = plan.dispatch.group_segment_size;
        d.private_segment_size = plan.dispatch.private_segment_size;
        auto status            = (payload->replay_set_dispatch != nullptr)
                                     ? payload->replay_set_dispatch(&d)
                                     : ROCPROFILER_STATUS_ERROR_NOT_AVAILABLE;
        // The remapper reads this pass's logs at PASS_EXIT and fails the stage on missing ones.
        ROCP_ERROR_IF(status != ROCPROFILER_STATUS_SUCCESS)
            << "--att-remap: dispatch " << s->dispatch_id << " pass " << pass
            << ": replacing the dispatch failed: " << rocprofiler_get_status_string(status);
    }

    if(plan.trace != 0)
    {
        // The pass is the only work on the agent between here and its PASS PHASE_EXIT: kernel
        // replay holds other dispatches off the agent for the whole loop, and restores device
        // memory after PHASE_EXIT.
        const auto target  = att_replay_target{plan.target_cu, plan.simd, plan.shader_engine_mask};
        const auto context = agent_traces.rlock(
            [](const auto& _traces, uint64_t _handle, const att_replay_target& _target) {
                auto itr = _traces.find(_handle);
                if(itr == _traces.end() || _target.simd_select >= num_simds ||
                   _target.target_cu != itr->second.target_cu ||
                   _target.shader_engine_mask != itr->second.shader_engine_mask)
                    return std::optional<rocprofiler_context_id_t>{};
                return std::make_optional(itr->second.contexts.at(_target.simd_select));
            },
            s->agent.handle,
            target);

        auto status = ROCPROFILER_STATUS_ERROR_CONTEXT_NOT_FOUND;
        if(context)
        {
            traced_passes.wlock(
                [](auto& _passes, uint64_t _handle, const traced_pass& _pass) {
                    _passes[_handle] = _pass;
                },
                s->agent.handle,
                traced_pass{s->dispatch_id, pass, target});
            status = rocprofiler_start_context(*context);
        }
        s->traced = (status == ROCPROFILER_STATUS_SUCCESS);
        if(s->traced)
        {
            s->trace_context = *context;
        }
        else
        {
            traced_passes.wlock([](auto& _passes, uint64_t _handle) { _passes.erase(_handle); },
                                s->agent.handle);
            ROCP_ERROR << fmt::format(
                "--att-remap: dispatch {} pass {} is not traced (target_cu={}, simd_select={}, "
                "shader_engine_mask={:#x}): {}",
                s->dispatch_id,
                pass,
                plan.target_cu,
                plan.simd,
                plan.shader_engine_mask,
                context ? rocprofiler_get_status_string(status)
                        : "no thread trace context is configured for this target");
        }
    }
}

void
pass_exit(rocprofiler_callback_tracing_kernel_replay_data_t* payload,
          rocprofiler_user_data_t                            user_data)
{
    auto* s = static_cast<session_t*>(user_data.ptr);
    if(s == nullptr) return;

    stop_trace(*s, payload->current_pass);

    auto result = library().pass_exit(s->handle, payload->current_pass);
    if(result < 0)
    {
        ROCP_ERROR << "--att-remap: dispatch " << s->dispatch_id << " pass "
                   << payload->current_pass << ": " << last_error();
        result = 0;
    }
    s->last_exit = result;
}

void
end(rocprofiler_user_data_t user_data)
{
    auto* s = static_cast<session_t*>(user_data.ptr);
    if(s == nullptr) return;

    auto& lib = library();
    auto  dir = std::string{lib.session_dir(s->handle)};
    switch(lib.end_dispatch(s->handle))
    {
        case 0:
            ROCP_INFO << "--att-remap: dispatch " << s->dispatch_id << " remapped: " << dir;
            break;
        case 1:
            ROCP_WARNING << "--att-remap: dispatch " << s->dispatch_id
                         << " ran with its own kernel after the remap failed; see " << dir
                         << "/report.json";
            break;
        default:
            ROCP_ERROR << "--att-remap: dispatch " << s->dispatch_id
                       << " replay loop ended before the remapper's schedule; see " << dir
                       << "/report.json";
            break;
    }
    finished.wlock([](auto& _dirs, std::string&& _dir) { _dirs.emplace_back(std::move(_dir)); },
                   std::move(dir));
    if(tl_session == s) tl_session = nullptr;
    delete s;
}

std::optional<traced_pass>
traced_pass_on(rocprofiler_agent_id_t agent)
{
    return traced_passes.rlock(
        [](const auto& _passes, uint64_t _handle) {
            auto itr = _passes.find(_handle);
            return (itr != _passes.end()) ? std::make_optional(itr->second) : std::nullopt;
        },
        agent.handle);
}

std::optional<std::vector<uint8_t>>
code_object_bytes(const rocprofiler_callback_tracing_code_object_load_data_t& code_object)
{
    if(code_object.storage_type == ROCPROFILER_CODE_OBJECT_STORAGE_TYPE_MEMORY)
    {
        if(code_object.memory_base == 0 || code_object.memory_size == 0) return std::nullopt;
        // NOLINTNEXTLINE(performance-no-int-to-ptr)
        const auto* begin = reinterpret_cast<const uint8_t*>(code_object.memory_base);
        return std::vector<uint8_t>(begin, begin + code_object.memory_size);
    }
    if(code_object.storage_type != ROCPROFILER_CODE_OBJECT_STORAGE_TYPE_FILE) return std::nullopt;

    constexpr auto prefix = std::string_view{"file://"};
    auto           text   = std::string{(code_object.uri != nullptr) ? code_object.uri : ""};
    if(text.compare(0, prefix.size(), prefix) != 0) return std::nullopt;
    text = text.substr(prefix.size());

    uint64_t offset = 0;
    uint64_t size   = 0;
    if(auto hash = text.find('#'); hash != std::string::npos)
    {
        auto fragment = text.substr(hash + 1);
        text          = text.substr(0, hash);
        for(size_t pos = 0; pos <= fragment.size();)
        {
            auto end   = std::min(fragment.find('&', pos), fragment.size());
            auto field = fragment.substr(pos, end - pos);
            auto eq    = field.find('=');
            if(eq != std::string::npos)
            {
                auto key   = field.substr(0, eq);
                auto value = std::strtoull(field.c_str() + eq + 1, nullptr, 0);
                if(key == "offset") offset = value;
                if(key == "size") size = value;
            }
            pos = end + 1;
        }
    }

    auto        fd    = code_object.storage_file;
    auto        owned = false;
    struct stat st    = {};
    if(fd < 0 || fstat(fd, &st) != 0)
    {
        fd    = open(percent_decode(text).c_str(), O_RDONLY | O_CLOEXEC);
        owned = true;
        if(fd < 0 || fstat(fd, &st) != 0)
        {
            if(fd >= 0) close(fd);
            return std::nullopt;
        }
    }
    const auto file_size = static_cast<uint64_t>(st.st_size);
    auto       bytes     = std::optional<std::vector<uint8_t>>{};
    if(offset <= file_size)
    {
        if(size == 0) size = file_size - offset;
        if(size <= file_size - offset)
        {
            bytes = std::vector<uint8_t>(size);
            for(uint64_t done = 0; bytes && done < size;)
            {
                auto n =
                    pread(fd, bytes->data() + done, size - done, static_cast<off_t>(offset + done));
                if(n <= 0)
                    bytes.reset();
                else
                    done += static_cast<uint64_t>(n);
            }
        }
    }
    if(owned) close(fd);
    return bytes;
}
}  // namespace att_remap
}  // namespace tool
}  // namespace rocprofiler
