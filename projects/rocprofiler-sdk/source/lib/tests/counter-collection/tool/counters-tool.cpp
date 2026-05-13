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

#include "counters-tool.hpp"
#include <rocprofiler-sdk/agent.h>
#include <rocprofiler-sdk/callback_tracing.h>
#include <rocprofiler-sdk/context.h>
#include <rocprofiler-sdk/counter_config.h>
#include <rocprofiler-sdk/counters.h>
#include <rocprofiler-sdk/dispatch_counting_service.h>
#include <rocprofiler-sdk/fwd.h>
#include <rocprofiler-sdk/registration.h>
#include <rocprofiler-sdk/rocprofiler.h>

#include <cxxabi.h>
#include <fmt/core.h>

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <future>
#include <list>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#define ROCPROFILER_CALL(result, msg)                                                              \
    {                                                                                              \
        rocprofiler_status_t _status = result;                                                     \
        if(_status != ROCPROFILER_STATUS_SUCCESS)                                                  \
        {                                                                                          \
            auto errmsg = fmt::format("[" #result "][{}:{}] {} failed with error code {}: {}",     \
                                      __FILE__,                                                    \
                                      __LINE__,                                                    \
                                      msg,                                                         \
                                      static_cast<int>(_status),                                   \
                                      rocprofiler_get_status_string(_status));                     \
            fmt::println(stderr, "{}", errmsg);                                                    \
            throw std::runtime_error(errmsg);                                                      \
        }                                                                                          \
    }

namespace
{
bool
verbose()
{
    static bool _v = (std::getenv("ROCPROFILER_LOG_LEVEL") != nullptr);
    return _v;
}

std::string
demangle(const char* mangled)
{
    int   status    = 0;
    char* demangled = abi::__cxa_demangle(mangled, nullptr, nullptr, &status);

    std::string result{mangled};
    if(status == 0 && demangled)
    {
        result = demangled;
    }

    std::free(demangled);
    return result;
}

template <typename T>
class synced
{
public:
    synced() = default;
    explicit synced(T v)
    : _data{std::move(v)}
    {}

    synced(const synced&) = delete;
    synced& operator=(const synced&) = delete;
    synced(synced&&)                 = delete;
    synced& operator=(synced&&) = delete;

    template <typename Fn>
    decltype(auto) wlock(Fn&& fn)
    {
        auto _lk = std::unique_lock{_mut};
        return fn(_data);
    }

    template <typename Fn>
    decltype(auto) rlock(Fn&& fn) const
    {
        auto _lk = std::unique_lock{_mut};
        return fn(_data);
    }

private:
    mutable std::mutex _mut{};
    T                  _data{};
};

void
code_object_tracing_callback(rocprofiler_callback_tracing_record_t record,
                             rocprofiler_user_data_t* /* user_data */,
                             void* /* data */);

// called before kernel is enqueued into HSA queue
void
kernel_dispatch_callback(rocprofiler_dispatch_counting_service_data_t dispatch_data,
                         rocprofiler_counter_config_id_t*             config,
                         rocprofiler_user_data_t*                     user_data,
                         void* /*callback_data_args*/);

// called when counters are ready for a given dispatch
void
counter_record_callback(rocprofiler_dispatch_counting_service_data_t dispatch_data,
                        rocprofiler_counter_record_t*                record_data,
                        size_t                                       record_count,
                        rocprofiler_user_data_t                      user_data,
                        void* /*callback_data_args*/);
}  // namespace

namespace tool
{
struct kernel_name_t
{
    std::string pretty{};
    std::string mangled{};
};

using counter_info_t = rocprofiler_counter_info_v1_t;

struct counters_cache_t
{
    std::unordered_map<std::string, counter_info_t>           by_name{};
    std::unordered_map<rocprofiler_counter_id_t, std::string> by_handle{};

    void insert(const counter_info_t& info)
    {
        by_name.emplace(info.name, info);
        by_handle.emplace(info.id, info.name);
    }
};

struct collection_state_t
{
    using agent_config_map_t =
        std::unordered_map<rocprofiler_agent_id_t, rocprofiler_counter_config_id_t>;

    agent_config_map_t                 agent_configs{};
    std::atomic<int>                   pending_dispatches{0};
    std::atomic<bool>                  sealed{false};
    std::atomic<bool>                  fulfilled{false};
    synced<collection_results_t>       results{};
    std::promise<collection_results_t> promise{};

    void try_fulfill()
    {
        if(!sealed.load(std::memory_order_acquire)) return;
        if(pending_dispatches.load(std::memory_order_acquire) != 0) return;
        if(fulfilled.exchange(true, std::memory_order_acq_rel)) return;
        results.rlock([&](const auto& r) { promise.set_value(r); });
    }
};

using agent_info_map_t    = std::unordered_map<rocprofiler_agent_id_t, rocprofiler_agent_t>;
using agent_counter_map_t = std::unordered_map<rocprofiler_agent_id_t, counters_cache_t>;

struct tool_ctx_t
{
    rocprofiler_client_id_t*      id       = {nullptr};
    rocprofiler_client_finalize_t finalize = {nullptr};

    rocprofiler_context_id_t kernel_cb_ctx{};
    rocprofiler_context_id_t codeobj_cb_ctx{};

    synced<std::unordered_map<uint64_t, kernel_name_t>> kernel_map{};

    agent_info_map_t    agent_info_map{};
    agent_counter_map_t agent_counter_map{};

    synced<std::list<collection_state_t>> collections{};
    collection_state_t*                   active_collection = nullptr;

    tool_ctx_t()                  = delete;
    tool_ctx_t(const tool_ctx_t&) = delete;
    tool_ctx_t& operator=(const tool_ctx_t&) = delete;
    tool_ctx_t(tool_ctx_t&&)                 = delete;
    tool_ctx_t& operator=(tool_ctx_t&&) = delete;

    static tool_ctx_t& ctx();

    static int  init(rocprofiler_client_finalize_t finalize_func, void* tool_data);
    static void fini(void* tool_data);
};

tool_ctx_t&
ctx()
{
    static tool_ctx_t _ctx{};
    return _ctx;
}

int
init(rocprofiler_client_finalize_t finalize_func, void* /* tool_data */)
{
    auto& ctx = tool::ctx();

    ctx.finalize = finalize_func;

    // initialize
    // * ROCPROFILER_CALLBACK_TRACING_CODE_OBJECT       => kernel id to name mapping
    // * ROCPROFILER_CALLBACK_TRACING_KERNEL_DISPATCH   => kernel dispatch callback to set counters

    ROCPROFILER_CALL(rocprofiler_create_context(&ctx.kernel_cb_ctx), "failed to create context");
    ROCPROFILER_CALL(rocprofiler_create_context(&ctx.codeobj_cb_ctx), "failed to create context");

    ROCPROFILER_CALL(
        rocprofiler_configure_callback_tracing_service(
            ctx.codeobj_cb_ctx,                       /* rocprofiler_context_id_t               */
            ROCPROFILER_CALLBACK_TRACING_CODE_OBJECT, /* rocprofiler_callback_tracing_kind_t    */
            nullptr,                                  /* const rocprofiler_tracing_operation_t* */
            0,                                        /* size_t operations_count                */
            code_object_tracing_callback,             /* rocprofiler_callback_tracing_cb_t      */
            &ctx.codeobj_cb_ctx                       /* void* callback_args                    */
            ),
        "failed to configure kernel code-object tracing service");

    ROCPROFILER_CALL(rocprofiler_configure_callback_dispatch_counting_service(
                         ctx.kernel_cb_ctx,        /* rocprofiler_context_id_t                    */
                         kernel_dispatch_callback, /* rocprofiler_dispatch_counting_service_cb_t  */
                         nullptr,                  /* void* dispatch_callback_args                */
                         counter_record_callback,  /* rocprofiler_dispatch_counting_record_cb_t   */
                         nullptr                   /* void* record_callback_args                  */
                         ),
                     "failed to configure kernel dispatch counters tracing service");

    ROCPROFILER_CALL(rocprofiler_start_context(ctx.codeobj_cb_ctx), "failed to start context");

    const auto iterate_agents = [](rocprofiler_agent_version_t version,
                                   const void**                _agents,
                                   size_t                      _num_agents,
                                   void* /* user_data */) {
        const auto iterate_counters = [](rocprofiler_agent_id_t    agent_id,
                                         rocprofiler_counter_id_t* counters,
                                         size_t                    num_counters,
                                         void* /* user_data */) {
            auto& cache = tool::ctx().agent_counter_map[agent_id];

            for(size_t i = 0; i < num_counters; ++i)
            {
                counter_info_t counter_info{};

                ROCPROFILER_CALL(
                    rocprofiler_query_counter_info(
                        counters[i], ROCPROFILER_COUNTER_INFO_VERSION_1, &counter_info),
                    fmt::format("failed to query counters[{}] info {} on agent {}",
                                i,
                                counters[i].handle,
                                agent_id.handle));

                cache.insert(counter_info);
            }
            return ROCPROFILER_STATUS_SUCCESS;
        };

        const auto* agents = reinterpret_cast<const rocprofiler_agent_t* const*>(_agents);

        // std::span, my beloved
        for(size_t i = 0; i < _num_agents; ++i)
        {
            const auto& agent = *agents[i];
            if(agent.type != ROCPROFILER_AGENT_TYPE_GPU) continue;

            if(verbose()) fmt::println("enumerating agent[{}] : {}", i, agent.product_name);

            tool::ctx().agent_info_map.emplace(agent.id, agent);

            ROCPROFILER_CALL(rocprofiler_iterate_agent_supported_counters(
                                 agent.id,          // rocprofiler_agent_id_t
                                 iterate_counters,  // rocprofiler_available_counters_cb_t
                                 nullptr            // void* user_data
                                 ),
                             fmt::format("failed to query counters for agent[{}] {} ({:05x})",
                                         i,
                                         agent.product_name,
                                         agent.id.handle));
        }

        return ROCPROFILER_STATUS_SUCCESS;
    };

    // query available counters
    ROCPROFILER_CALL(
        rocprofiler_query_available_agents(
            ROCPROFILER_AGENT_INFO_VERSION_0, iterate_agents, sizeof(rocprofiler_agent_t), nullptr),
        "failed to query available agents");

    return ROCPROFILER_STATUS_SUCCESS;
}

void
fini(void* /* tool_data */)
{
    const auto& cctx = tool::ctx();

    size_t total_collections = 0;
    size_t total_kernels     = 0;
    cctx.collections.rlock([&](const auto& list) { total_collections = list.size(); });
    cctx.kernel_map.rlock([&](const auto& map) { total_kernels = map.size(); });

    fmt::println("{} fini: {} collections, {} kernels, {} agents",
                 tool::ctx().id->name,
                 total_collections,
                 total_kernels,
                 cctx.agent_info_map.size());
}

std::future<collection_results_t>
set_counters_config(const counters_t& config)
{
    auto& ctx = tool::ctx();

    for(const auto& counter : config)
    {
        for(const auto& [agent_id, cache] : ctx.agent_counter_map)
        {
            if(cache.by_name.find(counter) == cache.by_name.end())
            {
                throw std::runtime_error(fmt::format(
                    "counter '{}' not found on agent {:05x}", counter, agent_id.handle));
            }
        }
    }

    return ctx.collections.wlock([&](auto& list) -> std::future<collection_results_t> {
        list.emplace_back();
        auto& coll            = list.back();
        ctx.active_collection = &coll;

        for(const auto& [agent_id, cache] : ctx.agent_counter_map)
        {
            auto counter_ids = std::vector<rocprofiler_counter_id_t>{};
            counter_ids.reserve(config.size());
            for(const auto& name : config)
                counter_ids.push_back(cache.by_name.at(name).id);

            rocprofiler_counter_config_id_t config_id{};
            ROCPROFILER_CALL(
                rocprofiler_create_counter_config(
                    agent_id, counter_ids.data(), counter_ids.size(), &config_id),
                fmt::format("failed to create counter config on agent {:05x}", agent_id.handle));

            coll.agent_configs[agent_id] = config_id;

            if(verbose())
                fmt::println(stderr,
                             "set_counters_config: agent={:05x} config={:#x} ({} counters)",
                             agent_id.handle,
                             config_id.handle,
                             counter_ids.size());
        }

        return coll.promise.get_future();
    });
}

void
start_counter_collection()
{
    ROCPROFILER_CALL(rocprofiler_start_context(tool::ctx().kernel_cb_ctx),
                     "failed to start kernel callback context");
}

void
stop_counter_collection()
{
    ROCPROFILER_CALL(rocprofiler_stop_context(tool::ctx().kernel_cb_ctx),
                     "failed to stop kernel callback context");

    auto* coll = tool::ctx().active_collection;
    if(coll)
    {
        coll->sealed.store(true, std::memory_order_release);
        coll->try_fulfill();
    }
}

}  // namespace tool

namespace
{
using namespace tool;

// called when kernel code object / kernel is loaded
void
code_object_tracing_callback(rocprofiler_callback_tracing_record_t record,
                             rocprofiler_user_data_t* /* user_data */,
                             void* /* data */)
{
    using kernel_codeobj_rec_t =
        rocprofiler_callback_tracing_code_object_kernel_symbol_register_data_t;
    auto& ctx = tool::ctx();
    if(record.kind == ROCPROFILER_CALLBACK_TRACING_CODE_OBJECT &&
       record.operation == ROCPROFILER_CODE_OBJECT_DEVICE_KERNEL_SYMBOL_REGISTER &&
       record.phase == ROCPROFILER_CALLBACK_PHASE_LOAD)
    {
        ctx.kernel_map.wlock([&](auto& map) {
            const auto* rec      = static_cast<kernel_codeobj_rec_t*>(record.payload);
            const auto& inserted = map.emplace(
                rec->kernel_id, kernel_name_t{demangle(rec->kernel_name), rec->kernel_name});

            if(verbose())
            {
                fmt::println(stderr,
                             "code_object_tracing_callback: Kernel id {:2} : {:40} : {}",
                             rec->kernel_id,
                             inserted.first->second.mangled,
                             inserted.first->second.pretty);
            }
        });
    }
}

// called before kernel is enqueued into HSA queue
void
kernel_dispatch_callback(rocprofiler_dispatch_counting_service_data_t dispatch_data,
                         rocprofiler_counter_config_id_t*             config,
                         rocprofiler_user_data_t*                     user_data,
                         void* /*callback_data_args*/)
{
    const auto& ctx      = tool::ctx();
    auto*       coll     = ctx.active_collection;
    auto        agent_id = dispatch_data.dispatch_info.agent_id;

    bool is_internal = false;
    ctx.kernel_map.rlock([&](const auto& map) {
        auto it = map.find(dispatch_data.dispatch_info.kernel_id);
        if(it != map.end()) is_internal = (it->second.mangled.rfind("__", 0) == 0);
    });

    if(is_internal) return;

    *config = coll->agent_configs.at(agent_id);
    coll->pending_dispatches.fetch_add(1, std::memory_order_relaxed);
    user_data->ptr = coll;

    if(verbose())
    {
        ctx.kernel_map.rlock([&](const auto& map) {
            fmt::println(
                stderr,
                "kernel_dispatch_callback: kernel={} dispatch={} agent={:05x} config={:#x} : {}",
                dispatch_data.dispatch_info.kernel_id,
                dispatch_data.dispatch_info.dispatch_id,
                agent_id.handle,
                config->handle,
                map.at(dispatch_data.dispatch_info.kernel_id).pretty);
        });
    }
}

// called when counters are ready for a given dispatch
void
counter_record_callback(rocprofiler_dispatch_counting_service_data_t dispatch_data,
                        rocprofiler_counter_record_t*                record_data,
                        size_t                                       record_count,
                        rocprofiler_user_data_t                      user_data,
                        void* /*callback_data_args*/)
{
    auto&       ctx       = tool::ctx();
    auto*       coll      = static_cast<tool::collection_state_t*>(user_data.ptr);
    const auto  agent_id  = dispatch_data.dispatch_info.agent_id;
    const auto  kernel_id = dispatch_data.dispatch_info.kernel_id;
    const auto& cache     = ctx.agent_counter_map.at(agent_id);

    const auto dispatch_id = dispatch_data.dispatch_info.dispatch_id;

    if(verbose())
    {
        ctx.kernel_map.rlock([&](const auto& map) {
            fmt::println(stderr,
                         "counter_record_callback: kernel={} dispatch={} agent={:05x} records={}",
                         map.at(kernel_id).pretty,
                         dispatch_id,
                         agent_id.handle,
                         record_count);
        });
    }

    coll->results.wlock([&](auto& results) {
        auto& counters = results[agent_id][kernel_id][dispatch_id];

        for(size_t i = 0; i < record_count; ++i)
        {
            rocprofiler_counter_id_t counter_id{};
            ROCPROFILER_CALL(rocprofiler_query_record_counter_id(record_data[i].id, &counter_id),
                             "failed to query counter record");

            const auto& name = cache.by_handle.at(counter_id);
            counters[name].values.push_back(record_data[i].counter_value);
        }
    });

    coll->pending_dispatches.fetch_sub(1, std::memory_order_acq_rel);
    coll->try_fulfill();
}
}  // namespace

namespace tool
{
double
counter_results_t::sum() const
{
    double s = 0;
    for(auto v : values)
        s += v;
    return s;
}

double
counter_results_t::min() const
{
    if(values.empty()) return 0;
    double m = values.front();
    for(auto v : values)
        m = std::min(m, v);
    return m;
}

double
counter_results_t::max() const
{
    if(values.empty()) return 0;
    double m = values.front();
    for(auto v : values)
        m = std::max(m, v);
    return m;
}

size_t
counter_results_t::size() const
{
    return values.size();
}

std::string
kernel_name(uint64_t kernel_id)
{
    std::string name{};
    tool::ctx().kernel_map.rlock([&](const auto& map) {
        auto it = map.find(kernel_id);
        if(it == map.end())
            throw std::runtime_error(fmt::format("kernel_name: unknown kernel_id {}", kernel_id));
        name = it->second.pretty;
    });
    return name;
}
}  // namespace tool

extern "C" {
rocprofiler_tool_configure_result_t*
rocprofiler_configure(uint32_t                 version,
                      const char*              runtime_version,
                      uint32_t                 priority,
                      rocprofiler_client_id_t* id) __attribute__((used));

rocprofiler_tool_configure_result_t*
rocprofiler_configure(uint32_t                 version,
                      const char*              runtime_version,
                      uint32_t                 priority,
                      rocprofiler_client_id_t* id)
{
    tool::ctx().id = id;
    id->name       = "aql-counters-test";

    uint32_t major = version / 10000;
    uint32_t minor = (version % 10000) / 100;
    uint32_t patch = version % 100;

    fmt::println("{} (priority={}) is using rocprofiler-sdk v{}.{}.{} ({})",
                 id->name,
                 priority,
                 major,
                 minor,
                 patch,
                 runtime_version);

    static auto cfg = rocprofiler_tool_configure_result_t{
        sizeof(rocprofiler_tool_configure_result_t), &tool::init, &tool::fini, nullptr};
    return &cfg;
}
}
