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

/**
 * @file tests/kernel-replay-dispatch/client.cpp
 *
 * @brief LD_PRELOAD tool that replays the app's `bump` dispatch three times and, on pass 1,
 * submits its own copy of `substitute` through replay_set_dispatch, with its own kernel argument
 * buffer and twice the grid. It loads that copy through HSA during CONFIG PHASE_ENTER, the way a
 * kernel rewriter would, and reads the app's output buffer at every PASS PHASE_EXIT, before replay
 * restores it.
 *
 * Environment:
 *   KR_DISPATCH_ATT=1      also trace every pass with dispatch thread trace and check that pass 1
 *                          reports the substitute's kernel id
 *   KR_DISPATCH_CLUSTER=1  the app launched a cluster dispatch; require app_dispatch to say so
 */

#include <rocprofiler-sdk/agent.h>
#include <rocprofiler-sdk/callback_tracing.h>
#include <rocprofiler-sdk/experimental/kernel_replay.h>
#include <rocprofiler-sdk/experimental/thread_trace.h>
#include <rocprofiler-sdk/fwd.h>
#include <rocprofiler-sdk/registration.h>
#include <rocprofiler-sdk/rocprofiler.h>

#include <hsa/hsa.h>
#include <hsa/hsa_ext_amd.h>

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#define RC(call)                                                                                   \
    do                                                                                             \
    {                                                                                              \
        rocprofiler_status_t _s = (call);                                                          \
        if(_s != ROCPROFILER_STATUS_SUCCESS)                                                       \
        {                                                                                          \
            fprintf(stderr,                                                                        \
                    "[kd] error '%s' @%d: %s\n",                                                   \
                    #call,                                                                         \
                    __LINE__,                                                                      \
                    rocprofiler_get_status_string(_s));                                            \
            std::abort();                                                                          \
        }                                                                                          \
    } while(0)

namespace
{
constexpr uint32_t kBlock          = 67;
constexpr uint64_t kPasses         = 3;
constexpr uint64_t kSubstitutePass = 1;
constexpr char     kSubstitute[]   = "substitute";

rocprofiler_context_id_t g_replay_ctx{0};
rocprofiler_context_id_t g_code_ctx{0};
rocprofiler_context_id_t g_att_ctx{0};
bool                     g_att     = false;
bool                     g_cluster = false;

struct kernel_symbol
{
    uint64_t code_object_id = 0;
    uint64_t kernel_object  = 0;
    uint64_t kernel_id      = 0;
    uint64_t kernarg_size   = 0;
};

std::mutex                                           g_code_mut{};
std::map<uint64_t, std::vector<uint8_t>>             g_code_objects{};  // code object id -> bytes
std::map<std::string, kernel_symbol>                 g_app_kernels{};   // first symbol of each name
std::unordered_map<uint64_t, uint64_t>               g_kernel_ids{};   // kernel object -> kernel id
std::unordered_map<uint64_t, rocprofiler_agent_id_t> g_code_agents{};  // code object id -> agent

// Per replayed dispatch, set at CONFIG PHASE_ENTER.
uint64_t g_substitute_object = 0;
void*    g_tool_kernarg      = nullptr;
int*     g_app_out           = nullptr;
int      g_n                 = 0;
uint64_t g_bump_kernel_id    = 0;

std::atomic<int64_t>        g_current_pass{-1};
std::atomic<int>            g_failures{0};
std::atomic<int>            g_replayed{0};
std::atomic<int>            g_checked_passes{0};
std::mutex                  g_att_mut{};
std::map<int64_t, uint64_t> g_att_kernel_ids{};  // pass -> kernel id thread trace saw

void
fail(const char* what)
{
    fprintf(stderr, "[kd] FAIL: %s\n", what);
    g_failures.fetch_add(1);
}

void
expect_status(const char* what, rocprofiler_status_t got, rocprofiler_status_t want)
{
    if(got == want) return;
    fprintf(stderr,
            "[kd] FAIL: %s returned %s, expected %s\n",
            what,
            rocprofiler_get_status_string(got),
            rocprofiler_get_status_string(want));
    g_failures.fetch_add(1);
}

bool
covers(const rocprofiler_callback_tracing_kernel_replay_data_t* p, size_t end)
{
    return p->size >= end;
}

constexpr size_t kSetDispatchEnd =
    offsetof(rocprofiler_callback_tracing_kernel_replay_data_t, replay_set_dispatch) +
    sizeof(rocprofiler_kernel_replay_dispatch_cb_t);

void
code_object_cb(rocprofiler_callback_tracing_record_t record, rocprofiler_user_data_t*, void*)
{
    if(record.kind != ROCPROFILER_CALLBACK_TRACING_CODE_OBJECT) return;
    if(record.phase != ROCPROFILER_CALLBACK_PHASE_LOAD) return;

    if(record.operation == ROCPROFILER_CODE_OBJECT_LOAD)
    {
        auto* data =
            static_cast<rocprofiler_callback_tracing_code_object_load_data_t*>(record.payload);
        std::lock_guard<std::mutex> lk{g_code_mut};
        g_code_agents[data->code_object_id] = data->rocp_agent;
        if(data->storage_type == ROCPROFILER_CODE_OBJECT_STORAGE_TYPE_MEMORY &&
           data->memory_base != 0 && data->memory_size != 0)
        {
            const auto* base = reinterpret_cast<const uint8_t*>(data->memory_base);
            g_code_objects[data->code_object_id].assign(base, base + data->memory_size);
        }
    }
    else if(record.operation == ROCPROFILER_CODE_OBJECT_DEVICE_KERNEL_SYMBOL_REGISTER)
    {
        auto* data =
            static_cast<rocprofiler_callback_tracing_code_object_kernel_symbol_register_data_t*>(
                record.payload);
        std::lock_guard<std::mutex> lk{g_code_mut};
        g_kernel_ids[data->kernel_object] = data->kernel_id;
        auto name                         = std::string{data->kernel_name ? data->kernel_name : ""};
        if(auto pos = name.rfind(".kd"); pos != std::string::npos && pos + 3 == name.size())
            name.resize(pos);
        g_app_kernels.emplace(name,
                              kernel_symbol{data->code_object_id,
                                            data->kernel_object,
                                            data->kernel_id,
                                            data->kernarg_segment_size});
    }
}

// The HSA agent behind a rocprofiler agent, matched by KFD node id.
bool
find_hsa_agent(rocprofiler_agent_id_t id, hsa_agent_t& out)
{
    auto match = std::pair<uint64_t, uint32_t>{id.handle, UINT32_MAX};
    RC(rocprofiler_query_available_agents(
        ROCPROFILER_AGENT_INFO_VERSION_0,
        [](rocprofiler_agent_version_t, const void** agents, size_t n, void* data) {
            auto* m = static_cast<std::pair<uint64_t, uint32_t>*>(data);
            for(size_t i = 0; i < n; ++i)
            {
                auto* a = static_cast<const rocprofiler_agent_v0_t*>(agents[i]);
                if(a->id.handle == m->first) m->second = a->node_id;
            }
            return ROCPROFILER_STATUS_SUCCESS;
        },
        sizeof(rocprofiler_agent_v0_t),
        &match));
    if(match.second == UINT32_MAX) return false;

    struct search
    {
        uint32_t    node  = UINT32_MAX;
        hsa_agent_t agent = {};
        bool        found = false;
    };
    auto s = search{match.second};
    hsa_iterate_agents(
        [](hsa_agent_t agent, void* data) {
            auto*             ss   = static_cast<search*>(data);
            hsa_device_type_t type = HSA_DEVICE_TYPE_CPU;
            hsa_agent_get_info(agent, HSA_AGENT_INFO_DEVICE, &type);
            if(type != HSA_DEVICE_TYPE_GPU) return HSA_STATUS_SUCCESS;
            uint32_t node = UINT32_MAX;
            hsa_agent_get_info(
                agent, static_cast<hsa_agent_info_t>(HSA_AMD_AGENT_INFO_DRIVER_NODE_ID), &node);
            if(node != ss->node) return HSA_STATUS_SUCCESS;
            ss->agent = agent;
            ss->found = true;
            return HSA_STATUS_INFO_BREAK;
        },
        &s);
    out = s.agent;
    return s.found;
}

// The agent's coarse-grained global memory pool.
bool
find_device_pool(hsa_agent_t agent, hsa_amd_memory_pool_t& out)
{
    struct search
    {
        hsa_amd_memory_pool_t pool{};
        bool                  found = false;
    } s;
    hsa_amd_agent_iterate_memory_pools(
        agent,
        [](hsa_amd_memory_pool_t pool, void* data) {
            auto*             ss = static_cast<search*>(data);
            hsa_amd_segment_t segment{};
            hsa_amd_memory_pool_get_info(pool, HSA_AMD_MEMORY_POOL_INFO_SEGMENT, &segment);
            if(segment != HSA_AMD_SEGMENT_GLOBAL) return HSA_STATUS_SUCCESS;
            uint32_t flags = 0;
            hsa_amd_memory_pool_get_info(pool, HSA_AMD_MEMORY_POOL_INFO_GLOBAL_FLAGS, &flags);
            if((flags & HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_COARSE_GRAINED) == 0)
                return HSA_STATUS_SUCCESS;
            ss->pool  = pool;
            ss->found = true;
            return HSA_STATUS_INFO_BREAK;
        },
        &s);
    out = s.pool;
    return s.found;
}

// Load this process's copy of `substitute` into a new executable, as a kernel rewriter loads its
// rewritten code object. Returns its kernel object, or 0.
uint64_t
load_substitute(hsa_agent_t agent, std::vector<uint8_t>& code)
{
    hsa_code_object_reader_t reader{};
    if(hsa_code_object_reader_create_from_memory(code.data(), code.size(), &reader) !=
       HSA_STATUS_SUCCESS)
        return 0;
    hsa_executable_t exe{};
    if(hsa_executable_create_alt(
           HSA_PROFILE_FULL, HSA_DEFAULT_FLOAT_ROUNDING_MODE_DEFAULT, nullptr, &exe) !=
       HSA_STATUS_SUCCESS)
        return 0;
    if(hsa_executable_load_agent_code_object(exe, agent, reader, nullptr, nullptr) !=
           HSA_STATUS_SUCCESS ||
       hsa_executable_freeze(exe, nullptr) != HSA_STATUS_SUCCESS)
        return 0;

    hsa_executable_symbol_t sym{};
    if(hsa_executable_get_symbol_by_name(exe, "substitute.kd", &agent, &sym) !=
           HSA_STATUS_SUCCESS &&
       hsa_executable_get_symbol_by_name(exe, kSubstitute, &agent, &sym) != HSA_STATUS_SUCCESS)
        return 0;
    uint64_t object = 0;
    hsa_executable_symbol_get_info(sym, HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_OBJECT, &object);
    return object;
}

uint64_t replay_pass_count(rocprofiler_kernel_dispatch_info_t, rocprofiler_user_data_t)
{
    return kPasses;
}

void
config_enter(rocprofiler_callback_tracing_kernel_replay_data_t* p)
{
    if(!covers(p, kSetDispatchEnd) || p->app_dispatch == nullptr)
    {
        fprintf(stderr, "[kd] FAIL: CONFIG payload has no app_dispatch\n");
        g_failures.fetch_add(1);
        return;
    }

    const auto& app = *p->app_dispatch;
    if(app.grid_size.x != p->dispatch_info.grid_size.x ||
       app.workgroup_size.x != p->dispatch_info.workgroup_size.x || app.kernarg_address == nullptr)
        fail("app_dispatch does not describe the dispatch");
    if(p->replay_set_dispatch != nullptr)
        fail("replay_set_dispatch is only offered during PASS PHASE_ENTER");
    if(g_cluster && (app.cluster_size.x != 2 || app.cluster_size.y != 1 || app.cluster_size.z != 1))
        fail("the cluster dispatch was not described with its cluster size");
    // gfx1250 submits ordinary launches as extended dispatches with one workgroup per cluster.
    const bool one_per_cluster =
        (app.cluster_size.x == 0 && app.cluster_size.y == 0 && app.cluster_size.z == 0) ||
        (app.cluster_size.x == 1 && app.cluster_size.y == 1 && app.cluster_size.z == 1);
    if(!g_cluster && !one_per_cluster)
        fail("an ordinary launch reported a multi-workgroup cluster");

    hsa_agent_t           agent{};
    hsa_amd_memory_pool_t pool{};
    if(!find_hsa_agent(p->dispatch_info.agent_id, agent) || !find_device_pool(agent, pool))
    {
        fail("no HSA agent or device pool for the replayed agent");
        return;
    }

    auto code = std::vector<uint8_t>{};
    auto bump = kernel_symbol{};
    {
        std::lock_guard<std::mutex> lk{g_code_mut};
        auto                        itr = g_app_kernels.find(kSubstitute);
        auto                        b   = g_app_kernels.find("bump");
        if(itr == g_app_kernels.end() || b == g_app_kernels.end() ||
           g_code_objects.count(itr->second.code_object_id) == 0)
        {
            fprintf(stderr, "substitute code object unavailable\n");
            return;
        }
        code = g_code_objects.at(itr->second.code_object_id);
        bump = b->second;
    }
    g_bump_kernel_id = bump.kernel_id;

    // Our own executable; keep its bytes alive for as long as it is loaded.
    static auto substitute_code = std::vector<uint8_t>{};
    substitute_code             = std::move(code);
    g_substitute_object         = load_substitute(agent, substitute_code);
    if(g_substitute_object == 0)
    {
        fail("loading the substitute code object through HSA");
        return;
    }

    // Our kernel argument buffer: a copy of the app's (bump and substitute share a signature).
    const size_t bytes = bump.kernarg_size ? bump.kernarg_size : 16;
    if(hsa_amd_memory_pool_allocate(pool, bytes, 0, &g_tool_kernarg) != HSA_STATUS_SUCCESS ||
       hsa_amd_agents_allow_access(1, &agent, nullptr, g_tool_kernarg) != HSA_STATUS_SUCCESS)
    {
        fail("allocating the kernel argument buffer");
        return;
    }
    auto host = std::vector<uint8_t>(bytes);
    if(hsa_memory_copy(host.data(), app.kernarg_address, bytes) != HSA_STATUS_SUCCESS ||
       hsa_memory_copy(g_tool_kernarg, host.data(), bytes) != HSA_STATUS_SUCCESS)
    {
        fail("copying the app's kernel arguments");
        return;
    }
    std::memcpy(&g_app_out, host.data(), sizeof(g_app_out));
    std::memcpy(&g_n, host.data() + sizeof(int*), sizeof(g_n));

    p->replay_pass_count = replay_pass_count;
    g_replayed.fetch_add(1);
}

void
pass_enter(rocprofiler_callback_tracing_kernel_replay_data_t* p)
{
    g_current_pass.store(static_cast<int64_t>(p->current_pass));
    if(!covers(p, kSetDispatchEnd) || p->replay_set_dispatch == nullptr ||
       p->app_dispatch == nullptr)
    {
        fail("PASS payload has no replay_set_dispatch");
        return;
    }
    if(p->current_pass != kSubstitutePass) return;

    auto bad          = *p->app_dispatch;
    bad.kernel_object = 0x1234;
    expect_status("replacing with an unknown kernel",
                  p->replay_set_dispatch(&bad),
                  ROCPROFILER_STATUS_ERROR_INVALID_ARGUMENT);

    auto dispatch            = *p->app_dispatch;
    dispatch.kernel_object   = g_substitute_object;
    dispatch.kernarg_address = g_tool_kernarg;
    dispatch.grid_size.x *= 2;
    expect_status(
        "replacing the dispatch", p->replay_set_dispatch(&dispatch), ROCPROFILER_STATUS_SUCCESS);
    expect_status(
        "clearing the replacement", p->replay_set_dispatch(nullptr), ROCPROFILER_STATUS_SUCCESS);
    expect_status("replacing the dispatch again",
                  p->replay_set_dispatch(&dispatch),
                  ROCPROFILER_STATUS_SUCCESS);
}

// Read the app's buffer before replay restores it.
void
pass_exit(rocprofiler_callback_tracing_kernel_replay_data_t* p)
{
    g_current_pass.store(-1);
    if(g_app_out == nullptr || g_n <= 0) return;

    auto host = std::vector<int>(static_cast<size_t>(g_n) + 1, -1);
    if(hsa_memory_copy(host.data(), g_app_out, host.size() * sizeof(int)) != HSA_STATUS_SUCCESS)
    {
        fail("reading the output buffer at PASS PHASE_EXIT");
        return;
    }

    const bool substitute = p->current_pass == kSubstitutePass;
    const int  blocks     = (g_n + static_cast<int>(kBlock) - 1) / static_cast<int>(kBlock);
    for(int i = 0; i < g_n; ++i)
    {
        const int want = substitute ? 1000 + i : 1;
        if(host[i] != want)
        {
            fprintf(stderr,
                    "[kd] FAIL: pass %lu out[%d]=%d, expected %d\n",
                    static_cast<unsigned long>(p->current_pass),
                    i,
                    host[i],
                    want);
            g_failures.fetch_add(1);
            return;
        }
    }
    const int want_count = substitute ? 2 * blocks : 0;
    fprintf(stderr,
            "[kd] pass %lu ran %s, workgroups counted %d (expected %d)\n",
            static_cast<unsigned long>(p->current_pass),
            substitute ? "substitute" : "bump",
            host[g_n],
            want_count);
    if(host[g_n] != want_count) fail("the substitute did not run on the doubled grid");
    g_checked_passes.fetch_add(1);
}

void
kernel_replay_cb(rocprofiler_callback_tracing_record_t record, rocprofiler_user_data_t*, void*)
{
    if(record.kind != ROCPROFILER_CALLBACK_TRACING_KERNEL_REPLAY) return;
    auto* p = static_cast<rocprofiler_callback_tracing_kernel_replay_data_t*>(record.payload);
    if(p->dispatch_info.workgroup_size.x != kBlock) return;

    if(record.operation == ROCPROFILER_KERNEL_REPLAY_CONFIG &&
       record.phase == ROCPROFILER_CALLBACK_PHASE_ENTER)
        config_enter(p);
    else if(record.operation == ROCPROFILER_KERNEL_REPLAY_PASS)
    {
        if(record.phase == ROCPROFILER_CALLBACK_PHASE_ENTER)
            pass_enter(p);
        else
            pass_exit(p);
    }
}

rocprofiler_thread_trace_control_flags_t
att_dispatch_cb(rocprofiler_agent_id_t,
                rocprofiler_queue_id_t,
                rocprofiler_async_correlation_id_t,
                rocprofiler_kernel_id_t kernel_id,
                rocprofiler_dispatch_id_t,
                void*,
                rocprofiler_user_data_t*)
{
    const auto pass = g_current_pass.load();
    if(pass < 0) return ROCPROFILER_THREAD_TRACE_CONTROL_NONE;
    std::lock_guard<std::mutex> lk{g_att_mut};
    g_att_kernel_ids[pass] = kernel_id;
    return ROCPROFILER_THREAD_TRACE_CONTROL_START_AND_STOP;
}

void att_shader_cb(rocprofiler_thread_trace_shader_data_t, rocprofiler_user_data_t) {}

bool
configure_att()
{
    RC(rocprofiler_create_context(&g_att_ctx));
    auto agents = std::vector<rocprofiler_agent_id_t>{};
    RC(rocprofiler_query_available_agents(
        ROCPROFILER_AGENT_INFO_VERSION_0,
        [](rocprofiler_agent_version_t, const void** _agents, size_t n, void* data) {
            auto* out = static_cast<std::vector<rocprofiler_agent_id_t>*>(data);
            for(size_t i = 0; i < n; ++i)
            {
                auto* a = static_cast<const rocprofiler_agent_v0_t*>(_agents[i]);
                if(a->type == ROCPROFILER_AGENT_TYPE_GPU) out->push_back(a->id);
            }
            return ROCPROFILER_STATUS_SUCCESS;
        },
        sizeof(rocprofiler_agent_v0_t),
        &agents));
    auto params = std::vector<rocprofiler_thread_trace_parameter_t>{
        {ROCPROFILER_THREAD_TRACE_PARAMETER_SHADER_ENGINE_MASK, {0x1}}};
    bool any = false;
    for(auto id : agents)
        any |= rocprofiler_configure_dispatch_thread_trace_service(g_att_ctx,
                                                                   id,
                                                                   params.data(),
                                                                   params.size(),
                                                                   att_dispatch_cb,
                                                                   att_shader_cb,
                                                                   nullptr) ==
               ROCPROFILER_STATUS_SUCCESS;
    if(!any)
    {
        fprintf(stderr, "ATT unavailable\n");
        return false;
    }
    RC(rocprofiler_start_context(g_att_ctx));
    return true;
}

int
tool_init(rocprofiler_client_finalize_t, void*)
{
    const char* att     = std::getenv("KR_DISPATCH_ATT");
    g_att               = att && std::strcmp(att, "1") == 0;
    const char* cluster = std::getenv("KR_DISPATCH_CLUSTER");
    g_cluster           = cluster && std::strcmp(cluster, "1") == 0;

    RC(rocprofiler_create_context(&g_code_ctx));
    RC(rocprofiler_configure_callback_tracing_service(
        g_code_ctx, ROCPROFILER_CALLBACK_TRACING_CODE_OBJECT, nullptr, 0, code_object_cb, nullptr));
    RC(rocprofiler_start_context(g_code_ctx));

    RC(rocprofiler_create_context(&g_replay_ctx));
    RC(rocprofiler_configure_callback_tracing_service(g_replay_ctx,
                                                      ROCPROFILER_CALLBACK_TRACING_KERNEL_REPLAY,
                                                      nullptr,
                                                      0,
                                                      kernel_replay_cb,
                                                      nullptr));
    if(g_att && !configure_att()) return -1;
    RC(rocprofiler_start_context(g_replay_ctx));
    return 0;
}

void
tool_fini(void*)
{
    bool ok = g_failures.load() == 0;
    if(g_replayed.load() != 1)
    {
        fprintf(stderr, "[kd] FAIL: expected one replayed dispatch, saw %d\n", g_replayed.load());
        ok = false;
    }
    if(g_checked_passes.load() != static_cast<int>(kPasses))
    {
        fprintf(stderr,
                "[kd] FAIL: checked %d passes, expected %lu\n",
                g_checked_passes.load(),
                static_cast<unsigned long>(kPasses));
        ok = false;
    }
    if(g_att)
    {
        uint64_t substitute_id = 0;
        {
            std::lock_guard<std::mutex> lk{g_code_mut};
            if(auto itr = g_kernel_ids.find(g_substitute_object); itr != g_kernel_ids.end())
                substitute_id = itr->second;
        }
        std::lock_guard<std::mutex> lk{g_att_mut};
        for(uint64_t pass = 0; pass < kPasses; ++pass)
        {
            const auto want = pass == kSubstitutePass ? substitute_id : g_bump_kernel_id;
            const auto itr  = g_att_kernel_ids.find(static_cast<int64_t>(pass));
            const auto got  = itr != g_att_kernel_ids.end() ? itr->second : 0;
            fprintf(stderr,
                    "[kd] att pass %lu kernel id %lu (expected %lu)\n",
                    static_cast<unsigned long>(pass),
                    static_cast<unsigned long>(got),
                    static_cast<unsigned long>(want));
            if(want == 0 || got != want) ok = false;
        }
    }
    fprintf(stderr, ok ? "[kd] PASS\n" : "[kd] FAIL\n");
}
}  // namespace

extern "C" rocprofiler_tool_configure_result_t*
rocprofiler_configure(uint32_t, const char*, uint32_t priority, rocprofiler_client_id_t* id)
{
    if(priority > 0) return nullptr;
    id->name        = "kr-dispatch";
    static auto cfg = rocprofiler_tool_configure_result_t{
        sizeof(rocprofiler_tool_configure_result_t), &tool_init, &tool_fini, nullptr};
    return &cfg;
}
