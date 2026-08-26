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
 * @file tests/kernel-replay-local-context/client.cpp
 *
 * @brief LD_PRELOAD tool that replays a distinctive HIP kernel and locally starts/stops
 * per-service contexts across passes.
 *
 * Environment:
 *   KR_LC_SERVICES    comma list: counters, att, spm, pc-sampling
 *   KR_LC_PASSES      replay pass count (default 4)
 *   KR_LC_STOP_PASS   pass index at PHASE_ENTER where listed services are locally stopped.
 *                     0 = before any collection, 1 = after pass 0, <0 = never stop
 *   KR_LC_START_PASS  pass index at PHASE_ENTER where listed services are locally started
 *                     again (re-enable after a local stop). <0 = never local-start.
 *                     A local start cannot promote a globally-stopped context.
 *   KR_LC_KEEP        comma list of services that are never locally started/stopped
 */

#include <rocprofiler-sdk/agent.h>
#include <rocprofiler-sdk/buffer.h>
#include <rocprofiler-sdk/counters.h>
#include <rocprofiler-sdk/dispatch_counting_service.h>
#include <rocprofiler-sdk/experimental/kernel_replay.h>
#include <rocprofiler-sdk/experimental/spm.h>
#include <rocprofiler-sdk/experimental/thread_trace.h>
#include <rocprofiler-sdk/fwd.h>
#include <rocprofiler-sdk/internal_threading.h>
#include <rocprofiler-sdk/pc_sampling.h>
#include <rocprofiler-sdk/registration.h>
#include <rocprofiler-sdk/rocprofiler.h>

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <set>
#include <sstream>
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
                    "[lc] error '%s' @%d: %s\n",                                                   \
                    #call,                                                                         \
                    __LINE__,                                                                      \
                    rocprofiler_get_status_string(_s));                                            \
            std::abort();                                                                          \
        }                                                                                          \
    } while(0)

namespace
{
constexpr uint32_t TEST_BLOCK_X = 67;

rocprofiler_context_id_t g_replay_ctx{0};
rocprofiler_context_id_t g_counters_ctx{0};
rocprofiler_context_id_t g_att_ctx{0};
rocprofiler_context_id_t g_spm_ctx{0};
rocprofiler_context_id_t g_pcs_ctx{0};
rocprofiler_buffer_id_t  g_pcs_buffer{0};

std::set<std::string>   g_services{};
std::set<std::string>   g_keep{};
int64_t                 g_passes        = 4;
int64_t                 g_stop_pass     = 1;
int64_t                 g_start_pass    = -1;
rocprofiler_kernel_id_t g_target_kernel = UINT64_MAX;

std::atomic<int> g_counter_records{0};
std::atomic<int> g_spm_records{0};
std::atomic<int> g_att_shader{0};
std::atomic<int> g_pcs_samples{0};
std::atomic<int> g_local_stops{0};
std::atomic<int> g_local_starts{0};
std::atomic<int> g_replayed{0};

std::set<std::string>
parse_list(const char* env)
{
    std::set<std::string> out{};
    if(!env || *env == '\0') return out;
    std::stringstream ss{env};
    std::string       item{};
    while(std::getline(ss, item, ','))
    {
        if(!item.empty()) out.insert(item);
    }
    return out;
}

int64_t
env_i64(const char* name, int64_t fallback)
{
    const char* v = std::getenv(name);
    if(!v || *v == '\0') return fallback;
    return std::strtoll(v, nullptr, 10);
}

bool
wants(const char* name)
{
    return g_services.count(name) != 0;
}

bool
kept(const char* name)
{
    return g_keep.count(name) != 0;
}

std::vector<rocprofiler_agent_id_t>
gpu_agents()
{
    std::vector<rocprofiler_agent_id_t> agents{};
    RC(rocprofiler_query_available_agents(
        ROCPROFILER_AGENT_INFO_VERSION_0,
        [](rocprofiler_agent_version_t, const void** _agents, size_t n, void* data) {
            auto* out = static_cast<std::vector<rocprofiler_agent_id_t>*>(data);
            for(size_t i = 0; i < n; ++i)
            {
                auto* agent = static_cast<const rocprofiler_agent_v0_t*>(_agents[i]);
                if(agent->type == ROCPROFILER_AGENT_TYPE_GPU) out->push_back(agent->id);
            }
            return ROCPROFILER_STATUS_SUCCESS;
        },
        sizeof(rocprofiler_agent_v0_t),
        &agents));
    return agents;
}

uint64_t pass_count_cb(rocprofiler_kernel_dispatch_info_t, rocprofiler_user_data_t)
{
    return static_cast<uint64_t>(g_passes);
}

void
maybe_local_toggle(rocprofiler_callback_tracing_kernel_replay_data_t* p,
                   const char*                                        name,
                   rocprofiler_context_id_t                           ctx)
{
    if(ctx.handle == 0 || kept(name)) return;
    if(!p) return;

    if(g_start_pass >= 0 && static_cast<int64_t>(p->current_pass) == g_start_pass)
    {
        if(p->replay_local_enable_context_cb)
        {
            auto st = p->replay_local_enable_context_cb(ctx);
            if(st == ROCPROFILER_STATUS_SUCCESS)
                g_local_starts.fetch_add(1);
            else
                fprintf(stderr,
                        "[lc] local_start(%s) failed: %s\n",
                        name,
                        rocprofiler_get_status_string(st));
        }
    }

    if(g_stop_pass >= 0 && static_cast<int64_t>(p->current_pass) == g_stop_pass)
    {
        if(p->replay_local_disable_context_cb)
        {
            auto st = p->replay_local_disable_context_cb(ctx);
            if(st == ROCPROFILER_STATUS_SUCCESS)
                g_local_stops.fetch_add(1);
            else
                fprintf(stderr,
                        "[lc] local_stop(%s) failed: %s\n",
                        name,
                        rocprofiler_get_status_string(st));
        }
    }
}

void
kernel_replay_cb(rocprofiler_callback_tracing_record_t record, rocprofiler_user_data_t*, void*)
{
    if(record.kind != ROCPROFILER_CALLBACK_TRACING_KERNEL_REPLAY) return;
    auto* p = static_cast<rocprofiler_callback_tracing_kernel_replay_data_t*>(record.payload);
    if(p->dispatch_info.workgroup_size.x != TEST_BLOCK_X) return;

    if(record.operation == ROCPROFILER_KERNEL_REPLAY_CONFIG &&
       record.phase == ROCPROFILER_CALLBACK_PHASE_ENTER)
    {
        p->pass_count_cb = pass_count_cb;
        g_target_kernel  = p->dispatch_info.kernel_id;
        g_replayed.fetch_add(1);
        return;
    }

    if(record.operation != ROCPROFILER_KERNEL_REPLAY_PASS) return;
    if(record.phase != ROCPROFILER_CALLBACK_PHASE_ENTER) return;

    maybe_local_toggle(p, "counters", g_counters_ctx);
    maybe_local_toggle(p, "att", g_att_ctx);
    maybe_local_toggle(p, "spm", g_spm_ctx);
    maybe_local_toggle(p, "pc-sampling", g_pcs_ctx);
}

void
counter_record_cb(rocprofiler_dispatch_counting_service_data_t d,
                  rocprofiler_counter_record_t*,
                  size_t,
                  rocprofiler_user_data_t,
                  void*)
{
    if(d.dispatch_info.kernel_id == g_target_kernel) g_counter_records.fetch_add(1);
}

void
counter_dispatch_cb(rocprofiler_dispatch_counting_service_data_t d,
                    rocprofiler_counter_config_id_t*             config,
                    rocprofiler_user_data_t*,
                    void*)
{
    static std::mutex                                                    m{};
    static std::unordered_map<uint64_t, rocprofiler_counter_config_id_t> cache{};
    const auto agent = d.dispatch_info.agent_id;
    {
        std::lock_guard<std::mutex> lk{m};
        if(auto it = cache.find(agent.handle); it != cache.end())
        {
            *config = it->second;
            return;
        }
    }
    std::vector<rocprofiler_counter_id_t> all{};
    RC(rocprofiler_iterate_agent_supported_counters(
        agent,
        [](rocprofiler_agent_id_t, rocprofiler_counter_id_t* cs, size_t n, void* ud) {
            auto* v = static_cast<std::vector<rocprofiler_counter_id_t>*>(ud);
            for(size_t i = 0; i < n; ++i)
                v->push_back(cs[i]);
            return ROCPROFILER_STATUS_SUCCESS;
        },
        &all));
    std::vector<rocprofiler_counter_id_t> want{};
    for(auto cc : all)
    {
        rocprofiler_counter_info_v0_t info{};
        RC(rocprofiler_query_counter_info(cc, ROCPROFILER_COUNTER_INFO_VERSION_0, &info));
        if(info.name && std::string{info.name} == "SQ_WAVES") want.push_back(cc);
    }
    if(want.empty())
    {
        fprintf(stderr, "[lc] SQ_WAVES not found\n");
        std::abort();
    }
    rocprofiler_counter_config_id_t cfg{.handle = 0};
    RC(rocprofiler_create_counter_config(agent, want.data(), want.size(), &cfg));
    {
        std::lock_guard<std::mutex> lk{m};
        cache.emplace(agent.handle, cfg);
    }
    *config = cfg;
}

void
spm_record_cb(const rocprofiler_spm_dispatch_counting_service_data_t* d,
              const rocprofiler_spm_counter_record_t**,
              size_t,
              rocprofiler_spm_record_flag_t flags,
              rocprofiler_user_data_t,
              void*)
{
    if(!d) return;
    if((flags & ROCPROFILER_SPM_RECORD_FLAG_DISPATCH_END) == 0) return;
    if(d->dispatch_info.kernel_id == g_target_kernel) g_spm_records.fetch_add(1);
}

void
spm_dispatch_cb(const rocprofiler_spm_dispatch_counting_service_data_t* d,
                rocprofiler_counter_config_id_t*                        config,
                rocprofiler_user_data_t*,
                void*)
{
    static std::mutex                                                    m{};
    static std::unordered_map<uint64_t, rocprofiler_counter_config_id_t> cache{};
    const auto agent = d->dispatch_info.agent_id;
    {
        std::lock_guard<std::mutex> lk{m};
        if(auto it = cache.find(agent.handle); it != cache.end())
        {
            *config = it->second;
            return;
        }
    }
    std::vector<rocprofiler_counter_id_t> all{};
    RC(rocprofiler_spm_iterate_agent_supported_counters(
        agent,
        [](rocprofiler_agent_id_t, rocprofiler_counter_id_t* cs, size_t n, void* ud) {
            auto* v = static_cast<std::vector<rocprofiler_counter_id_t>*>(ud);
            for(size_t i = 0; i < n; ++i)
                v->push_back(cs[i]);
            return ROCPROFILER_STATUS_SUCCESS;
        },
        &all));
    std::vector<rocprofiler_counter_id_t> want{};
    for(auto cc : all)
    {
        rocprofiler_counter_info_v0_t info{};
        RC(rocprofiler_query_counter_info(cc, ROCPROFILER_COUNTER_INFO_VERSION_0, &info));
        if(info.name && std::string{info.name} == "SQ_WAVES") want.push_back(cc);
    }
    if(want.empty() && !all.empty()) want.push_back(all.front());
    if(want.empty())
    {
        fprintf(stderr, "[lc] no SPM counters\n");
        std::abort();
    }
    rocprofiler_spm_parameters_t param{
        .size  = sizeof(rocprofiler_spm_parameters_t),
        .type  = ROCPROFILER_SPM_PARAMETER_TYPE_SAMPLE_INTERVAL_SCLK_CYCLES,
        .value = 1200};
    rocprofiler_spm_parameters_t*   params[] = {&param};
    rocprofiler_counter_config_id_t cfg{.handle = 0};
    RC(rocprofiler_spm_create_counter_config(agent, want.data(), want.size(), params, 1, &cfg));
    {
        std::lock_guard<std::mutex> lk{m};
        cache.emplace(agent.handle, cfg);
    }
    *config = cfg;
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
    if(kernel_id != g_target_kernel) return ROCPROFILER_THREAD_TRACE_CONTROL_NONE;
    return ROCPROFILER_THREAD_TRACE_CONTROL_START_AND_STOP;
}

void att_shader_cb(rocprofiler_thread_trace_shader_data_t, rocprofiler_user_data_t)
{
    g_att_shader.fetch_add(1);
}

void
pcs_buffer_cb(rocprofiler_context_id_t,
              rocprofiler_buffer_id_t,
              rocprofiler_record_header_t** headers,
              size_t                        num_headers,
              void*,
              uint64_t)
{
    for(size_t i = 0; i < num_headers; ++i)
    {
        auto* h = headers[i];
        if(h && h->category == ROCPROFILER_BUFFER_CATEGORY_PC_SAMPLING) g_pcs_samples.fetch_add(1);
    }
}

bool
configure_counters()
{
    RC(rocprofiler_create_context(&g_counters_ctx));
    RC(rocprofiler_configure_callback_dispatch_counting_service(
        g_counters_ctx, counter_dispatch_cb, nullptr, counter_record_cb, nullptr));
    RC(rocprofiler_start_context(g_counters_ctx));
    return true;
}

bool
configure_att()
{
    RC(rocprofiler_create_context(&g_att_ctx));
    auto agents     = gpu_agents();
    auto parameters = std::vector<rocprofiler_thread_trace_parameter_t>{};
    parameters.push_back({ROCPROFILER_THREAD_TRACE_PARAMETER_SHADER_ENGINE_MASK, 0x1});
    bool any = false;
    for(auto id : agents)
    {
        auto st = rocprofiler_configure_dispatch_thread_trace_service(g_att_ctx,
                                                                      id,
                                                                      parameters.data(),
                                                                      parameters.size(),
                                                                      att_dispatch_cb,
                                                                      att_shader_cb,
                                                                      nullptr);
        if(st == ROCPROFILER_STATUS_SUCCESS)
            any = true;
        else
            fprintf(stderr,
                    "[lc] ATT configure agent %lu: %s\n",
                    static_cast<unsigned long>(id.handle),
                    rocprofiler_get_status_string(st));
    }
    if(!any)
    {
        fprintf(stderr, "ATT unavailable\n");
        return false;
    }
    RC(rocprofiler_start_context(g_att_ctx));
    return true;
}

bool
configure_spm()
{
    RC(rocprofiler_create_context(&g_spm_ctx));
    auto st = rocprofiler_spm_configure_callback_dispatch_service(
        g_spm_ctx, spm_dispatch_cb, nullptr, spm_record_cb, nullptr);
    if(st != ROCPROFILER_STATUS_SUCCESS)
    {
        fprintf(stderr, "SPM unavailable: %s\n", rocprofiler_get_status_string(st));
        return false;
    }
    RC(rocprofiler_start_context(g_spm_ctx));
    return true;
}

bool
configure_pcs()
{
    RC(rocprofiler_create_context(&g_pcs_ctx));
    RC(rocprofiler_create_buffer(g_pcs_ctx,
                                 8192,
                                 2048,
                                 ROCPROFILER_BUFFER_POLICY_LOSSLESS,
                                 pcs_buffer_cb,
                                 nullptr,
                                 &g_pcs_buffer));
    rocprofiler_callback_thread_t thread{};
    RC(rocprofiler_create_callback_thread(&thread));
    RC(rocprofiler_assign_callback_thread(g_pcs_buffer, thread));

    auto agents = gpu_agents();
    bool any    = false;
    for(auto id : agents)
    {
        std::vector<rocprofiler_pc_sampling_configuration_t> configs{};
        auto qst = rocprofiler_query_pc_sampling_agent_configurations(
            id,
            [](const rocprofiler_pc_sampling_configuration_t* cfgs, size_t n, void* ud) {
                auto* v = static_cast<std::vector<rocprofiler_pc_sampling_configuration_t>*>(ud);
                for(size_t i = 0; i < n; ++i)
                    v->push_back(cfgs[i]);
                return ROCPROFILER_STATUS_SUCCESS;
            },
            &configs);
        if(qst != ROCPROFILER_STATUS_SUCCESS || configs.empty()) continue;
        const auto& cfg = configs.front();
        auto        st  = rocprofiler_configure_pc_sampling_service(
            g_pcs_ctx, id, cfg.method, cfg.unit, cfg.min_interval, g_pcs_buffer, 0);
        if(st == ROCPROFILER_STATUS_SUCCESS) any = true;
    }
    if(!any)
    {
        fprintf(stderr, "PC sampling unavailable\n");
        return false;
    }
    RC(rocprofiler_start_context(g_pcs_ctx));
    return true;
}

int
expected_dispatch_records()
{
    // Contexts start globally active. PHASE_ENTER toggles apply before the pass
    // dispatches. Client applies start then stop, so a same-pass pair ends stopped.
    bool collecting = true;
    int  n          = 0;
    for(int pass = 0; pass < static_cast<int>(g_passes); ++pass)
    {
        if(g_start_pass >= 0 && pass == static_cast<int>(g_start_pass)) collecting = true;
        if(g_stop_pass >= 0 && pass == static_cast<int>(g_stop_pass)) collecting = false;
        if(collecting) ++n;
    }
    return n;
}

int
tool_init(rocprofiler_client_finalize_t, void*)
{
    g_services   = parse_list(std::getenv("KR_LC_SERVICES"));
    g_keep       = parse_list(std::getenv("KR_LC_KEEP"));
    g_passes     = env_i64("KR_LC_PASSES", 4);
    g_stop_pass  = env_i64("KR_LC_STOP_PASS", 1);
    g_start_pass = env_i64("KR_LC_START_PASS", -1);

    RC(rocprofiler_create_context(&g_replay_ctx));
    RC(rocprofiler_configure_callback_tracing_service(g_replay_ctx,
                                                      ROCPROFILER_CALLBACK_TRACING_KERNEL_REPLAY,
                                                      nullptr,
                                                      0,
                                                      kernel_replay_cb,
                                                      nullptr));

    if(g_services.empty())
    {
        fprintf(stderr, "[lc] KR_LC_SERVICES is empty\n");
        return -1;
    }
    if(wants("counters") && !configure_counters()) return -1;
    if(wants("att") && !configure_att()) return -1;
    if(wants("spm") && !configure_spm()) return -1;
    if(wants("pc-sampling") && !configure_pcs()) return -1;

    RC(rocprofiler_start_context(g_replay_ctx));
    return 0;
}

void
tool_fini(void*)
{
    if(g_pcs_buffer.handle != 0) rocprofiler_flush_buffer(g_pcs_buffer);

    bool ok = true;
    if(g_replayed.load() < 1)
    {
        fprintf(stderr, "[lc] FAIL: no replayed dispatch (workgroup %u)\n", TEST_BLOCK_X);
        ok = false;
    }

    auto check_exact = [&](const char* name, bool keep, int got) {
        int want = keep ? static_cast<int>(g_passes) : expected_dispatch_records();
        fprintf(stderr, "[lc] %s records=%d expected=%d keep=%d\n", name, got, want, keep);
        if(got != want)
        {
            fprintf(stderr, "[lc] FAIL: %s record count\n", name);
            ok = false;
        }
    };

    if(wants("counters")) check_exact("counters", kept("counters"), g_counter_records.load());
    if(wants("spm")) check_exact("spm", kept("spm"), g_spm_records.load());

    if(wants("att"))
    {
        const int  got         = g_att_shader.load();
        const bool keep        = kept("att");
        const bool expect_data = keep || expected_dispatch_records() > 0;
        fprintf(stderr, "[lc] att shader_callbacks=%d expect_data=%d\n", got, expect_data);
        if(expect_data && got == 0)
        {
            fprintf(stderr, "[lc] FAIL: ATT produced no shader data\n");
            ok = false;
        }
        if(!expect_data && got != 0)
        {
            fprintf(stderr, "[lc] FAIL: ATT ran while locally stopped from pass 0\n");
            ok = false;
        }
    }

    if(wants("pc-sampling"))
    {
        fprintf(stderr,
                "[lc] pc-sampling samples=%d local_starts=%d local_stops=%d "
                "(agent-wide service; local start/stop is a no-op for collection)\n",
                g_pcs_samples.load(),
                g_local_starts.load(),
                g_local_stops.load());
        const bool should_stop  = g_stop_pass >= 0 && !kept("pc-sampling");
        const bool should_start = g_start_pass >= 0 && !kept("pc-sampling");
        if(should_stop && g_local_stops.load() < 1)
        {
            fprintf(stderr, "[lc] FAIL: pc-sampling local_stop was not invoked successfully\n");
            ok = false;
        }
        if(should_start && g_local_starts.load() < 1)
        {
            fprintf(stderr, "[lc] FAIL: pc-sampling local_start was not invoked successfully\n");
            ok = false;
        }
    }

    fprintf(stderr, ok ? "[lc] PASS\n" : "[lc] FAIL\n");
}
}  // namespace

extern "C" rocprofiler_tool_configure_result_t*
rocprofiler_configure(uint32_t, const char*, uint32_t priority, rocprofiler_client_id_t* id)
{
    if(priority > 0) return nullptr;
    id->name        = "kr-local-context";
    static auto cfg = rocprofiler_tool_configure_result_t{
        sizeof(rocprofiler_tool_configure_result_t), &tool_init, &tool_fini, nullptr};
    return &cfg;
}
