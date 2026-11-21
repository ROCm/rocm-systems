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
 * @file tests/range-replay-local-context/client.cpp
 *
 * @brief LD_PRELOAD tool that replays a range with per-dispatch services enabled and locally
 * starts/stops their contexts across the replayed passes.
 *
 * The range replay counterpart of tests/kernel-replay-local-context. A range is replayed by
 * re-submitting its recording through the queue interceptor, so every per-dispatch service sees
 * each replayed dispatch through the same enter/exit hooks as a live one. Counting what each
 * service delivers for the range's kernel is what shows those hooks, and the localized toggles
 * they consult, hold up under range replay.
 *
 * Unlike kernel replay, pass 0 is the application's own execution of the range. It raises no PASS
 * callback, so it always collects with the contexts' global state; the toggles below apply only to
 * the replayed passes 1..N-1.
 *
 * Environment:
 *   RR_LC_SERVICES    comma list: counters, att, spm, pc-sampling
 *   RR_LC_PASSES      total passes including the application's own (default 4)
 *   RR_LC_STOP_PASS   replayed pass at whose PASS PHASE_ENTER the listed services are locally
 *                     stopped; < 1 = never
 *   RR_LC_START_PASS  replayed pass at whose PHASE_ENTER they are locally started again; < 1 =
 *                     never. A local start cannot promote a globally stopped context.
 *   RR_LC_KEEP        comma list of services that are never locally started/stopped
 */

#include "range.hpp"

#include <rocprofiler-sdk/agent.h>
#include <rocprofiler-sdk/buffer.h>
#include <rocprofiler-sdk/callback_tracing.h>
#include <rocprofiler-sdk/counters.h>
#include <rocprofiler-sdk/dispatch_counting_service.h>
#include <rocprofiler-sdk/experimental/range_replay.h>
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
                    "[rr-lc] error '%s' @%d: %s\n",                                                \
                    #call,                                                                         \
                    __LINE__,                                                                      \
                    rocprofiler_get_status_string(_s));                                            \
            std::abort();                                                                          \
        }                                                                                          \
    } while(0)

namespace
{
rocprofiler_context_id_t g_replay_ctx{0};
rocprofiler_context_id_t g_counters_ctx{0};
rocprofiler_context_id_t g_att_ctx{0};
rocprofiler_context_id_t g_spm_ctx{0};
rocprofiler_context_id_t g_pcs_ctx{0};
rocprofiler_buffer_id_t  g_pcs_buffer{0};

std::set<std::string> g_services{};
std::set<std::string> g_keep{};
int64_t               g_passes     = 4;
int64_t               g_stop_pass  = 1;
int64_t               g_start_pass = -1;

std::mutex         g_kernels_mtx{};
std::set<uint64_t> g_target_kernels{};

std::atomic<int> g_counter_records{0};
std::atomic<int> g_spm_records{0};
std::atomic<int> g_att_traced{0};
std::atomic<int> g_att_shader{0};
std::atomic<int> g_pcs_samples{0};
std::atomic<int> g_local_stops{0};
std::atomic<int> g_local_starts{0};

std::atomic<int>      g_pass_enters{0};
std::atomic<int>      g_closes{0};
std::atomic<int>      g_close_status{-1};
std::atomic<uint64_t> g_close_dispatches{0};
std::atomic<uint64_t> g_close_divergence{0};

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

bool
is_target_kernel(uint64_t kernel_id)
{
    auto lk = std::lock_guard<std::mutex>{g_kernels_mtx};
    return g_target_kernels.count(kernel_id) != 0;
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

uint64_t pass_count(uint64_t, rocprofiler_user_data_t) { return static_cast<uint64_t>(g_passes); }

void
maybe_local_toggle(const rocprofiler_callback_tracing_range_replay_data_t* p,
                   const char*                                             name,
                   rocprofiler_context_id_t                                ctx)
{
    if(ctx.handle == 0 || kept(name)) return;

    // Start before stop, so a pass that names both ends stopped, matching expected_passes().
    if(g_start_pass >= 1 && static_cast<int64_t>(p->current_pass) == g_start_pass &&
       p->replay_local_start_context_cb != nullptr)
    {
        auto st = p->replay_local_start_context_cb(ctx);
        if(st == ROCPROFILER_STATUS_SUCCESS)
            g_local_starts.fetch_add(1);
        else
            fprintf(stderr,
                    "[rr-lc] local_start(%s) failed: %s\n",
                    name,
                    rocprofiler_get_status_string(st));
    }

    if(g_stop_pass >= 1 && static_cast<int64_t>(p->current_pass) == g_stop_pass &&
       p->replay_local_stop_context_cb != nullptr)
    {
        auto st = p->replay_local_stop_context_cb(ctx);
        if(st == ROCPROFILER_STATUS_SUCCESS)
            g_local_stops.fetch_add(1);
        else
            fprintf(stderr,
                    "[rr-lc] local_stop(%s) failed: %s\n",
                    name,
                    rocprofiler_get_status_string(st));
    }
}

void
range_replay_cb(rocprofiler_callback_tracing_record_t record, rocprofiler_user_data_t*, void*)
{
    if(record.kind != ROCPROFILER_CALLBACK_TRACING_RANGE_REPLAY) return;
    auto* p = static_cast<rocprofiler_callback_tracing_range_replay_data_t*>(record.payload);
    if(p == nullptr || p->range_id != kRangeId) return;
    if(record.phase != ROCPROFILER_CALLBACK_PHASE_ENTER) return;

    switch(record.operation)
    {
        case ROCPROFILER_RANGE_REPLAY_CONFIG: p->pass_count_cb = pass_count; break;
        case ROCPROFILER_RANGE_REPLAY_PASS:
            g_pass_enters.fetch_add(1);
            maybe_local_toggle(p, "counters", g_counters_ctx);
            maybe_local_toggle(p, "att", g_att_ctx);
            maybe_local_toggle(p, "spm", g_spm_ctx);
            maybe_local_toggle(p, "pc-sampling", g_pcs_ctx);
            break;
        case ROCPROFILER_RANGE_REPLAY_CLOSE:
            g_closes.fetch_add(1);
            g_close_status.store(static_cast<int>(p->status));
            g_close_dispatches.store(p->dispatch_count);
            g_close_divergence.store(p->divergence_count);
            break;
        default: break;
    }
}

void
code_object_cb(rocprofiler_callback_tracing_record_t record, rocprofiler_user_data_t*, void*)
{
    if(record.kind != ROCPROFILER_CALLBACK_TRACING_CODE_OBJECT) return;
    if(record.operation != ROCPROFILER_CODE_OBJECT_DEVICE_KERNEL_SYMBOL_REGISTER) return;
    if(record.phase != ROCPROFILER_CALLBACK_PHASE_LOAD) return;

    const auto* data =
        static_cast<rocprofiler_callback_tracing_code_object_kernel_symbol_register_data_t*>(
            record.payload);
    if(data == nullptr || data->kernel_name == nullptr) return;
    if(std::strstr(data->kernel_name, kKernelName) == nullptr) return;

    auto lk = std::lock_guard<std::mutex>{g_kernels_mtx};
    g_target_kernels.insert(data->kernel_id);
}

void
counter_record_cb(rocprofiler_dispatch_counting_service_data_t d,
                  rocprofiler_counter_record_t*,
                  size_t,
                  rocprofiler_user_data_t,
                  void*)
{
    if(is_target_kernel(d.dispatch_info.kernel_id)) g_counter_records.fetch_add(1);
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
        fprintf(stderr, "[rr-lc] SQ_WAVES not found\n");
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
    if(is_target_kernel(d->dispatch_info.kernel_id)) g_spm_records.fetch_add(1);
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
        fprintf(stderr, "[rr-lc] no SPM counters\n");
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

// A locally stopped thread trace context returns before this is called, so counting the calls
// that start a trace counts the traced dispatches exactly.
rocprofiler_thread_trace_control_flags_t
att_dispatch_cb(rocprofiler_agent_id_t,
                rocprofiler_queue_id_t,
                rocprofiler_async_correlation_id_t,
                rocprofiler_kernel_id_t kernel_id,
                rocprofiler_dispatch_id_t,
                void*,
                rocprofiler_user_data_t*)
{
    if(!is_target_kernel(kernel_id)) return ROCPROFILER_THREAD_TRACE_CONTROL_NONE;
    g_att_traced.fetch_add(1);
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
                    "[rr-lc] ATT configure agent %lu: %s\n",
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

// Passes, out of RR_LC_PASSES, that a toggled service collects on. Pass 0 is the application's
// own run and always collects; the replayed passes follow the sticky toggles.
int
expected_passes()
{
    bool collecting = true;
    int  n          = 1;
    for(int64_t pass = 1; pass < g_passes; ++pass)
    {
        if(g_start_pass >= 1 && pass == g_start_pass) collecting = true;
        if(g_stop_pass >= 1 && pass == g_stop_pass) collecting = false;
        if(collecting) ++n;
    }
    return n;
}

int
tool_init(rocprofiler_client_finalize_t, void*)
{
    g_services   = parse_list(std::getenv("RR_LC_SERVICES"));
    g_keep       = parse_list(std::getenv("RR_LC_KEEP"));
    g_passes     = env_i64("RR_LC_PASSES", 4);
    g_stop_pass  = env_i64("RR_LC_STOP_PASS", 1);
    g_start_pass = env_i64("RR_LC_START_PASS", -1);

    if(g_services.empty())
    {
        fprintf(stderr, "[rr-lc] RR_LC_SERVICES is empty\n");
        return -1;
    }

    RC(rocprofiler_create_context(&g_replay_ctx));
    RC(rocprofiler_configure_callback_tracing_service(g_replay_ctx,
                                                      ROCPROFILER_CALLBACK_TRACING_RANGE_REPLAY,
                                                      nullptr,
                                                      0,
                                                      range_replay_cb,
                                                      nullptr));
    RC(rocprofiler_configure_callback_tracing_service(g_replay_ctx,
                                                      ROCPROFILER_CALLBACK_TRACING_CODE_OBJECT,
                                                      nullptr,
                                                      0,
                                                      code_object_cb,
                                                      nullptr));

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

    const auto status = g_close_status.load();
    fprintf(stderr,
            "[rr-lc] closes=%d status=%d dispatches=%lu divergence=%lu pass_enters=%d\n",
            g_closes.load(),
            status,
            static_cast<unsigned long>(g_close_dispatches.load()),
            static_cast<unsigned long>(g_close_divergence.load()),
            g_pass_enters.load());

    if(g_closes.load() != 1)
    {
        fprintf(stderr, "[rr-lc] FAIL: expected exactly one CLOSE for the range\n");
        ok = false;
    }
    if(status != static_cast<int>(ROCPROFILER_RANGE_REPLAY_STATUS_REPLAYED))
    {
        fprintf(stderr, "[rr-lc] FAIL: the range was not replayed (status %d)\n", status);
        ok = false;
    }
    if(g_close_dispatches.load() != kRangeDispatches)
    {
        fprintf(stderr, "[rr-lc] FAIL: CLOSE reported the wrong dispatch count\n");
        ok = false;
    }
    if(g_pass_enters.load() != static_cast<int>(g_passes - 1))
    {
        fprintf(stderr, "[rr-lc] FAIL: expected %ld replayed passes\n", (long) (g_passes - 1));
        ok = false;
    }
    // A service that perturbed the range's state during a replayed pass would show up here as a
    // region the final pass left different from the application's own run.
    if(std::getenv("ROCPROF_RANGE_REPLAY_VERIFY") != nullptr && g_close_divergence.load() != 0)
    {
        fprintf(stderr, "[rr-lc] FAIL: replayed passes diverged from the application's run\n");
        ok = false;
    }

    auto expected = [](const char* name) {
        const int passes = kept(name) ? static_cast<int>(g_passes) : expected_passes();
        return static_cast<int>(kRangeDispatches) * passes;
    };

    auto check_exact = [&](const char* service, const char* what, int got) {
        const int want = expected(service);
        fprintf(stderr,
                "[rr-lc] %s %s=%d expected=%d keep=%d\n",
                service,
                what,
                got,
                want,
                kept(service));
        if(got != want)
        {
            fprintf(stderr, "[rr-lc] FAIL: %s %s count\n", service, what);
            ok = false;
        }
    };

    if(wants("counters")) check_exact("counters", "records", g_counter_records.load());
    if(wants("spm")) check_exact("spm", "records", g_spm_records.load());

    if(wants("att"))
    {
        check_exact("att", "traced_dispatches", g_att_traced.load());
        fprintf(stderr, "[rr-lc] att shader_callbacks=%d\n", g_att_shader.load());
        if(g_att_shader.load() == 0)
        {
            fprintf(stderr, "[rr-lc] FAIL: ATT produced no shader data\n");
            ok = false;
        }
    }

    if(wants("pc-sampling"))
    {
        fprintf(stderr,
                "[rr-lc] pc-sampling samples=%d local_starts=%d local_stops=%d "
                "(agent-wide service; local start/stop is a no-op for collection)\n",
                g_pcs_samples.load(),
                g_local_starts.load(),
                g_local_stops.load());
        const bool should_stop = g_stop_pass >= 1 && g_stop_pass < g_passes && !kept("pc-sampling");
        const bool should_start =
            g_start_pass >= 1 && g_start_pass < g_passes && !kept("pc-sampling");
        if(should_stop && g_local_stops.load() < 1)
        {
            fprintf(stderr, "[rr-lc] FAIL: pc-sampling local_stop was not invoked successfully\n");
            ok = false;
        }
        if(should_start && g_local_starts.load() < 1)
        {
            fprintf(stderr, "[rr-lc] FAIL: pc-sampling local_start was not invoked successfully\n");
            ok = false;
        }
    }

    fprintf(stderr, ok ? "[rr-lc] PASS\n" : "[rr-lc] FAIL\n");
}
}  // namespace

extern "C" rocprofiler_tool_configure_result_t*
rocprofiler_configure(uint32_t, const char*, uint32_t priority, rocprofiler_client_id_t* id)
{
    if(priority > 0) return nullptr;
    id->name        = "rr-local-context";
    static auto cfg = rocprofiler_tool_configure_result_t{
        sizeof(rocprofiler_tool_configure_result_t), &tool_init, &tool_fini, nullptr};
    return &cfg;
}
