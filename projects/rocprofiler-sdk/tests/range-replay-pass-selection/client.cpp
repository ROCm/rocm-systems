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
 * @file tests/range-replay-pass-selection/client.cpp
 *
 * @brief LD_PRELOAD tool that replays a range with per-dispatch services enabled and decides, in
 * each service's own dispatch callback, which passes that service collects on.
 *
 * The range replay counterpart of tests/kernel-replay-pass-selection. A range is replayed by
 * re-submitting its recording through the queue interceptor on the thread that closes the range,
 * so every per-dispatch service sees each replayed dispatch through the same enter/exit hooks as a
 * live one and calls its dispatch callback on that thread while the pass is current. The PASS
 * callback publishes the pass index in thread-local state and the dispatch callbacks consult it.
 * Counting, per pass, what each service delivers for the range's kernel shows that the selection
 * holds up under range replay.
 *
 * Unlike kernel replay, pass 0 is the application's own execution of the range. It raises no PASS
 * callback, so its dispatches see no published pass and always collect; the selection below
 * applies only to the replayed passes 1..N-1.
 *
 * Environment:
 *   RR_PS_SERVICES    comma list: counters, att, spm, pc-sampling
 *   RR_PS_PASSES      total passes including the application's own (default 4)
 *   RR_PS_STOP_PASS   replayed pass from which the listed services stop collecting; < 1 = never
 *   RR_PS_START_PASS  replayed pass from which they collect again; < 1 = never
 *   RR_PS_KEEP        comma list of services that collect on every pass
 *
 * PC sampling is agent-wide rather than dispatch-scoped, so it has no dispatch callback to select
 * passes with: it samples every pass, and its run only checks that the range replays under it.
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

#include <array>
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
                    "[rr-ps] error '%s' @%d: %s\n",                                                \
                    #call,                                                                         \
                    __LINE__,                                                                      \
                    rocprofiler_get_status_string(_s));                                            \
            std::abort();                                                                          \
        }                                                                                          \
    } while(0)

namespace
{
constexpr int64_t kMaxPasses = 16;

using pass_counts_t = std::array<std::atomic<int>, kMaxPasses>;

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

pass_counts_t    g_counter_records{};
pass_counts_t    g_spm_records{};
pass_counts_t    g_att_traced{};
std::atomic<int> g_att_shader{0};
std::atomic<int> g_pcs_samples{0};

std::atomic<int>      g_pass_enters{0};
std::atomic<int>      g_closes{0};
std::atomic<int>      g_close_status{-1};
std::atomic<uint64_t> g_close_dispatches{0};
std::atomic<uint64_t> g_close_divergence{0};

// Replayed pass in flight on this thread: published at PASS PHASE_ENTER and withdrawn at
// PHASE_EXIT. The re-executed dispatches of that pass reach the dispatch callbacks on this thread
// in between; every other dispatch, including the application's own run of the range, sees -1.
thread_local int64_t tl_replay_pass = -1;

int64_t
current_pass()
{
    return (tl_replay_pass < 0) ? 0 : tl_replay_pass;
}

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

// Whether `name` collects on `pass`. The application's own run (pass 0) always collects; on the
// replayed passes a service stops from RR_PS_STOP_PASS and collects again from RR_PS_START_PASS,
// with the start applied first so a pass naming both ends stopped.
bool
collects_on(const char* name, int64_t pass)
{
    if(kept(name) || pass < 1) return true;
    bool collecting = true;
    for(int64_t p = 1; p <= pass; ++p)
    {
        if(g_start_pass >= 1 && p == g_start_pass) collecting = true;
        if(g_stop_pass >= 1 && p == g_stop_pass) collecting = false;
    }
    return collecting;
}

void
count_on_pass(pass_counts_t& counts, uint64_t pass)
{
    if(pass < static_cast<uint64_t>(kMaxPasses)) counts.at(pass).fetch_add(1);
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
range_replay_cb(rocprofiler_callback_tracing_record_t record, rocprofiler_user_data_t*, void*)
{
    if(record.kind != ROCPROFILER_CALLBACK_TRACING_RANGE_REPLAY) return;
    auto* p = static_cast<rocprofiler_callback_tracing_range_replay_data_t*>(record.payload);
    if(p == nullptr || p->range_id != kRangeId) return;

    const bool enter = (record.phase == ROCPROFILER_CALLBACK_PHASE_ENTER);
    switch(record.operation)
    {
        case ROCPROFILER_RANGE_REPLAY_CONFIG:
            if(enter) p->pass_count_cb = pass_count;
            break;
        case ROCPROFILER_RANGE_REPLAY_PASS:
            if(enter) g_pass_enters.fetch_add(1);
            tl_replay_pass = enter ? static_cast<int64_t>(p->current_pass) : -1;
            break;
        case ROCPROFILER_RANGE_REPLAY_CLOSE:
            if(!enter) break;
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

// The dispatch callback tags the dispatch with the pass it selected it on, and the SDK hands that
// user_data back with the dispatch's records, so a record is attributed to the pass that asked for
// it rather than to whichever pass happens to be current when the record arrives.
void
counter_record_cb(rocprofiler_dispatch_counting_service_data_t d,
                  rocprofiler_counter_record_t*,
                  size_t,
                  rocprofiler_user_data_t user_data,
                  void*)
{
    if(is_target_kernel(d.dispatch_info.kernel_id))
        count_on_pass(g_counter_records, user_data.value);
}

void
counter_dispatch_cb(rocprofiler_dispatch_counting_service_data_t d,
                    rocprofiler_counter_config_id_t*             config,
                    rocprofiler_user_data_t*                     user_data,
                    void*)
{
    const auto pass = current_pass();
    if(!collects_on("counters", pass)) return;
    user_data->value = static_cast<uint64_t>(pass);

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
        fprintf(stderr, "[rr-ps] SQ_WAVES not found\n");
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
              rocprofiler_user_data_t       user_data,
              void*)
{
    if(!d) return;
    if((flags & ROCPROFILER_SPM_RECORD_FLAG_DISPATCH_END) == 0) return;
    if(is_target_kernel(d->dispatch_info.kernel_id)) count_on_pass(g_spm_records, user_data.value);
}

void
spm_dispatch_cb(const rocprofiler_spm_dispatch_counting_service_data_t* d,
                rocprofiler_counter_config_id_t*                        config,
                rocprofiler_user_data_t*                                user_data,
                void*)
{
    const auto pass = current_pass();
    if(!collects_on("spm", pass)) return;
    user_data->value = static_cast<uint64_t>(pass);

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
        fprintf(stderr, "[rr-ps] no SPM counters\n");
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
    if(!is_target_kernel(kernel_id)) return ROCPROFILER_THREAD_TRACE_CONTROL_NONE;
    const auto pass = current_pass();
    if(!collects_on("att", pass)) return ROCPROFILER_THREAD_TRACE_CONTROL_NONE;
    count_on_pass(g_att_traced, static_cast<uint64_t>(pass));
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
                    "[rr-ps] ATT configure agent %lu: %s\n",
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
tool_init(rocprofiler_client_finalize_t, void*)
{
    g_services   = parse_list(std::getenv("RR_PS_SERVICES"));
    g_keep       = parse_list(std::getenv("RR_PS_KEEP"));
    g_passes     = env_i64("RR_PS_PASSES", 4);
    g_stop_pass  = env_i64("RR_PS_STOP_PASS", 1);
    g_start_pass = env_i64("RR_PS_START_PASS", -1);

    if(g_services.empty())
    {
        fprintf(stderr, "[rr-ps] RR_PS_SERVICES is empty\n");
        return -1;
    }
    if(g_passes < 2 || g_passes > kMaxPasses)
    {
        fprintf(stderr,
                "[rr-ps] RR_PS_PASSES=%ld is outside [2, %ld]\n",
                static_cast<long>(g_passes),
                static_cast<long>(kMaxPasses));
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
            "[rr-ps] closes=%d status=%d dispatches=%lu divergence=%lu pass_enters=%d\n",
            g_closes.load(),
            status,
            static_cast<unsigned long>(g_close_dispatches.load()),
            static_cast<unsigned long>(g_close_divergence.load()),
            g_pass_enters.load());

    if(g_closes.load() != 1)
    {
        fprintf(stderr, "[rr-ps] FAIL: expected exactly one CLOSE for the range\n");
        ok = false;
    }
    if(status != static_cast<int>(ROCPROFILER_RANGE_REPLAY_STATUS_REPLAYED))
    {
        fprintf(stderr, "[rr-ps] FAIL: the range was not replayed (status %d)\n", status);
        ok = false;
    }
    if(g_close_dispatches.load() != kRangeDispatches)
    {
        fprintf(stderr, "[rr-ps] FAIL: CLOSE reported the wrong dispatch count\n");
        ok = false;
    }
    if(g_pass_enters.load() != static_cast<int>(g_passes - 1))
    {
        fprintf(stderr, "[rr-ps] FAIL: expected %ld replayed passes\n", (long) (g_passes - 1));
        ok = false;
    }
    // A service that perturbed the range's state during a replayed pass would show up here as a
    // region the final pass left different from the application's own run.
    if(std::getenv("ROCPROF_RANGE_REPLAY_VERIFY") != nullptr && g_close_divergence.load() != 0)
    {
        fprintf(stderr, "[rr-ps] FAIL: replayed passes diverged from the application's run\n");
        ok = false;
    }

    auto check_passes = [&](const char* service, const char* what, const pass_counts_t& got) {
        for(int64_t pass = 0; pass < g_passes; ++pass)
        {
            const int want = collects_on(service, pass) ? static_cast<int>(kRangeDispatches) : 0;
            const int have = got.at(pass).load();
            fprintf(stderr,
                    "[rr-ps] %s pass=%ld %s=%d expected=%d\n",
                    service,
                    static_cast<long>(pass),
                    what,
                    have,
                    want);
            if(have != want)
            {
                fprintf(stderr,
                        "[rr-ps] FAIL: %s %s on pass %ld\n",
                        service,
                        what,
                        static_cast<long>(pass));
                ok = false;
            }
        }
    };

    if(wants("counters")) check_passes("counters", "records", g_counter_records);
    if(wants("spm")) check_passes("spm", "records", g_spm_records);

    if(wants("att"))
    {
        check_passes("att", "traced_dispatches", g_att_traced);
        fprintf(stderr, "[rr-ps] att shader_callbacks=%d\n", g_att_shader.load());
        if(g_att_shader.load() == 0)
        {
            fprintf(stderr, "[rr-ps] FAIL: ATT produced no shader data\n");
            ok = false;
        }
    }

    if(wants("pc-sampling"))
        fprintf(stderr,
                "[rr-ps] pc-sampling samples=%d (agent-wide service: samples every pass)\n",
                g_pcs_samples.load());

    fprintf(stderr, ok ? "[rr-ps] PASS\n" : "[rr-ps] FAIL\n");
}
}  // namespace

extern "C" rocprofiler_tool_configure_result_t*
rocprofiler_configure(uint32_t, const char*, uint32_t priority, rocprofiler_client_id_t* id)
{
    if(priority > 0) return nullptr;
    id->name        = "rr-pass-selection";
    static auto cfg = rocprofiler_tool_configure_result_t{
        sizeof(rocprofiler_tool_configure_result_t), &tool_init, &tool_fini, nullptr};
    return &cfg;
}
