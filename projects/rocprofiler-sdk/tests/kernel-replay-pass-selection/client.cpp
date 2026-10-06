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
 * @file tests/kernel-replay-pass-selection/client.cpp
 *
 * @brief LD_PRELOAD tool that replays a distinctive HIP kernel and decides, in each service's own
 * dispatch callback, which passes that service collects on.
 *
 * Every pass of a replayed dispatch is submitted through the queue interceptor on the replaying
 * thread, so each dispatch-scoped service calls its dispatch callback once per pass, on that
 * thread, between the pass's PASS PHASE_ENTER and PHASE_EXIT. The PASS callback publishes the pass
 * index in thread-local state and the dispatch callbacks consult it; that is the whole of per-pass
 * service selection. The checks count, per pass, what each service delivered for the replayed
 * kernel, and fail if any of its dispatch callbacks ran outside a pass.
 *
 * Environment:
 *   KR_PS_SERVICES    comma list: counters, att, spm, pc-sampling
 *   KR_PS_PASSES      replay pass count (default 4)
 *   KR_PS_STOP_PASS   pass from which the listed services stop collecting.
 *                     0 = before any collection, 1 = after pass 0, <0 = never stop
 *   KR_PS_START_PASS  pass from which they collect again after a stop. <0 = never.
 *   KR_PS_KEEP        comma list of services that collect on every pass
 *
 * PC sampling is agent-wide rather than dispatch-scoped, so it has no dispatch callback to select
 * passes with: it samples every pass, and its run only checks that replay completes under it.
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
                    "[kr-ps] error '%s' @%d: %s\n",                                                \
                    #call,                                                                         \
                    __LINE__,                                                                      \
                    rocprofiler_get_status_string(_s));                                            \
            std::abort();                                                                          \
        }                                                                                          \
    } while(0)

namespace
{
constexpr uint32_t TEST_BLOCK_X = 67;
constexpr int64_t  kMaxPasses   = 16;

using pass_counts_t = std::array<std::atomic<int>, kMaxPasses>;

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

pass_counts_t    g_counter_records{};
pass_counts_t    g_spm_records{};
pass_counts_t    g_att_traced{};
std::atomic<int> g_att_shader{0};
std::atomic<int> g_pcs_samples{0};
std::atomic<int> g_replayed{0};
std::atomic<int> g_pass_enters{0};
std::atomic<int> g_outside_pass{0};

// Pass in flight on this thread: published at PASS PHASE_ENTER and withdrawn at PHASE_EXIT. The
// SDK submits each pass between the two on this thread, so the service dispatch callbacks fired by
// that submit read the pass from here; any other dispatch sees -1.
thread_local int64_t tl_replay_pass = -1;

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

// Whether `name` collects on `pass`: from KR_PS_STOP_PASS it stops and from KR_PS_START_PASS it
// collects again, with the start applied first so a pass naming both ends stopped.
bool
collects_on(const char* name, int64_t pass)
{
    if(kept(name)) return true;
    bool collecting = true;
    for(int64_t p = 0; p <= pass; ++p)
    {
        if(g_start_pass >= 0 && p == g_start_pass) collecting = true;
        if(g_stop_pass >= 0 && p == g_stop_pass) collecting = false;
    }
    return collecting;
}

// The pass a dispatch callback for the replayed kernel runs on, or -1 (and a recorded failure) if
// it ran outside a pass, which would leave the tool nothing to select with.
int64_t
replay_pass_for(rocprofiler_kernel_id_t kernel_id)
{
    if(kernel_id != g_target_kernel) return -1;
    if(tl_replay_pass < 0) g_outside_pass.fetch_add(1);
    return tl_replay_pass;
}

void
count_on_pass(pass_counts_t& counts, uint64_t pass)
{
    if(pass < static_cast<uint64_t>(kMaxPasses)) counts.at(pass).fetch_add(1);
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

uint64_t replay_pass_count(rocprofiler_kernel_dispatch_info_t, rocprofiler_user_data_t)
{
    return static_cast<uint64_t>(g_passes);
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
        p->replay_pass_count = replay_pass_count;
        g_target_kernel      = p->dispatch_info.kernel_id;
        g_replayed.fetch_add(1);
        return;
    }

    if(record.operation != ROCPROFILER_KERNEL_REPLAY_PASS) return;
    if(record.phase == ROCPROFILER_CALLBACK_PHASE_ENTER)
    {
        g_pass_enters.fetch_add(1);
        tl_replay_pass = static_cast<int64_t>(p->current_pass);
    }
    else
    {
        tl_replay_pass = -1;
    }
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
    if(d.dispatch_info.kernel_id == g_target_kernel)
        count_on_pass(g_counter_records, user_data.value);
}

void
counter_dispatch_cb(rocprofiler_dispatch_counting_service_data_t d,
                    rocprofiler_counter_config_id_t*             config,
                    rocprofiler_user_data_t*                     user_data,
                    void*)
{
    const auto pass = replay_pass_for(d.dispatch_info.kernel_id);
    if(pass < 0 || !collects_on("counters", pass)) return;
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
        fprintf(stderr, "[kr-ps] SQ_WAVES not found\n");
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
    if(d->dispatch_info.kernel_id == g_target_kernel) count_on_pass(g_spm_records, user_data.value);
}

void
spm_dispatch_cb(const rocprofiler_spm_dispatch_counting_service_data_t* d,
                rocprofiler_counter_config_id_t*                        config,
                rocprofiler_user_data_t*                                user_data,
                void*)
{
    const auto pass = replay_pass_for(d->dispatch_info.kernel_id);
    if(pass < 0 || !collects_on("spm", pass)) return;
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
        fprintf(stderr, "[kr-ps] no SPM counters\n");
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
    const auto pass = replay_pass_for(kernel_id);
    if(pass < 0 || !collects_on("att", pass)) return ROCPROFILER_THREAD_TRACE_CONTROL_NONE;
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
                    "[kr-ps] ATT configure agent %lu: %s\n",
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
    g_services   = parse_list(std::getenv("KR_PS_SERVICES"));
    g_keep       = parse_list(std::getenv("KR_PS_KEEP"));
    g_passes     = env_i64("KR_PS_PASSES", 4);
    g_stop_pass  = env_i64("KR_PS_STOP_PASS", 1);
    g_start_pass = env_i64("KR_PS_START_PASS", -1);

    if(g_services.empty())
    {
        fprintf(stderr, "[kr-ps] KR_PS_SERVICES is empty\n");
        return -1;
    }
    if(g_passes < 2 || g_passes > kMaxPasses)
    {
        fprintf(stderr,
                "[kr-ps] KR_PS_PASSES=%ld is outside [2, %ld]\n",
                static_cast<long>(g_passes),
                static_cast<long>(kMaxPasses));
        return -1;
    }

    RC(rocprofiler_create_context(&g_replay_ctx));
    RC(rocprofiler_configure_callback_tracing_service(g_replay_ctx,
                                                      ROCPROFILER_CALLBACK_TRACING_KERNEL_REPLAY,
                                                      nullptr,
                                                      0,
                                                      kernel_replay_cb,
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
    if(g_replayed.load() != 1)
    {
        fprintf(stderr,
                "[kr-ps] FAIL: expected one replayed dispatch (workgroup %u), saw %d\n",
                TEST_BLOCK_X,
                g_replayed.load());
        ok = false;
    }
    if(g_pass_enters.load() != static_cast<int>(g_passes))
    {
        fprintf(stderr,
                "[kr-ps] FAIL: expected %ld passes, saw %d\n",
                static_cast<long>(g_passes),
                g_pass_enters.load());
        ok = false;
    }
    if(g_outside_pass.load() != 0)
    {
        fprintf(stderr,
                "[kr-ps] FAIL: %d dispatch callbacks for the replayed kernel ran outside a pass\n",
                g_outside_pass.load());
        ok = false;
    }

    auto check_passes = [&](const char* service, const char* what, const pass_counts_t& got) {
        for(int64_t pass = 0; pass < g_passes; ++pass)
        {
            const int want = collects_on(service, pass) ? 1 : 0;
            const int have = got.at(pass).load();
            fprintf(stderr,
                    "[kr-ps] %s pass=%ld %s=%d expected=%d\n",
                    service,
                    static_cast<long>(pass),
                    what,
                    have,
                    want);
            if(have != want)
            {
                fprintf(stderr,
                        "[kr-ps] FAIL: %s %s on pass %ld\n",
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

        bool any_traced = false;
        for(int64_t pass = 0; pass < g_passes; ++pass)
            any_traced = any_traced || collects_on("att", pass);

        const int got = g_att_shader.load();
        fprintf(stderr, "[kr-ps] att shader_callbacks=%d expect_data=%d\n", got, any_traced);
        if(any_traced && got == 0)
        {
            fprintf(stderr, "[kr-ps] FAIL: ATT produced no shader data\n");
            ok = false;
        }
        if(!any_traced && got != 0)
        {
            fprintf(stderr, "[kr-ps] FAIL: ATT produced shader data with no pass selected\n");
            ok = false;
        }
    }

    if(wants("pc-sampling"))
        fprintf(stderr,
                "[kr-ps] pc-sampling samples=%d (agent-wide service: samples every pass)\n",
                g_pcs_samples.load());

    fprintf(stderr, ok ? "[kr-ps] PASS\n" : "[kr-ps] FAIL\n");
}
}  // namespace

extern "C" rocprofiler_tool_configure_result_t*
rocprofiler_configure(uint32_t, const char*, uint32_t priority, rocprofiler_client_id_t* id)
{
    if(priority > 0) return nullptr;
    id->name        = "kr-pass-selection";
    static auto cfg = rocprofiler_tool_configure_result_t{
        sizeof(rocprofiler_tool_configure_result_t), &tool_init, &tool_fini, nullptr};
    return &cfg;
}
