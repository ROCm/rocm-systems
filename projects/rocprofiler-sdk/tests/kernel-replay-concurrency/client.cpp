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
 * @file tests/kernel-replay-concurrency/client.cpp
 *
 * @brief Client tool for the deterministic kernel-replay P0-1 concurrency regression test.
 *
 * LD_PRELOAD-ed alongside main.cpp. It replays the "hog" kernel (block 256, pass_count=5) and opts
 * the concurrent "victim" kernel (block 64) out of replay. A dummy SQ_WAVES counter service is
 * registered purely to force queue interposition.
 *
 * To make the P0-1 race deterministic, the tool exposes a handshake block via kr_coord_get() and
 * signals the replay-window phase boundaries so the workload can land the victim's write exactly
 * inside the snapshot -> restore window:
 *   - CONFIG enter (hog): wait until the victim has stamped V=OLD (old_ready) so the snapshot
 *     captures OLD.
 *   - PASS enter, pass 0 (hog): snapshot is done -> snapshot_done (victim may now write V=NEW).
 *   - CONFIG exit (hog): all passes + restores done -> window_done (victim may now read V).
 */

#include <rocprofiler-sdk/experimental/kernel_replay.h>
#include <rocprofiler-sdk/registration.h>
#include <rocprofiler-sdk/rocprofiler.h>

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <shared_mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#define RC(call)                                                                                   \
    do                                                                                             \
    {                                                                                              \
        rocprofiler_status_t _s = (call);                                                          \
        if(_s != ROCPROFILER_STATUS_SUCCESS)                                                       \
        {                                                                                          \
            fprintf(stderr,                                                                        \
                    "[tool] error '%s' @%d: %s\n",                                                 \
                    #call,                                                                         \
                    __LINE__,                                                                      \
                    rocprofiler_get_status_string(_s));                                            \
            std::abort();                                                                          \
        }                                                                                          \
    } while(0)

// Handshake block shared with the workload (resolved there via dlsym(RTLD_DEFAULT,
// "kr_coord_get")).
extern "C" {
struct kr_coord_t
{
    std::atomic<long> old_ready{0};      // victim -> tool/A: V=OLD ready for cycle k
    std::atomic<long> snapshot_done{0};  // tool -> victim: snapshot captured (write NEW now)
    std::atomic<long> window_done{0};    // tool -> victim: restores done (read now)
    std::atomic<long> cycle_done{0};     // victim -> A: cycle k fully finished
};

// Exported with default visibility on purpose: the tests build compiles with hidden visibility,
// but the workload resolves this symbol via dlsym(RTLD_DEFAULT, "kr_coord_get").
__attribute__((visibility("default"))) kr_coord_t*
kr_coord_get()
{
    static kr_coord_t c;
    return &c;
}
}

namespace
{
constexpr uint32_t       HOG_BLOCK_X = 256;
rocprofiler_context_id_t g_ctx{0};
std::atomic<long>        g_hog_cfg{0};
std::atomic<long>        g_victim_cfg{0};
std::atomic<long>        g_id_checks{0};  // KERNEL_REPLAY records whose dispatch_id was inspected
std::atomic<long>        g_id_bad{0};     // records with a bad (zero or changed) dispatch_id

// Reserved id of the hog replay currently in flight. Only the hog launcher thread runs the replay
// (one at a time, synchronously in the WriteInterceptor), so a plain global is safe.
rocprofiler_dispatch_id_t g_hog_dispatch_id{0};

uint64_t hog_pass_count(rocprofiler_kernel_dispatch_info_t, rocprofiler_user_data_t) { return 5; }

// A replay dispatch_id must be nonzero (0 is the "unset" sentinel that make_dispatch_info leaves)
// and identical across the CONFIG and every PASS of one dispatch. A violation prints a [repro] FAIL
// line (fails the ctest via FAIL_REGULAR_EXPRESSION) and bumps g_id_bad.
void
flag_bad_dispatch_id(const char* what, rocprofiler_dispatch_id_t id, uint64_t pass)
{
    fprintf(stderr,
            "[repro] FAIL: kernel-replay dispatch_id %s (id=%lu pass=%lu)\n",
            what,
            static_cast<unsigned long>(id),
            static_cast<unsigned long>(pass));
    g_id_bad.fetch_add(1, std::memory_order_relaxed);
}

void
kernel_replay_cb(rocprofiler_callback_tracing_record_t record, rocprofiler_user_data_t*, void*)
{
    if(record.kind != ROCPROFILER_CALLBACK_TRACING_KERNEL_REPLAY) return;
    auto*      p = static_cast<rocprofiler_callback_tracing_kernel_replay_data_t*>(record.payload);
    const bool hog = (p->dispatch_info.workgroup_size.x == HOG_BLOCK_X);
    auto*      c   = kr_coord_get();

    // Every replay record (hog or victim, any phase) must carry the reserved, nonzero dispatch id
    // -- these CONFIG/PASS callback records are not in rocprofv3 JSON, so this is where that is
    // checked.
    g_id_checks.fetch_add(1, std::memory_order_relaxed);
    if(p->dispatch_info.dispatch_id == 0)
        flag_bad_dispatch_id("is zero", p->dispatch_info.dispatch_id, p->current_pass);

    if(record.operation == ROCPROFILER_KERNEL_REPLAY_CONFIG &&
       record.phase == ROCPROFILER_CALLBACK_PHASE_ENTER)
    {
        if(!hog)  // victim: leave replay_pass_count NULL -> NOT replayed
        {
            g_victim_cfg.fetch_add(1, std::memory_order_relaxed);
            return;
        }
        p->replay_pass_count = hog_pass_count;
        g_hog_dispatch_id    = p->dispatch_info.dispatch_id;  // reused by every pass of this replay
        g_hog_cfg.fetch_add(1, std::memory_order_relaxed);
        // snapshot happens right after this returns -> make sure V=OLD is already set
        while(c->old_ready.load() == 0)
            std::this_thread::yield();
    }
    else if(!hog)
    {
        return;
    }
    else if(record.operation == ROCPROFILER_KERNEL_REPLAY_PASS)
    {
        // Every pass (enter and exit) must reuse the id reserved at CONFIG, unchanged.
        if(p->dispatch_info.dispatch_id != g_hog_dispatch_id)
            flag_bad_dispatch_id(
                "changed across passes", p->dispatch_info.dispatch_id, p->current_pass);
        if(record.phase == ROCPROFILER_CALLBACK_PHASE_ENTER && p->current_pass == 0)
            c->snapshot_done.store(
                c->old_ready.load());  // snapshot captured OLD -> victim writes NEW
    }
    else if(record.operation == ROCPROFILER_KERNEL_REPLAY_CONFIG &&
            record.phase == ROCPROFILER_CALLBACK_PHASE_EXIT)
    {
        if(p->dispatch_info.dispatch_id != g_hog_dispatch_id)
            flag_bad_dispatch_id(
                "changed at config exit", p->dispatch_info.dispatch_id, p->current_pass);
        c->window_done.store(c->old_ready.load());  // all restores done -> victim reads
    }
}

// Counter service: registered only to force queue interposition (and to exercise counter-buffer
// allocation during replay). The records themselves are ignored.
void
counter_record_cb(rocprofiler_dispatch_counting_service_data_t,
                  rocprofiler_counter_record_t*,
                  size_t,
                  rocprofiler_user_data_t,
                  void*)
{}

void
counter_dispatch_cb(rocprofiler_dispatch_counting_service_data_t d,
                    rocprofiler_counter_config_id_t*             config,
                    rocprofiler_user_data_t*,
                    void*)
{
    static std::shared_mutex                                             m{};
    static std::unordered_map<uint64_t, rocprofiler_counter_config_id_t> cache{};
    const auto agent = d.dispatch_info.agent_id;
    {
        auto rl = std::shared_lock{m};
        if(auto it = cache.find(agent.handle); it != cache.end())
        {
            *config = it->second;
            return;
        }
    }
    auto wl = std::unique_lock{m};
    if(auto it = cache.find(agent.handle); it != cache.end())
    {
        *config = it->second;
        return;
    }
    std::vector<rocprofiler_counter_id_t> all;
    RC(rocprofiler_iterate_agent_supported_counters(
        agent,
        [](rocprofiler_agent_id_t, rocprofiler_counter_id_t* cs, size_t n, void* ud) {
            auto* v = static_cast<std::vector<rocprofiler_counter_id_t>*>(ud);
            for(size_t i = 0; i < n; ++i)
                v->push_back(cs[i]);
            return ROCPROFILER_STATUS_SUCCESS;
        },
        &all));
    std::vector<rocprofiler_counter_id_t> want;
    for(auto cc : all)
    {
        rocprofiler_counter_info_v0_t info{};
        RC(rocprofiler_query_counter_info(cc, ROCPROFILER_COUNTER_INFO_VERSION_0, &info));
        if(info.name && std::string{info.name} == "SQ_WAVES") want.push_back(cc);
    }
    if(want.empty())
    {
        fprintf(stderr, "[tool] SQ_WAVES not found\n");
        std::abort();
    }
    rocprofiler_counter_config_id_t cfg{.handle = 0};
    RC(rocprofiler_create_counter_config(agent, want.data(), want.size(), &cfg));
    cache.emplace(agent.handle, cfg);
    *config = cfg;
}

int
tool_init(rocprofiler_client_finalize_t, void*)
{
    RC(rocprofiler_create_context(&g_ctx));
    RC(rocprofiler_configure_callback_dispatch_counting_service(
        g_ctx, counter_dispatch_cb, nullptr, counter_record_cb, nullptr));
    RC(rocprofiler_configure_callback_tracing_service(
        g_ctx, ROCPROFILER_CALLBACK_TRACING_KERNEL_REPLAY, nullptr, 0, kernel_replay_cb, nullptr));
    RC(rocprofiler_start_context(g_ctx));
    return 0;
}

void
tool_fini(void*)
{
    fprintf(stderr,
            "[tool] fini hog_replayed=%ld victim_optouts=%ld id_checks=%ld id_bad=%ld\n",
            g_hog_cfg.load(),
            g_victim_cfg.load(),
            g_id_checks.load(),
            g_id_bad.load());
}
}  // namespace

extern "C" rocprofiler_tool_configure_result_t*
rocprofiler_configure(uint32_t, const char*, uint32_t, rocprofiler_client_id_t* id)
{
    id->name        = "kr-p01-deterministic";
    static auto cfg = rocprofiler_tool_configure_result_t{
        sizeof(rocprofiler_tool_configure_result_t), &tool_init, &tool_fini, nullptr};
    return &cfg;
}
