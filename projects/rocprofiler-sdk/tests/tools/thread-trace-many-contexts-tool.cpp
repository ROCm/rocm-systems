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
//
// Stress test for the per-agent shared trace buffer and queue. Allocates hundreds of
// thread-trace contexts with varying parameters, each sized to ~1GB. Only one trace is
// active per agent at a time, so all contexts on an agent share a single output buffer
// and a single HSA submission queue, keeping hundreds of contexts within device memory
// and the HSA per-agent queue limit (a separate buffer/queue per context would not fit).
//
// Each context's emitted trace is decoded to verify it holds real waves (not just a
// non-empty header): most decoded instructions must resolve to a valid PC (a loaded code
// object), every data chunk must come from a shader engine the context selected, and on
// gfx10+ every wave must come from the selected SIMD. Runs in both device and dispatch
// thread-trace modes (ATT_TRACE_MODE).
//
// undefine NDEBUG so asserts are implemented
#ifdef NDEBUG
#    undef NDEBUG
#endif

#include "thread-trace-callbacks.hpp"

#include <atomic>
#include <cstdlib>
#include <mutex>
#include <set>
#include <string>
#include <vector>

namespace ATTTest
{
namespace ManyContexts
{
rocprofiler_client_id_t* client_id = nullptr;

// Selects which thread-trace service the stress test exercises. Both share the same
// per-agent buffer/queue, so both must scale to hundreds of contexts.
enum class trace_mode_t
{
    device,
    dispatch
};

trace_mode_t
trace_mode()
{
    const char* m = std::getenv("ATT_TRACE_MODE");
    if(m != nullptr && std::string{m} == "dispatch") return trace_mode_t::dispatch;
    return trace_mode_t::device;
}

// Number of SIMDs per CU/WGP that SQTT can target (SIMD_SEL is 2 bits on gfx10+).
constexpr uint32_t NUM_SIMDS = 4;

// gfx major version from an agent's gfx_target_version (major = (v / 10000) % 100); >=10 is
// RDNA. Mirrors the encoding documented on rocprofiler_agent_v0_t::gfx_target_version.
uint32_t
gfx_major(uint32_t gfx_target_version)
{
    return (gfx_target_version / 10000) % 100;
}

// Encode a logical SIMD target [0,NUM_SIMDS) for the agent's arch: gfx9 takes a SIMD_EN
// bitmask (one bit per SIMD), gfx10+ takes a SIMD_SEL index. aqlprofile writes this value
// straight into the arch's register field, so the caller must supply the right form.
uint32_t
simd_select_param(uint32_t gfx_target_version, uint32_t simd_index)
{
    return gfx_major(gfx_target_version) >= 10 ? simd_index : (1u << simd_index);
}

// Per-context configuration, derived deterministically from the context index so the
// shader-data callback can recompute a context's config (e.g. its shader-engine mask or
// target SIMD) from the index carried in userdata.
struct ctx_config_t
{
    uint32_t target_cu;
    uint32_t se_mask;
    uint32_t simd_index;  // logical SIMD [0,NUM_SIMDS); encoded per-arch at configure time
    uint64_t buffer_size;
};

ctx_config_t
config_for(size_t i)
{
    constexpr uint64_t GB             = 1ull << 30;
    const uint64_t     buffer_sizes[] = {1 * GB, 512ull << 20};
    return ctx_config_t{static_cast<uint32_t>(i % 4),          // target_cu in [0,3]
                        (i % 2) != 0 ? 0x3u : 0x1u,            // shader-engine mask
                        static_cast<uint32_t>(i % NUM_SIMDS),  // target SIMD [0,NUM_SIMDS)
                        buffer_sizes[i % 2]};
}

rocprofiler_thread_trace_decoder_id_t decoder{};
bool                                  decoder_ok = false;  // decoder library loaded

// Aggregate results across every context/agent.
std::atomic<size_t> g_waves{0};     // total decoded waves
std::atomic<size_t> g_valid_pc{0};  // instructions whose PC resolved to a code object
std::atomic<size_t> g_total_pc{0};  // total instructions inspected
std::atomic<size_t> g_chunks{0};    // shader-data chunks delivered
// Chunks delivered for a shader engine the context's shader_engine_mask did not select.
// The shader engine id is on the chunk itself, so this is checkable without the decoder.
std::atomic<size_t> g_wrong_se{0};

// Decoded waves whose SIMD != the one the context selected. Only meaningful on gfx10+
// (see the note below); gated by g_check_simd.
std::atomic<size_t> g_wrong_simd{0};
std::atomic<bool>   g_check_simd{false};  // set once in tool_init: true iff all agents gfx10+

// Note on SIMD/CU targeting: SIMD_SELECT is encoded per-arch (see simd_select_param). gfx9
// SQTT emits wave records for all four SIMDs regardless of the request, so SIMD exclusivity
// cannot be checked there; it is validated on gfx10+ (g_wrong_simd), where the hardware
// honors per-SIMD selection. The decoder always reports a wave's cu as the configured
// target_cu, so CU exclusivity is never checkable. Shader-engine targeting is honored and
// validated on all archs.

// Heap-allocated so we control destruction order relative to rocprofiler shutdown.
struct ToolState
{
    rocprofiler_context_id_t              tracing_ctx{};
    std::vector<rocprofiler_context_id_t> contexts{};

    // Sequencing of the one-active-trace-per-agent rotation. Kernel-dispatch
    // callbacks fire concurrently, so all rotation state is guarded by seq_mut.
    std::mutex seq_mut{};
    size_t     active_index{0};  // next context to run
    bool       trace_active{false};
    uint64_t   start_dispatch{0};

    // Contexts (by index) that delivered at least one real (decoded) wave.
    std::mutex       mut{};
    std::set<size_t> captured{};
};

ToolState* state = nullptr;

// Number of contexts to allocate. Overridable so CI can tune memory/time.
size_t
num_contexts()
{
    static const size_t n = [] {
        const char* env = std::getenv("ATT_NUM_CONTEXTS");
        return env != nullptr ? std::strtoul(env, nullptr, 10) : size_t{400};
    }();
    return n;
}

// Keep each context active across a few dispatches so it reliably captures one.
constexpr uint64_t CAPTURE_WINDOW = 2;

// Per-decode scratch passed through rocprofiler_trace_decode to decode_record().
struct decode_scratch_t
{
    size_t  waves{0};
    size_t  valid_pc{0};
    size_t  total_pc{0};
    size_t  wrong_simd{0};     // waves whose SIMD != expected_simd (only when check_simd)
    uint8_t expected_simd{0};  // SIMD [0,NUM_SIMDS) this context selected (gfx10+)
    bool    check_simd{false};
};

void
decode_record(rocprofiler_thread_trace_decoder_record_type_t record_type_id,
              void*                                          trace_events,
              uint64_t                                       trace_size,
              void*                                          userdata)
{
    if(record_type_id != ROCPROFILER_THREAD_TRACE_DECODER_RECORD_WAVE) return;

    auto* scratch = static_cast<decode_scratch_t*>(userdata);
    auto* waves   = static_cast<rocprofiler_thread_trace_decoder_wave_t*>(trace_events);

    for(uint64_t w = 0; w < trace_size; w++)
    {
        const auto& wave = waves[w];
        scratch->waves++;

        // gfx10+ honors per-SIMD selection, so every wave must be from the selected SIMD.
        if(scratch->check_simd && wave.simd != scratch->expected_simd) scratch->wrong_simd++;

        // A wave is "real" if its instructions resolve to a loaded code object. The
        // decoder sets pc.code_object_id != 0 only when it matched the PC to a
        // registered code object, so this rejects header-only / garbage buffers.
        for(uint64_t i = 0; i < wave.instructions_size; i++)
        {
            scratch->total_pc++;
            if(wave.instructions_array[i].pc.code_object_id != 0) scratch->valid_pc++;
        }
    }
}

void
shader_data_callback(rocprofiler_thread_trace_shader_data_t shader_data,
                     rocprofiler_user_data_t                userdata)
{
    if(shader_data.data_size == 0 || state == nullptr) return;
    // userdata carries the context index this service was configured with.
    auto       idx = reinterpret_cast<size_t>(userdata.ptr);
    const auto cfg = config_for(idx);

    // Shader-engine targeting: every chunk must come from an engine the context
    // selected. The engine id is on the chunk, so this needs no decoder.
    g_chunks.fetch_add(1);
    const auto se_id = shader_data.shader_engine_id;
    if(se_id < 0 || se_id >= 32 || ((cfg.se_mask >> se_id) & 0x1u) == 0u) g_wrong_se.fetch_add(1);

    bool got_waves = false;
    if(decoder_ok)
    {
        decode_scratch_t scratch{};
        scratch.check_simd    = g_check_simd.load();
        scratch.expected_simd = static_cast<uint8_t>(cfg.simd_index);
        DECODER_CALL(rocprofiler_trace_decode(
            decoder, decode_record, shader_data.data, shader_data.data_size, &scratch));
        g_waves.fetch_add(scratch.waves);
        g_valid_pc.fetch_add(scratch.valid_pc);
        g_total_pc.fetch_add(scratch.total_pc);
        g_wrong_simd.fetch_add(scratch.wrong_simd);
        got_waves = scratch.waves > 0;
    }

    // "captured" means real decoded waves when the decoder is available, otherwise any
    // delivered (non-empty) data, matching the convention used by the other ATT tools.
    if(!decoder_ok || got_waves)
    {
        std::unique_lock<std::mutex> lk(state->mut);
        state->captured.insert(idx);
    }
}

// Register/unregister loaded code objects with the decoder so instruction PCs can be
// resolved (required for the wave-validity check above).
void
codeobj_tracing_callback(rocprofiler_callback_tracing_record_t record,
                         rocprofiler_user_data_t* /* user_data */,
                         void* /* userdata */)
{
    if(record.kind != ROCPROFILER_CALLBACK_TRACING_CODE_OBJECT) return;
    if(record.operation != ROCPROFILER_CODE_OBJECT_LOAD) return;

    auto* data = static_cast<rocprofiler_callback_tracing_code_object_load_data_t*>(record.payload);
    if(data->storage_type == ROCPROFILER_CODE_OBJECT_STORAGE_TYPE_FILE) return;

    if(record.phase != ROCPROFILER_CALLBACK_PHASE_LOAD)
    {
        DECODER_CALL(
            rocprofiler_thread_trace_decoder_codeobj_unload(decoder, data->code_object_id));
        return;
    }

    DECODER_CALL(rocprofiler_thread_trace_decoder_codeobj_load(
        decoder,
        data->code_object_id,
        data->load_delta,
        data->load_size,
        reinterpret_cast<const void*>(data->memory_base),
        data->memory_size));
}

// Dispatch-mode ATT callback for the single started context; traces only the first few
// dispatches (enough to capture without serializing the whole app). Dispatch mode can't
// rotate hundreds of contexts like device mode: only one dispatch interceptor is active
// per agent (so only the started context's callback fires) and start/stop of an ATT
// dispatch context from within a callback deadlocks. So sharing is proven at config time
// (every context allocates the shared buffer/queue) and one running context validates it.
rocprofiler_thread_trace_control_flags_t
att_dispatch_callback(rocprofiler_agent_id_t /* agent */,
                      rocprofiler_queue_id_t /* queue_id */,
                      rocprofiler_async_correlation_id_t /* correlation_id */,
                      rocprofiler_kernel_id_t /* kernel_id */,
                      rocprofiler_dispatch_id_t /* dispatch_id */,
                      void*                    userdata_config,
                      rocprofiler_user_data_t* userdata_shader)
{
    if(state == nullptr) return ROCPROFILER_THREAD_TRACE_CONTROL_NONE;

    static std::atomic<uint64_t> traced{0};
    if(traced.fetch_add(1) >= CAPTURE_WINDOW) return ROCPROFILER_THREAD_TRACE_CONTROL_NONE;

    // Forward the context index (configure-time userdata) to the shader callback.
    if(userdata_shader != nullptr) userdata_shader->ptr = userdata_config;
    return ROCPROFILER_THREAD_TRACE_CONTROL_START_AND_STOP;
}

// Drives the rotation: start the next context on a dispatch, stop it a couple of
// dispatches later, then advance. Only one context is ever active at a time.
void
dispatch_tracing_callback(rocprofiler_callback_tracing_record_t record,
                          rocprofiler_user_data_t* /* user_data */,
                          void* /* userdata */)
{
    if(record.kind != ROCPROFILER_CALLBACK_TRACING_KERNEL_DISPATCH) return;
    if(record.phase == ROCPROFILER_CALLBACK_PHASE_EXIT) return;
    if(state == nullptr) return;

    assert(record.payload);
    auto* rdata = static_cast<rocprofiler_callback_tracing_kernel_dispatch_data_t*>(record.payload);
    auto  dispatch_id = rdata->dispatch_info.dispatch_id;

    // Serialize the whole check-then-act so concurrent dispatch callbacks can't
    // double-start or double-stop a context.
    std::unique_lock<std::mutex> lk(state->seq_mut);

    if(record.phase == ROCPROFILER_CALLBACK_PHASE_ENTER)
    {
        if(!state->trace_active && state->active_index < state->contexts.size())
        {
            // Skip past any context that refuses to start rather than aborting.
            while(state->active_index < state->contexts.size() &&
                  rocprofiler_start_context(state->contexts[state->active_index]) !=
                      ROCPROFILER_STATUS_SUCCESS)
                ++state->active_index;

            if(state->active_index < state->contexts.size())
            {
                state->trace_active   = true;
                state->start_dispatch = dispatch_id;
            }
        }
        return;
    }

    // Completion phase: stop the active context once its capture window has elapsed.
    assert(record.phase == ROCPROFILER_CALLBACK_PHASE_NONE);
    if(state->trace_active && dispatch_id >= state->start_dispatch + CAPTURE_WINDOW)
    {
        rocprofiler_stop_context(state->contexts[state->active_index]);
        state->trace_active = false;
        ++state->active_index;
    }
}

// A GPU agent plus its gfx arch version, needed to encode SIMD_SELECT per-arch.
struct gpu_agent_t
{
    rocprofiler_agent_id_t id{};
    uint32_t               gfx_target_version{};
};

std::vector<gpu_agent_t>
get_gpu_agents()
{
    std::vector<gpu_agent_t> agents{};
    ROCPROFILER_CALL(
        rocprofiler_query_available_agents(
            ROCPROFILER_AGENT_INFO_VERSION_0,
            [](rocprofiler_agent_version_t, const void** _agents, size_t _num, void* _data) {
                auto* out = static_cast<std::vector<gpu_agent_t>*>(_data);
                for(size_t i = 0; i < _num; ++i)
                {
                    const auto* agent = static_cast<const rocprofiler_agent_v0_t*>(_agents[i]);
                    if(agent->type == ROCPROFILER_AGENT_TYPE_GPU)
                        out->push_back({agent->id, agent->gfx_target_version});
                }
                return ROCPROFILER_STATUS_SUCCESS;
            },
            sizeof(rocprofiler_agent_v0_t),
            &agents),
        "query agents");
    return agents;
}

int
tool_init(rocprofiler_client_finalize_t /* fini_func */, void* /* tool_data */)
{
    state = new ToolState{};

    // Directory containing librocprof-trace-decoder.so. Overridable so environments
    // that install the decoder outside the default ROCm lib can still decode. When the
    // decoder is unavailable, wave/PC validation is skipped (as the other ATT tools do).
    const char* decoder_path = std::getenv("ROCPROF_TRACE_DECODER_PATH");
    if(decoder_path == nullptr) decoder_path = "/opt/rocm/lib";
    decoder_ok = rocprofiler_thread_trace_decoder_create(&decoder, decoder_path) ==
                     ROCPROFILER_STATUS_SUCCESS &&
                 decoder.handle != 0;
    if(!decoder_ok)
        std::cerr << "[many-contexts] trace decoder unavailable at " << decoder_path
                  << "; wave/PC validation skipped (set ROCPROF_TRACE_DECODER_PATH)" << std::endl;

    auto agents = get_gpu_agents();
    if(agents.empty()) return 0;

    // SIMD exclusivity is only observable where the hardware honors per-SIMD selection
    // (gfx10+); enable the check only if every GPU agent is gfx10+.
    bool all_gfx10_plus = true;
    for(const auto& a : agents)
        if(gfx_major(a.gfx_target_version) < 10) all_gfx10_plus = false;
    g_check_simd = all_gfx10_plus;

    const bool   dispatch = trace_mode() == trace_mode_t::dispatch;
    const size_t N        = num_contexts();
    state->contexts.reserve(N);

    ROCPROFILER_CALL(rocprofiler_create_context(&state->tracing_ctx), "context creation");
    // Device mode rotates by starting/stopping ATT contexts over kernel dispatches, which
    // is safe for device thread trace. Dispatch mode cannot do that (starting an ATT
    // dispatch context from within a dispatch callback deadlocks), so it starts all
    // contexts up front and rotates inside att_dispatch_callback instead.
    if(!dispatch)
        ROCPROFILER_CALL(rocprofiler_configure_callback_tracing_service(
                             state->tracing_ctx,
                             ROCPROFILER_CALLBACK_TRACING_KERNEL_DISPATCH,
                             nullptr,
                             0,
                             dispatch_tracing_callback,
                             nullptr),
                         "dispatch tracing service configure");
    // Register code objects with the decoder so instruction PCs resolve.
    ROCPROFILER_CALL(
        rocprofiler_configure_callback_tracing_service(state->tracing_ctx,
                                                       ROCPROFILER_CALLBACK_TRACING_CODE_OBJECT,
                                                       nullptr,
                                                       0,
                                                       codeobj_tracing_callback,
                                                       nullptr),
        "code object tracing service configure");

    for(size_t i = 0; i < N; ++i)
    {
        rocprofiler_context_id_t ctx{};
        ROCPROFILER_CALL(rocprofiler_create_context(&ctx), "context creation");

        const auto cfg = config_for(i);

        rocprofiler_user_data_t user{};
        user.ptr = reinterpret_cast<void*>(i);

        bool ok = true;
        for(const auto& agent : agents)
        {
            // SIMD_SELECT is encoded per-arch, so build the params for each agent.
            auto params = std::vector<rocprofiler_thread_trace_parameter_t>{};
            params.push_back({ROCPROFILER_THREAD_TRACE_PARAMETER_TARGET_CU, {cfg.target_cu}});
            params.push_back({ROCPROFILER_THREAD_TRACE_PARAMETER_SIMD_SELECT,
                              {simd_select_param(agent.gfx_target_version, cfg.simd_index)}});
            params.push_back(
                {ROCPROFILER_THREAD_TRACE_PARAMETER_SHADER_ENGINE_MASK, {cfg.se_mask}});
            params.push_back({ROCPROFILER_THREAD_TRACE_PARAMETER_BUFFER_SIZE, {cfg.buffer_size}});

            auto status =
                dispatch
                    ? rocprofiler_configure_dispatch_thread_trace_service(ctx,
                                                                          agent.id,
                                                                          params.data(),
                                                                          params.size(),
                                                                          att_dispatch_callback,
                                                                          shader_data_callback,
                                                                          user.ptr)
                    : rocprofiler_configure_device_thread_trace_service(
                          ctx, agent.id, params.data(), params.size(), shader_data_callback, user);
            if(status != ROCPROFILER_STATUS_SUCCESS) ok = false;
        }

        int valid = 0;
        rocprofiler_context_is_valid(ctx, &valid);
        if(ok && valid != 0) state->contexts.emplace_back(ctx);
    }

    std::cerr << "[many-contexts] mode=" << (dispatch ? "dispatch" : "device") << " configured "
              << state->contexts.size() << " / " << N << " contexts" << std::endl;

    // Dispatch mode: all contexts are already configured (proving the shared buffer/queue
    // scale). Start one to validate the dispatch trace path; rotating hundreds is not
    // possible in dispatch mode (see att_dispatch_callback).
    if(dispatch && !state->contexts.empty())
    {
        ROCPROFILER_CALL(rocprofiler_start_context(state->contexts[0]), "dispatch context start");
        std::unique_lock<std::mutex> lk(state->seq_mut);
        state->active_index = 1;  // one context is given a turn
    }

    ROCPROFILER_CALL(rocprofiler_start_context(state->tracing_ctx), "tracing context start");
    return 0;
}

void
tool_fini(void* /* tool_data */)
{
    if(state == nullptr) return;

    const bool dispatch     = trace_mode() == trace_mode_t::dispatch;
    size_t     active_index = 0;
    {
        std::unique_lock<std::mutex> lk(state->seq_mut);
        active_index = state->active_index;
    }

    if(dispatch)
    {
        // Only the first context was started in dispatch mode.
        if(!state->contexts.empty()) rocprofiler_stop_context(state->contexts[0]);
    }
    else
    {
        std::unique_lock<std::mutex> lk(state->seq_mut);
        // Device mode: only the rotation's currently-active context may be running.
        if(state->trace_active && state->active_index < state->contexts.size())
            rocprofiler_stop_context(state->contexts[state->active_index]);
    }
    rocprofiler_stop_context(state->tracing_ctx);

    size_t configured = state->contexts.size();
    size_t ran        = std::min(active_index, configured);
    size_t captured   = 0;
    {
        std::unique_lock<std::mutex> lk(state->mut);
        captured = state->captured.size();
    }

    const size_t waves      = g_waves.load();
    const size_t valid_pc   = g_valid_pc.load();
    const size_t total_pc   = g_total_pc.load();
    const size_t chunks     = g_chunks.load();
    const size_t wrong_se   = g_wrong_se.load();
    const size_t wrong_simd = g_wrong_simd.load();

    std::cerr << "[many-contexts] configured=" << configured << " ran=" << ran
              << " captured=" << captured << " chunks=" << chunks << " wrong_se=" << wrong_se
              << " wrong_simd=" << wrong_simd << " waves=" << waves << " valid_pc=" << valid_pc
              << "/" << total_pc << " decoder=" << (decoder_ok ? "on" : "off") << std::endl;

    // Every context must have configured and allocated its (shared) buffer and queue,
    // which is the sharing proof: this would OOM or exhaust HSA queues per-context.
    assert(configured == num_contexts() && "not all contexts configured/allocated");

    if(dispatch)
    {
        // Dispatch mode proves the sharing at configuration time (all contexts above);
        // one running context validates the dispatch trace path.
        assert(ran >= 1 && "dispatch context was not run");
        assert(captured >= 1 && "dispatch context captured no data");
    }
    else
    {
        // Device mode rotates through every context; each should have had a turn.
        assert(ran == configured && "not every context was run (need more dispatches)");
        // Not every target_cu is guaranteed to have active waves, so require that MOST
        // configs produced data rather than all.
        assert(captured * 4 >= configured * 3 && "most configs should capture data");
    }

    // Shader-engine targeting must be honored: no chunk may be delivered for an engine
    // the context's shader_engine_mask did not select. Checkable without the decoder.
    assert(chunks > 0 && "no shader-data chunks were delivered");
    assert(wrong_se == 0 && "captured a chunk from a shader engine that was not selected");

    // With the decoder available, verify the captured data contains real waves with
    // resolvable instruction PCs, not just non-empty header buffers. A small fraction of
    // instructions may not resolve (trap handler, and the first few on some archs such as
    // gfx1250). SIMD/CU exclusivity is intentionally not asserted (see note above).
    if(decoder_ok)
    {
        assert(waves > 0 && total_pc > 0 && "no decodable waves/instructions were captured");
        assert(valid_pc * 10 >= total_pc * 9 && "most instructions should have a valid PC");
        // gfx10+ only (g_check_simd); on gfx9 wrong_simd stays 0 since the check is disabled.
        assert(wrong_simd == 0 && "captured a wave from a SIMD that was not selected");
    }

    delete state;
    state = nullptr;

    if(decoder_ok) rocprofiler_thread_trace_decoder_destroy(decoder);
}

}  // namespace ManyContexts
}  // namespace ATTTest

extern "C" rocprofiler_tool_configure_result_t*
rocprofiler_configure(uint32_t /* version */,
                      const char* /* runtime_version */,
                      uint32_t                 priority,
                      rocprofiler_client_id_t* id)
{
    if(priority > 0) return nullptr;

    id->name                         = "ATT_test_many_contexts";
    ATTTest::ManyContexts::client_id = id;

    static auto cfg =
        rocprofiler_tool_configure_result_t{sizeof(rocprofiler_tool_configure_result_t),
                                            &ATTTest::ManyContexts::tool_init,
                                            &ATTTest::ManyContexts::tool_fini,
                                            nullptr};
    return &cfg;
}
