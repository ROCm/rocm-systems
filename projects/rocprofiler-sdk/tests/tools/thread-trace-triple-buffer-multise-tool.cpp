// MIT License
//
// Copyright (c) 2026 Advanced Micro Devices, Inc.
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

#include <rocprofiler-sdk/callback_tracing.h>
#include <rocprofiler-sdk/experimental/thread_trace.h>
#include <rocprofiler-sdk/registration.h>
#include <rocprofiler-sdk/rocprofiler.h>

#include <array>
#include <atomic>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <map>
#include <memory>
#include <mutex>
#include <vector>

namespace ATTTest
{
namespace MultiSE
{
constexpr size_t   GPU_BYTES_PER_SE    = 4ul << 20;
constexpr size_t   MAX_CAPTURE_BYTES   = 1ul << 30;
constexpr size_t   MAX_CHUNKS_PER_SE   = 1024;
constexpr size_t   MIN_SWAPS_PER_SE    = 2;
constexpr uint32_t SHADER_ENGINE_SHIFT = 20;
constexpr uint32_t DISPATCH_MASK       = (1u << SHADER_ENGINE_SHIFT) - 1;

void
require(bool condition, const char* message)
{
    if(condition) return;
    std::cerr << "Multi-SE thread trace: " << message << std::endl;
    std::abort();
}

void
check(rocprofiler_status_t status, const char* operation)
{
    require(status == ROCPROFILER_STATUS_SUCCESS, operation);
}

struct chunk_t
{
    std::vector<char> bytes{};
    bool              end{};
    bool              retired_buffer{};
};

struct engine_trace_t
{
    std::mutex                  mutex{};
    std::map<uint64_t, chunk_t> chunks{};
    size_t                      byte_count{};
};

struct agent_trace_t
{
    rocprofiler_agent_id_t                       id{};
    std::vector<std::unique_ptr<engine_trace_t>> engines{};
};

struct capture_t
{
    rocprofiler_context_id_t                    trace_context{};
    rocprofiler_context_id_t                    control_context{};
    rocprofiler_thread_trace_decoder_id_t       decoder{};
    std::vector<std::unique_ptr<agent_trace_t>> agents{};
    std::atomic<size_t>                         byte_count{0};
    bool                                        started{};
    bool                                        validated{};
};

// Owned from tool_init through tool_fini, including all SDK worker callbacks.
capture_t* capture = nullptr;

void
collect(rocprofiler_thread_trace_shader_data_t data, rocprofiler_user_data_t userdata)
{
    auto& agent = *static_cast<agent_trace_t*>(userdata.ptr);
    require(data.agent.handle == agent.id.handle, "chunk belongs to the wrong GPU");
    require(data.shader_engine_id >= 0 &&
                static_cast<uint64_t>(data.shader_engine_id) < agent.engines.size(),
            "unexpected shader engine");
    constexpr auto full_flags = ROCPROFILER_THREAD_TRACE_SHADER_DATA_FLAGS_GPU_BUFFER_FULL |
                                ROCPROFILER_THREAD_TRACE_SHADER_DATA_FLAGS_CPU_BUFFER_FULL;
    require((data.flags & full_flags) == 0, "CPU or GPU trace buffer overflowed");
    require(data.read_offset <= data.data_size && (data.data_size == 0 || data.data != nullptr),
            "invalid chunk data or read offset");

    const auto size = data.data_size - data.read_offset;
    require(size <= MAX_CAPTURE_BYTES, "chunk exceeds capture budget");
    const auto previous_size = capture->byte_count.fetch_add(size);
    require(previous_size <= MAX_CAPTURE_BYTES - size, "capture exceeds host memory budget");

    // Copy before returning the SDK-owned buffer. No callback waits for another chunk.
    auto chunk = chunk_t{};
    chunk.bytes.resize(size);
    if(size > 0)
        std::memcpy(
            chunk.bytes.data(), static_cast<const char*>(data.data) + data.read_offset, size);
    chunk.end = (data.flags & ROCPROFILER_THREAD_TRACE_SHADER_DATA_FLAGS_END) != 0;
    // Supported GFX9/GFX12 hardware retires a full per-SE buffer on each swap.
    // Neither the small GFX9 header nor the final END/drain chunk counts as a swap.
    chunk.retired_buffer = !chunk.end && data.data_size == GPU_BYTES_PER_SE;

    auto& engine = *agent.engines.at(data.shader_engine_id);
    auto  lock   = std::lock_guard{engine.mutex};
    require(engine.chunks.size() < MAX_CHUNKS_PER_SE, "too many chunks for one engine");
    require(engine.chunks.emplace(data.chunk_index, std::move(chunk)).second,
            "duplicate chunk index");
    engine.byte_count += size;
}

struct parse_state_t
{
    uint32_t engine_id{};
    uint32_t last_dispatch{};
    size_t   records{};
};

void
parse(rocprofiler_thread_trace_decoder_record_type_t type,
      void*                                          events,
      uint64_t                                       count,
      void*                                          userdata)
{
    if(type == ROCPROFILER_THREAD_TRACE_DECODER_RECORD_INFO)
    {
        auto* infos = static_cast<rocprofiler_thread_trace_decoder_info_t*>(events);
        for(size_t i = 0; i < count; ++i)
        {
            require(infos[i] != ROCPROFILER_THREAD_TRACE_DECODER_INFO_DATA_LOST,
                    "decoder reported lost trace data");
            if(infos[i] != ROCPROFILER_THREAD_TRACE_DECODER_INFO_NONE)
                std::cerr << rocprofiler_thread_trace_decoder_info_string(capture->decoder,
                                                                          infos[i])
                          << std::endl;
        }
    }
    if(type != ROCPROFILER_THREAD_TRACE_DECODER_RECORD_SHADERDATA) return;

    auto& state   = *static_cast<parse_state_t*>(userdata);
    auto* records = static_cast<rocprofiler_thread_trace_decoder_shaderdata_t*>(events);
    for(size_t i = 0; i < count; ++i)
    {
        const auto engine   = records[i].value >> SHADER_ENGINE_SHIFT;
        const auto dispatch = records[i].value & DISPATCH_MASK;
        require(engine == state.engine_id, "shaderdata belongs to the wrong engine");
        require(dispatch >= state.last_dispatch, "shaderdata dispatch order regressed");
        state.last_dispatch = dispatch;
        ++state.records;
    }
}

void
validate()
{
    // stop_context has joined the workers, so collection is no longer concurrent.
    bool complete_gpu = false;
    for(auto& agent : capture->agents)
    {
        size_t complete_engines = 0;
        for(size_t se = 0; se < agent->engines.size(); ++se)
        {
            auto& engine = *agent->engines.at(se);
            if(engine.chunks.empty()) continue;  // A visible GPU may have been idle.

            std::vector<char> bytes;
            bytes.reserve(engine.byte_count);
            uint64_t next_index = 0;
            size_t   swaps      = 0;
            bool     ended      = false;
            for(const auto& [index, chunk] : engine.chunks)
            {
                require(index == next_index++, "missing chunk index");
                require(!ended, "chunk received after END");
                ended = chunk.end;
                swaps += chunk.retired_buffer;
                bytes.insert(bytes.end(), chunk.bytes.begin(), chunk.bytes.end());
            }
            require(ended, "missing END chunk");
            engine.chunks.clear();

            auto state      = parse_state_t{};
            state.engine_id = se;
            if(!bytes.empty())
                check(rocprofiler_trace_decode(
                          capture->decoder, parse, bytes.data(), bytes.size(), &state),
                      "trace decode failed");
            std::cout << "GPU " << agent->id.handle << " SE " << se << ": " << state.records
                      << " shaderdata records, " << swaps << " buffer swaps\n";
            if(state.records > 0 && swaps >= MIN_SWAPS_PER_SE) ++complete_engines;
        }
        complete_gpu |= agent->engines.size() >= 2 && complete_engines == agent->engines.size();
    }
    require(complete_gpu, "no GPU has shaderdata and at least two swaps on every engine");
}

void
control(rocprofiler_callback_tracing_record_t record, rocprofiler_user_data_t*, void*)
{
    if(record.kind != ROCPROFILER_CALLBACK_TRACING_MARKER_CONTROL_API) return;
    if(record.operation == ROCPROFILER_MARKER_CONTROL_API_ID_roctxProfilerResume &&
       record.phase == ROCPROFILER_CALLBACK_PHASE_EXIT)
    {
        require(!capture->started, "expected a single capture");
        check(rocprofiler_start_context(capture->trace_context), "start context failed");
        capture->started = true;
    }
    else if(record.operation == ROCPROFILER_MARKER_CONTROL_API_ID_roctxProfilerPause &&
            record.phase == ROCPROFILER_CALLBACK_PHASE_ENTER)
    {
        require(capture->started && !capture->validated, "unexpected capture stop");
        check(rocprofiler_stop_context(capture->trace_context), "stop context failed");
        validate();
        capture->validated = true;
    }
}

rocprofiler_status_t
configure_agents(rocprofiler_agent_version_t, const void** agents, size_t count, void*)
{
    for(size_t i = 0; i < count; ++i)
    {
        const auto& info = *static_cast<const rocprofiler_agent_v0_t*>(agents[i]);
        if(info.type != ROCPROFILER_AGENT_TYPE_GPU || !info.runtime_visibility.hsa) continue;
        require(info.num_shader_banks > 0 && info.num_shader_banks <= 32,
                "shader-engine count is outside the supported mask width");
        auto agent = std::make_unique<agent_trace_t>();
        agent->id  = info.id;
        for(size_t se = 0; se < info.num_shader_banks; ++se)
            agent->engines.emplace_back(std::make_unique<engine_trace_t>());

        const auto engines = agent->engines.size();
        // NUM_BUFFERS controls CPU staging slots; the SDK uses three GPU buffers per SE.
        auto parameters = std::array<rocprofiler_thread_trace_parameter_t, 5>{
            {{ROCPROFILER_THREAD_TRACE_PARAMETER_SIMD_SELECT, {1}},
             {ROCPROFILER_THREAD_TRACE_PARAMETER_BUFFER_SIZE, {GPU_BYTES_PER_SE * engines}},
             {ROCPROFILER_THREAD_TRACE_PARAMETER_SHADER_ENGINE_MASK,
              {(uint64_t{1} << engines) - 1}},
             {ROCPROFILER_THREAD_TRACE_PARAMETER_NUM_BUFFERS, {4 + engines}},
             {ROCPROFILER_THREAD_TRACE_PARAMETER_NO_DETAIL, {1}}}};
        check(rocprofiler_configure_device_thread_trace_service(capture->trace_context,
                                                                info.id,
                                                                parameters.data(),
                                                                parameters.size(),
                                                                collect,
                                                                {.ptr = agent.get()}),
              "thread trace configuration failed");
        capture->agents.emplace_back(std::move(agent));
    }
    return ROCPROFILER_STATUS_SUCCESS;
}

int
tool_init(rocprofiler_client_finalize_t, void*)
{
    capture     = new capture_t{};
    auto status = rocprofiler_thread_trace_decoder_create(&capture->decoder, "");
    if(status != ROCPROFILER_STATUS_SUCCESS)
        status = rocprofiler_thread_trace_decoder_create(&capture->decoder, "/opt/rocm/lib");
    check(status, "decoder creation failed");
    check(rocprofiler_create_context(&capture->trace_context), "trace context creation failed");
    check(rocprofiler_create_context(&capture->control_context), "control context creation failed");
    // NO_DETAIL shaderdata decoding does not need code-object registration/disassembly.
    check(rocprofiler_configure_callback_tracing_service(
              capture->control_context,
              ROCPROFILER_CALLBACK_TRACING_MARKER_CONTROL_API,
              nullptr,
              0,
              control,
              nullptr),
          "control callback configuration failed");
    check(rocprofiler_query_available_agents(ROCPROFILER_AGENT_INFO_VERSION_0,
                                             configure_agents,
                                             sizeof(rocprofiler_agent_t),
                                             nullptr),
          "agent enumeration failed");
    check(rocprofiler_start_context(capture->control_context), "control context start failed");
    return 0;
}

void
tool_fini(void*)
{
    require(capture->validated, "capture was not stopped and validated");
    rocprofiler_thread_trace_decoder_destroy(capture->decoder);
    delete capture;
    capture = nullptr;
    std::cout << "Multi-SE triple-buffer validation passed" << std::endl;
}
}  // namespace MultiSE
}  // namespace ATTTest

extern "C" rocprofiler_tool_configure_result_t*
rocprofiler_configure(uint32_t, const char*, uint32_t priority, rocprofiler_client_id_t* id)
{
    if(priority > 0) return nullptr;
    id->name = "ATT_test_multise";
    static auto result =
        rocprofiler_tool_configure_result_t{sizeof(rocprofiler_tool_configure_result_t),
                                            ATTTest::MultiSE::tool_init,
                                            ATTTest::MultiSE::tool_fini,
                                            nullptr};
    return &result;
}
