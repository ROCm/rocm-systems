// Copyright (c) 2018-2023 Advanced Micro Devices, Inc.
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
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

#pragma once

#include <hsa/hsa.h>
#include <hsa/hsa_api_trace.h>
#include <hsa/hsa_ext_amd.h>

#include "fmt/core.h"
#include "fmt/ranges.h"

#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include "lib/common/utility.hpp"

namespace rocprofiler
{
namespace hsa
{
static const uint32_t LDS_BLOCK_SIZE = 128 * 4;

class AgentInfo
{
public:
    AgentInfo() = default;
    AgentInfo(const hsa_agent_t agent, const ::CoreApiTable& table);
    uint64_t               getIndex() const;
    hsa_device_type_t      getType() const;
    bool                   isGpu() const { return getType() == HSA_DEVICE_TYPE_GPU; }
    bool                   isCpu() const { return getType() == HSA_DEVICE_TYPE_CPU; }
    uint64_t               getHandle() const;
    const std::string_view getName() const;
    const char*            getNameChar() const { return name_; }
    std::string            getGfxip() const;
    uint32_t               getMaxWaveSize() const;
    uint32_t               getMaxQueueSize() const;
    uint32_t               getCUCount() const;
    uint32_t               getSimdCountPerCU() const;
    uint32_t               getShaderEngineCount() const;
    uint32_t               getShaderArraysPerSE() const;
    uint32_t               getMaxWavesPerCU() const;
    uint32_t               getCUCountPerSH() const;
    uint32_t               getWaveSlotsPerSimd() const;
    uint32_t               getPCIDomain() const;
    uint32_t               getPCILocationID() const;
    uint32_t               getXccCount() const;

    void setIndex(uint64_t index);
    void setType(hsa_device_type_t type);
    void setHandle(uint64_t handle);
    void setName(const std::string& name);

    void     setNumaNode(uint32_t numa_node);
    uint32_t getNumaNode() const;

    void        setNearCpuAgent(hsa_agent_t near_cpu_agent);
    hsa_agent_t getNearCpuAgent();
    hsa_agent_t getAgent() const { return agent_; }

    hsa_amd_memory_pool_t cpu_pool;
    hsa_amd_memory_pool_t kernarg_pool;
    hsa_amd_memory_pool_t gpu_pool;

    static std::shared_ptr<const std::vector<AgentInfo>> getAgents(const CoreApiTable&,
                                                                   const AmdExtTable&);

    // Keep move constuctors (i.e. std::move())
    AgentInfo(AgentInfo&& other) noexcept = default;
    AgentInfo& operator=(AgentInfo&& other) noexcept = default;

    // Do not allow copying this class
    AgentInfo(const AgentInfo&) = delete;
    AgentInfo& operator=(const AgentInfo&) = delete;

private:
    uint64_t          index_     = 0;
    hsa_device_type_t type_      = HSA_DEVICE_TYPE_CPU;  // Agent type - Cpu = 0, Gpu = 1 or Dsp = 2
    uint64_t          handle_    = 0;
    char              name_[64]  = {'\0'};
    char              gfxip_[64] = {'\0'};
    uint32_t          max_wave_size_        = 0;
    uint32_t          max_queue_size_       = 0;
    uint32_t          cu_num_               = 0;
    uint32_t          simds_per_cu_         = 0;
    uint32_t          se_num_               = 0;
    uint32_t          shader_arrays_per_se_ = 0;
    uint32_t          waves_per_cu_         = 0;
    // CUs per SH/SA
    uint32_t compute_units_per_sh_ = 0;
    uint32_t wave_slots_per_simd_  = 0;
    // Number of XCCs on the GPU
    uint32_t xcc_num_ = 0;

    uint32_t pci_domain_      = 0;
    uint32_t pci_location_id_ = 0;

    uint32_t    numa_node_      = 0;
    hsa_agent_t near_cpu_agent_ = {};
    hsa_agent_t agent_          = {};
};

const std::vector<AgentInfo>&
all_agents();
}  // namespace hsa
}  // namespace rocprofiler

namespace fmt
{
template <>
struct formatter<rocprofiler::hsa::AgentInfo>
{
    template <typename ParseContext>
    constexpr auto parse(ParseContext& ctx)
    {
        return ctx.begin();
    }

    template <typename Ctx>
    auto format(rocprofiler::hsa::AgentInfo const& agent, Ctx& ctx) const
    {
        auto device_type_name = [](auto dev) -> std::string_view {
            switch(dev)
            {
                case HSA_DEVICE_TYPE_CPU: return "CPU";
                case HSA_DEVICE_TYPE_GPU: return "GPU";
                case HSA_DEVICE_TYPE_DSP: return "DSP";
            }
            return "UNKNOWN";
        };

        return fmt::format_to(
            ctx.out(),
            R"({{"index":"{}","type":"{}","handle":"{}","name":"{}","gfxip":"{}","MaxWaveSize":"{}","MaxQueueSize":"{}","CUCount":"{}","SimdCountPerCU":"{}","ShaderEngineCount":"{}","ShaderArraysPerSE":"{}","MaxWavesPerCU":"{}","CUCountPerSH":"{}","WaveSlotsPerSimd":"{}","PCIDomain":"{}","PCILocationID":"{}","XccCount":"{}"}})",
            agent.getIndex(),
            device_type_name(agent.getType()),
            agent.getHandle(),
            agent.getName(),
            agent.getGfxip(),
            agent.getMaxWaveSize(),
            agent.getMaxQueueSize(),
            agent.getCUCount(),
            agent.getSimdCountPerCU(),
            agent.getShaderEngineCount(),
            agent.getShaderArraysPerSE(),
            agent.getMaxWavesPerCU(),
            agent.getCUCountPerSH(),
            agent.getWaveSlotsPerSimd(),
            agent.getPCIDomain(),
            agent.getPCILocationID(),
            agent.getXccCount());
    }
};
}  // namespace fmt
