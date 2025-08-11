/* Copyright (c) 2023 Advanced Micro Devices, Inc.

 Permission is hereby granted, free of charge, to any person obtaining a copy
 of this software and associated documentation files (the "Software"), to deal
 in the Software without restriction, including without limitation the rights
 to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 copies of the Software, and to permit persons to whom the Software is
 furnished to do so, subject to the following conditions:

 The above copyright notice and this permission notice shall be included in
 all copies or substantial portions of the Software.

 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 THE SOFTWARE. */

#include "counters_sampler.h"
#include "src/core/hsa/hsa_support.h"
#include "src/api/rocprofiler_singleton.h"
#include "src/core/counters/mmio/pcie_counters_mi200.h"
#include "src/core/counters/mmio/df_counters_mi200.h"
#include "src/utils/libpci_helper.h"

namespace rocprofiler {

CountersSampler::CountersSampler(rocprofiler_buffer_id_t buffer_id,
                                 rocprofiler_filter_id_t filter_id,
                                 rocprofiler_session_id_t session_id)
    : buffer_id_(buffer_id),
      filter_id_(filter_id),
      session_id_(session_id),
      pci_system_initialized_(false)
{
  pci_system_initialized_ = GetPciAccessLibApi()->pci_system_init() == 0;
  params_ = rocprofiler::ROCProfiler_Singleton::GetInstance()
                .GetSession(session_id_)
                ->GetFilter(filter_id_)
                ->GetCountersSamplerParameterData();

  std::vector<hsa_agent_t> agents;
  HSASupport_Singleton::GetInstance().GetCoreApiTable().hsa_iterate_agents_fn(
      [](hsa_agent_t agent, void* arg) {
        auto& agents = *reinterpret_cast<std::vector<hsa_agent_t>*>(arg);
        const auto& ai = HSASupport_Singleton::GetInstance().GetHSAAgentInfo(agent.handle);
        if (ai.GetType() == HSA_DEVICE_TYPE_GPU) {
          agents.emplace_back(agent);
        }
        return HSA_STATUS_SUCCESS;
      },
      &agents);

  // create perfmon instances for the counter types specfied

  // PCIE counters
  std::vector<std::string> pcie_counter_names;
  for (int i = 0; i < params_.counters_num; i++) {
    if (params_.counters[i].type == ROCPROFILER_COUNTERS_SAMPLER_PCIE_COUNTERS)
      pcie_counter_names.push_back(params_.counters[i].name);
  }

  if (pcie_counter_names.size() > 0) {
    auto agentInfo = HSASupport_Singleton::GetInstance().GetHSAAgentInfo(agents[params_.gpu_agent_index].handle);
    if (agentInfo.GetDeviceInfo().getName()== "gfx90a") {
      PciePerfMonMI200* perfmon = new PciePerfMonMI200(agentInfo);
      perfmon->SetCounterNames(pcie_counter_names);
      perfmon_instances_.push_back(perfmon);
    }
  }
  // XGMI counters
  std::vector<std::string> xgmi_counter_names;
  for (int i = 0; i < params_.counters_num; i++) {
    if (params_.counters[i].type == ROCPROFILER_COUNTERS_SAMPLER_XGMI_COUNTERS)
      xgmi_counter_names.push_back(params_.counters[i].name);
  }

  if (xgmi_counter_names.size() > 0) {
    auto agentInfo = HSASupport_Singleton::GetInstance().GetHSAAgentInfo(agents[params_.gpu_agent_index].handle);
    if (agentInfo.GetDeviceInfo().getName() == "gfx90a") {
      DFPerfMonMI200* perfmon = new DFPerfMonMI200(agentInfo);
      perfmon->SetCounterNames(xgmi_counter_names);
      perfmon_instances_.push_back(perfmon);
    }
  }
}

CountersSampler::~CountersSampler() {
  // cleanup perfmon instancess
  for (auto& perfmon : perfmon_instances_) {
    if (perfmon != nullptr) delete perfmon;
  }
  // clean up libpcieaccess resources
  // TODO: should be part of mmio class in future
  if (pci_system_initialized_) {
    GetPciAccessLibApi()->pci_system_cleanup();
    pci_system_initialized_ = false;
    UnLoadPcieAccessLibAPI();
  }
}

void CountersSampler::Start() {
  if (sampler_thread_.joinable()) {
    return;
  }

  std::cout << "Sampler Start\n";
  // Start all Perfmons
  for (auto& perfmon : perfmon_instances_) {
    perfmon->Start();
  }

  // Start polling thread
  keep_running_ = true;
  sampler_thread_ = std::thread([this]() { SamplerLoop(); });
}

void CountersSampler::Stop() {
  if (!sampler_thread_.joinable()) {
    return;
  }

  std::cout << "Sampler Stop\n";
  // Stop all Perfmons
  for (auto& perfmon : perfmon_instances_) {
    perfmon->Stop();
  }

  // Stop polling thread
  keep_running_ = false;
  sampler_thread_.join();
}

void CountersSampler::AddRecord(rocprofiler_record_counters_sampler_t& record) {
  rocprofiler::ROCProfiler_Singleton& tool = rocprofiler::ROCProfiler_Singleton::GetInstance();
  const auto session = tool.GetSession(session_id_);
  const auto buffer = session->GetBuffer(buffer_id_);

  std::lock_guard<std::mutex> lk(session->GetSessionLock());

  record.header = {ROCPROFILER_COUNTERS_SAMPLER_RECORD, {tool.GetUniqueRecordId()}};

  // Add the record to the buffer(a deep-copy operation) along with
  // a lambda function to deep-copy the record.counters member to
  // the newly created buffer record
  buffer->AddRecord(
      record, record.counters,
      (record.num_counters * (sizeof(rocprofiler_counters_sampler_counter_output_t) + 1)),
      [](auto& buff_record, const void* data) {
        buff_record.counters = const_cast<rocprofiler_counters_sampler_counter_output_t*>(
            static_cast<const rocprofiler_counters_sampler_counter_output_t*>(data));
      });
}

void CountersSampler::SamplerLoop() {
  std::this_thread::sleep_until(std::chrono::steady_clock::now() +
                                std::chrono::milliseconds(params_.initial_delay));
  uint32_t elapsed = 0;
  while (keep_running_ && (elapsed <= params_.sampling_duration)) {
    auto next_tick =
        std::chrono::steady_clock::now() + std::chrono::milliseconds(params_.sampling_rate);

    rocprofiler_record_counters_sampler_t record;
    std::vector<rocprofiler_counters_sampler_counter_output_t> values;
    for (auto& perfmon : perfmon_instances_) {
      perfmon->Read(values);
    }
    record.counters = static_cast<rocprofiler_counters_sampler_counter_output_t*>(
        malloc(values.size() * sizeof(rocprofiler_counters_sampler_counter_output_t)));
    ::memcpy(record.counters, &(values)[0],
             values.size() * sizeof(rocprofiler_counters_sampler_counter_output_t));
    record.num_counters = values.size();
    rocprofiler_counters_sampler_counter_output_t* record_counters = record.counters;
    AddRecord(record);
    free(record_counters);

    std::this_thread::sleep_until(next_tick);
    elapsed += params_.sampling_rate;
  }
}

}  // namespace rocprofiler