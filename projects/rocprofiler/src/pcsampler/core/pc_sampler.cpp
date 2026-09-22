/* Copyright (c) 2022 Advanced Micro Devices, Inc.

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

#include <cassert>
#include <atomic>
#include <mutex>
#include <thread>
#include <unordered_map>

#include <hsa/hsa.h>

#include "src/api/rocprofiler_singleton.h"
#include "src/pcsampler/session/pc_sampler.h"
#include "src/pcsampler/gfxip/gfxip.h"
#include "src/core/hsa/hsa_support.h"
#include "src/utils/libpci_helper.h"

namespace rocprofiler::pc_sampler {

PCSampler::PCSampler(rocprofiler_buffer_id_t buffer_id, rocprofiler_filter_id_t filter_id,
                     rocprofiler_session_id_t session_id)
    : buffer_id_(buffer_id),
      filter_id_(filter_id),
      session_id_(session_id),
      pci_system_initialized_(false) {
  pci_system_initialized_ = GetPciAccessLibApi()->pci_system_init() == 0;
}

PCSampler::~PCSampler() {
  if (pci_system_initialized_) {
    GetPciAccessLibApi()->pci_system_cleanup();

    pci_system_initialized_ = false;
    UnLoadPcieAccessLibAPI();
  }
}

void PCSampler::Start() {
  if (sampler_thread_.joinable()) {
    return;
  }

  devices_.clear();

  using agents_t = std::vector<hsa_agent_t>;

  agents_t agents;
  HSASupport_Singleton& hsasupport_singleton = HSASupport_Singleton::GetInstance();
  hsasupport_singleton.GetCoreApiTable().hsa_iterate_agents_fn(
   [](hsa_agent_t agent, void *arg){
     auto &agents = *reinterpret_cast<agents_t *>(arg);
     agents.emplace_back(agent);
     return HSA_STATUS_SUCCESS;
   },
   &agents);

  for (const auto &agent : agents) {
    const auto& ai = hsasupport_singleton.GetHSAAgentInfo(agent.handle);
    if (ai.GetType() !=  HSA_DEVICE_TYPE_GPU) {
      continue;
    }
    devices_.emplace(agent.handle, gfxip::device_t{pci_system_initialized_, ai});
  }

  keep_running_ = true;
  sampler_thread_ = std::thread([this]() { SamplerLoop(); });
}

void PCSampler::Stop() {
  if (!sampler_thread_.joinable()) {
    return;
  }

  keep_running_ = false;
  sampler_thread_.join();
}

void PCSampler::AddRecord(rocprofiler_record_pc_sample_t& record) {
  rocprofiler::ROCProfiler_Singleton&  rocprofiler_instance = rocprofiler::ROCProfiler_Singleton::GetInstance();
  const auto session = rocprofiler_instance.GetSession(session_id_);
  const auto buffer = session->GetBuffer(buffer_id_);

  std::lock_guard<std::mutex> lk(session->GetSessionLock());

  record.header = {ROCPROFILER_PC_SAMPLING_RECORD, {rocprofiler_instance.GetUniqueRecordId()}};
  buffer->AddRecord(record);
}

void PCSampler::SamplerLoop() {
  while (keep_running_) {
    auto next_tick = std::chrono::steady_clock::now() + std::chrono::milliseconds(10);
    for (auto& agent : devices_) {
      auto& device = agent.second;
      if (device.fd_.mmio2.get() >= 0) {
        gfxip::read_pc_samples_v9_ioctl(device, this);
      } else {
        gfxip::read_pc_samples_v9(device, this);
      }
    }
    std::this_thread::sleep_until(next_tick);
  }
}

}  // namespace rocprofiler::pc_sampler
