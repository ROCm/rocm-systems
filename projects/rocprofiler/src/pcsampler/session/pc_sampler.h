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

#ifndef SRC_PCSAMPLER_PC_SAMPLER_H_
#define SRC_PCSAMPLER_PC_SAMPLER_H_

#include <thread>
#include <unordered_map>

#include <hsa/hsa.h>

#include "rocprofiler.h"
#include "src/pcsampler/gfxip/gfxip.h"

namespace rocprofiler::pc_sampler {

struct PCSampler {
  PCSampler(rocprofiler_buffer_id_t buffer_id, rocprofiler_filter_id_t filter_id,
            rocprofiler_session_id_t session_id);
  ~PCSampler();

  PCSampler(const PCSampler&) = delete;
  PCSampler& operator=(const PCSampler&) = delete;

  void Start();
  void Stop();
  void AddRecord(rocprofiler_record_pc_sample_t& record);

 private:
  void SamplerLoop();

  rocprofiler_buffer_id_t buffer_id_;
  rocprofiler_filter_id_t filter_id_;
  rocprofiler_session_id_t session_id_;
  bool pci_system_initialized_{false};
  std::unordered_map<decltype(hsa_agent_t::handle), gfxip::device_t> devices_;
  std::atomic<bool> keep_running_{false};
  std::thread sampler_thread_;
};

}  // namespace rocprofiler::pc_sampler

#endif  // SRC_PCSAMPLER_PC_SAMPLER_H_
