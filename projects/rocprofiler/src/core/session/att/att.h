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

#ifndef SRC_CORE_SESSION_ATT_ATT_H_
#define SRC_CORE_SESSION_ATT_ATT_H_

#include <hsa/hsa_ven_amd_aqlprofile.h>

#include <map>
#include <mutex>
#include <string>
#include <vector>

#include "rocprofiler.h"

namespace rocprofiler {

typedef struct {
  uint64_t kernel_descriptor;
  hsa_signal_t original_signal;
  hsa_signal_t new_signal;
  rocprofiler_session_id_t session_id;
  rocprofiler_buffer_id_t buffer_id;
  hsa_ven_amd_aqlprofile_profile_t* profile;
  rocprofiler_kernel_properties_t kernel_properties;
  uint32_t thread_id;
  uint64_t queue_index;
} att_pending_signal_t;

enum rocprofiler_att_isa_dump_mode {
  ISA_MODE_DUMP_ALL=0,
  ISA_MODE_DUMP_OBJ,
  ISA_MODE_DUMP_KERNEL,
  ISA_MODE_DUMP_NONE
};

union att_header_packet_t {
  struct {
    uint64_t reserved : 14;
    uint64_t navi : 1;
    uint64_t enable : 1;
    uint64_t DSIMDM : 4;
    uint64_t DCU : 5;
    uint64_t DSA : 1;
    uint64_t SEID : 6;
    uint64_t isadumpmode : 3;
  };
  uint64_t raw;
};

namespace att {

class AttTracer {
 public:
  AttTracer(rocprofiler_buffer_id_t buffer_id, rocprofiler_filter_id_t filter_id,
            rocprofiler_session_id_t session_id);

  void AddPendingSignals(uint32_t writer_id, uint64_t kernel_object,
                         const hsa_signal_t& original_completion_signal,
                         const hsa_signal_t& new_completion_signal,
                         rocprofiler_session_id_t session_id, rocprofiler_buffer_id_t buffer_id,
                         hsa_ven_amd_aqlprofile_profile_t* profile,
                         rocprofiler_kernel_properties_t kernel_properties, uint32_t thread_id,
                         uint64_t queue_index);

  const std::vector<att_pending_signal_t>& GetPendingSignals(uint32_t writer_id);

 private:
  rocprofiler_buffer_id_t buffer_id_;
  rocprofiler_filter_id_t filter_id_;
  rocprofiler_session_id_t session_id_;

  std::mutex sessions_pending_signals_lock_;
  std::map<uint32_t, std::vector<att_pending_signal_t>> sessions_pending_signals_;
};

}  // namespace att

}  // namespace rocprofiler


#endif  // SRC_CORE_SESSION_ATT_ATT_H_
