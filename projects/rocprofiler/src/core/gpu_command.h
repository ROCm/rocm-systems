/******************************************************************************
Copyright (c) 2018 Advanced Micro Devices, Inc. All rights reserved.

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
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*******************************************************************************/

#ifndef SRC_CORE_GPU_COMMAND_H_
#define SRC_CORE_GPU_COMMAND_H_

#include <hsa/hsa.h>

#include "core/types.h"
#include "util/exception.h"
#include "util/hsa_rsrc_factory.h"

namespace rocprofiler {
enum gpu_cmd_op_t {
  PMC_ENABLE_GPU_CMD_OP = 0,
  PMC_DISABLE_GPU_CMD_OP = 1,
  WAIT_IDLE_GPU_CMD_OP = 2,
  NUMBER_GPU_CMD_OP
};

size_t GetGpuCommand(gpu_cmd_op_t op, const rocprofiler::util::AgentInfo* agent_info,
                     packet_t** command_out);

static inline size_t IssueGpuCommand(gpu_cmd_op_t op,
                                     const rocprofiler::util::AgentInfo* agent_info,
                                     hsa_queue_t* queue) {
  packet_t* command;
  const size_t size = GetGpuCommand(op, agent_info, &command);
  hsa_status_t status = hsa_signal_create(1, 0, NULL, &(command->completion_signal));
  if (status != HSA_STATUS_SUCCESS) EXC_RAISING(status, "signal_create " << std::hex << status);
  rocprofiler::util::HsaRsrcFactory::Instance().Submit(queue, command, size);
  rocprofiler::util::HsaRsrcFactory::Instance().SignalWait(command->completion_signal, 1);
  status = hsa_signal_destroy(command->completion_signal);
  if (status != HSA_STATUS_SUCCESS) EXC_ABORT(status, "signal_destroy " << std::hex << status);
  return HSA_STATUS_SUCCESS;
}

static inline size_t IssueGpuCommand(gpu_cmd_op_t op, hsa_agent_t agent, hsa_queue_t* queue) {
  rocprofiler::util::HsaRsrcFactory* hsa_rsrc = &rocprofiler::util::HsaRsrcFactory::Instance();
  const rocprofiler::util::AgentInfo* agent_info = hsa_rsrc->GetAgentInfo(agent);
  return IssueGpuCommand(op, agent_info, queue);
}

}  // namespace rocprofiler

#endif  // SRC_CORE_GPU_COMMAND_H_
