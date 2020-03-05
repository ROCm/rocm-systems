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

#include <dirent.h>
#include <hsa.h>
#include <hsakmt.h>
#include <stdio.h>
#include <string.h>

#include <fstream>
#include <iostream>
#include <thread>

#include "ctrl/run_kernel.h"
#include "ctrl/test_aql.h"
#include "dummy_kernel/dummy_kernel.h"
#include "simple_convolution/simple_convolution.h"

int get_gpu_node_id() {
  int gpu_node = - 1;

#if 0
  // find a valid gpu node from /sys/class/kfd/kfd/topology/nodes
  std::string path = "/sys/class/kfd/kfd/topology/nodes";
  DIR *dir;
  struct dirent *ent;

  if ((dir = opendir(path.c_str())) != NULL) {
    while ((ent = readdir(dir)) != NULL) {

      std::string dir = ent->d_name;

      if (dir.find_first_not_of("0123456789") == std::string::npos) {
        std::string file = path + "/" + ent->d_name + "/gpu_id";
        std::ifstream infile(file);
        int id;

        infile >> id;
        if (id != 0) {
          gpu_node = atoi(ent->d_name);
          break;
        }
      }
    }
    closedir (dir);
  }
#else
  HsaSystemProperties m_SystemProperties;
  memset(&m_SystemProperties, 0, sizeof(m_SystemProperties));

  HSAKMT_STATUS status = hsaKmtAcquireSystemProperties(&m_SystemProperties);
  if (status != HSAKMT_STATUS_SUCCESS) {
    std::cerr << "Error in hsaKmtAcquireSystemProperties"<< std::endl;
    return 1;
  }

  // tranverse all CPU and GPU nodes and break when a GPU node is found
  for (unsigned i = 0; i < m_SystemProperties.NumNodes; ++i) {
    HsaNodeProperties nodeProperties;
    memset(&nodeProperties, 0, sizeof(HsaNodeProperties));

    status = hsaKmtGetNodeProperties(i, &nodeProperties);
    if (status != HSAKMT_STATUS_SUCCESS) {
      std::cerr << "Error in hsaKmtAcquireSystemProperties"<< std::endl;
      break;
    } else if(nodeProperties.NumFComputeCores) {
      gpu_node = i;
      break;
    }
  }
#endif

  printf ("GPU node id(%d)\n", gpu_node);
  return gpu_node;
}

void thread_fun(const int kiter, const int diter, const uint32_t agents_number) {
  const AgentInfo* agent_info[agents_number];
  hsa_queue_t* queue[agents_number];
  HsaRsrcFactory* rsrc = &HsaRsrcFactory::Instance();

  for (uint32_t n = 0; n < agents_number; ++n) {
    uint32_t agent_id = n % rsrc->GetCountOfGpuAgents();
    if (rsrc->GetGpuAgentInfo(agent_id, &agent_info[n]) == false) {
      fprintf(stderr, "AgentInfo failed\n");
      abort();
    }
    if (rsrc->CreateQueue(agent_info[n], 128, &queue[n]) == false) {
      fprintf(stderr, "CreateQueue failed\n");
      abort();
    }
  }

  for (int i = 0; i < kiter; ++i) {
    for (uint32_t n = 0; n < agents_number; ++n) {
      // RunKernel<DummyKernel, TestAql>(0, NULL, agent_info[n], queue[n], diter);
      RunKernel<SimpleConvolution, TestAql>(0, NULL, agent_info[n], queue[n], diter);
    }
  }

  for (uint32_t n = 0; n < agents_number; ++n) {
    hsa_queue_destroy(queue[n]);
  }
}

int main(int argc, char** argv) {
  const char* kiter_s = getenv("ROCP_KITER");
  const char* diter_s = getenv("ROCP_DITER");
  const char* agents_s = getenv("ROCP_AGENTS");
  const char* thrs_s = getenv("ROCP_THRS");
  const char* spm_enabled = getenv("ROCP_SPM");
  int gpu_node_id = -1;

  const int kiter = (kiter_s != NULL) ? atol(kiter_s) : 1;
  const int diter = (diter_s != NULL) ? atol(diter_s) : 1;
  const uint32_t agents_number = (agents_s != NULL) ? (uint32_t)atol(agents_s) : 1;
  const int thrs = (thrs_s != NULL) ? atol(thrs_s) : 1;

  if (spm_enabled != NULL) {
    if (hsa_init() != HSA_STATUS_SUCCESS) {
      std::cerr << "Error in hsa_init()" << std::endl;
      return 1;
    }
    gpu_node_id = get_gpu_node_id();
    if (gpu_node_id == -1) {
      std::cerr << "Error in get_gpu_node_id()" << std::endl;
      return 1;
    }
    HSAKMT_STATUS status = hsaKmtEnableDebugTrap(gpu_node_id, INVALID_QUEUEID);
    if (status != HSAKMT_STATUS_SUCCESS) {
      std::cerr << "Error in enabling debug trap" << std::endl;
      return 1;
    }
  }

  TestHsa::HsaInstantiate();

  std::vector<std::thread> t(thrs);
  for (int n = 0; n < thrs; ++n) {
    t[n] = std::thread(thread_fun, kiter, diter, agents_number);
  }
  for (int n = 0; n < thrs; ++n) {
    t[n].join();
  }

  if (spm_enabled != NULL) {
    if (gpu_node_id == -1) {
      std::cerr << "Invalid GPU node id" << std::endl;
      return 1;
    }
    HSAKMT_STATUS status = hsaKmtDisableDebugTrap(gpu_node_id);
    if (status != HSAKMT_STATUS_SUCCESS) {
      std::cerr << "Error in disabling debug" << std::endl;
      return 1;
    }
  }

  TestHsa::HsaShutdown();
  return 0;
}
