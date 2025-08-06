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

#include <hsa/hsa.h>
#include <stdio.h>
#include <string.h>
#include <iostream>
#include <thread>

#include "ctrl/run_kernel.h"
#include "ctrl/test_aql.h"
#include "dummy_kernel/dummy_kernel.h"
#include "simple_convolution/simple_convolution.h"

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
      //RunKernel<DummyKernel, TestAql>(0, NULL, agent_info[n], queue[n], diter);
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

  const int kiter = (kiter_s != NULL) ? atol(kiter_s) : 1;
  const int diter = (diter_s != NULL) ? atol(diter_s) : 1;
  const uint32_t agents_number = (agents_s != NULL) ? (uint32_t)atol(agents_s) : 1;
  const int thrs = (thrs_s != NULL) ? atol(thrs_s) : 1;

  TestHsa::HsaInstantiate();

  std::vector<std::thread> t(thrs);
  for (int n = 0; n < thrs; ++n) {
    t[n] = std::thread(thread_fun, kiter, diter, agents_number);
  }
  for (int n = 0; n < thrs; ++n) {
    t[n].join();
  }

  TestHsa::HsaShutdown();
  return 0;
}
