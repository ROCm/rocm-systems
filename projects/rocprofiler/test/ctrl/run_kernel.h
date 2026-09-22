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

#ifndef TEST_CTRL_RUN_KERNEL_H_
#define TEST_CTRL_RUN_KERNEL_H_

#include "ctrl/test_hsa.h"
#include "util/test_assert.h"

template <class Kernel, class Test>
bool RunKernel(int argc = 0, char* argv[] = NULL, const AgentInfo* agent_info = NULL,
               hsa_queue_t* queue = NULL, int count = 1) {
  bool ret_val = false;

  if (getenv("ROC_TEST_TRACE") == NULL) std::clog.rdbuf(NULL);


  // Create test kernel object
  Kernel test_kernel;

  TestHsa* test_hsa = new TestHsa(&test_kernel);
  test_hsa->SetAgentInfo(agent_info);
  test_hsa->SetQueue(queue);

  TestAql* test_aql = new Test(test_hsa);
  TEST_ASSERT(test_aql != NULL);
  if (test_aql == NULL) return 1;

  // Initialization of Hsa Runtime
  ret_val = test_aql->Initialize(argc, argv);
  if (ret_val == false) {
    std::cerr << "Error in the test initialization" << std::endl;
    // TEST_ASSERT(ret_val);
    return false;
  }

  // Setup Hsa resources needed for execution
  ret_val = test_aql->Setup();
  if (ret_val == false) {
    std::cerr << "Error in creating hsa resources" << std::endl;
    TEST_ASSERT(ret_val);
    return false;
  }

  // Kernel dspatch iterations
  for (int i = 0; i < count; ++i) {
    // Run test kernel
    ret_val = test_aql->Run();
    if (ret_val == false) {
      std::cerr << "Error in running the test kernel" << std::endl;
      TEST_ASSERT(ret_val);
      return false;
    }

    // Verify the results of the execution
    ret_val = test_aql->VerifyResults();
    if (ret_val) {
      std::clog << "Test : Passed" << std::endl;
    } else {
      std::clog << "Test : Failed" << std::endl;
    }
  }

  // Print time taken by sample
  test_aql->PrintTime();

  test_aql->Cleanup();
  delete test_aql;

  return ret_val;
}

#endif  // TEST_CTRL_RUN_KERNEL_H_
