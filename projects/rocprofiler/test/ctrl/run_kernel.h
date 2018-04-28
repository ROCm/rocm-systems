/******************************************************************************

Copyright ©2013 Advanced Micro Devices, Inc. All rights reserved.

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

Redistributions of source code must retain the above copyright notice, this list
of conditions and the following disclaimer.

Redistributions in binary form must reproduce the above copyright notice, this
list of conditions and the following disclaimer in the documentation and/or
other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE
OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED
OF THE POSSIBILITY OF SUCH DAMAGE.

*******************************************************************************/

#ifndef TEST_CTRL_RUN_KERNEL_H_
#define TEST_CTRL_RUN_KERNEL_H_

#include "ctrl/test_hsa.h"
#include "util/test_assert.h"

template <class Kernel, class Test> bool RunKernel(int argc, char* argv[], int count = 1) {
  bool ret_val = false;

  // Create test kernel object
  Kernel test_kernel;
  TestAql* test_aql = new TestHsa(&test_kernel);
  test_aql = new Test(test_aql);
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
