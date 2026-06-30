// MIT License
//
// Copyright (c) 2017-2025 Advanced Micro Devices, Inc.
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
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.


#ifndef TEST_CTRL_TEST_AQL_H_
#define TEST_CTRL_TEST_AQL_H_

#include <hsa/hsa.h>
#include <hsa/hsa_ven_amd_aqlprofile.h>

#include "util/hsa_rsrc_factory.h"

// Test AQL interface
class TestAql {
 public:
  explicit TestAql(TestAql* t = 0) : test_(t) {}
  virtual ~TestAql() {
    if (test_) delete test_;
  }

  TestAql* Test() { return test_; }
  virtual const AgentInfo* GetAgentInfo() { return (test_) ? test_->GetAgentInfo() : 0; }
  virtual hsa_queue_t* GetQueue() { return (test_) ? test_->GetQueue() : 0; }
  virtual HsaRsrcFactory* GetRsrcFactory() { return (test_) ? test_->GetRsrcFactory() : 0; }

  // Initialize application environment including setting
  // up of various configuration parameters based on
  // command line arguments
  // @return bool true on success and false on failure
  virtual bool Initialize(int argc, char** argv) {
    return (test_) ? test_->Initialize(argc, argv) : true;
  }

  // Setup application parameters for exectuion
  // @return bool true on success and false on failure
  virtual bool Setup() { return (test_) ? test_->Setup() : true; }

  // Run the kernel
  // @return bool true on success and false on failure
  virtual bool Run() { return (test_) ? test_->Run() : true; }
  virtual bool RunSdma(size_t sdma_size) { return (test_) ? test_->RunSdma(sdma_size) : true; }

  // Verify results
  // @return bool true on success and false on failure
  virtual bool VerifyResults() { return (test_) ? test_->VerifyResults() : true; }

  // Print to console the time taken to execute kernel
  virtual void PrintTime() {
    if (test_) test_->PrintTime();
  }

  // Release resources e.g. memory allocations
  // @return bool true on success and false on failure
  virtual bool Cleanup() { return (test_) ? test_->Cleanup() : true; }

  // To get test name
  // @return test name
  virtual const char* Name() { return (test_) ? test_->Name() : NULL; }

 private:
  TestAql* const test_;
};

#endif  // TEST_CTRL_TEST_AQL_H_
