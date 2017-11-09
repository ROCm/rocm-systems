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

#ifndef TEST_CTRL_TEST_AQL_H_
#define TEST_CTRL_TEST_AQL_H_

#include <hsa.h>
#include <hsa_ven_amd_aqlprofile.h>

#include "util/hsa_rsrc_factory.h"

// Test AQL interface
class TestAql {
 public:
  explicit TestAql(TestAql* t = 0) : test_(t) {}
  virtual ~TestAql() { if (test_) delete test_; }

  TestAql* Test() { return test_; }
  virtual AgentInfo* GetAgentInfo() { return (test_) ? test_->GetAgentInfo() : 0; }
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

 private:
  TestAql* const test_;
};

#endif  // TEST_CTRL_TEST_AQL_H_
