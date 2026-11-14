/*
Copyright (c) 2026 - present Advanced Micro Devices, Inc. All rights reserved.

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
*/
#ifndef TESTS_RDC_TESTS_FUNCTIONAL_RDC_POLL_LATENCY_H_
#define TESTS_RDC_TESTS_FUNCTIONAL_RDC_POLL_LATENCY_H_

#include <string>

#include "rdc_tests/test_base.h"

// Reports how long RDC takes to read telemetry from all GPUs on this system.
class TestRdcPollLatency : public TestBase {
 public:
  TestRdcPollLatency();

  // @Brief: Destructor for test case of TestRdcPollLatency
  virtual ~TestRdcPollLatency();

  // @Brief: Setup the environment for measurement
  virtual void SetUp();

  // @Brief: Core measurement execution
  virtual void Run();

  // @Brief: Clean up and retrieve the resource
  virtual void Close();

  // @Brief: Display  results
  virtual void DisplayResults() const;

  // @Brief: Display information about what this test does
  virtual void DisplayTestInfo(void);

 private:
  std::string report_;
  // What Run() created, for Close() to release even after a failed assertion.
  bool started_ = false;
  bool group_created_ = false;
  bool field_group_created_ = false;
  bool watching_ = false;
  rdc_gpu_group_t group_id_ = 0;
  rdc_field_grp_t field_group_id_ = 0;
};

#endif  // TESTS_RDC_TESTS_FUNCTIONAL_RDC_POLL_LATENCY_H_
