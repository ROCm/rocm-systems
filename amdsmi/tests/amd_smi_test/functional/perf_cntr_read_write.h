/*
 * Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */

#ifndef TESTS_AMD_SMI_TEST_FUNCTIONAL_PERF_CNTR_READ_WRITE_H_
#define TESTS_AMD_SMI_TEST_FUNCTIONAL_PERF_CNTR_READ_WRITE_H_

#include <string>

#include "../test_base.h"

class TestPerfCntrReadWrite : public TestBase {
 public:
    TestPerfCntrReadWrite();

  // @Brief: Destructor for test case of TestPerfCntrReadWrite
  virtual ~TestPerfCntrReadWrite();

  // @Brief: Setup the environment for measurement
  virtual void SetUp();

  // @Brief: Core measurement execution
  virtual void Run();

  // @Brief: Clean up and retrive the resource
  virtual void Close();

  // @Brief: Display  results
  virtual void DisplayResults() const;

  // @Brief: Display information about what this test does
  virtual void DisplayTestInfo(void);

 private:
  void CountEvents(amdsmi_processor_handle dv_ind,
       amdsmi_event_type_t evnt, amdsmi_counter_value_t *val,
                                                   int32_t sleep_sec = 1);
  void testEventsIndividually(amdsmi_processor_handle dv_ind);
  void testEventsSimultaneously(amdsmi_processor_handle dv_ind);
};

class PerfCntrEvtGrp {
 public:
    explicit PerfCntrEvtGrp(amdsmi_event_group_t grp,
                             uint32_t first, uint32_t last, std::string name);
    ~PerfCntrEvtGrp();

    amdsmi_event_group_t group(void) const { return grp_;}
    uint32_t first_evt(void) const {return first_evt_;}
    uint32_t last_evt(void) const {return last_evt_;}
    uint32_t num_events(void) const {return num_events_;}
    std::string name(void) const { return name_;}
 private:
    amdsmi_event_group_t grp_;
    uint32_t first_evt_;
    uint32_t last_evt_;
    uint32_t num_events_;
    std::string name_;
};

#endif  // TESTS_AMD_SMI_TEST_FUNCTIONAL_PERF_CNTR_READ_WRITE_H_
