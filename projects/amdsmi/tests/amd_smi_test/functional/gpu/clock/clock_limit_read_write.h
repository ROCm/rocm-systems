// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef TESTS_AMD_SMI_TEST_FUNCTIONAL_CLOCK_LIMIT_READ_WRITE_H_
#define TESTS_AMD_SMI_TEST_FUNCTIONAL_CLOCK_LIMIT_READ_WRITE_H_

#include <cstdint>
#include <functional>
#include <string>
#include <utility>
#include <vector>

#include "test_base.h"

class TestClockLimitReadWrite : public TestBase {
 public:
  TestClockLimitReadWrite();

  // @Brief: Destructor for test case of TestClockLimitReadWrite
  virtual ~TestClockLimitReadWrite();

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
  struct ClockCtx {
    amdsmi_processor_handle handle;
    amdsmi_clk_type_t clk;
    const char* name;
    uint32_t dv_ind;
    std::string desc;  // "gpu=N, clk_type=X" for API output
  };
  struct Limits {
    uint64_t min_mhz;
    uint64_t max_mhz;
  };
  // One limit to set, and the status(es) the set is expected to return.
  struct Step {
    amdsmi_clk_limit_type_t type;
    std::string label;
    uint64_t target_mhz;
    amdsmi_status_t want;
    amdsmi_status_t or_want;

    static Step Valid(amdsmi_clk_limit_type_t t, std::string l, uint64_t mhz) {
      return {t, std::move(l), mhz, AMDSMI_STATUS_SUCCESS, AMDSMI_STATUS_SUCCESS};
    }
    static Step Rejected(amdsmi_clk_limit_type_t t, std::string l, uint64_t mhz) {
      return {t, std::move(l), mhz, AMDSMI_STATUS_NOT_SUPPORTED, AMDSMI_STATUS_NOT_SUPPORTED};
    }
    // Accepted, or rejected because this limit cannot be set on this clock.
    static Step Probe(amdsmi_clk_limit_type_t t, std::string l, uint64_t mhz) {
      return {t, std::move(l), mhz, AMDSMI_STATUS_SUCCESS, AMDSMI_STATUS_NOT_SUPPORTED};
    }
  };
  // What the sweep tracks as it goes.
  struct SweepState {
    const ClockCtx& clock;
    const amdsmi_frequencies_t& baseline;  // the table before any limit was set
    Limits expect;                         // the limits the device should report now
    bool manual_checked;
  };

  amdsmi_status_t CheckCall(const char* api, const std::string& args,
                            const std::function<amdsmi_status_t()>& call,
                            amdsmi_status_t want = AMDSMI_STATUS_SUCCESS,
                            amdsmi_status_t or_want = AMDSMI_STATUS_SUCCESS);
  amdsmi_status_t GetFreq(const ClockCtx& c, amdsmi_frequencies_t* f,
                          amdsmi_status_t or_want = AMDSMI_STATUS_SUCCESS);
  amdsmi_status_t GetInfo(const ClockCtx& c, amdsmi_clk_info_t* info,
                          amdsmi_status_t or_want = AMDSMI_STATUS_SUCCESS);
  amdsmi_status_t GetPerf(const ClockCtx& c, amdsmi_dev_perf_level_t* pfl);

  void RunNegativeTests(uint32_t dv_ind);
  void RunClockTest(uint32_t dv_ind, amdsmi_clk_type_t clk);
  void SweepLimits(const ClockCtx& c, const amdsmi_frequencies_t& f, const amdsmi_clk_info_t& info);
  static std::vector<uint64_t> BuildValues(const amdsmi_frequencies_t& f, const Limits& start);
  static std::vector<Step> BuildPlan(const std::vector<uint64_t>& values, const Limits& start,
                                     bool max_ok, bool min_ok);
  amdsmi_status_t ApplyStep(SweepState* sw, const Step& step);
  void CheckCurrentLevel(const SweepState& sw, const std::string& label);
  bool ReadFreqUntil(const ClockCtx& c,
                     const std::function<bool(const amdsmi_frequencies_t&)>& settled,
                     amdsmi_frequencies_t* out);
};

#endif  // TESTS_AMD_SMI_TEST_FUNCTIONAL_CLOCK_LIMIT_READ_WRITE_H_
