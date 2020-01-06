/*
Copyright (c) 2019 - Advanced Micro Devices, Inc. All rights reserved.

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
#ifndef TESTS_RDC_TESTS_TEST_BASE_H_
#define TESTS_RDC_TESTS_TEST_BASE_H_

#include <string>
#include "rdc/rdc_client.h"

class TestBase {
 public:
  TestBase(void);

  virtual ~TestBase(void);

  enum VerboseLevel {VERBOSE_MIN = 0, VERBOSE_STANDARD, VERBOSE_PROGRESS};

  // @Brief: Before run the core measure codes, do something to set up
  // i.e. init runtime, prepare packet...
  virtual void SetUp(void);

  // @Brief: Core measurement codes executing here
  virtual void Run(void);

  // @Brief: Do something clean up
  virtual void Close(void);

  // @Brief: Display the results
  virtual void DisplayResults(void) const;

  // @Brief: Display information about the test
  virtual void DisplayTestInfo(void);

  const std::string & description(void) const {return description_;}

  void set_description(std::string d);

  void set_title(std::string name) {
    title_ = name;
  }
  std::string title(void) const {
    return title_;
  }
  void set_verbosity(uint32_t v) {
    verbosity_ = v;
  }
  uint32_t verbosity(void) const {
    return verbosity_;
  }
  void set_dont_fail(bool f) {
    dont_fail_ = f;
  }
  bool dont_fail(void) const {
    return dont_fail_;
  }
  void set_num_monitor_devs(uint32_t i) {
    num_monitor_devs_ = i;
  }
  uint32_t num_monitor_devs(void) const {
    return num_monitor_devs_;
  }
  void set_init_options(uint64_t x) {
    init_options_ = x;
  }
  uint64_t init_options(void) const {
    return init_options_;
  }
  void set_rdc_channel(rdc_channel_t c) {
    rdc_channel_ = c;
  }
  rdc_channel_t rdc_channel(void) const {
    return rdc_channel_;
  }
  void set_monitor_server_ip(std::string ip) {
    monitor_server_ip_ = ip;
  }
  std::string monitor_server_ip(void) const {
    return monitor_server_ip_;
  }
  void set_monitor_server_port(std::string port) {
    monitor_server_port_ = port;
  }
  std::string monitor_server_port(void) const {
    return monitor_server_port_;
  }


 protected:
  void PrintDeviceHeader(uint32_t dv_ind);
  rdc_status_t AllocateRDCChannel(void);

 private:
  uint64_t num_monitor_devs_;  ///< Number of monitor devices found
  std::string description_;
  std::string title_;   ///< Displayed title of test
  uint32_t verbosity_;   ///< How much additional output to produce
  bool dont_fail_;       ///< Don't quit test on individual failure if true
  uint64_t init_options_;  ///< rsmi initialization options
  std::string monitor_server_ip_;
  std::string monitor_server_port_;
  rdc_channel_t rdc_channel_;
};

#define IF_VERB(VB) if (verbosity() && verbosity() >= (TestBase::VERBOSE_##VB))

// Macros to be used within TestBase classes
#define CHK_ERR_ASRT(RET) { \
    if (dont_fail() && ((RET) != RDC_STATUS_SUCCESS)) { \
        std::cout << std::endl << "\t===> TEST FAILURE." << std::endl; \
        DISPLAY_RDC_ERR(RET); \
        std::cout << \
         "\t===> Abort is over-ridden due to dont_fail command line option." \
                                                               << std::endl; \
        return; \
    } else { \
      ASSERT_EQ(RDC_STATUS_SUCCESS, (RET)); \
    } \
}

#endif  // TESTS_RDC_TESTS_TEST_BASE_H_
