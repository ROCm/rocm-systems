/*
Copyright (c) 2021 - present Advanced Micro Devices, Inc. All rights reserved.

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
#ifndef RDC_MODULES_RDC_ROCR_TESTBASE_H_
#define RDC_MODULES_RDC_ROCR_TESTBASE_H_

#include <memory>
#include <string>
#include <vector>

#include "rdc_modules/rdc_rocr/RdcRocrBase.h"

namespace amd {
namespace rdc {

class TestBase : public RdcRocrBase {
 public:
  explicit TestBase(uint32_t gpu_index);

  virtual ~TestBase(void);

  enum VerboseLevel { VERBOSE_MIN = 0, VERBOSE_STANDARD, VERBOSE_PROGRESS };

  // @Brief: Before run the core measure codes, do something to set up
  // i.e. init runtime, prepare packet...
  virtual hsa_status_t SetUp(void);

  // @Brief: Core measurement codes executing here
  virtual void Run(void);

  // @Brief: Do something clean up
  virtual void Close(void);

  // @Brief: Display the results
  virtual void DisplayResults(void) const;

  // @Brief: Display information about the test
  virtual void DisplayTestInfo(void);

  const std::string& description(void) const { return description_; }

  void set_description(std::string d);

  const std::string& get_gpu_info() const { return gpu_info_; }
  const std::string& get_per_gpu_info() const { return per_gpu_info_; }

  hsa_status_t FindGPUIndex(hsa_agent_t agent, void* data);
  // Return the agent by GPU index in amd_smi
  hsa_status_t get_agent_by_gpu_index(uint32_t gpu_index, hsa_agent_t* agent);

 protected:
  uint32_t gpu_index_;
  std::string gpu_info_;
  std::string per_gpu_info_;

 private:
  std::string description_;
};

}  // namespace rdc
}  // namespace amd

#endif  // RDC_MODULES_RDC_ROCR_TESTBASE_H_
