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

#ifndef RDC_MODULES_RDC_ROCR_RDCROCRBASE_H_
#define RDC_MODULES_RDC_ROCR_RDCROCRBASE_H_
#include <stdint.h>
#include <stdio.h>

#include <string>

#include "hsa/hsa.h"
#include "hsa/hsa_ext_amd.h"
#include "rdc_lib/RdcPerfTimer.h"

namespace amd {
namespace rdc {

/// Common interface for RocR tests and samples
class RdcRocrBase {
 public:
  RdcRocrBase(void);

  virtual ~RdcRocrBase(void);

  ///< Setters and Getters

  void set_gpu_device1(hsa_agent_t in_dev) { gpu_device1_.handle = in_dev.handle; }
  hsa_agent_t* gpu_device1(void) { return &gpu_device1_; }

  void set_cpu_device(hsa_agent_t in_dev) { cpu_device_.handle = in_dev.handle; }
  hsa_agent_t* cpu_device(void) { return &cpu_device_; }

  void set_kernel_file_name(const char* in_file_name) { kernel_file_name_ = in_file_name; }
  std::string const kernel_file_name(void) const { return kernel_file_name_; }

  void set_kernel_name(std::string in_kernel_name) { kernel_name_ = in_kernel_name; }
  std::string const kernel_name(void) const { return kernel_name_; }

  void set_agent_name(std::string in_agent_name) { agent_name_ = in_agent_name; }

  std::string const get_agent_name(void) const { return agent_name_; }

  void set_kernel_object(uint64_t in_kernel_object) { kernel_object_ = in_kernel_object; }
  uint64_t kernel_object(void) const { return kernel_object_; }

  void set_profile(hsa_profile_t in_prof) { profile_ = in_prof; }
  hsa_profile_t profile(void) const { return profile_; }

  uint32_t private_segment_size(void) const { return private_segment_size_; }
  void set_private_segment_size(uint32_t sz) { private_segment_size_ = sz; }

  void set_group_segment_size(uint32_t sz) { group_segment_size_ = sz; }
  uint32_t group_segment_size(void) const { return group_segment_size_; }

  void set_group_size(uint32_t sz) { group_size_ = sz; }
  uint32_t group_size(void) const { return group_size_; }

  void set_main_queue(hsa_queue_t* q) { main_queue_ = q; }
  hsa_queue_t* main_queue(void) const { return main_queue_; }

  hsa_kernel_dispatch_packet_t& aql(void) { return aql_; }

  void set_num_iteration(int num) { num_iteration_ = num; }
  uint32_t num_iteration(void) const { return num_iteration_; }

  hsa_amd_memory_pool_t& device_pool(void) { return device_pool_; }

  hsa_amd_memory_pool_t& cpu_pool(void) { return cpu_pool_; }

  hsa_amd_memory_pool_t& kern_arg_pool(void) { return kern_arg_pool_; }

  void set_kernarg_size(uint32_t sz) { kernarg_size_ = sz; }
  uint32_t kernarg_size(void) const { return kernarg_size_; }

  void set_kernarg_align(uint32_t align) { kernarg_align_ = align; }
  uint32_t kernarg_align(void) const { return kernarg_align_; }

  void* kernarg_buffer(void) const { return kernarg_buffer_; }
  void set_kernarg_buffer(void* buffer) { kernarg_buffer_ = buffer; }

  int32_t requires_profile(void) const { return requires_profile_; }

  char* orig_hsa_enable_interrupt() const { return orig_hsa_enable_interrupt_; }

  bool enable_interrupt() const { return enable_interrupt_; }

  void set_title(std::string name) { title_ = name; }
  std::string title(void) const { return title_; }

  RdcPerfTimer* hsa_timer(void) { return &hsa_timer_; }

  void set_verbosity(uint32_t v) { verbosity_ = v; }
  uint32_t verbosity(void) const { return verbosity_; }

  void set_monitor_verbosity(uint32_t m) { monitor_verbosity_ = m; }
  uint32_t monitor_verbosity(void) const { return monitor_verbosity_; }

 protected:
  void set_requires_profile(int32_t reqd_prof) { requires_profile_ = reqd_prof; }

  void set_enable_interrupt(bool doEnable) { enable_interrupt_ = doEnable; }

 private:
  uint64_t num_iteration_;  ///< Number of times to execute test

  hsa_queue_t* main_queue_;  ///< AQL queue used for packets

  hsa_agent_t gpu_device1_;  ///< Handle to first GPU found

  hsa_agent_t cpu_device_;  ///< Handle to CPU

  hsa_amd_memory_pool_t device_pool_;  ///< Memory pool on gpu pool list

  hsa_amd_memory_pool_t cpu_pool_;  ///< Memory pool on cpu pool list

  hsa_amd_memory_pool_t kern_arg_pool_;  ///< Memory pool suitable for args

  uint64_t kernel_object_;  ///< Handle to kernel code

  std::string kernel_file_name_;  ///< Code object file name

  std::string kernel_name_;  ///< Kernel name

  std::string agent_name_;  ///< Agent name

  hsa_kernel_dispatch_packet_t aql_;  ///< Kernel dispatch packet

  uint32_t group_segment_size_;  ///< Kernel group seg size

  uint32_t kernarg_size_;  ///< Kernarg memory size

  uint32_t kernarg_align_;  ///< Alignment for kern argument memory

  void* kernarg_buffer_;  ///< Unaligned allocated kernel arg. buffer

  hsa_profile_t profile_;  ///< Device profile.

  uint32_t group_size_;  ///< Number of work items in one group

  uint32_t private_segment_size_;  ///< Kernel private seg size

  int32_t requires_profile_;  ///< Profile required by test (-1 if no req.)

  char* orig_hsa_enable_interrupt_;  ///< Orig. value of HSA_ENABLE_INTERRUPT

  bool enable_interrupt_;  ///< Whether to enable/disable interrupts for test

  std::string title_;  ///< Displayed title of test

  uint32_t verbosity_;  ///< How much additional output to produce

  uint32_t monitor_verbosity_;  ///< verbose or not

  RdcPerfTimer hsa_timer_;  ///< Timer to be used for timing parts of test
};

}  // namespace rdc
}  // namespace amd

#endif  // RDC_MODULES_RDC_ROCR_RDCROCRBASE_H_
