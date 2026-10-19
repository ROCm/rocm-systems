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

namespace amd {
namespace rdc {

/// Common interface for RocR tests and samples
class RdcRocrBase {
 public:
  RdcRocrBase(void);

  virtual ~RdcRocrBase(void);

  ///< Setters and Getters

  void set_kernel_file_name(const char* in_file_name) { kernel_file_name_ = in_file_name; }
  std::string const kernel_file_name(void) const { return kernel_file_name_; }

  void set_kernel_name(std::string in_kernel_name) { kernel_name_ = in_kernel_name; }
  std::string const kernel_name(void) const { return kernel_name_; }

  void set_agent_name(std::string in_agent_name) { agent_name_ = in_agent_name; }

  std::string const get_agent_name(void) const { return agent_name_; }

  void set_kernel_object(uint64_t in_kernel_object) { kernel_object_ = in_kernel_object; }
  uint64_t kernel_object(void) const { return kernel_object_; }

  uint32_t private_segment_size(void) const { return private_segment_size_; }
  void set_private_segment_size(uint32_t sz) { private_segment_size_ = sz; }

  void set_group_segment_size(uint32_t sz) { group_segment_size_ = sz; }
  uint32_t group_segment_size(void) const { return group_segment_size_; }

  void set_kernarg_size(uint32_t sz) { kernarg_size_ = sz; }
  uint32_t kernarg_size(void) const { return kernarg_size_; }

  void set_kernarg_align(uint32_t align) { kernarg_align_ = align; }
  uint32_t kernarg_align(void) const { return kernarg_align_; }

 private:
  uint64_t kernel_object_ = 0;  ///< Handle to kernel code

  std::string kernel_file_name_;  ///< Code object file name

  std::string kernel_name_;  ///< Kernel name

  std::string agent_name_;  ///< Agent name

  uint32_t group_segment_size_ = 0;  ///< Kernel group seg size

  uint32_t kernarg_size_ = 0;  ///< Kernarg memory size

  uint32_t kernarg_align_ = 0;  ///< Alignment for kern argument memory

  uint32_t private_segment_size_ = 0;  ///< Kernel private seg size
};

}  // namespace rdc
}  // namespace amd

#endif  // RDC_MODULES_RDC_ROCR_RDCROCRBASE_H_
