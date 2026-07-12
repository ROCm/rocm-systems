////////////////////////////////////////////////////////////////////////////////
//
// The University of Illinois/NCSA
// Open Source License (NCSA)
//
// Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
// Developed by:
//
//                 AMD Research and AMD HSA Software Development
//
//                 Advanced Micro Devices, Inc.
//
//                 www.amd.com
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to
// deal with the Software without restriction, including without limitation
// the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the
// Software is furnished to do so, subject to the following conditions:
//
//  - Redistributions of source code must retain the above copyright notice,
//    this list of conditions and the following disclaimers.
//  - Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimers in
//    the documentation and/or other materials provided with the distribution.
//  - Neither the names of Advanced Micro Devices, Inc,
//    nor the names of its contributors may be used to endorse or promote
//    products derived from this Software without specific prior written
//    permission.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
// THE CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR
// OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
// ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
// DEALINGS WITH THE SOFTWARE.
//
////////////////////////////////////////////////////////////////////////////////

#ifndef HSA_RUNTIME_CORE_INC_AMD_AIE_CODE_HPP_
#define HSA_RUNTIME_CORE_INC_AMD_AIE_CODE_HPP_

#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace rocr {
namespace amd { namespace elf { class Image; } }
namespace AMD {

struct AieKernelInfo {
  std::string name;
  const uint8_t* insts_data = nullptr;  // into ELF buffer; never null after parse
  uint64_t insts_size = 0;              // > 0
  const uint8_t* pdi_data = nullptr;    // null if no PDI
  uint64_t pdi_size = 0;                // 0 if no PDI
  uint32_t kernarg_size = 0;
  uint32_t num_cols = 0;
};

class AieCode {
 public:
  static std::unique_ptr<AieCode> Create(const void* data, size_t size);
  static bool IsAieCodeObject(const void* data, size_t size);

  const std::string& GetArchSectionName() const { return arch_section_name_; }
  std::vector<std::string> GetKernelNames() const;
  const AieKernelInfo* GetKernel(const std::string& name) const;
  size_t GetKernelCount() const { return kernels_.size(); }

 private:
  AieCode() = default;
  bool Parse();

  std::unique_ptr<amd::elf::Image> elf_;
  const uint8_t* section_base_ = nullptr;  // start of arch section in ELF buffer
  uint64_t section_size_ = 0;
  std::string arch_section_name_;
  std::map<std::string, AieKernelInfo> kernels_;
};

}  // namespace AMD
}  // namespace rocr

#endif  // HSA_RUNTIME_CORE_INC_AMD_AIE_CODE_HPP_
