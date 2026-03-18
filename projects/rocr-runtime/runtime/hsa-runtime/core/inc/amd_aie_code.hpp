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
namespace amd {
namespace elf {
class Image;
class Section;
class Symbol;
}  // namespace elf
}  // namespace amd

namespace AMD {

/// @brief Information about an AIE kernel extracted from the ELF.
struct AieKernelInfo {
  std::string name;

  /// Offset into instruction data where this kernel starts.
  uint64_t instr_offset = 0;

  /// Size of instruction data for this kernel.
  uint64_t instr_size = 0;

  /// Offset into control packet data.
  uint64_t ctrl_packet_offset = 0;

  /// Size of control packet data.
  uint64_t ctrl_packet_size = 0;

  /// Kernel argument buffer size.
  uint32_t kernarg_size = 0;

  /// Number of columns used by this kernel.
  uint32_t num_cols = 0;
};

/// @brief Parser for NPU/AIE ELF code objects.
///
/// This class parses NPU ELF files to extract kernel metadata, instruction
/// buffers, and control packet data.
///
/// NPU ELF files contain:
/// - .ctrltext section: Control/instruction data for the NPU
/// - .ctrldata section: Control packet data
/// - Symbol table: Kernel names and metadata
/// - Notes: Additional metadata (kernel arguments, column counts, etc.)
class AieCode {
 public:
  /// @brief Creates an AieCode instance from a memory buffer.
  ///
  /// @param data Pointer to the ELF data.
  /// @param size Size of the ELF data in bytes.
  /// @return Unique pointer to AieCode on success, nullptr on failure.
  static std::unique_ptr<AieCode> Create(const void* data, size_t size);

  /// @brief Checks if the given ELF data is an AIE code object.
  ///
  /// @param data Pointer to the ELF data.
  /// @param size Size of the ELF data in bytes.
  /// @return true if this is an AIE ELF, false otherwise.
  static bool IsAieCodeObject(const void* data, size_t size);

  /// @brief Returns the instruction buffer data.
  const uint8_t* GetInstructionData() const;

  /// @brief Returns the size of the instruction buffer.
  size_t GetInstructionSize() const;

  /// @brief Returns the control packet data.
  const uint8_t* GetCtrlPacketData() const;

  /// @brief Returns the size of the control packet buffer.
  size_t GetCtrlPacketSize() const;

  /// @brief Returns kernel info by name.
  ///
  /// @param name The kernel name.
  /// @return Pointer to kernel info if found, nullptr otherwise.
  const AieKernelInfo* GetKernel(const std::string& name) const;

  /// @brief Returns a list of all kernel names in this code object.
  std::vector<std::string> GetKernelNames() const;

  /// @brief Returns the number of kernels in this code object.
  size_t GetKernelCount() const { return kernels_.size(); }

  /// @brief Returns the raw ELF data.
  const void* GetElfData() const { return elf_data_; }

  /// @brief Returns the raw ELF size.
  size_t GetElfSize() const { return elf_size_; }

 private:
  AieCode() = default;

  /// @brief Parses the ELF and extracts kernel metadata.
  bool Parse();

  /// @brief Extracts kernel symbols from the symbol table.
  bool ExtractKernelSymbols();

  /// @brief Loads section data into memory.
  bool LoadSectionData(amd::elf::Section* section, std::vector<uint8_t>& dest);

  /// Internal ELF image.
  std::unique_ptr<amd::elf::Image> elf_;

  /// Raw ELF data pointer (not owned).
  const void* elf_data_ = nullptr;

  /// Size of raw ELF data.
  size_t elf_size_ = 0;

  /// Parsed kernel metadata.
  std::map<std::string, AieKernelInfo> kernels_;

  /// Instruction buffer data (from .ctrltext section).
  std::vector<uint8_t> instr_data_;

  /// Control packet data (from .ctrldata section).
  std::vector<uint8_t> ctrl_packet_data_;
};

}  // namespace AMD
}  // namespace rocr

#endif  // HSA_RUNTIME_CORE_INC_AMD_AIE_CODE_HPP_
