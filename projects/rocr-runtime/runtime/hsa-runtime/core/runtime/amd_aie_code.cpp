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

#include "core/inc/amd_aie_code.hpp"

#include <cstring>
#include <elf.h>

#include "core/inc/amd_elf_image.hpp"

namespace rocr {
namespace AMD {

namespace {

// Section names used in NPU ELF files.
constexpr const char* kInstrSectionName = ".ctrltext";
constexpr const char* kCtrlPacketSectionName = ".ctrldata";
constexpr const char* kGroupSectionPrefix = ".group";

// NPU ELF uses a distinct OS ABI or machine type to distinguish from GPU code.
// AIE ELF files typically use EM_NONE or a specific AIE machine type.
// For now, we detect based on section presence.
constexpr uint16_t kAieMachine = 0;  // EM_NONE - AIE ELF files use this

// Symbol types for AIE kernels
constexpr uint8_t kAieKernelSymbolType = STT_FUNC;

}  // namespace

bool AieCode::IsAieCodeObject(const void* data, size_t size) {
  if (!data || size < sizeof(Elf64_Ehdr)) {
    return false;
  }

  const auto* ehdr = static_cast<const Elf64_Ehdr*>(data);

  // Check ELF magic
  if (memcmp(ehdr->e_ident, ELFMAG, SELFMAG) != 0) {
    return false;
  }

  // AIE code objects typically use EM_NONE (0) as the machine type,
  // or they have specific NPU-related sections. Check both.
  if (ehdr->e_machine == kAieMachine) {
    return true;
  }

  // Also check for presence of .ctrltext section which is specific to NPU code.
  // This requires parsing the section headers.
  // For a quick check, we verify machine type != EM_AMDGPU (224) to avoid
  // treating GPU code as NPU code.
  constexpr uint16_t kEmAmdgpu = 224;
  if (ehdr->e_machine == kEmAmdgpu) {
    return false;  // This is GPU code, not NPU
  }

  // If machine type is something else, we need to verify by looking at sections.
  // For now, accept any non-AMDGPU ELF as potentially AIE and let Parse() verify.
  return true;
}

std::unique_ptr<AieCode> AieCode::Create(const void* data, size_t size) {
  if (!data || size == 0) {
    return nullptr;
  }

  auto code = std::unique_ptr<AieCode>(new AieCode());
  code->elf_data_ = data;
  code->elf_size_ = size;

  // Create ELF image and parse
  code->elf_.reset(amd::elf::NewElf64Image());
  if (!code->elf_) {
    return nullptr;
  }

  if (!code->elf_->initFromBuffer(data, size)) {
    return nullptr;
  }

  if (!code->Parse()) {
    return nullptr;
  }

  return code;
}

bool AieCode::Parse() {
  // Find and load instruction section (.ctrltext)
  amd::elf::Section* instr_section = nullptr;
  amd::elf::Section* ctrl_packet_section = nullptr;

  for (size_t i = 0; i < elf_->sectionCount(); ++i) {
    amd::elf::Section* sec = elf_->section(i);
    if (!sec) continue;

    std::string name = sec->Name();

    if (name == kInstrSectionName) {
      instr_section = sec;
    } else if (name == kCtrlPacketSectionName) {
      ctrl_packet_section = sec;
    }
  }

  // Load instruction data if present
  if (instr_section) {
    if (!LoadSectionData(instr_section, instr_data_)) {
      return false;
    }
  }

  // Load control packet data if present
  if (ctrl_packet_section) {
    if (!LoadSectionData(ctrl_packet_section, ctrl_packet_data_)) {
      return false;
    }
  }

  // Extract kernel symbols
  if (!ExtractKernelSymbols()) {
    // If no kernels found but we have instruction data, create a default kernel
    if (!instr_data_.empty()) {
      AieKernelInfo default_kernel;
      default_kernel.name = "_default_";
      default_kernel.instr_offset = 0;
      default_kernel.instr_size = instr_data_.size();
      default_kernel.ctrl_packet_offset = 0;
      default_kernel.ctrl_packet_size = ctrl_packet_data_.size();
      default_kernel.kernarg_size = 0;
      default_kernel.num_cols = 1;
      kernels_["_default_"] = default_kernel;
    }
  }

  // Verify we have at least instruction data
  return !instr_data_.empty();
}

bool AieCode::ExtractKernelSymbols() {
  amd::elf::SymbolTable* symtab = elf_->symtab();
  if (!symtab) {
    return false;
  }

  bool found_kernel = false;

  for (size_t i = 0; i < symtab->symbolCount(); ++i) {
    amd::elf::Symbol* sym = symtab->symbol(i);
    if (!sym) continue;

    // Look for function symbols (potential kernels)
    if (sym->type() != kAieKernelSymbolType) {
      continue;
    }

    // Skip local/undefined symbols
    if (sym->binding() == STB_LOCAL || sym->value() == 0) {
      continue;
    }

    amd::elf::Section* sec = sym->section();
    if (!sec) continue;

    // Check if symbol is in instruction section
    std::string sec_name = sec->Name();
    if (sec_name != kInstrSectionName) {
      continue;
    }

    AieKernelInfo kernel;
    kernel.name = sym->name();
    kernel.instr_offset = sym->value();
    kernel.instr_size = sym->size();
    kernel.ctrl_packet_offset = 0;
    kernel.ctrl_packet_size = ctrl_packet_data_.size();
    kernel.kernarg_size = 0;  // TODO: Extract from metadata/notes
    kernel.num_cols = 1;      // TODO: Extract from metadata/notes

    kernels_[kernel.name] = kernel;
    found_kernel = true;
  }

  return found_kernel;
}

bool AieCode::LoadSectionData(amd::elf::Section* section,
                              std::vector<uint8_t>& dest) {
  if (!section) {
    return false;
  }

  uint64_t size = section->size();
  if (size == 0) {
    return true;  // Empty section is valid
  }

  dest.resize(size);
  return section->getData(0, dest.data(), size);
}

const uint8_t* AieCode::GetInstructionData() const {
  return instr_data_.empty() ? nullptr : instr_data_.data();
}

size_t AieCode::GetInstructionSize() const {
  return instr_data_.size();
}

const uint8_t* AieCode::GetCtrlPacketData() const {
  return ctrl_packet_data_.empty() ? nullptr : ctrl_packet_data_.data();
}

size_t AieCode::GetCtrlPacketSize() const {
  return ctrl_packet_data_.size();
}

const AieKernelInfo* AieCode::GetKernel(const std::string& name) const {
  auto it = kernels_.find(name);
  return (it != kernels_.end()) ? &it->second : nullptr;
}

std::vector<std::string> AieCode::GetKernelNames() const {
  std::vector<std::string> names;
  names.reserve(kernels_.size());
  for (const auto& kv : kernels_) {
    names.push_back(kv.first);
  }
  return names;
}

}  // namespace AMD
}  // namespace rocr
