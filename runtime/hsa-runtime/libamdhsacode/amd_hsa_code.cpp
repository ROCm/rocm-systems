////////////////////////////////////////////////////////////////////////////////
//
// The University of Illinois/NCSA
// Open Source License (NCSA)
//
// Copyright (c) 2014-2020, Advanced Micro Devices, Inc. All rights reserved.
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

#include <assert.h>
#include <cstring>
#include <iomanip>
#include <algorithm>
#include "core/inc/amd_hsa_code.hpp"
#include "amd_hsa_code_util.hpp"
#include <libelf.h>
#include "inc/amd_hsa_elf.h"
#include <fstream>
#include <sstream>
#include <cstdlib>
#include <algorithm>

#ifdef SP3_STATIC_LIB
#include "sp3.h"
#endif // SP3_STATIC_LIB

#ifndef _WIN32
#define _alloca alloca
#endif

namespace rocr {
namespace amd {
namespace hsa {
namespace code {

    using amd::elf::GetNoteString;

    bool Symbol::IsDeclaration() const
    {
      return elfsym->type() == STT_COMMON;
    }

    bool Symbol::IsDefinition() const
    {
      return !IsDeclaration();
    }

    bool Symbol::IsAgent() const
    {
      return elfsym->section()->flags() & SHF_AMDGPU_HSA_AGENT ? true : false;
    }

    hsa_symbol_linkage_t Symbol::Linkage() const
    {
      return elfsym->binding() == STB_GLOBAL ? HSA_SYMBOL_LINKAGE_PROGRAM : HSA_SYMBOL_LINKAGE_MODULE;
    }

    hsa_variable_allocation_t Symbol::Allocation() const
    {
      return IsAgent() ? HSA_VARIABLE_ALLOCATION_AGENT : HSA_VARIABLE_ALLOCATION_PROGRAM;
    }

    hsa_variable_segment_t Symbol::Segment() const
    {
      return elfsym->section()->flags() & SHF_AMDGPU_HSA_READONLY ? HSA_VARIABLE_SEGMENT_READONLY : HSA_VARIABLE_SEGMENT_GLOBAL;
    }

    uint64_t Symbol::Size() const
    {
      return elfsym->size();
    }

    uint32_t Symbol::Size32() const
    {
      assert(elfsym->size() < UINT32_MAX);
      return (uint32_t) Size();
    }

    uint32_t Symbol::Alignment() const
    {
      assert(elfsym->section()->addralign() < UINT32_MAX);
      return uint32_t(elfsym->section()->addralign());
    }

    bool Symbol::IsConst() const
    {
      return elfsym->section()->flags() & SHF_WRITE ? true : false;
    }

    hsa_status_t Symbol::GetInfo(hsa_code_symbol_info_t attribute, void *value)
    {
      if (!value) {
          return HSA_STATUS_ERROR_INVALID_ARGUMENT;
      }

      switch (attribute) {
        case HSA_CODE_SYMBOL_INFO_TYPE: {
          *((hsa_symbol_kind_t*)value) = Kind();
          break;
        }
        case HSA_CODE_SYMBOL_INFO_NAME_LENGTH: {
          *((uint32_t*)value) = GetSymbolName().size();
          break;
        }
        case HSA_CODE_SYMBOL_INFO_NAME: {
          std::string SymbolName = GetSymbolName();
          memset(value, 0x0, SymbolName.size());
          memcpy(value, SymbolName.c_str(), SymbolName.size());
          break;
        }
        case HSA_CODE_SYMBOL_INFO_MODULE_NAME_LENGTH: {
          *((uint32_t*)value) = GetModuleName().size();
          break;
        }
        case HSA_CODE_SYMBOL_INFO_MODULE_NAME: {
          std::string ModuleName = GetModuleName();
          memset(value, 0x0, ModuleName.size());
          memcpy(value, ModuleName.c_str(), ModuleName.size());
          break;
        }
        case HSA_CODE_SYMBOL_INFO_LINKAGE: {
          *((hsa_symbol_linkage_t*)value) = Linkage();
          break;
        }
        case HSA_CODE_SYMBOL_INFO_IS_DEFINITION: {
          *((bool*)value) = IsDefinition();
          break;
        }
        default: {
          return HSA_STATUS_ERROR_INVALID_ARGUMENT;
        }
      }
      return HSA_STATUS_SUCCESS;
    }

    std::string Symbol::GetModuleName() const {
      std::string FullName = Name();
      return FullName.rfind(":") != std::string::npos ?
        FullName.substr(0, FullName.find(":")) : "";
    }

    std::string Symbol::GetSymbolName() const {
      std::string FullName = Name();
      return FullName.rfind(":") != std::string::npos ?
        std::move(FullName.substr(FullName.rfind(":") + 1)) : std::move(FullName);
    }

    hsa_code_symbol_t Symbol::ToHandle(Symbol* sym)
    {
      hsa_code_symbol_t s;
      s.handle = reinterpret_cast<uint64_t>(sym);
      return s;
    }

    Symbol* Symbol::FromHandle(hsa_code_symbol_t s)
    {
      return reinterpret_cast<Symbol*>(s.handle);
    }

    KernelSymbol::KernelSymbol(amd::elf::Symbol* elfsym_, const amd_kernel_code_t* akc)
        : Symbol(elfsym_)
        , kernarg_segment_size(0)
        , kernarg_segment_alignment(0)
        , group_segment_size(0)
        , private_segment_size(0)
        , is_dynamic_callstack(0)
    {
      if (akc) {
        kernarg_segment_size = (uint32_t) akc->kernarg_segment_byte_size;
        kernarg_segment_alignment = (uint32_t) (1 << akc->kernarg_segment_alignment);
        group_segment_size = uint32_t(akc->workgroup_group_segment_byte_size);
        private_segment_size = uint32_t(akc->workitem_private_segment_byte_size);
        is_dynamic_callstack =
          AMD_HSA_BITS_GET(akc->kernel_code_properties, AMD_KERNEL_CODE_PROPERTIES_IS_DYNAMIC_CALLSTACK) ? true : false;
      }
    }

    hsa_status_t KernelSymbol::GetInfo(hsa_code_symbol_info_t attribute, void *value)
    {
      assert(value);
      switch (attribute) {
        case HSA_CODE_SYMBOL_INFO_KERNEL_KERNARG_SEGMENT_SIZE: {
          *((uint32_t*)value) = kernarg_segment_size;
          break;
        }
        case HSA_CODE_SYMBOL_INFO_KERNEL_KERNARG_SEGMENT_ALIGNMENT: {
          *((uint32_t*)value) = kernarg_segment_alignment;
          break;
        }
        case HSA_CODE_SYMBOL_INFO_KERNEL_GROUP_SEGMENT_SIZE: {
          *((uint32_t*)value) = group_segment_size;
          break;
        }
        case HSA_CODE_SYMBOL_INFO_KERNEL_PRIVATE_SEGMENT_SIZE: {
          *((uint32_t*)value) = private_segment_size;
          break;
        }
        case HSA_CODE_SYMBOL_INFO_KERNEL_DYNAMIC_CALLSTACK: {
          *((bool*)value) = is_dynamic_callstack;
          break;
        }
        default: {
          return Symbol::GetInfo(attribute, value);
        }
      }
      return HSA_STATUS_SUCCESS;
    }

    hsa_status_t VariableSymbol::GetInfo(hsa_code_symbol_info_t attribute, void *value)
    {
      assert(value);
      switch (attribute) {
        case HSA_CODE_SYMBOL_INFO_VARIABLE_ALLOCATION: {
          *((hsa_variable_allocation_t*)value) = Allocation();
          break;
        }
        case HSA_CODE_SYMBOL_INFO_VARIABLE_SEGMENT: {
          *((hsa_variable_segment_t*)value) = Segment();
          break;
        }
        case HSA_CODE_SYMBOL_INFO_VARIABLE_ALIGNMENT: {
          *((uint32_t*)value) = Alignment();
          break;
        }
        case HSA_CODE_SYMBOL_INFO_VARIABLE_SIZE: {
          *((uint32_t*)value) = Size();
          break;
        }
        case HSA_CODE_SYMBOL_INFO_VARIABLE_IS_CONST: {
          *((bool*)value) = IsConst();
          break;
        }
        default: {
          return Symbol::GetInfo(attribute, value);
        }
      }
      return HSA_STATUS_SUCCESS;
    }

    AmdHsaCode::AmdHsaCode(bool combineDataSegments_)
      : img(nullptr),
        combineDataSegments(combineDataSegments_),
        hsatext(0), imageInit(0), samplerInit(0),
        debugInfo(0), debugLine(0), debugAbbrev(0)
    {
      for (unsigned i = 0; i < AMDGPU_HSA_SEGMENT_LAST; ++i) {
        for (unsigned j = 0; j < 2; ++j) {
          hsaSegments[i][j] = 0;
        }
      }
      for (unsigned i = 0; i < AMDGPU_HSA_SECTION_LAST; ++i) {
        hsaSections[i] = 0;
      }
    }

    AmdHsaCode::~AmdHsaCode()
    {
      for (Symbol* sym : symbols) { delete sym; }
    }

    bool AmdHsaCode::PullElf()
    {
      uint32_t majorVersion, minorVersion;
      if (!GetCodeObjectVersion(&majorVersion, &minorVersion)) {
        return false;
      }
      if (majorVersion >= 2) {
        return PullElfV2();
      } else {
        return PullElfV1();
      }
    }

    bool AmdHsaCode::PullElfV1()
    {
      for (size_t i = 0; i < img->segmentCount(); ++i) {
        Segment* s = img->segment(i);
        if (s->type() == PT_AMDGPU_HSA_LOAD_GLOBAL_PROGRAM ||
            s->type() == PT_AMDGPU_HSA_LOAD_GLOBAL_AGENT ||
            s->type() == PT_AMDGPU_HSA_LOAD_READONLY_AGENT ||
            s->type() == PT_AMDGPU_HSA_LOAD_CODE_AGENT) {
          dataSegments.push_back(s);
        }
      }
      for (size_t i = 0; i < img->sectionCount(); ++i) {
        Section* sec = img->section(i);
        if (!sec) { continue; }
        if ((sec->type() == SHT_PROGBITS || sec->type() == SHT_NOBITS) &&
            (sec->flags() & (SHF_AMDGPU_HSA_AGENT | SHF_AMDGPU_HSA_GLOBAL | SHF_AMDGPU_HSA_READONLY | SHF_AMDGPU_HSA_CODE))) {
          dataSections.push_back(sec);
        } else if (sec->type() == SHT_RELA) {
          relocationSections.push_back(sec->asRelocationSection());
        }
        if (sec->Name() == ".hsatext") {
          hsatext = sec;
        }
      }
      for (size_t i = 0; i < img->symtab()->symbolCount(); ++i) {
        amd::elf::Symbol* elfsym = img->symtab()->symbol(i);
        Symbol* sym = 0;
        switch (elfsym->type()) {
        case STT_AMDGPU_HSA_KERNEL: {
          amd::elf::Section* sec = elfsym->section();
          amd_kernel_code_t akc;
          if (!sec) {
            out << "Failed to find section for symbol " << elfsym->name() << std::endl;
            return false;
          }
          if (!(sec->flags() & (SHF_AMDGPU_HSA_AGENT | SHF_AMDGPU_HSA_CODE | SHF_EXECINSTR))) {
            out << "Invalid code section for symbol " << elfsym->name() << std::endl;
            return false;
          }
          if (!sec->getData(elfsym->value(), &akc, sizeof(amd_kernel_code_t))) {
            out << "Failed to get AMD Kernel Code for symbol " << elfsym->name() << std::endl;
            return false;
          }
          sym = new KernelSymbol(elfsym, &akc);
          break;
        }
        case STT_OBJECT:
        case STT_COMMON:
          sym = new VariableSymbol(elfsym);
          break;
        default:
          break; // Skip unknown symbols.
        }
        if (sym) { symbols.push_back(sym); }
      }

      return true;
    }

    bool AmdHsaCode::LoadFromFile(const std::string& filename)
    {
      if (!img) { img.reset(amd::elf::NewElf64Image()); }
      if (!img->loadFromFile(filename)) { return ElfImageError(); }
      if (!PullElf()) { return ElfImageError(); }
      return true;
    }

    bool AmdHsaCode::SaveToFile(const std::string& filename)
    {
      return img->saveToFile(filename) || ElfImageError();
    }

    bool AmdHsaCode::WriteToBuffer(void* buffer)
    {
      return img->copyToBuffer(buffer, ElfSize()) || ElfImageError();
    }


    bool AmdHsaCode::InitFromBuffer(const void* buffer, size_t size)
    {
      if (!img) { img.reset(amd::elf::NewElf64Image()); }
      if (!img->initFromBuffer(buffer, size)) { return ElfImageError(); }
      if (!PullElf()) { return ElfImageError(); }
      return true;
    }

    bool AmdHsaCode::InitAsBuffer(const void* buffer, size_t size)
    {
      if (!img) { img.reset(amd::elf::NewElf64Image()); }
      if (!img->initAsBuffer(buffer, size)) { return ElfImageError(); }
      if (!PullElf()) { return ElfImageError(); }
      return true;
    }

    bool AmdHsaCode::InitAsHandle(hsa_code_object_t code_object)
    {
      void *elfmemrd = reinterpret_cast<void*>(code_object.handle);
      if (!elfmemrd) { return false; }
      return InitAsBuffer(elfmemrd, 0);
    }

    bool AmdHsaCode::InitNew(bool xnack)
    {
      if (!img) {
        img.reset(amd::elf::NewElf64Image());
        uint32_t flags = 0;
        if (xnack) { flags |= ELF::EF_AMDGPU_FEATURE_XNACK_V2; }
        return img->initNew(ELF::EM_AMDGPU, ET_EXEC, ELF::ELFOSABI_AMDGPU_HSA, ELF::ELFABIVERSION_AMDGPU_HSA_V2, flags) ||
          ElfImageError(); // FIXME: elfutils libelf does not allow program headers in ET_REL file type, so change it later in finalizer.
      }
      return false;
    }

    bool AmdHsaCode::Freeze()
    {
      return img->Freeze() || ElfImageError();
    }

    hsa_code_object_t AmdHsaCode::GetHandle()
    {
      hsa_code_object_t code_object;
      code_object.handle = reinterpret_cast<uint64_t>(img->data());
      return code_object;
    }

    const char* AmdHsaCode::ElfData()
    {
      return img->data();
    }

    uint64_t AmdHsaCode::ElfSize()
    {
      return img->size();
    }

    bool AmdHsaCode::Validate()
    {
      if (!img->Validate()) { return ElfImageError(); }
      if (img->Machine() != ELF::EM_AMDGPU) {
        out << "ELF error: Invalid machine" << std::endl;
        return false;
      }
      return true;
    }

    void AmdHsaCode::AddAmdNote(uint32_t type, const void* desc, uint32_t desc_size)
    {
      img->note()->addNote("AMD", type, desc, desc_size);
    }

    void AmdHsaCode::AddNoteCodeObjectVersion(uint32_t major, uint32_t minor)
    {
      amdgpu_hsa_note_code_object_version_t desc;
      desc.major_version = major;
      desc.minor_version = minor;
      AddAmdNote(NT_AMD_HSA_CODE_OBJECT_VERSION, &desc, sizeof(desc));
    }

    bool AmdHsaCode::GetCodeObjectVersion(uint32_t* major, uint32_t* minor)
    {
      switch (img->ABIVersion()) {
      case ELF::ELFABIVERSION_AMDGPU_HSA_V2:
        amdgpu_hsa_note_code_object_version_t* desc;
        if (GetAmdNote(NT_AMD_HSA_CODE_OBJECT_VERSION, &desc)) {
          *major = desc->major_version;
          *minor = desc->minor_version;
          return *major <= 2;
        }
        return false;
      case ELF::ELFABIVERSION_AMDGPU_HSA_V3:
        *major = 3;
        *minor = 0;
        return true;
      case ELF::ELFABIVERSION_AMDGPU_HSA_V4:
        *major = 4;
        *minor = 0;
        return true;
      case ELF::ELFABIVERSION_AMDGPU_HSA_V5:
        *major = 5;
        *minor = 0;
        return true;
      case ELF::ELFABIVERSION_AMDGPU_HSA_V6:
        *major = 6;
        *minor = 0;
        return true;
      }

      return false;
    }

    bool AmdHsaCode::GetNoteCodeObjectVersion(std::string& version)
    {
      amdgpu_hsa_note_code_object_version_t* desc;
      if (!GetAmdNote(NT_AMD_HSA_CODE_OBJECT_VERSION, &desc)) { return false; }
      version.clear();
      version += std::to_string(desc->major_version);
      version += ".";
      version += std::to_string(desc->minor_version);
      return true;
    }

    void AmdHsaCode::AddNoteHsail(uint32_t hsail_major, uint32_t hsail_minor, hsa_profile_t profile, hsa_machine_model_t machine_model, hsa_default_float_rounding_mode_t rounding_mode)
    {
      amdgpu_hsa_note_hsail_t desc;
      memset(&desc, 0, sizeof(desc));
      desc.hsail_major_version = hsail_major;
      desc.hsail_minor_version = hsail_minor;
      desc.profile = uint8_t(profile);
      desc.machine_model = uint8_t(machine_model);
      desc.default_float_round = uint8_t(rounding_mode);
      AddAmdNote(NT_AMD_HSA_HSAIL, &desc, sizeof(desc));
    }

    bool AmdHsaCode::GetNoteHsail(uint32_t* hsail_major, uint32_t* hsail_minor, hsa_profile_t* profile, hsa_machine_model_t* machine_model, hsa_default_float_rounding_mode_t* default_float_round)
    {
      amdgpu_hsa_note_hsail_t *desc;
      if (!GetAmdNote(NT_AMD_HSA_HSAIL, &desc)) { return false; }
      *hsail_major = desc->hsail_major_version;
      *hsail_minor = desc->hsail_minor_version;
      *profile = (hsa_profile_t) desc->profile;
      *machine_model = (hsa_machine_model_t) desc->machine_model;
      *default_float_round = (hsa_default_float_rounding_mode_t) desc->default_float_round;
      return true;
    }

    void AmdHsaCode::AddNoteIsa(const std::string& vendor_name, const std::string& architecture_name, uint32_t major, uint32_t minor, uint32_t stepping)
    {
      size_t size = sizeof(amdgpu_hsa_note_producer_t) + vendor_name.length() + architecture_name.length() + 1;
      amdgpu_hsa_note_isa_t* desc = (amdgpu_hsa_note_isa_t*) _alloca(size);
      memset(desc, 0, size);
      desc->vendor_name_size = vendor_name.length()+1;
      desc->architecture_name_size = architecture_name.length()+1;
      desc->major = major;
      desc->minor = minor;
      desc->stepping = stepping;
      memcpy(desc->vendor_and_architecture_name, vendor_name.c_str(), vendor_name.length() + 1);
      memcpy(desc->vendor_and_architecture_name + desc->vendor_name_size, architecture_name.c_str(), architecture_name.length() + 1);
      AddAmdNote(NT_AMD_HSA_ISA_VERSION, desc, size);
    }

    bool AmdHsaCode::GetNoteIsa(std::string& vendor_name, std::string& architecture_name, uint32_t* major_version, uint32_t* minor_version, uint32_t* stepping)
    {
      amdgpu_hsa_note_isa_t *desc;
      if (!GetAmdNote(NT_AMD_HSA_ISA_VERSION, &desc)) { return false; }
      vendor_name = GetNoteString(desc->vendor_name_size, desc->vendor_and_architecture_name);
      architecture_name = GetNoteString(desc->architecture_name_size, desc->vendor_and_architecture_name + vendor_name.length() + 1);
      *major_version = desc->major;
      *minor_version = desc->minor;
      *stepping = desc->stepping;
      return true;
    }

    struct MachInfo {
      std::string Name = "";
      bool XnackSupported = false;
      bool SrameccSupported = false;
    };

    // TODO: Move isa registry into the loader.
    static bool GetMachInfo(unsigned Mach, MachInfo &MI) {
      switch (Mach) {
      case ELF::EF_AMDGPU_MACH_AMDGCN_GFX600:  MI.Name = "gfx600";  MI.XnackSupported = false; MI.SrameccSupported = false; break;
      case ELF::EF_AMDGPU_MACH_AMDGCN_GFX601:  MI.Name = "gfx601";  MI.XnackSupported = false; MI.SrameccSupported = false; break;
      case ELF::EF_AMDGPU_MACH_AMDGCN_GFX602:  MI.Name = "gfx602";  MI.XnackSupported = false; MI.SrameccSupported = false; break;
      case ELF::EF_AMDGPU_MACH_AMDGCN_GFX700:  MI.Name = "gfx700";  MI.XnackSupported = false; MI.SrameccSupported = false; break;
      case ELF::EF_AMDGPU_MACH_AMDGCN_GFX701:  MI.Name = "gfx701";  MI.XnackSupported = false; MI.SrameccSupported = false; break;
      case ELF::EF_AMDGPU_MACH_AMDGCN_GFX702:  MI.Name = "gfx702";  MI.XnackSupported = false; MI.SrameccSupported = false; break;
      case ELF::EF_AMDGPU_MACH_AMDGCN_GFX703:  MI.Name = "gfx703";  MI.XnackSupported = false; MI.SrameccSupported = false; break;
      case ELF::EF_AMDGPU_MACH_AMDGCN_GFX704:  MI.Name = "gfx704";  MI.XnackSupported = false; MI.SrameccSupported = false; break;
      case ELF::EF_AMDGPU_MACH_AMDGCN_GFX705:  MI.Name = "gfx705";  MI.XnackSupported = false; MI.SrameccSupported = false; break;
      case ELF::EF_AMDGPU_MACH_AMDGCN_GFX801:  MI.Name = "gfx801";  MI.XnackSupported = true;  MI.SrameccSupported = false; break;
      case ELF::EF_AMDGPU_MACH_AMDGCN_GFX802:  MI.Name = "gfx802";  MI.XnackSupported = false; MI.SrameccSupported = false; break;
      case ELF::EF_AMDGPU_MACH_AMDGCN_GFX803:  MI.Name = "gfx803";  MI.XnackSupported = false; MI.SrameccSupported = false; break;
      case ELF::EF_AMDGPU_MACH_AMDGCN_GFX805:  MI.Name = "gfx805";  MI.XnackSupported = false; MI.SrameccSupported = false; break;
      case ELF::EF_AMDGPU_MACH_AMDGCN_GFX810:  MI.Name = "gfx810";  MI.XnackSupported = true;  MI.SrameccSupported = false; break;
      case ELF::EF_AMDGPU_MACH_AMDGCN_GFX900:  MI.Name = "gfx900";  MI.XnackSupported = true;  MI.SrameccSupported = false; break;
      case ELF::EF_AMDGPU_MACH_AMDGCN_GFX902:  MI.Name = "gfx902";  MI.XnackSupported = true;  MI.SrameccSupported = false; break;
      case ELF::EF_AMDGPU_MACH_AMDGCN_GFX904:  MI.Name = "gfx904";  MI.XnackSupported = true;  MI.SrameccSupported = false; break;
      case ELF::EF_AMDGPU_MACH_AMDGCN_GFX906:  MI.Name = "gfx906";  MI.XnackSupported = true;  MI.SrameccSupported = true;  break;
      case ELF::EF_AMDGPU_MACH_AMDGCN_GFX908:  MI.Name = "gfx908";  MI.XnackSupported = true;  MI.SrameccSupported = true;  break;
      case ELF::EF_AMDGPU_MACH_AMDGCN_GFX909:  MI.Name = "gfx909";  MI.XnackSupported = true;  MI.SrameccSupported = false; break;
      case ELF::EF_AMDGPU_MACH_AMDGCN_GFX90A:  MI.Name = "gfx90a";  MI.XnackSupported = true;  MI.SrameccSupported = true;  break;
      case ELF::EF_AMDGPU_MACH_AMDGCN_GFX90C:  MI.Name = "gfx90c";  MI.XnackSupported = true;  MI.SrameccSupported = false; break;
      case ELF::EF_AMDGPU_MACH_AMDGCN_GFX942:  MI.Name = "gfx942";  MI.XnackSupported = true;  MI.SrameccSupported = true;  break;
      case ELF::EF_AMDGPU_MACH_AMDGCN_GFX950:  MI.Name = "gfx950";  MI.XnackSupported = true;  MI.SrameccSupported = true;  break;
      case ELF::EF_AMDGPU_MACH_AMDGCN_GFX1010: MI.Name = "gfx1010"; MI.XnackSupported = true;  MI.SrameccSupported = false; break;
      case ELF::EF_AMDGPU_MACH_AMDGCN_GFX1011: MI.Name = "gfx1011"; MI.XnackSupported = true;  MI.SrameccSupported = false; break;
      case ELF::EF_AMDGPU_MACH_AMDGCN_GFX1012: MI.Name = "gfx1012"; MI.XnackSupported = true;  MI.SrameccSupported = false; break;
      case ELF::EF_AMDGPU_MACH_AMDGCN_GFX1013: MI.Name = "gfx1013"; MI.XnackSupported = true;  MI.SrameccSupported = false; break;
      case ELF::EF_AMDGPU_MACH_AMDGCN_GFX1030: MI.Name = "gfx1030"; MI.XnackSupported = false; MI.SrameccSupported = false; break;
      case ELF::EF_AMDGPU_MACH_AMDGCN_GFX1031: MI.Name = "gfx1031"; MI.XnackSupported = false; MI.SrameccSupported = false; break;
      case ELF::EF_AMDGPU_MACH_AMDGCN_GFX1032: MI.Name = "gfx1032"; MI.XnackSupported = false; MI.SrameccSupported = false; break;
      case ELF::EF_AMDGPU_MACH_AMDGCN_GFX1033: MI.Name = "gfx1033"; MI.XnackSupported = false; MI.SrameccSupported = false; break;
      case ELF::EF_AMDGPU_MACH_AMDGCN_GFX1034: MI.Name = "gfx1034"; MI.XnackSupported = false; MI.SrameccSupported = false; break;
      case ELF::EF_AMDGPU_MACH_AMDGCN_GFX1035: MI.Name = "gfx1035"; MI.XnackSupported = false; MI.SrameccSupported = false; break;
      case ELF::EF_AMDGPU_MACH_AMDGCN_GFX1036: MI.Name = "gfx1036"; MI.XnackSupported = false; MI.SrameccSupported = false; break;
      case ELF::EF_AMDGPU_MACH_AMDGCN_GFX1100: MI.Name = "gfx1100"; MI.XnackSupported = false; MI.SrameccSupported = false; break;
      case ELF::EF_AMDGPU_MACH_AMDGCN_GFX1101: MI.Name = "gfx1101"; MI.XnackSupported = false; MI.SrameccSupported = false; break;
      case ELF::EF_AMDGPU_MACH_AMDGCN_GFX1102: MI.Name = "gfx1102"; MI.XnackSupported = false; MI.SrameccSupported = false; break;
      case ELF::EF_AMDGPU_MACH_AMDGCN_GFX1103: MI.Name = "gfx1103"; MI.XnackSupported = false; MI.SrameccSupported = false; break;
      case ELF::EF_AMDGPU_MACH_AMDGCN_GFX1150: MI.Name = "gfx1150"; MI.XnackSupported = false; MI.SrameccSupported = false; break;
      case ELF::EF_AMDGPU_MACH_AMDGCN_GFX1151: MI.Name = "gfx1151"; MI.XnackSupported = false; MI.SrameccSupported = false; break;
      case ELF::EF_AMDGPU_MACH_AMDGCN_GFX1152: MI.Name = "gfx1152"; MI.XnackSupported = false; MI.SrameccSupported = false; break;
      case ELF::EF_AMDGPU_MACH_AMDGCN_GFX1153: MI.Name = "gfx1153"; MI.XnackSupported = false; MI.SrameccSupported = false; break;
      case ELF::EF_AMDGPU_MACH_AMDGCN_GFX1200: MI.Name = "gfx1200"; MI.XnackSupported = false; MI.SrameccSupported = false; break;
      case ELF::EF_AMDGPU_MACH_AMDGCN_GFX1201: MI.Name = "gfx1201"; MI.XnackSupported = false; MI.SrameccSupported = false; break;
      case ELF::EF_AMDGPU_MACH_AMDGCN_GFX9_GENERIC:    MI.Name = "gfx9-generic";    MI.XnackSupported = true; MI.SrameccSupported = false; break;
      case ELF::EF_AMDGPU_MACH_AMDGCN_GFX9_4_GENERIC:  MI.Name = "gfx9-4-generic";  MI.XnackSupported = true;  MI.SrameccSupported = true; break;
      case ELF::EF_AMDGPU_MACH_AMDGCN_GFX10_1_GENERIC: MI.Name = "gfx10-1-generic"; MI.XnackSupported = true; MI.SrameccSupported = false; break;
      case ELF::EF_AMDGPU_MACH_AMDGCN_GFX10_3_GENERIC: MI.Name = "gfx10-3-generic"; MI.XnackSupported = false; MI.SrameccSupported = false; break;
      case ELF::EF_AMDGPU_MACH_AMDGCN_GFX11_GENERIC:   MI.Name = "gfx11-generic";   MI.XnackSupported = false; MI.SrameccSupported = false; break;
      case ELF::EF_AMDGPU_MACH_AMDGCN_GFX12_GENERIC:   MI.Name = "gfx12-generic";   MI.XnackSupported = false; MI.SrameccSupported = false; break;
      default: return false;
      }
      return true;
    }

    // This fuction is also copied to the Code Object Manager library.
    static std::string ConvertOldTargetNameToNew(const std::string &old_name, bool is_finalizer, uint32_t e_flags) {
      assert(!old_name.empty() && "Expecting non-empty old name");

      unsigned mach = 0;
      if (old_name == "AMD:AMDGPU:6:0:0")
        mach = ELF::EF_AMDGPU_MACH_AMDGCN_GFX600;
      else if (old_name == "AMD:AMDGPU:6:0:1")
        mach = ELF::EF_AMDGPU_MACH_AMDGCN_GFX601;
      else if (old_name == "AMD:AMDGPU:6:0:2")
        mach = ELF::EF_AMDGPU_MACH_AMDGCN_GFX602;
      else if (old_name == "AMD:AMDGPU:7:0:0")
        mach = ELF::EF_AMDGPU_MACH_AMDGCN_GFX700;
      else if (old_name == "AMD:AMDGPU:7:0:1")
        mach = ELF::EF_AMDGPU_MACH_AMDGCN_GFX701;
      else if (old_name == "AMD:AMDGPU:7:0:2")
        mach = ELF::EF_AMDGPU_MACH_AMDGCN_GFX702;
      else if (old_name == "AMD:AMDGPU:7:0:3")
        mach = ELF::EF_AMDGPU_MACH_AMDGCN_GFX703;
      else if (old_name == "AMD:AMDGPU:7:0:4")
        mach = ELF::EF_AMDGPU_MACH_AMDGCN_GFX704;
      else if (old_name == "AMD:AMDGPU:7:0:5")
        mach = ELF::EF_AMDGPU_MACH_AMDGCN_GFX705;
      else if (old_name == "AMD:AMDGPU:8:0:1")
        mach = ELF::EF_AMDGPU_MACH_AMDGCN_GFX801;
      else if (old_name == "AMD:AMDGPU:8:0:0" || old_name == "AMD:AMDGPU:8:0:2")
        mach = ELF::EF_AMDGPU_MACH_AMDGCN_GFX802;
      else if (old_name == "AMD:AMDGPU:8:0:3" || old_name == "AMD:AMDGPU:8:0:4")
        mach = ELF::EF_AMDGPU_MACH_AMDGCN_GFX803;
      else if (old_name == "AMD:AMDGPU:8:0:5")
        mach = ELF::EF_AMDGPU_MACH_AMDGCN_GFX805;
      else if (old_name == "AMD:AMDGPU:8:1:0")
        mach = ELF::EF_AMDGPU_MACH_AMDGCN_GFX810;
      else if (old_name == "AMD:AMDGPU:9:0:0" || old_name == "AMD:AMDGPU:9:0:1")
        mach = ELF::EF_AMDGPU_MACH_AMDGCN_GFX900;
      else if (old_name == "AMD:AMDGPU:9:0:2" || old_name == "AMD:AMDGPU:9:0:3")
        mach = ELF::EF_AMDGPU_MACH_AMDGCN_GFX902;
      else if (old_name == "AMD:AMDGPU:9:0:4" || old_name == "AMD:AMDGPU:9:0:5")
        mach = ELF::EF_AMDGPU_MACH_AMDGCN_GFX904;
      else if (old_name == "AMD:AMDGPU:9:0:6" || old_name == "AMD:AMDGPU:9:0:7")
        mach = ELF::EF_AMDGPU_MACH_AMDGCN_GFX906;
      else if (old_name == "AMD:AMDGPU:9:0:12")
        mach = ELF::EF_AMDGPU_MACH_AMDGCN_GFX90C;
      else {
        // Code object v2 only supports asics up to gfx906 plus gfx90c. Do NOT
        // add handling of new asics into this if-else-if* block.
        return "";
      }
      MachInfo MI;
      if (!GetMachInfo(mach, MI))
        return "";

      // Only "AMD:AMDGPU:9:0:6" and "AMD:AMDGPU:9:0:7" supports SRAMECC for
      // code object V2, and it must be OFF.
      if (MI.SrameccSupported)
        MI.Name += ":sramecc-";

      if (is_finalizer) {
        if (e_flags & ELF::EF_AMDGPU_FEATURE_XNACK_V2)
          MI.Name += ":xnack+";
        else if (MI.XnackSupported)
          MI.Name += ":xnack-";
      } else {
        if (old_name == "AMD:AMDGPU:8:0:1")
          MI.Name += ":xnack+";
        else if (old_name == "AMD:AMDGPU:8:1:0")
          MI.Name += ":xnack+";
        else if (old_name == "AMD:AMDGPU:9:0:1")
          MI.Name += ":xnack+";
        else if (old_name == "AMD:AMDGPU:9:0:3")
          MI.Name += ":xnack+";
        else if (old_name == "AMD:AMDGPU:9:0:5")
          MI.Name += ":xnack+";
        else if (old_name == "AMD:AMDGPU:9:0:7")
          MI.Name += ":xnack+";
        else if (MI.XnackSupported)
          MI.Name += ":xnack-";
      }

      return MI.Name;
    }

    bool AmdHsaCode::GetIsa(std::string& isa_name, unsigned *genericVersion)
    {
      isa_name.clear();

      uint32_t code_object_major_version = 0;
      uint32_t code_object_minor_version = 0;

      // Generic versioning starts at 1, so zero means no generic version.
      if (genericVersion)
        *genericVersion = 0;

      switch (img->EClass()) {
      case ELFCLASS64:
        // There is no e_machine and/or OS ABI for R600 so rely on checking
        // the ELFCLASS to determine if AMDGCN versus R600. AMDHSA always uses
        // ELFCLASS64 and R600 always uses ELFCLASS32.
        isa_name += "amdgcn";
        break;
      default:
        return false;
      }
      if (img->Machine() != ELF::EM_AMDGPU)
        return false;
      isa_name += "-amd-";

      if (!GetCodeObjectVersion(&code_object_major_version, &code_object_minor_version)) {
        return false;
      }
      if (code_object_major_version >= 3) {

        switch (img->OsAbi()) {
        case ELF::ELFOSABI_AMDGPU_HSA:
          isa_name += "amdhsa";
          break;
        default:
          // Only support AMDHSA in the ROCm runtime.
          return false;
        }

        isa_name += "--";

        unsigned mach = img->EFlags() & ELF::EF_AMDGPU_MACH;
        MachInfo MI;

        if (!GetMachInfo(mach, MI))
          return false;

        if (code_object_major_version == 3) {
          if (img->EFlags() & ELF::EF_AMDGPU_FEATURE_SRAMECC_V3)
            MI.Name += ":sramecc+";
          else if (MI.SrameccSupported)
            MI.Name += ":sramecc-";

          if (img->EFlags() & ELF::EF_AMDGPU_FEATURE_XNACK_V3)
            MI.Name += ":xnack+";
          else if (MI.XnackSupported)
            MI.Name += ":xnack-";
        } else if (code_object_major_version >= 4) {
          switch (img->EFlags() & ELF::EF_AMDGPU_FEATURE_SRAMECC_V4) {
          case ELF::EF_AMDGPU_FEATURE_SRAMECC_OFF_V4:
            MI.Name += ":sramecc-";
            break;
          case ELF::EF_AMDGPU_FEATURE_SRAMECC_ON_V4:
            MI.Name += ":sramecc+";
            break;
          }

          switch (img->EFlags() & ELF::EF_AMDGPU_FEATURE_XNACK_V4) {
          case ELF::EF_AMDGPU_FEATURE_XNACK_OFF_V4:
            MI.Name += ":xnack-";
            break;
          case ELF::EF_AMDGPU_FEATURE_XNACK_ON_V4:
            MI.Name += ":xnack+";
            break;
          }

          // Generic version is not part of the ISA name.
          // Only parse it when the caller wants it.
          if (genericVersion && code_object_major_version >= 6) {
            *genericVersion = (img->EFlags() & ELF::EF_AMDGPU_GENERIC_VERSION) >> ELF::EF_AMDGPU_GENERIC_VERSION_OFFSET;
          }
        } else {
          return false;
        }

        isa_name += MI.Name;

        return true;
      } else {

        std::string vendor_name, architecture_name;
        uint32_t major_version, minor_version, stepping;
        if (!GetNoteIsa(vendor_name, architecture_name, &major_version, &minor_version, &stepping))
          return false;

        isa_name += "amdhsa--";

        std::string target_name = vendor_name + ':' + architecture_name + ':' +
            std::to_string(major_version) + ':' + std::to_string(minor_version) + ':' +
            std::to_string(stepping);

        amdgpu_hsa_note_hsail_t *hsail_note;
        bool is_finalizer = GetAmdNote(NT_AMD_HSA_HSAIL, &hsail_note);
        target_name = ConvertOldTargetNameToNew(target_name, is_finalizer, img->EFlags());
        if (target_name.empty()) return false;

        isa_name += target_name;

        return true;
      }
    }

    void AmdHsaCode::AddNoteProducer(uint32_t major, uint32_t minor, const std::string& producer)
    {
      size_t size = sizeof(amdgpu_hsa_note_producer_t) + producer.length();
      amdgpu_hsa_note_producer_t* desc = (amdgpu_hsa_note_producer_t*) _alloca(size);
      memset(desc, 0, size);
      desc->producer_name_size = producer.length();
      desc->producer_major_version = major;
      desc->producer_minor_version = minor;
      memcpy(desc->producer_name, producer.c_str(), producer.length() + 1);
      AddAmdNote(NT_AMD_HSA_PRODUCER, desc, size);
    }

    bool AmdHsaCode::GetNoteProducer(uint32_t* major, uint32_t* minor, std::string& producer_name)
    {
      amdgpu_hsa_note_producer_t* desc;
      if (!GetAmdNote(NT_AMD_HSA_PRODUCER, &desc)) { return false; }
      *major = desc->producer_major_version;
      *minor = desc->producer_minor_version;
      producer_name = GetNoteString(desc->producer_name_size, desc->producer_name);
      return true;
    }

    void AmdHsaCode::AddNoteProducerOptions(const std::string& options)
    {
      size_t size = sizeof(amdgpu_hsa_note_producer_options_t) + options.length();
      amdgpu_hsa_note_producer_options_t *desc = (amdgpu_hsa_note_producer_options_t*) _alloca(size);
      desc->producer_options_size = options.length();
      memcpy(desc->producer_options, options.c_str(), options.length() + 1);
      AddAmdNote(NT_AMD_HSA_PRODUCER_OPTIONS, desc, size);
    }

    void AmdHsaCode::AddNoteProducerOptions(int32_t call_convention, const hsa_ext_control_directives_t& user_directives, const std::string& user_options)
    {
      using namespace code_options;
      std::ostringstream ss;
      ss <<
        space << "-hsa_call_convention=" << call_convention <<
        control_directives(user_directives);
      if (!user_options.empty()) {
        ss << space << user_options;
      }

      AddNoteProducerOptions(ss.str());
    }

    bool AmdHsaCode::GetNoteProducerOptions(std::string& options)
    {
      amdgpu_hsa_note_producer_options_t* desc;
      if (!GetAmdNote(NT_AMD_HSA_PRODUCER_OPTIONS, &desc)) { return false; }
      options = GetNoteString(desc->producer_options_size, desc->producer_options);
      return true;
    }

    hsa_status_t AmdHsaCode::GetInfo(hsa_code_object_info_t attribute, void *value)
    {
      assert(value);
      switch (attribute) {
      case HSA_CODE_OBJECT_INFO_VERSION: {
        std::string version;
        if (!GetNoteCodeObjectVersion(version)) { return HSA_STATUS_ERROR_INVALID_CODE_OBJECT; }
        char *svalue = (char*)value;
        memset(svalue, 0x0, 64);
        memcpy(svalue, version.c_str(), (std::min)(size_t(63), version.length()));
        break;
      }
      case HSA_CODE_OBJECT_INFO_ISA: {
        // TODO: Currently returns string representation instead of hsa_isa_t
        // which is unavailable here.
        std::string isa;
        if (!GetIsa(isa)) { return HSA_STATUS_ERROR_INVALID_CODE_OBJECT; }
        char *svalue = (char*)value;
        memset(svalue, 0x0, 64);
        memcpy(svalue, isa.c_str(), (std::min)(size_t(63), isa.length()));
        break;
      }
      case HSA_CODE_OBJECT_INFO_MACHINE_MODEL:
      case HSA_CODE_OBJECT_INFO_PROFILE:
      case HSA_CODE_OBJECT_INFO_DEFAULT_FLOAT_ROUNDING_MODE: {
        uint32_t hsail_major, hsail_minor;
        hsa_profile_t profile;
        hsa_machine_model_t machine_model;
        hsa_default_float_rounding_mode_t default_float_round;
        if (!GetNoteHsail(&hsail_major, &hsail_minor, &profile, &machine_model, &default_float_round)) {
          return HSA_STATUS_ERROR_INVALID_CODE_OBJECT;
        }
        switch (attribute) {
        case HSA_CODE_OBJECT_INFO_MACHINE_MODEL:
           *((hsa_machine_model_t*)value) = machine_model; break;
        case HSA_CODE_OBJECT_INFO_PROFILE:
          *((hsa_profile_t*)value) = profile; break;
        case HSA_CODE_OBJECT_INFO_DEFAULT_FLOAT_ROUNDING_MODE:
          *((hsa_default_float_rounding_mode_t*)value) = default_float_round; break;
        default: break;
        }
        break;
      }
      default:
        assert(false);
        return HSA_STATUS_ERROR_INVALID_ARGUMENT;
      }
      return HSA_STATUS_SUCCESS;
    }

    hsa_status_t AmdHsaCode::GetSymbol(const char *module_name, const char *symbol_name, hsa_code_symbol_t *s)
    {
      std::string mname = MangleSymbolName(module_name ? module_name : "", symbol_name);
      for (Symbol* sym : symbols) {
        if (sym->Name() == mname) {
          *s = Symbol::ToHandle(sym);
          return HSA_STATUS_SUCCESS;
        }
      }
      return HSA_STATUS_ERROR_INVALID_SYMBOL_NAME;
    }

    hsa_status_t AmdHsaCode::IterateSymbols(hsa_code_object_t code_object,
                                  hsa_status_t (*callback)(
                                  hsa_code_object_t code_object,
                                  hsa_code_symbol_t symbol,
                                  void* data),
                                void* data)
    {
      for (Symbol* sym : symbols) {
        hsa_code_symbol_t s = Symbol::ToHandle(sym);
        hsa_status_t status = callback(code_object, s, data);
        if (status != HSA_STATUS_SUCCESS) { return status; }
      }
      return HSA_STATUS_SUCCESS;
    }

    Section* AmdHsaCode::ImageInitSection()
    {
      if (!imageInit) {
        imageInit = img->addSection(
          ".hsaimage_imageinit",
          SHT_PROGBITS,
          SHF_MERGE,
          sizeof(amdgpu_hsa_image_descriptor_t));
      }
      return imageInit;
    }

    void AmdHsaCode::AddImageInitializer(Symbol* image, uint64_t destOffset, const amdgpu_hsa_image_descriptor_t& desc)
    {
      uint64_t offset = ImageInitSection()->addData(&desc, sizeof(desc), 8);
      amd::elf::Symbol* imageInit =
        img->symtab()->addSymbol(ImageInitSection(), "", offset, 0, STT_AMDGPU_HSA_METADATA, STB_LOCAL);
      image->elfSym()->section()->relocationSection()->addRelocation(R_AMDGPU_V1_INIT_IMAGE, imageInit, image->elfSym()->value() + destOffset, 0);
    }

    void AmdHsaCode::AddImageInitializer(
      Symbol* image, uint64_t destOffset,
      amdgpu_hsa_metadata_kind16_t kind,
      amdgpu_hsa_image_geometry8_t geometry,
      amdgpu_hsa_image_channel_order8_t channel_order, amdgpu_hsa_image_channel_type8_t channel_type,
      uint64_t width, uint64_t height, uint64_t depth, uint64_t array)
    {
      amdgpu_hsa_image_descriptor_t desc;
      desc.size = (uint16_t) sizeof(amdgpu_hsa_image_descriptor_t);
      desc.kind = kind;
      desc.geometry = geometry;
      desc.channel_order = channel_order;
      desc.channel_type = channel_type;
      desc.width = width;
      desc.height = height;
      desc.depth = depth;
      desc.array = array;
      AddImageInitializer(image, destOffset, desc);
    }


    Section* AmdHsaCode::SamplerInitSection()
    {
      if (!samplerInit) {
        samplerInit = img->addSection(
          ".hsaimage_samplerinit",
          SHT_PROGBITS,
          SHF_MERGE,
          sizeof(amdgpu_hsa_sampler_descriptor_t));
      }
      return samplerInit;
    }

    void AmdHsaCode::AddSamplerInitializer(Symbol* sampler, uint64_t destOffset, const amdgpu_hsa_sampler_descriptor_t& desc)
    {
      uint64_t offset = SamplerInitSection()->addData(&desc, sizeof(desc), 8);
      amd::elf::Symbol* samplerInit =
        img->symtab()->addSymbol(SamplerInitSection(), "", offset, 0, STT_AMDGPU_HSA_METADATA, STB_LOCAL);
      sampler->elfSym()->section()->relocationSection()->addRelocation(R_AMDGPU_V1_INIT_SAMPLER, samplerInit, sampler->elfSym()->value() + destOffset, 0);
    }

    void AmdHsaCode::AddSamplerInitializer(Symbol* sampler, uint64_t destOffset,
        amdgpu_hsa_sampler_coord8_t coord,
        amdgpu_hsa_sampler_filter8_t filter,
        amdgpu_hsa_sampler_addressing8_t addressing)
    {
      amdgpu_hsa_sampler_descriptor_t desc;
      desc.size = (uint16_t) sizeof(amdgpu_hsa_sampler_descriptor_t);
      desc.kind = AMDGPU_HSA_METADATA_KIND_INIT_SAMP;
      desc.coord = coord;
      desc.filter = filter;
      desc.addressing = addressing;
      AddSamplerInitializer(sampler, destOffset, desc);
    }

    void AmdHsaCode::AddInitVarWithAddress(bool large, Symbol* dest, uint64_t destOffset, Symbol* addrOf, uint64_t addrAddend)
    {
      uint32_t rtype = large ? R_AMDGPU_V1_64 : R_AMDGPU_V1_32_LOW;
      dest->elfSym()->section()->relocationSection()->addRelocation(rtype, addrOf->elfSym(), dest->elfSym()->value() + destOffset, addrAddend);
    }

    uint64_t AmdHsaCode::NextKernelCodeOffset() const
    {
      return HsaText()->nextDataOffset(256);
    }

    bool AmdHsaCode::AddKernelCode(KernelSymbol* sym, const void* code, size_t size)
    {
      assert(nullptr != sym);

      uint64_t offset = HsaText()->addData(code, size, 256);
      sym->setValue(offset);
      sym->setSize(size);
      return true;
    }

    Section* AmdHsaCode::AddEmptySection()
    {
      dataSections.push_back(nullptr); return nullptr;
    }

    Section* AmdHsaCode::AddCodeSection(Segment* segment)
    {
      if (nullptr == img) { return nullptr; }
      Section *sec = img->addSection(
        ".hsatext",
        SHT_PROGBITS,
        SHF_ALLOC | SHF_EXECINSTR | SHF_WRITE | SHF_AMDGPU_HSA_CODE | SHF_AMDGPU_HSA_AGENT,
        0,
        segment);
      dataSections.push_back(sec);
      hsatext = sec;
      return sec;
    }

    Section* AmdHsaCode::AddDataSection(const std::string &name,
                                        uint32_t type,
                                        uint64_t flags,
                                        Segment* segment)
    {
      if (nullptr == img) { return nullptr; }
      Section *sec = img->addSection(name, type, flags, 0, segment);
      dataSections.push_back(sec);
      return sec;
    }

    void AmdHsaCode::InitHsaSectionSegment(amdgpu_hsa_elf_section_t section, bool combineSegments)
    {
      InitHsaSegment(AmdHsaElfSectionSegment(section), combineSegments || !IsAmdHsaElfSectionROData(section));
    }

    Section* AmdHsaCode::HsaDataSection(amdgpu_hsa_elf_section_t sec, bool combineSegments)
    {
      if (!hsaSections[sec]) {
        bool writable = combineSegments || !IsAmdHsaElfSectionROData(sec);
        Segment* segment = HsaSegment(AmdHsaElfSectionSegment(sec), writable);
        assert(segment); // Expected to be init the segment via InitHsaSegment.
        Section* section;
        switch (sec) {
        case AMDGPU_HSA_RODATA_GLOBAL_PROGRAM:
          section = AddDataSection(".hsarodata_global_program", SHT_PROGBITS, SHF_ALLOC | SHF_AMDGPU_HSA_GLOBAL, segment); break;
        case AMDGPU_HSA_RODATA_GLOBAL_AGENT:
          section = AddDataSection(".hsarodata_global_agent", SHT_PROGBITS, SHF_ALLOC | SHF_AMDGPU_HSA_GLOBAL | SHF_AMDGPU_HSA_AGENT, segment); break;
        case AMDGPU_HSA_RODATA_READONLY_AGENT:
          section = AddDataSection(".hsarodata_readonly_agent", SHT_PROGBITS, SHF_ALLOC | SHF_AMDGPU_HSA_READONLY | SHF_AMDGPU_HSA_AGENT, segment); break;
        case AMDGPU_HSA_DATA_GLOBAL_PROGRAM:
          section = AddDataSection(".hsadata_global_program", SHT_PROGBITS, SHF_ALLOC | SHF_WRITE | SHF_AMDGPU_HSA_GLOBAL, segment); break;
        case AMDGPU_HSA_DATA_GLOBAL_AGENT:
          section = AddDataSection(".hsadata_global_agent", SHT_PROGBITS, SHF_ALLOC | SHF_WRITE | SHF_AMDGPU_HSA_GLOBAL | SHF_AMDGPU_HSA_AGENT, segment); break;
        case AMDGPU_HSA_DATA_READONLY_AGENT:
          section = AddDataSection(".hsadata_readonly_agent", SHT_PROGBITS, SHF_ALLOC | SHF_WRITE | SHF_AMDGPU_HSA_READONLY | SHF_AMDGPU_HSA_AGENT, segment); break;
        case AMDGPU_HSA_BSS_GLOBAL_PROGRAM:
          section = AddDataSection(".hsabss_global_program", SHT_NOBITS, SHF_ALLOC | SHF_WRITE | SHF_AMDGPU_HSA_GLOBAL, segment); break;
        case AMDGPU_HSA_BSS_GLOBAL_AGENT:
          section = AddDataSection(".hsabss_global_agent", SHT_NOBITS, SHF_ALLOC | SHF_WRITE | SHF_AMDGPU_HSA_GLOBAL | SHF_AMDGPU_HSA_AGENT, segment); break;
        case AMDGPU_HSA_BSS_READONLY_AGENT:
          section = AddDataSection(".hsabss_readonly_agent", SHT_NOBITS, SHF_ALLOC | SHF_WRITE | SHF_AMDGPU_HSA_READONLY | SHF_AMDGPU_HSA_AGENT, segment); break;
        default:
          assert(false); return 0;
        }
        hsaSections[sec] = section;
      }
      return hsaSections[sec];
    }

    void AmdHsaCode::InitHsaSegment(amdgpu_hsa_elf_segment_t segment, bool writable)
    {
      if (!hsaSegments[segment][writable]) {
        uint32_t flags = PF_R;
        if (writable) { flags |= PF_W; }
        if (segment == AMDGPU_HSA_SEGMENT_CODE_AGENT) { flags |= PF_X; }
        uint32_t type = PT_LOOS + segment;
        assert(segment < AMDGPU_HSA_SEGMENT_LAST);
        hsaSegments[segment][writable] = img->initSegment(type, flags);
      }
    }

    bool AmdHsaCode::AddHsaSegments()
    {
      if (!img->addSegments()) { return ElfImageError(); }
      return true;
    }

    Segment* AmdHsaCode::HsaSegment(amdgpu_hsa_elf_segment_t segment, bool writable)
    {
      return hsaSegments[segment][writable];
    }

    Symbol* AmdHsaCode::AddExecutableSymbol(const std::string &name,
                                            unsigned char type,
                                            unsigned char binding,
                                            unsigned char other,
                                            Section *section)
    {
      if (nullptr == img) { return nullptr; }
      if (!section) { section = HsaText(); }
      symbols.push_back(new KernelSymbol(img->symtab()->addSymbol(section, name, 0, 0, type, binding, other), nullptr));
      return symbols.back();
    }

    Symbol* AmdHsaCode::AddVariableSymbol(const std::string &name,
                                          unsigned char type,
                                          unsigned char binding,
                                          unsigned char other,
                                          Section *section,
                                          uint64_t value,
                                          uint64_t size)
    {
      if (nullptr == img) { return nullptr; }
      symbols.push_back(new VariableSymbol(img->symtab()->addSymbol(section, name, value, size, type, binding, other)));
      return symbols.back();
    }

    void AmdHsaCode::AddSectionSymbols()
    {
      if (nullptr == img) { return; }
      for (size_t i = 0; i < dataSections.size(); ++i) {
        if (dataSections[i] && dataSections[i]->flags() & SHF_ALLOC) {
          symbols.push_back(new VariableSymbol(img->symtab()->addSymbol(dataSections[i], "__hsa_section" + dataSections[i]->Name(), 0, 0, STT_SECTION, STB_LOCAL)));
        }
      }
    }

    Symbol* AmdHsaCode::GetSymbolByElfIndex(size_t index)
    {
      for (auto &s : symbols) {
        if (s && index == s->Index()) {
          return s;
        }
      }
      return nullptr;
    }

    Symbol* AmdHsaCode::FindSymbol(const std::string &n)
    {
      for (auto &s : symbols) {
        if (s && n == s->Name()) {
          return s;
        }
      }
      return nullptr;
    }

    void AmdHsaCode::AddData(amdgpu_hsa_elf_section_t s, const void* data, size_t size)
    {
//      getDataSection(s)->addData(data, size);
    }

    Section* AmdHsaCode::DebugInfo()
    {
      if (!debugInfo) {
        debugInfo = img->addSection(".debug_info", SHT_PROGBITS);
      }
      return debugInfo;
    }

    Section* AmdHsaCode::DebugLine()
    {
      if (!debugLine) {
        debugLine = img->addSection(".debug_line", SHT_PROGBITS);
      }
      return debugLine;
    }

    Section* AmdHsaCode::DebugAbbrev()
    {
      if (!debugAbbrev) {
        debugAbbrev = img->addSection(".debug_abbrev", SHT_PROGBITS);
      }
      return debugAbbrev;
    }

    Section* AmdHsaCode::AddHsaHlDebug(const std::string& name, const void* data, size_t size)
    {
      Section* section = img->addSection(name, SHT_PROGBITS, SHF_OS_NONCONFORMING);
      section->addData(data, size, 1);
      return section;
    }

    bool AmdHsaCode::PrintToFile(const std::string& filename)
    {
      std::ofstream out(filename);
      if (out.fail()) { return false; }
      Print(out);
      return out.fail();
    }

    void AmdHsaCode::Print(std::ostream& out)
    {
      PrintNotes(out);
      out << std::endl;
      PrintSegments(out);
      out << std::endl;
      PrintSections(out);
      out << std::endl;
      PrintSymbols(out);
      out << std::endl;
      PrintMachineCode(out);
      out << std::endl;
      out << "AMD HSA Code Object End" << std::endl;
    }

    void AmdHsaCode::PrintNotes(std::ostream& out)
    {
      {
        uint32_t major_version, minor_version;
        if (GetCodeObjectVersion(&major_version, &minor_version)) {
          out << "AMD HSA Code Object" << std::endl
              << "  Version " << major_version << "." << minor_version << std::endl;
        }
      }
      {
        uint32_t hsail_major, hsail_minor;
        hsa_profile_t profile;
        hsa_machine_model_t machine_model;
        hsa_default_float_rounding_mode_t rounding_mode;
        if (GetNoteHsail(&hsail_major, &hsail_minor, &profile, &machine_model, &rounding_mode)) {
          out << "HSAIL " << std::endl
              << "  Version: " << hsail_major << "." << hsail_minor << std::endl
              << "  Profile: " << HsaProfileToString(profile)
              << "  Machine model: " << HsaMachineModelToString(machine_model)
              << "  Default float rounding: " << HsaFloatRoundingModeToString(rounding_mode) << std::endl;
        }
      }
      {
        std::string vendor_name, architecture_name;
        uint32_t major_version, minor_version, stepping;
        if (GetNoteIsa(vendor_name, architecture_name, &major_version, &minor_version, &stepping)) {
          out << "ISA" << std::endl
              << "  Vendor " << vendor_name
              << "  Arch " << architecture_name
              << "  Version " << major_version << ":" << minor_version << ":" << stepping << std::endl;
        }
      }
      {
        std::string producer_name, producer_options;
        uint32_t major, minor;
        if (GetNoteProducer(&major, &minor, producer_name)) {
          out << "Producer '" << producer_name << "' " << "Version " << major << ":" << minor << std::endl;
        }
      }
      {
        std::string producer_options;
        if (GetNoteProducerOptions(producer_options)) {
          out << "Producer options" << std::endl
              << "  '" << producer_options << "'" << std::endl;
        }
      }
    }

    void AmdHsaCode::PrintSegments(std::ostream& out)
    {
      out << "Segments (total " << DataSegmentCount() << "):" << std::endl;
      for (size_t i = 0; i < DataSegmentCount(); ++i) {
        PrintSegment(out, DataSegment(i));
      }
    }

    void AmdHsaCode::PrintSections(std::ostream& out)
    {
      out << "Data Sections (total " << DataSectionCount() << "):" << std::endl;
      for (size_t i = 0; i < DataSectionCount(); ++i) {
        PrintSection(out, DataSection(i));
      }
      out << std::endl;
      out << "Relocation Sections (total " << RelocationSectionCount() << "):" << std::endl;
      for (size_t i = 0; i < RelocationSectionCount(); ++i) {
        PrintSection(out, GetRelocationSection(i));
      }
    }

    void AmdHsaCode::PrintSymbols(std::ostream& out)
    {
      out << "Symbols (total " << SymbolCount() << "):" << std::endl;
      for (size_t i = 0; i < SymbolCount(); ++i) {
        PrintSymbol(out, GetSymbol(i));
      }
    }

    void AmdHsaCode::PrintMachineCode(std::ostream& out)
    {
      if (HasHsaText()) {
        out << std::dec;
        for (size_t i = 0; i < SymbolCount(); ++i) {
          Symbol* sym = GetSymbol(i);
          if (sym->IsKernelSymbol() && sym->IsDefinition()) {
            amd_kernel_code_t kernel_code;
            HsaText()->getData(sym->SectionOffset(), &kernel_code, sizeof(amd_kernel_code_t));
            out << "AMD Kernel Code for " << sym->Name() << ": " << std::endl << std::dec;
            PrintAmdKernelCode(out, &kernel_code);
            out << std::endl;
          }
        }

        std::vector<uint8_t> isa(HsaText()->size(), 0);
        HsaText()->getData(0, isa.data(), HsaText()->size());

        out << "Disassembly:" << std::endl;
        PrintDisassembly(out, isa.data(), HsaText()->size(), 0);
        out << std::endl << std::dec;
      } else {
        out << "Machine code section is not present" << std::endl << std::endl;
      }
    }

    void AmdHsaCode::PrintSegment(std::ostream& out, Segment* segment)
    {
      out << "  Segment (" << segment->getSegmentIndex() << ")" << std::endl;
      out << "    Type: " << AmdPTLoadToString(segment->type())
          << " "
          << "    Flags: " << "0x" << std::hex << std::setw(8) << std::setfill('0') << segment->flags() << std::dec
          << std::endl
          << "    Image Size: " << segment->imageSize()
          << " "
          << "    Memory Size: " << segment->memSize()
          << " "
          << "    Align: " << segment->align()
          << " "
          << "    VAddr: " << segment->vaddr()
          << std::endl;
      out << std::dec;
    }

    void AmdHsaCode::PrintSection(std::ostream& out, Section* section)
    {
      out << "  Section " << section->Name() << " (Index " << section->getSectionIndex() << ")" << std::endl;
      out << "    Type: " << section->type()
          << " "
          << "    Flags: " << "0x" << std::hex << std::setw(8) << std::setfill('0') << section->flags() << std::dec
          << std::endl
          << "    Size:  " << section->size()
          << " "
          << "    Address: " << section->addr()
          << " "
          << "    Align: " << section->addralign()
          << std::endl;
      out << std::dec;

      if (section->flags() & SHF_AMDGPU_HSA_CODE) {
        // Printed separately.
        return;
      }

      switch (section->type()) {
      case SHT_NOBITS:
        return;
      case SHT_RELA:
        PrintRelocationData(out, section->asRelocationSection());
        return;
      default:
        PrintRawData(out, section);
      }
    }

    void AmdHsaCode::PrintRawData(std::ostream& out, Section* section)
    {
      out << "    Data:" << std::endl;
      unsigned char *sdata = (unsigned char*)alloca(section->size());
      section->getData(0, sdata, section->size());
      PrintRawData(out, sdata, section->size());
    }

    void AmdHsaCode::PrintRawData(std::ostream& out, const unsigned char *data, size_t size)
    {
      out << std::hex << std::setfill('0');
      for (size_t i = 0; i < size; i += 16) {
        out << "      " << std::setw(7) << i << ":";

        for (size_t j = 0; j < 16; j += 1) {
          uint32_t value = i + j < size ? (uint32_t)data[i + j] : 0;
          if (j % 2 == 0) { out << ' '; }
          out << std::setw(2) << value;
        }
        out << "  ";

        for (size_t j = 0; i + j < size && j < 16; j += 1) {
          char value = (char)data[i + j] >= 32 && (char)data[i + j] <= 126 ? (char)data[i + j] : '.';
          out << value;
        }
        out << std::endl;
      }
      out << std::dec;
    }

    void AmdHsaCode::PrintRelocationData(std::ostream& out, RelocationSection* section)
    {
      if (section->targetSection()) {
        out << "    Relocation Entries for " << section->targetSection()->Name() << " Section (total " << section->relocationCount() << "):" << std::endl;
      } else {
        // Dynamic relocations do not have a target section, they work with
        // virtual addresses.
        out << "    Dynamic Relocation Entries (total " << section->relocationCount() << "):" << std::endl;
      }
      for (size_t i = 0; i < section->relocationCount(); ++i) {
        out << "      Relocation (Index " << i << "):" << std::endl;
        out << "        Type: " << section->relocation(i)->type() << std::endl;
        out << "        Symbol: " << section->relocation(i)->symbol()->name() << std::endl;
        out << "        Offset: " << section->relocation(i)->offset() << " Addend: " << section->relocation(i)->addend() << std::endl;
      }
      out << std::dec;
    }

    void AmdHsaCode::PrintSymbol(std::ostream& out, Symbol* sym)
    {
      out << "  Symbol " << sym->Name() << " (Index " << sym->Index() << "):" << std::endl;
      if (sym->IsKernelSymbol() || sym->IsVariableSymbol()) {
        out << "    Section: " << sym->GetSection()->Name() << " ";
        out << "    Section Offset: " << sym->SectionOffset() << std::endl;
        out << "    VAddr: " << sym->VAddr() << " ";
        out << "    Size: " << sym->Size() << " ";
        out << "    Alignment: " << sym->Alignment() << std::endl;
        out << "    Kind: " << HsaSymbolKindToString(sym->Kind()) << " ";
        out << "    Linkage: " << HsaSymbolLinkageToString(sym->Linkage()) << " ";
        out << "    Definition: " << (sym->IsDefinition() ? "TRUE" : "FALSE") << std::endl;
      }
      if (sym->IsVariableSymbol()) {
        out << "    Allocation: " << HsaVariableAllocationToString(sym->Allocation()) << " ";
        out << "    Segment: " << HsaVariableSegmentToString(sym->Segment()) << " ";
        out << "    Constant: " << (sym->IsConst() ? "TRUE" : "FALSE") << std::endl;
      }
      out << std::dec;
    }

    void AmdHsaCode::PrintMachineCode(std::ostream& out, KernelSymbol* sym)
    {
      assert(HsaText());
      amd_kernel_code_t kernel_code;
      HsaText()->getData(sym->SectionOffset(), &kernel_code, sizeof(amd_kernel_code_t));

      out << "AMD Kernel Code for " << sym->Name() << ": " << std::endl << std::dec;
      PrintAmdKernelCode(out, &kernel_code);
      out << std::endl;

      std::vector<uint8_t> isa(HsaText()->size(), 0);
      HsaText()->getData(0, isa.data(), HsaText()->size());
      uint64_t isa_offset = sym->SectionOffset() + kernel_code.kernel_code_entry_byte_offset;

      out << "Disassembly for " << sym->Name() << ": " << std::endl;
      PrintDisassembly(out, isa.data(), HsaText()->size(), isa_offset);
      out << std::endl << std::dec;
    }

    void AmdHsaCode::PrintDisassembly(std::ostream& out, const unsigned char *isa, size_t size, uint32_t isa_offset)
    {
    #ifdef SP3_STATIC_LIB
      // Default asic is ci.
      std::string asic = "CI";
      std::string vendor_name, architecture_name;
      uint32_t major_version, minor_version, stepping;
      if (GetNoteIsa(vendor_name, architecture_name, &major_version, &minor_version, &stepping)) {
        if (major_version == 7) {
          asic = "CI";
        } else if (major_version == 8) {
          asic = "VI";
        } else if (major_version == 9) {
          asic = "GFX9";
        } else if (major_version == 10) {
          asic = "GFX10";
        } else {
          assert(!"unknown compute capability");
        }
      }

      struct sp3_context *dis_state = sp3_new();
      sp3_setasic(dis_state, asic.c_str());

      sp3_vma *dis_vma = sp3_vm_new_ptr(0, size / 4, (const uint32_t*)isa);

      std::vector<uint32_t> comments(HsaText()->size() / 4, 0);
      for (size_t i = 0; i < SymbolCount(); ++i) {
        Symbol* sym = GetSymbol(i);
        if (sym->IsKernelSymbol() && sym->IsDefinition()) {
          comments[sym->SectionOffset() / 4] = COMMENT_AMD_KERNEL_CODE_T_BEGIN;
          comments[(sym->SectionOffset() + 252) / 4] = COMMENT_AMD_KERNEL_CODE_T_END;
          amd_kernel_code_t kernel_code;
          HsaText()->getData(sym->SectionOffset(), &kernel_code, sizeof(amd_kernel_code_t));
          comments[(kernel_code.kernel_code_entry_byte_offset + sym->SectionOffset()) / 4] = COMMENT_KERNEL_ISA_BEGIN;
        }
      }
      sp3_vma *comment_vma = sp3_vm_new_ptr(0, comments.size(), (const uint32_t*)comments.data());
      sp3_setcomments(dis_state, comment_vma, CommentTopCallBack, CommentRightCallBack, this);

      // When isa_offset == 0 disassembly full hsatext section.
      // Otherwise disassembly only from this offset till endpgm instruction.
      char *text = sp3_disasm(
        dis_state,
        dis_vma,
        isa_offset / 4,
        nullptr,
        SP3_SHTYPE_CS,
        nullptr,
        (unsigned)(size / 4),
        isa_offset == 0 ? SP3DIS_FORCEVALID | SP3DIS_COMMENTS : SP3DIS_COMMENTS);

      enum class IsaState {
        UNKNOWN,
        AMD_KERNEL_CODE_T_BEGIN,
        AMD_KERNEL_CODE_T,
        AMD_KERNEL_CODE_T_END,
        ISA_BEGIN,
        ISA,
        PADDING,
      };

      std::string line;
      char *text_ptr = text;
      IsaState state = IsaState::UNKNOWN;

      uint32_t offset = 0;
      uint32_t padding_end = 0;
      std::string padding;

      while (text_ptr && text_ptr[0] != '\0') {
        line.clear();
        while (text_ptr[0] != '\0' && text_ptr[0] != '\n') {
          line.push_back(text_ptr[0]);
          ++text_ptr;
        }
        ltrim(line);
        if (text_ptr[0] == '\n') {
          ++text_ptr;
        }
        switch (state) {
        case IsaState::UNKNOWN:
          assert(line != "// amd_kernel_code_t end");
          padding.clear();
          if (line == "// amd_kernel_code_t begin") {
            state = IsaState::AMD_KERNEL_CODE_T_BEGIN;
          } else if (line == "// isa begin") {
            state = IsaState::ISA_BEGIN;
          } else if (line == "end") {
            out << line << std::endl;
          } else if (line.find("v_cndmask_b32  v0, s0, v0, vcc") != std::string::npos) {
            padding += "  " + line + "\n";
            offset = ParseInstructionOffset(line);
            padding_end = ParseInstructionOffset(line);
            state = IsaState::PADDING;
          } else if (line != "shader (null)") {
            out << "  " << line << std::endl;
          }
          break;

        case IsaState::AMD_KERNEL_CODE_T_BEGIN:
          assert(line != "// amd_kernel_code_t begin");
          assert(line != "// amd_kernel_code_t end");
          assert(line != "// isa begin");
          assert(line != "end");
          padding.clear();
          offset = ParseInstructionOffset(line);
          state = IsaState::AMD_KERNEL_CODE_T;
          break;

        case IsaState::AMD_KERNEL_CODE_T:
          assert(line != "// amd_kernel_code_t begin");
          assert(line != "// isa begin");
          assert(line != "end");
          assert(padding.empty());
          if (line == "// amd_kernel_code_t end") {
            state = IsaState::AMD_KERNEL_CODE_T_END;
          }
          break;

        case IsaState::AMD_KERNEL_CODE_T_END:
          assert(line != "// amd_kernel_code_t begin");
          assert(line != "// amd_kernel_code_t end");
          assert(line != "// isa begin");
          assert(line != "end");
          assert(padding.empty());
          for (size_t i = 0; i < SymbolCount(); ++i) {
            Symbol* sym = GetSymbol(i);
            if (sym->IsKernelSymbol() && sym->IsDefinition() && sym->SectionOffset() == offset) {
              std::ostream::fmtflags flags = out.flags();
              char fill = out.fill();
              out << "  //" << std::endl;
              out << "  // amd_kernel_code_t for " << sym->Name()
                  << " (" << std::hex << std::setw(12) << std::setfill('0') << std::right << offset
                  << " - " << std::setw(12) << (offset + 256) << ')' << std::endl;
              out << "  //" << std::endl;
              out << std::setfill(fill);
              out.flags(flags);
              break;
            }
          }
          state = IsaState::UNKNOWN;
          break;

        case IsaState::ISA_BEGIN:
          assert(line != "// amd_kernel_code_t begin");
          assert(line != "// amd_kernel_code_t end");
          assert(line != "// isa begin");
          padding.clear();
          offset = ParseInstructionOffset(line);
          for (size_t i = 0; i < SymbolCount(); ++i) {
            Symbol* sym = GetSymbol(i);
            if (sym->IsKernelSymbol() && sym->IsDefinition()) {
              amd_kernel_code_t kernel_code;
              HsaText()->getData(sym->SectionOffset(), &kernel_code, sizeof(amd_kernel_code_t));
              if ((sym->SectionOffset() + kernel_code.kernel_code_entry_byte_offset) == offset) {
                out << "  //" << std::endl;
                out << "  // " << sym->Name() << ':' << std::endl;
                out << "  //" << std::endl;
                break;
              }
            }
          }
          if (line == "end") {
            out << line << std::endl;
            state = IsaState::UNKNOWN;
          } else {
            out << "  " << line << std::endl;
            state = IsaState::ISA;
          }
          break;

        case IsaState::ISA:
          assert(line != "// amd_kernel_code_t end");
          if (!padding.empty()) {
            out << padding;
            out.flush();
            padding.clear();
          }
          if (line == "// amd_kernel_code_t begin") {
            state = IsaState::AMD_KERNEL_CODE_T_BEGIN;
          } else if (line == "// isa begin") {
            state = IsaState::ISA_BEGIN;
          } else if (line == "end") {
            out << line << std::endl;
            state = IsaState::UNKNOWN;
          } else if (line.find("v_cndmask_b32  v0, s0, v0, vcc") != std::string::npos) {
            padding += "  " + line + "\n";
            offset = ParseInstructionOffset(line);
            padding_end = offset;
            state = IsaState::PADDING;
          } else {
            out << "  " << line << std::endl;
          }
          break;

        case IsaState::PADDING:
          assert(line != "// amd_kernel_code_t end");
          if (line.find("v_cndmask_b32  v0, s0, v0, vcc") != std::string::npos) {
            padding += "  " + line + "\n";
            padding_end = ParseInstructionOffset(line);
          } else if (line == "// amd_kernel_code_t begin" || line == "// isa begin" || line == "end") {
              padding.clear();
              std::ostream::fmtflags flags = out.flags();
              char fill = out.fill();
              out << "  //" << std::endl;
              out << "  // padding ("
                  << std::hex << std::setw(12) << std::setfill('0') << std::right << offset
                  << " - " << std::setw(12) << (padding_end + 4) << ')' << std::endl;
              out << "  //" << std::endl;
              out << std::setfill(fill);
              out.flags(flags);
              if (line == "// amd_kernel_code_t begin") {
                state = IsaState::AMD_KERNEL_CODE_T_BEGIN;
              } else if (line == "// isa begin") {
                state = IsaState::ISA_BEGIN;
              } else if (line == "end") {
                out << line << std::endl;
                state = IsaState::UNKNOWN;
              }
          } else {
            padding += "  " + line + "\n";
            state = IsaState::ISA;
          }
          break;

        default:
          assert(false);
          break;
        }
      }

      sp3_free(text);
      sp3_close(dis_state);
      sp3_vm_free(dis_vma);
      sp3_vm_free(comment_vma);
    #else
      PrintRawData(out, isa, size);
    #endif // SP3_STATIC_LIB
      out << std::dec;
    }

    std::string AmdHsaCode::MangleSymbolName(const std::string& module_name, const std::string symbol_name)
    {
      if (module_name.empty()) {
        return std::move(symbol_name);
      } else {
        return module_name + "::" + symbol_name;
      }
    }

    bool AmdHsaCode::ElfImageError()
    {
      out << img->output();
      return false;
    }

      AmdHsaCode* AmdHsaCodeManager::FromHandle(hsa_code_object_t c)
      {
        CodeMap::iterator i = codeMap.find(c.handle);
        if (i == codeMap.end()) {
          AmdHsaCode* code = new AmdHsaCode();
          const void* buffer = reinterpret_cast<const void*>(c.handle);
          if (!code->InitAsBuffer(buffer, 0)) {
            delete code;
            return 0;
          }
          codeMap[c.handle] = code;
          return code;
        }
        return i->second;
      }

      bool AmdHsaCodeManager::Destroy(hsa_code_object_t c)
      {
        CodeMap::iterator i = codeMap.find(c.handle);
        if (i == codeMap.end()) {
          // Currently, we do not always create map entry for every code object buffer.
          return true;
        }
        delete i->second;
        codeMap.erase(i);
        return true;
      }

    bool AmdHsaCode::PullElfV2()
    {
      for (size_t i = 0; i < img->segmentCount(); ++i) {
        Segment* s = img->segment(i);
        if (s->type() == PT_LOAD) {
          dataSegments.push_back(s);
        }
      }
      for (size_t i = 0; i < img->sectionCount(); ++i) {
        Section* sec = img->section(i);
        if (!sec) { continue; }
        if ((sec->type() == SHT_PROGBITS || sec->type() == SHT_NOBITS) &&
            !(sec->flags() & SHF_EXECINSTR)) {
          dataSections.push_back(sec);
        } else if (sec->type() == SHT_RELA) {
          relocationSections.push_back(sec->asRelocationSection());
        }
        if (sec->Name() == ".text") {
          hsatext = sec;
        }
      }
      for (size_t i = 0; i < img->getSymbolTable()->symbolCount(); ++i) {
        amd::elf::Symbol* elfsym = img->getSymbolTable()->symbol(i);
        Symbol* sym = 0;
        switch (elfsym->type()) {
        case STT_AMDGPU_HSA_KERNEL: {
          amd::elf::Section* sec = elfsym->section();
          amd_kernel_code_t akc;
          if (!sec) {
            out << "Failed to find section for symbol " << elfsym->name() << std::endl;
            return false;
          }
          if (!(sec->flags() & (SHF_ALLOC | SHF_EXECINSTR))) {
            out << "Invalid code section for symbol " << elfsym->name() << std::endl;
            return false;
          }
          if (!sec->getData(elfsym->value() - sec->addr(), &akc, sizeof(amd_kernel_code_t))) {
            out << "Failed to get AMD Kernel Code for symbol " << elfsym->name() << std::endl;
            return false;
          }
          sym = new KernelSymbolV2(elfsym, &akc);
          break;
        }
        case STT_OBJECT:
        case STT_COMMON:
          sym = new VariableSymbolV2(elfsym);
          break;
        default:
          break; // Skip unknown symbols.
        }
        if (sym) { symbols.push_back(sym); }
      }

      return true;
    }

    KernelSymbolV2::KernelSymbolV2(amd::elf::Symbol* elfsym_, const amd_kernel_code_t* akc) :
      KernelSymbol(elfsym_, akc) { }
}   // namespace code
}   // namespace hsa
}   // namespace amd
}   // namespace rocr
