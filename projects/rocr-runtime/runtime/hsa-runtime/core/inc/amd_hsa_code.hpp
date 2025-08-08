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

#ifndef AMD_HSA_CODE_HPP_
#define AMD_HSA_CODE_HPP_

#include "core/inc/amd_elf_image.hpp"
#include "inc/amd_hsa_elf.h"
#include "inc/amd_hsa_kernel_code.h"
#include "inc/hsa.h"
#include "inc/hsa_ext_finalize.h"
#include <memory>
#include <sstream>
#include <cassert>
#include <unordered_map>

namespace rocr {
namespace amd {
namespace hsa {
namespace common {

template<uint64_t signature>
class Signed {
public:
  static const uint64_t CT_SIGNATURE;
  const uint64_t RT_SIGNATURE;

protected:
  Signed(): RT_SIGNATURE(signature) {}
  virtual ~Signed() {}
};

template<uint64_t signature>
const uint64_t Signed<signature>::CT_SIGNATURE = signature;

bool IsAccessibleMemoryAddress(uint64_t address);

template<typename class_type, typename member_type>
size_t OffsetOf(member_type class_type::*member)
{
  return (char*)&((class_type*)nullptr->*member) - (char*)nullptr;
}

template<typename class_type>
class_type* ObjectAt(uint64_t address)
{
  if (!IsAccessibleMemoryAddress(address)) {
    return nullptr;
  }

  const uint64_t *rt_signature =
    (const uint64_t*)(address + OffsetOf(&class_type::RT_SIGNATURE));
  if (nullptr == rt_signature) {
    return nullptr;
  }
  if (class_type::CT_SIGNATURE != *rt_signature) {
    return nullptr;
  }

  return (class_type*)address;
}

}   //  namespace common

namespace code {

    typedef amd::elf::Segment Segment;
    typedef amd::elf::Section Section;
    typedef amd::elf::RelocationSection RelocationSection;
    typedef amd::elf::Relocation Relocation;

    class KernelSymbol;
    class VariableSymbol;

    class Symbol {
    protected:
      amd::elf::Symbol* elfsym;

    public:
      explicit Symbol(amd::elf::Symbol* elfsym_)
        : elfsym(elfsym_) { }
      virtual ~Symbol() { }
      virtual bool IsKernelSymbol() const { return false; }
      virtual KernelSymbol* AsKernelSymbol() { assert(false); return 0; }
      virtual bool IsVariableSymbol() const { return false; }
      virtual VariableSymbol* AsVariableSymbol() { assert(false); return 0; }
      amd::elf::Symbol* elfSym() { return elfsym; }
      std::string Name() const { return elfsym ? elfsym->name() : ""; }
      Section* GetSection() { return elfsym->section(); }
      virtual uint64_t SectionOffset() const { return elfsym->value(); }
      virtual uint64_t VAddr() const { return elfsym->section()->addr() + elfsym->value(); }
      uint32_t Index() const { return elfsym ? elfsym->index() : 0; }
      bool IsDeclaration() const;
      bool IsDefinition() const;
      virtual bool IsAgent() const;
      virtual hsa_symbol_kind_t Kind() const = 0;
      hsa_symbol_linkage_t Linkage() const;
      hsa_variable_allocation_t Allocation() const;
      hsa_variable_segment_t Segment() const;
      uint64_t Size() const;
      uint32_t Size32() const;
      uint32_t Alignment() const;
      bool IsConst() const;
      virtual hsa_status_t GetInfo(hsa_code_symbol_info_t attribute, void *value);
      static hsa_code_symbol_t ToHandle(Symbol* sym);
      static Symbol* FromHandle(hsa_code_symbol_t handle);
      void setValue(uint64_t value) { elfsym->setValue(value); }
      void setSize(uint32_t size) { elfsym->setSize(size); }

      std::string GetModuleName() const;
      std::string GetSymbolName() const;
    };

    class KernelSymbol : public Symbol {
    private:
      uint32_t kernarg_segment_size, kernarg_segment_alignment;
      uint32_t group_segment_size, private_segment_size;
      bool is_dynamic_callstack;

    public:
      explicit KernelSymbol(amd::elf::Symbol* elfsym_, const amd_kernel_code_t* akc);
      bool IsKernelSymbol() const override { return true; }
      KernelSymbol* AsKernelSymbol() override { return this; }
      hsa_symbol_kind_t Kind() const override { return HSA_SYMBOL_KIND_KERNEL; }
      hsa_status_t GetInfo(hsa_code_symbol_info_t attribute, void *value) override;
    };

    class VariableSymbol : public Symbol {
    public:
      explicit VariableSymbol(amd::elf::Symbol* elfsym_)
        : Symbol(elfsym_) { }
      bool IsVariableSymbol() const override { return true; }
      VariableSymbol* AsVariableSymbol() override { return this; }
      hsa_symbol_kind_t Kind() const override { return HSA_SYMBOL_KIND_VARIABLE; }
      hsa_status_t GetInfo(hsa_code_symbol_info_t attribute, void *value) override;
    };

    class AmdHsaCode {
    private:
      std::ostringstream out;
      std::unique_ptr<amd::elf::Image> img;
      std::vector<Segment*> dataSegments;
      std::vector<Section*> dataSections;
      std::vector<RelocationSection*> relocationSections;
      std::vector<Symbol*> symbols;
      bool combineDataSegments;
      Segment* hsaSegments[AMDGPU_HSA_SEGMENT_LAST][2];
      Section* hsaSections[AMDGPU_HSA_SECTION_LAST];

      amd::elf::Section* hsatext;
      amd::elf::Section* imageInit;
      amd::elf::Section* samplerInit;
      amd::elf::Section* debugInfo;
      amd::elf::Section* debugLine;
      amd::elf::Section* debugAbbrev;

      bool PullElf();
      bool PullElfV1();
      bool PullElfV2();

      void AddAmdNote(uint32_t type, const void* desc, uint32_t desc_size);
      template <typename S>
      bool GetAmdNote(uint32_t type, S** desc)
      {
        uint32_t desc_size;
        if (!img->note()->getNote("AMD", type, (void**) desc, &desc_size)) {
          out << "Failed to find note, type: " << type << std::endl;
          return false;
        }
        if (desc_size < sizeof(S)) {
          out << "Note size mismatch, type: " << type << " size: " << desc_size << " expected at least " << sizeof(S) << std::endl;
          return false;
        }
        return true;
      }

      void PrintSegment(std::ostream& out, Segment* segment);
      void PrintSection(std::ostream& out, Section* section);
      void PrintRawData(std::ostream& out, Section* section);
      void PrintRawData(std::ostream& out, const unsigned char *data, size_t size);
      void PrintRelocationData(std::ostream& out, RelocationSection* section);
      void PrintSymbol(std::ostream& out, Symbol* sym);
      void PrintDisassembly(std::ostream& out, const unsigned char *isa, size_t size, uint32_t isa_offset = 0);
      std::string MangleSymbolName(const std::string& module_name, const std::string& symbol_name);
      bool ElfImageError();

    public:
      bool HasHsaText() const { return hsatext != 0; }
      amd::elf::Section* HsaText() { assert(hsatext); return hsatext; }
      const amd::elf::Section* HsaText() const { assert(hsatext); return hsatext; }
      amd::elf::SymbolTable* Symtab() { assert(img); return img->symtab(); }
      uint16_t Machine() const { return img->Machine(); }
      uint32_t EFlags() const { return img->EFlags(); }
      uint32_t EClass() const { return img->EClass(); }
      uint32_t OsAbi() const { return img->OsAbi(); }

      AmdHsaCode(bool combineDataSegments = true);
      virtual ~AmdHsaCode();

      std::string output() { return out.str(); }
      bool LoadFromFile(const std::string& filename);
      bool SaveToFile(const std::string& filename);
      bool WriteToBuffer(void* buffer);
      bool InitFromBuffer(const void* buffer, size_t size);
      bool InitAsBuffer(const void* buffer, size_t size);
      bool InitAsHandle(hsa_code_object_t code_handle);
      bool InitNew(bool xnack = false);
      bool Freeze();
      hsa_code_object_t GetHandle();
      const char* ElfData();
      uint64_t ElfSize();
      bool Validate();
      void Print(std::ostream& out);
      void PrintNotes(std::ostream& out);
      void PrintSegments(std::ostream& out);
      void PrintSections(std::ostream& out);
      void PrintSymbols(std::ostream& out);
      void PrintMachineCode(std::ostream& out);
      void PrintMachineCode(std::ostream& out, KernelSymbol* sym);
      bool PrintToFile(const std::string& filename);

      void AddNoteCodeObjectVersion(uint32_t major, uint32_t minor);
      bool GetNoteCodeObjectVersion(std::string& version);
      void AddNoteHsail(uint32_t hsail_major, uint32_t hsail_minor, hsa_profile_t profile, hsa_machine_model_t machine_model, hsa_default_float_rounding_mode_t rounding_mode);
      bool GetNoteHsail(uint32_t* hsail_major, uint32_t* hsail_minor, hsa_profile_t* profile, hsa_machine_model_t* machine_model, hsa_default_float_rounding_mode_t* default_float_round);
      void AddNoteIsa(const std::string& vendor_name, const std::string& architecture_name, uint32_t major, uint32_t minor, uint32_t stepping);
      bool GetNoteIsa(std::string& vendor_name, std::string& architecture_name, uint32_t* major_version, uint32_t* minor_version, uint32_t* stepping);
      void AddNoteProducer(uint32_t major, uint32_t minor, const std::string& producer);
      bool GetNoteProducer(uint32_t* major, uint32_t* minor, std::string& producer_name);
      void AddNoteProducerOptions(const std::string& options);
      void AddNoteProducerOptions(int32_t call_convention, const hsa_ext_control_directives_t& user_directives, const std::string& user_options);
      bool GetNoteProducerOptions(std::string& options);

      bool GetIsa(std::string& isaName, unsigned *genericVersion = nullptr);
      bool GetCodeObjectVersion(uint32_t* major, uint32_t* minor);
      hsa_status_t GetInfo(hsa_code_object_info_t attribute, void *value);
      hsa_status_t GetSymbol(const char *module_name, const char *symbol_name, hsa_code_symbol_t *sym);
      hsa_status_t IterateSymbols(hsa_code_object_t code_object,
                                  hsa_status_t (*callback)(
                                    hsa_code_object_t code_object,
                                    hsa_code_symbol_t symbol,
                                    void* data),
                                  void* data);

      void AddHsaTextData(const void* buffer, size_t size);
      uint64_t NextKernelCodeOffset() const;
      bool AddKernelCode(KernelSymbol* sym, const void* code, size_t size);

      Symbol* AddKernelDefinition(const std::string& name, const void* isa, size_t isa_size);

      size_t DataSegmentCount() const { return dataSegments.size(); }
      Segment* DataSegment(size_t i) const { return dataSegments[i]; }

      size_t DataSectionCount() { return dataSections.size(); }
      Section* DataSection(size_t i) { return dataSections[i]; }

      Section* AddEmptySection();
      Section* AddCodeSection(Segment* segment);
      Section* AddDataSection(const std::string &name,
                              uint32_t type,
                              uint64_t flags,
                              Segment* segment);

      bool HasImageInitSection() const { return imageInit != 0; }
      Section* ImageInitSection();
      void AddImageInitializer(Symbol* image, uint64_t destOffset, const amdgpu_hsa_image_descriptor_t& init);
      void AddImageInitializer(Symbol* image, uint64_t destOffset,
        amdgpu_hsa_metadata_kind16_t kind,
        amdgpu_hsa_image_geometry8_t geometry,
        amdgpu_hsa_image_channel_order8_t channel_order, amdgpu_hsa_image_channel_type8_t channel_type,
        uint64_t width, uint64_t height, uint64_t depth, uint64_t array);


      bool HasSamplerInitSection() const { return samplerInit != 0; }
      amd::elf::Section* SamplerInitSection();
      amd::elf::Section* AddSamplerInit();
      void AddSamplerInitializer(Symbol* sampler, uint64_t destOffset, const amdgpu_hsa_sampler_descriptor_t& init);
      void AddSamplerInitializer(Symbol* sampler, uint64_t destOffset,
        amdgpu_hsa_sampler_coord8_t coord,
        amdgpu_hsa_sampler_filter8_t filter,
        amdgpu_hsa_sampler_addressing8_t addressing);

      void AddInitVarWithAddress(bool large, Symbol* dest, uint64_t destOffset, Symbol* addrOf, uint64_t addrAddend);

      void InitHsaSegment(amdgpu_hsa_elf_segment_t segment, bool writable);
      bool AddHsaSegments();
      Segment* HsaSegment(amdgpu_hsa_elf_segment_t segment, bool writable);

      void InitHsaSectionSegment(amdgpu_hsa_elf_section_t section, bool combineSegments = true);
      Section* HsaDataSection(amdgpu_hsa_elf_section_t section, bool combineSegments = true);

      Symbol* AddExecutableSymbol(const std::string &name,
                                  unsigned char type,
                                  unsigned char binding,
                                  unsigned char other,
                                  Section *section = 0);

      Symbol* AddVariableSymbol(const std::string &name,
                                unsigned char type,
                                unsigned char binding,
                                unsigned char other,
                                Section *section,
                                uint64_t value,
                                uint64_t size);
      void AddSectionSymbols();

      size_t RelocationSectionCount() { return relocationSections.size(); }
      RelocationSection* GetRelocationSection(size_t i) { return relocationSections[i]; }

      size_t SymbolCount() { return symbols.size(); }
      Symbol* GetSymbol(size_t i) { return symbols[i]; }
      Symbol* GetSymbolByElfIndex(size_t index);
      Symbol* FindSymbol(const std::string &n);

      void AddData(amdgpu_hsa_elf_section_t section, const void* data = 0, size_t size = 0);

      Section* DebugInfo();
      Section* DebugLine();
      Section* DebugAbbrev();

      Section* AddHsaHlDebug(const std::string& name, const void* data, size_t size);
    };

    class AmdHsaCodeManager {
    private:
      typedef std::unordered_map<uint64_t, AmdHsaCode*> CodeMap;
      CodeMap codeMap;

    public:
      AmdHsaCode* FromHandle(hsa_code_object_t handle);
      bool Destroy(hsa_code_object_t handle);
    };

    class KernelSymbolV2 : public KernelSymbol {
    private:
    public:
      explicit KernelSymbolV2(amd::elf::Symbol* elfsym_, const amd_kernel_code_t* akc);
      bool IsAgent() const override { return true; }
      uint64_t SectionOffset() const override { return elfsym->value() - elfsym->section()->addr(); }
      uint64_t VAddr() const override { return elfsym->value(); }
    };

    class VariableSymbolV2 : public VariableSymbol {
    private:
    public:
      explicit VariableSymbolV2(amd::elf::Symbol* elfsym_) : VariableSymbol(elfsym_) { }
      bool IsAgent() const override { return false; }
      uint64_t SectionOffset() const override { return elfsym->value() - elfsym->section()->addr(); }
      uint64_t VAddr() const override { return elfsym->value(); }
    };
}   //  namespace code
}   //  namespace hsa
}   //  namespace amd
}   //  namespace rocr

#endif // AMD_HSA_CODE_HPP_
