/*
Copyright (c) 2015 - 2021 Advanced Micro Devices, Inc. All rights reserved.

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
#include "hip_code_object.hpp"
#include "amd_hsa_elf.hpp"

#include <cstring>

#include <hip/driver_types.h>
#include "hip/hip_runtime_api.h"
#include "hip/hip_runtime.h"
#include "hip_internal.hpp"
#include "platform/program.hpp"
#include <elf/elf.hpp>
#include "comgrctx.hpp"
#include "hip_comgr_helper.hpp"

namespace hip {
hipError_t ihipFree(void* ptr);
// forward declaration of methods required for managed variables
hipError_t ihipMallocManaged(void** ptr, size_t size, unsigned int align = 0);
namespace {
// In uncompressed mode
constexpr char kOffloadBundleUncompressedMagicStr[] = "__CLANG_OFFLOAD_BUNDLE__";
static constexpr size_t kOffloadBundleUncompressedMagicStrSize =
    sizeof(kOffloadBundleUncompressedMagicStr);

// In compressed mode
constexpr char kOffloadBundleCompressedMagicStr[] = "CCOB";
static constexpr size_t kOffloadBundleCompressedMagicStrSize =
    sizeof(kOffloadBundleCompressedMagicStr);

constexpr char kOffloadKindHip[] = "hip";
constexpr char kOffloadKindHipv4[] = "hipv4";
constexpr char kOffloadKindHcc[] = "hcc";
constexpr char kAmdgcnTargetTriple[] = "amdgcn-amd-amdhsa-";
constexpr char kHipFatBinName[] = "hipfatbin";
constexpr char kHipFatBinName_[] = "hipfatbin-";
constexpr char kOffloadKindHipv4_[] = "hipv4-";  // bundled code objects need the prefix
constexpr char kOffloadHipV4FatBinName_[] = "hipfatbin-hipv4-";

// Clang Offload bundler description & Header in uncompressed mode.
struct __ClangOffloadBundleInfo {
  uint64_t offset;
  uint64_t size;
  uint64_t bundleEntryIdSize;
  const char bundleEntryId[1];
};

struct __ClangOffloadBundleUncompressedHeader {
  const char magic[kOffloadBundleUncompressedMagicStrSize - 1];
  uint64_t numOfCodeObjects;
  __ClangOffloadBundleInfo desc[1];
};

// Clang Offload bundler description & Header in compressed mode.
struct __ClangOffloadBundleCompressedHeader {
  const char magic[kOffloadBundleCompressedMagicStrSize - 1];
  uint16_t versionNumber;
  uint16_t compressionMethod;
  uint32_t totalSize;
  uint32_t uncompressedBinarySize;
  uint64_t Hash;
  const char compressedBinarydesc[1];
};
}  // namespace

bool CodeObject::IsClangOffloadMagicBundle(const void* data, bool& isCompressed) {
  std::string magic(reinterpret_cast<const char*>(data),
                    kOffloadBundleUncompressedMagicStrSize - 1);
  if (!magic.compare(kOffloadBundleUncompressedMagicStr)) {
    isCompressed = false;
    return true;
  }
  std::string magic1(reinterpret_cast<const char*>(data),
                    kOffloadBundleCompressedMagicStrSize - 1);
  if (!magic1.compare(kOffloadBundleCompressedMagicStr)) {
    isCompressed = true;
    return true;
  }
  return false;
}

uint32_t CodeObject::getGenericVersion(const void* image) {
  const Elf64_Ehdr* ehdr = reinterpret_cast<const Elf64_Ehdr*>(image);
  return (ehdr->e_machine == EM_AMDGPU && ehdr->e_ident[EI_OSABI] == ELFOSABI_AMDGPU_HSA &&
      ehdr->e_ident[EI_ABIVERSION] == ELFABIVERSION_AMDGPU_HSA_V6) ?
      ((ehdr->e_flags & EF_AMDGPU_GENERIC_VERSION) >> EF_AMDGPU_GENERIC_VERSION_OFFSET) : 0;
}

bool CodeObject::isGenericTarget(const void* image) {
  return getGenericVersion(image) >= EF_AMDGPU_GENERIC_VERSION_MIN;
}

bool CodeObject::containGenericTarget(const void *data) {
  const auto obheader = reinterpret_cast<const __ClangOffloadBundleUncompressedHeader*>(data);
  const auto* desc = &obheader->desc[0];
  for (uint64_t i = 0; i < obheader->numOfCodeObjects; ++i,
        desc = reinterpret_cast<const __ClangOffloadBundleInfo*>(
           reinterpret_cast<uintptr_t>(&desc->bundleEntryId[0]) + desc->bundleEntryIdSize)) {
    if (desc->size == 0) continue;
    const void* image =
         reinterpret_cast<const void*>(reinterpret_cast<uintptr_t>(obheader) + desc->offset);
    if (isGenericTarget(image)) {
      return true;
    }
  }
  return false;
}

uint64_t CodeObject::ElfSize(const void* emi) { return amd::Elf::getElfSize(emi); }

static bool getProcName(uint32_t EFlags, std::string& proc_name, bool& xnackSupported,
                        bool& sramEccSupported) {
  switch (EFlags & EF_AMDGPU_MACH) {
    case EF_AMDGPU_MACH_AMDGCN_GFX700:
      xnackSupported = false;
      sramEccSupported = false;
      proc_name = "gfx700";
      break;
    case EF_AMDGPU_MACH_AMDGCN_GFX701:
      xnackSupported = false;
      sramEccSupported = false;
      proc_name = "gfx701";
      break;
    case EF_AMDGPU_MACH_AMDGCN_GFX702:
      xnackSupported = false;
      sramEccSupported = false;
      proc_name = "gfx702";
      break;
    case EF_AMDGPU_MACH_AMDGCN_GFX703:
      xnackSupported = false;
      sramEccSupported = false;
      proc_name = "gfx703";
      break;
    case EF_AMDGPU_MACH_AMDGCN_GFX704:
      xnackSupported = false;
      sramEccSupported = false;
      proc_name = "gfx704";
      break;
    case EF_AMDGPU_MACH_AMDGCN_GFX705:
      xnackSupported = false;
      sramEccSupported = false;
      proc_name = "gfx705";
      break;
    case EF_AMDGPU_MACH_AMDGCN_GFX801:
      xnackSupported = true;
      sramEccSupported = false;
      proc_name = "gfx801";
      break;
    case EF_AMDGPU_MACH_AMDGCN_GFX802:
      xnackSupported = false;
      sramEccSupported = false;
      proc_name = "gfx802";
      break;
    case EF_AMDGPU_MACH_AMDGCN_GFX803:
      xnackSupported = false;
      sramEccSupported = false;
      proc_name = "gfx803";
      break;
    case EF_AMDGPU_MACH_AMDGCN_GFX805:
      xnackSupported = false;
      sramEccSupported = false;
      proc_name = "gfx805";
      break;
    case EF_AMDGPU_MACH_AMDGCN_GFX810:
      xnackSupported = true;
      sramEccSupported = false;
      proc_name = "gfx810";
      break;
    case EF_AMDGPU_MACH_AMDGCN_GFX900:
      xnackSupported = true;
      sramEccSupported = false;
      proc_name = "gfx900";
      break;
    case EF_AMDGPU_MACH_AMDGCN_GFX902:
      xnackSupported = true;
      sramEccSupported = false;
      proc_name = "gfx902";
      break;
    case EF_AMDGPU_MACH_AMDGCN_GFX904:
      xnackSupported = true;
      sramEccSupported = false;
      proc_name = "gfx904";
      break;
    case EF_AMDGPU_MACH_AMDGCN_GFX906:
      xnackSupported = true;
      sramEccSupported = true;
      proc_name = "gfx906";
      break;
    case EF_AMDGPU_MACH_AMDGCN_GFX908:
      xnackSupported = true;
      sramEccSupported = true;
      proc_name = "gfx908";
      break;
    case EF_AMDGPU_MACH_AMDGCN_GFX909:
      xnackSupported = true;
      sramEccSupported = false;
      proc_name = "gfx909";
      break;
    case EF_AMDGPU_MACH_AMDGCN_GFX90A:
      xnackSupported = true;
      sramEccSupported = true;
      proc_name = "gfx90a";
      break;
    case EF_AMDGPU_MACH_AMDGCN_GFX90C:
      xnackSupported = true;
      sramEccSupported = false;
      proc_name = "gfx90c";
      break;
    case EF_AMDGPU_MACH_AMDGCN_GFX942:
      xnackSupported = true;
      sramEccSupported = true;
      proc_name = "gfx942";
      break;
    case EF_AMDGPU_MACH_AMDGCN_GFX950:
      xnackSupported = true;
      sramEccSupported = true;
      proc_name = "gfx950";
      break;
    case EF_AMDGPU_MACH_AMDGCN_GFX1010:
      xnackSupported = true;
      sramEccSupported = false;
      proc_name = "gfx1010";
      break;
    case EF_AMDGPU_MACH_AMDGCN_GFX1011:
      xnackSupported = true;
      sramEccSupported = false;
      proc_name = "gfx1011";
      break;
    case EF_AMDGPU_MACH_AMDGCN_GFX1012:
      xnackSupported = true;
      sramEccSupported = false;
      proc_name = "gfx1012";
      break;
    case EF_AMDGPU_MACH_AMDGCN_GFX1013:
      xnackSupported = true;
      sramEccSupported = false;
      proc_name = "gfx1013";
      break;
    case EF_AMDGPU_MACH_AMDGCN_GFX1030:
      xnackSupported = false;
      sramEccSupported = false;
      proc_name = "gfx1030";
      break;
    case EF_AMDGPU_MACH_AMDGCN_GFX1031:
      xnackSupported = false;
      sramEccSupported = false;
      proc_name = "gfx1031";
      break;
    case EF_AMDGPU_MACH_AMDGCN_GFX1032:
      xnackSupported = false;
      sramEccSupported = false;
      proc_name = "gfx1032";
      break;
    case EF_AMDGPU_MACH_AMDGCN_GFX1033:
      xnackSupported = false;
      sramEccSupported = false;
      proc_name = "gfx1033";
      break;
    case EF_AMDGPU_MACH_AMDGCN_GFX1034:
      xnackSupported = false;
      sramEccSupported = false;
      proc_name = "gfx1034";
      break;
    case EF_AMDGPU_MACH_AMDGCN_GFX1035:
      xnackSupported = false;
      sramEccSupported = false;
      proc_name = "gfx1035";
      break;
    case EF_AMDGPU_MACH_AMDGCN_GFX1036:
      xnackSupported = false;
      sramEccSupported = false;
      proc_name = "gfx1036";
      break;
    case EF_AMDGPU_MACH_AMDGCN_GFX1100:
      xnackSupported = false;
      sramEccSupported = false;
      proc_name = "gfx1100";
      break;
    case EF_AMDGPU_MACH_AMDGCN_GFX1101:
      xnackSupported = false;
      sramEccSupported = false;
      proc_name = "gfx1101";
      break;
    case EF_AMDGPU_MACH_AMDGCN_GFX1102:
      xnackSupported = false;
      sramEccSupported = false;
      proc_name = "gfx1102";
      break;
    case EF_AMDGPU_MACH_AMDGCN_GFX1103:
      xnackSupported = false;
      sramEccSupported = false;
      proc_name = "gfx1103";
      break;
    case EF_AMDGPU_MACH_AMDGCN_GFX1150:
      xnackSupported = false;
      sramEccSupported = false;
      proc_name = "gfx1150";
      break;
    case EF_AMDGPU_MACH_AMDGCN_GFX1151:
      xnackSupported = false;
      sramEccSupported = false;
      proc_name = "gfx1151";
      break;
    case EF_AMDGPU_MACH_AMDGCN_GFX1200:
      xnackSupported = false;
      sramEccSupported = false;
      proc_name = "gfx1200";
      break;
    case EF_AMDGPU_MACH_AMDGCN_GFX1201:
      xnackSupported = false;
      sramEccSupported = false;
      proc_name = "gfx1201";
      break;
    case EF_AMDGPU_MACH_AMDGCN_GFX9_GENERIC:
      xnackSupported = true;
      sramEccSupported = false;
      proc_name = "gfx9-generic";
      break;
    case EF_AMDGPU_MACH_AMDGCN_GFX9_4_GENERIC:
      xnackSupported = true;
      sramEccSupported = true;
      proc_name = "gfx9-4-generic";
      break;
    case EF_AMDGPU_MACH_AMDGCN_GFX10_1_GENERIC:
      xnackSupported = true;
      sramEccSupported = false;
      proc_name = "gfx10-1-generic";
      break;
    case EF_AMDGPU_MACH_AMDGCN_GFX10_3_GENERIC:
      xnackSupported = false;
      sramEccSupported = false;
      proc_name = "gfx10-3-generic";
      break;
    case EF_AMDGPU_MACH_AMDGCN_GFX11_GENERIC:
      xnackSupported = false;
      sramEccSupported = false;
      proc_name = "gfx11-generic";
      break;
    case EF_AMDGPU_MACH_AMDGCN_GFX12_GENERIC:
      xnackSupported = false;
      sramEccSupported = false;
      proc_name = "gfx12-generic";
      break;
    default:
      return false;
  }
  return true;
}

static bool getTripleTargetIDFromCodeObject(const void* code_object, std::string& target_id) {
  if (!code_object) return false;
  const Elf64_Ehdr* ehdr = reinterpret_cast<const Elf64_Ehdr*>(code_object);
  if (ehdr->e_machine != EM_AMDGPU) return false;
  if (ehdr->e_ident[EI_OSABI] != ELFOSABI_AMDGPU_HSA) return false;

  bool isXnackSupported{false}, isSramEccSupported{false};
  const char* vstr = nullptr;
  std::string proc_name;
  if (!getProcName(ehdr->e_flags, proc_name, isXnackSupported, isSramEccSupported)) return false;
  target_id = std::string(kAmdgcnTargetTriple) + '-' + proc_name;

  switch (ehdr->e_ident[EI_ABIVERSION]) {
    case ELFABIVERSION_AMDGPU_HSA_V2: {
      ClPrint(amd::LOG_INFO, amd::LOG_COMGR, "[Code Object V2, target id:%s]", target_id.c_str());
      return false;
    }

    case ELFABIVERSION_AMDGPU_HSA_V3: {
      ClPrint(amd::LOG_INFO, amd::LOG_COMGR, "[Code Object V3, target id:%s]", target_id.c_str());
      if (isSramEccSupported) {
        if (ehdr->e_flags & EF_AMDGPU_FEATURE_SRAMECC_V3)
          target_id += ":sramecc+";
        else
          target_id += ":sramecc-";
      }
      if (isXnackSupported) {
        if (ehdr->e_flags & EF_AMDGPU_FEATURE_XNACK_V3)
          target_id += ":xnack+";
        else
          target_id += ":xnack-";
      }
      break;
    }

    case ELFABIVERSION_AMDGPU_HSA_V4:
    case ELFABIVERSION_AMDGPU_HSA_V5:
    case ELFABIVERSION_AMDGPU_HSA_V6: {
      if (ehdr->e_ident[EI_ABIVERSION] & ELFABIVERSION_AMDGPU_HSA_V4) {
        vstr = "V4";
      } else if (ehdr->e_ident[EI_ABIVERSION] & ELFABIVERSION_AMDGPU_HSA_V5) {
        vstr = "V5";
      } else if (ehdr->e_ident[EI_ABIVERSION] & ELFABIVERSION_AMDGPU_HSA_V6) {
        vstr = "V6";
      }
      unsigned co_sram_value = (ehdr->e_flags) & EF_AMDGPU_FEATURE_SRAMECC_V4;
      if (co_sram_value == EF_AMDGPU_FEATURE_SRAMECC_OFF_V4)
        target_id += ":sramecc-";
      else if (co_sram_value == EF_AMDGPU_FEATURE_SRAMECC_ON_V4)
        target_id += ":sramecc+";

      unsigned co_xnack_value = (ehdr->e_flags) & EF_AMDGPU_FEATURE_XNACK_V4;
      if (co_xnack_value == EF_AMDGPU_FEATURE_XNACK_OFF_V4)
        target_id += ":xnack-";

      else if (co_xnack_value == EF_AMDGPU_FEATURE_XNACK_ON_V4)
        target_id += ":xnack+";
      ClPrint(amd::LOG_INFO, amd::LOG_COMGR,
        "[Code Object %s, target id: %s]", vstr, target_id.c_str());
      break;
    }

    default: {
      return false;
    }
  }
  return true;
}

// Consumes the string 'consume_' from the starting of the given input
// eg: input = amdgcn-amd-amdhsa--gfx908 and consume_ is amdgcn-amd-amdhsa--
// input will become gfx908.
static bool consume(std::string& input, std::string consume_) {
  if (input.substr(0, consume_.size()) != consume_) {
    return false;
  }
  input = input.substr(consume_.size());
  return true;
}

// Trim String till character, will be used to get gpuname
// example: input is gfx908:sram-ecc+ and trim char is :
// input will become :sram-ecc+.
static std::string trimName(std::string& input, char trim) {
  auto pos_ = input.find(trim);
  auto res = input;
  if (pos_ == std::string::npos) {
    input = "";
  } else {
    res = input.substr(0, pos_);
    input = input.substr(pos_);
  }
  return res;
}

// Trim String till character, will be used to get bundle entry ID.
// example: input is amdgcn-amd-amdhsa--gfx1035.bc and trim char is .
// input will become amdgcn-amd-amdhsa--gfx1035
static bool trimNameTail(std::string& input, char trim) {
  auto pos_ = input.rfind(trim);
  if (pos_ == std::string::npos) {
    return false;
  }
  input = input.substr(0, pos_);
  return true;
}

static char getFeatureValue(std::string& input, std::string feature) {
  char res = ' ';
  if (consume(input, std::move(feature))) {
    res = input[0];
    input = input.substr(1);
  }
  return res;
}

static bool getTargetIDValue(std::string& input, std::string& processor, char& sramecc_value,
                             char& xnack_value) {
  processor = trimName(input, ':');
  sramecc_value = getFeatureValue(input, std::string(":sramecc"));
  if (sramecc_value != ' ' && sramecc_value != '+' && sramecc_value != '-') return false;
  xnack_value = getFeatureValue(input, std::string(":xnack"));
  if (xnack_value != ' ' && xnack_value != '+' && xnack_value != '-') return false;
  return true;
}

static bool getTripleTargetID(std::string bundled_co_entry_id, const void* code_object,
                          std::string& co_triple_target_id) {
  std::string offload_kind = trimName(bundled_co_entry_id, '-');
  if (offload_kind != kOffloadKindHipv4 && offload_kind != kOffloadKindHip &&
      offload_kind != kOffloadKindHcc)
    return false;
  if (offload_kind != kOffloadKindHipv4)
    return getTripleTargetIDFromCodeObject(code_object, co_triple_target_id);

  // For code object V4 onwards the bundled code object entry ID correctly
  // specifies the target triple.
  co_triple_target_id = bundled_co_entry_id.substr(1);
  return true;
}

static bool isCodeObjectCompatibleWithDevice(std::string co_triple_target_id,
                std::string agent_triple_target_id, unsigned int genericVersion) {
  // Primitive Check
  if (co_triple_target_id == agent_triple_target_id) return true;

  // Parse code object triple target id
  if (!consume(co_triple_target_id, std::string(kAmdgcnTargetTriple) + '-')) {
    return false;
  }

  std::string co_processor;
  char co_sram_ecc, co_xnack;
  if (!getTargetIDValue(co_triple_target_id, co_processor, co_sram_ecc, co_xnack)) {
    return false;
  }

  if (!co_triple_target_id.empty()) return false;

  // Parse agent isa triple target id
  if (!consume(agent_triple_target_id, std::string(kAmdgcnTargetTriple) + '-')) {
    return false;
  }

  std::string agent_isa_processor;
  char isa_sram_ecc, isa_xnack;
  if (!getTargetIDValue(agent_triple_target_id, agent_isa_processor, isa_sram_ecc, isa_xnack)) {
    return false;
  }

  if (!agent_triple_target_id.empty()) return false;

  // Check for compatibility
  if (genericVersion >= EF_AMDGPU_GENERIC_VERSION_MIN) {
    // co_processor is generic target
    if (!helpers::IsCompatibleWithGenericTarget(co_processor, agent_isa_processor))
      return false;
  } else if (agent_isa_processor != co_processor) {
    return false;
  }
  if (co_sram_ecc != ' ') {
    if (co_sram_ecc != isa_sram_ecc) return false;
  }
  if (co_xnack != ' ') {
    if (co_xnack != isa_xnack) return false;
  }
  return true;
}

bool CodeObject::QueryGenericTarget(std::string agentTarget, std::string& processor,
		char& sram_ecc, char& xnack) {
  static const std::string head = std::string(kAmdgcnTargetTriple) + '-';
  // Parse agent isa triple target id
  if (!consume(agentTarget, head)) {
    return false;
  }
  if (!getTargetIDValue(agentTarget, processor, sram_ecc, xnack)) {
    return false;
  }
  if (processor.empty()) return false;
  auto &map = helpers::GenericTargetMapping();
  auto search = map.find(processor);
  if (search == map.end()) return false;
  processor = head + search->second;
  return true;
}

// This will be moved to COMGR eventually
hipError_t CodeObject::ExtractCodeObjectFromFile(
    amd::Os::FileDesc fdesc, size_t fsize, const void** image,
    const std::vector<std::string>& device_names,
    std::vector<std::pair<const void*, size_t>>& code_objs, size_t foffset) {
  if (!amd::Os::isValidFileDesc(fdesc)) {
    return hipErrorFileNotFound;
  }

  // Map the file to memory, with offset 0.
  // file will be unmapped in ModuleUnload
  // const void* image = nullptr;
  if (!amd::Os::MemoryMapFileDesc(fdesc, fsize, foffset, image)) {
    return hipErrorInvalidValue;
  }

  // retrieve code_objs{binary_image, binary_size} for devices
  return extractCodeObjectFromFatBinary(*image, device_names, code_objs);
}

// This will be moved to COMGR eventually
hipError_t CodeObject::ExtractCodeObjectFromMemory(
    const void* data, const std::vector<std::string>& device_names,
    std::vector<std::pair<const void*, size_t>>& code_objs, std::string& uri) {
  // Get the URI from memory
  if (!amd::Os::GetURIFromMemory(data, 0, uri)) {
    return hipErrorInvalidValue;
  }

  return extractCodeObjectFromFatBinary(data, device_names, code_objs);
}

// This will be moved to COMGR eventually
hipError_t CodeObject::extractCodeObjectFromFatBinary(
    const void* data, const std::vector<std::string>& agent_triple_target_ids,
    std::vector<std::pair<const void*, size_t>>& code_objs) {
  bool isCompressed = false;
  if (!IsClangOffloadMagicBundle(data, isCompressed) || isCompressed) {
    LogPrintfError("IsClangOffloadMagicBundle(%p) return false or isCompressed is true", data);
    return hipErrorInvalidKernelFile;
  }

  // Initialize Code objects
  code_objs.reserve(agent_triple_target_ids.size());
  for (size_t i = 0; i < agent_triple_target_ids.size(); i++) {
    code_objs.push_back(std::make_pair(nullptr, 0));
  }

  const auto obheader = reinterpret_cast<const __ClangOffloadBundleUncompressedHeader*>(data);
  const auto* desc = &obheader->desc[0];
  size_t num_code_objs = code_objs.size();
  for (uint64_t i = 0; i < obheader->numOfCodeObjects; ++i,
                desc = reinterpret_cast<const __ClangOffloadBundleInfo*>(
                    reinterpret_cast<uintptr_t>(&desc->bundleEntryId[0]) +
                    desc->bundleEntryIdSize)) {
    const void* image =
        reinterpret_cast<const void*>(reinterpret_cast<uintptr_t>(obheader) + desc->offset);
    const size_t image_size = desc->size;

    if (num_code_objs == 0) break;
    std::string bundleEntryId{desc->bundleEntryId, desc->bundleEntryIdSize};

    std::string co_triple_target_id;
    uint32_t genericVersion = getGenericVersion(image);
    if (!getTripleTargetID(bundleEntryId, image, co_triple_target_id)) continue;
    ClPrint(amd::LOG_INFO, amd::LOG_COMGR,
      "bundleEntryId=%s, co_triple_target_id=%s, genericVersion=%u\n",
      bundleEntryId.c_str(), co_triple_target_id.c_str(), genericVersion);

    for (size_t dev = 0; dev < agent_triple_target_ids.size(); ++dev) {
      if (code_objs[dev].first != nullptr) {
        if (!isGenericTarget(code_objs[dev].first)) {
          continue; // Specific target already found
        } else if(genericVersion >= EF_AMDGPU_GENERIC_VERSION_MIN) {
          continue; // Generic target already found, no need to check another generic
        }
      }
      if (isCodeObjectCompatibleWithDevice(co_triple_target_id, agent_triple_target_ids[dev],
        genericVersion)) {
        if (code_objs[dev].first == nullptr) --num_code_objs;
        code_objs[dev] = std::make_pair(image, image_size);
      }
    }
  }
  if (num_code_objs == 0) {
    return hipSuccess;
  } else {
    LogPrintfError("%s",
                   "hipErrorNoBinaryForGpu: Unable to find code object for all current devices!");
    LogPrintfError("%s", "  Devices:");
    for (size_t i = 0; i < agent_triple_target_ids.size(); i++) {
      LogPrintfError("    %s - [%s]", agent_triple_target_ids[i].c_str(),
                     ((code_objs[i].first) ? "Found" : "Not Found"));
    }
    const auto obheader = reinterpret_cast<const __ClangOffloadBundleUncompressedHeader*>(data);
    const auto* desc = &obheader->desc[0];
    LogPrintfError("%s", "  Bundled Code Objects:");
    for (uint64_t i = 0; i < obheader->numOfCodeObjects; ++i,
                  desc = reinterpret_cast<const __ClangOffloadBundleInfo*>(
                      reinterpret_cast<uintptr_t>(&desc->bundleEntryId[0]) +
                      desc->bundleEntryIdSize)) {
      std::string bundleEntryId{desc->bundleEntryId, desc->bundleEntryIdSize};
      const void* image =
          reinterpret_cast<const void*>(reinterpret_cast<uintptr_t>(obheader) + desc->offset);

      std::string co_triple_target_id;
      bool valid_co = getTripleTargetID(bundleEntryId, image, co_triple_target_id);

      if (valid_co) {
        LogPrintfError("    %s - [Code object targetID is %s]", bundleEntryId.c_str(),
                       co_triple_target_id.c_str());
      } else {
        LogPrintfError("    %s - [Unsupported]", bundleEntryId.c_str());
      }
    }
    return hipErrorNoBinaryForGpu;
  }
}

// ================================================================================================
size_t CodeObject::getFatbinSize(const void* data, const bool isCompressed) {
  if (isCompressed) {
    const auto obheader = reinterpret_cast<const __ClangOffloadBundleCompressedHeader*>(data);
    return obheader->totalSize;
  } else {
    const auto obheader = reinterpret_cast<const __ClangOffloadBundleUncompressedHeader*>(data);
    const __ClangOffloadBundleInfo* desc = &obheader->desc[0];
    uint64_t i = 0;
    while (++i < obheader->numOfCodeObjects) {
      desc = reinterpret_cast<const __ClangOffloadBundleInfo*>(
          reinterpret_cast<uintptr_t>(&desc->bundleEntryId[0]) + desc->bundleEntryIdSize);
    }
    return desc->offset + desc->size;
  }
}

// ================================================================================================
hipError_t CodeObject::extractCodeObjectFromFatBinaryUsingComgr(
    const void* data, size_t size, const std::vector<std::string>& agent_triple_target_ids,
    std::vector<std::pair<const void*, size_t>>& code_objs) {
  hipError_t hipStatus = hipSuccess;
  amd_comgr_status_t comgrStatus = AMD_COMGR_STATUS_SUCCESS;

  const size_t num_devices = agent_triple_target_ids.size();
  size_t num_code_objs = num_devices;
  bool isCompressed = false;
  if (!IsClangOffloadMagicBundle(data, isCompressed)) {
    ClPrint(amd::LOG_INFO, amd::LOG_COMGR, "IsClangOffloadMagicBundle(%p) return false", data);
    // hipModuleLoadData() will possibly call here
    return hipErrorInvalidKernelFile;
  }

  if (size == 0) size = getFatbinSize(data, isCompressed);

  amd_comgr_data_t dataCodeObj{0};
  amd_comgr_data_set_t dataSetBundled{0};
  amd_comgr_data_set_t dataSetUnbundled{0};
  amd_comgr_action_info_t actionInfoUnbundle{0};
  amd_comgr_data_t item{0};


  std::set<std::string> devicesSet{};  // To make sure device is unique
  std::set<std::string> genericDevicesSet{}; // Used to record generic targets

  std::vector<const char*> bundleEntryIDs{};
  static const std::string hipv4 = kOffloadKindHipv4_;  // bundled code objects need the prefix
  for (size_t i = 0; i < num_devices; i++) {
    auto res = devicesSet.insert(hipv4 + agent_triple_target_ids[i]);
    if (res.second) {
      // This is a new device in devicesSet
      bundleEntryIDs.push_back(res.first->c_str());
      std::string processor;
      char sram_ecc = ' ', xnack = ' ';
      if (!QueryGenericTarget(agent_triple_target_ids[i], processor, sram_ecc, xnack)) {
        continue; // No generic target for this device
      }
      // Now processor is generic such as
      // amdgcn-amd-amdhsa--gfx9-4-generic, amdgcn-amd-amdhsa--gfx11-generic
      processor = hipv4 + processor;
      auto ret = genericDevicesSet.insert(processor);
      if (ret.second) {
        // Without feature
        bundleEntryIDs.push_back(ret.first->c_str());
      }
      if (xnack != ' ') {
        ret = genericDevicesSet.insert(processor + ":xnack" + xnack);
        if (ret.second) {
          // Generic target with xnack feature
          bundleEntryIDs.push_back(ret.first->c_str());
        }
      }
      if (sram_ecc != ' ') {
        processor += ":sramecc";
        processor += sram_ecc;
        ret = genericDevicesSet.insert(processor);
        if (ret.second) {
          // Generic target with sramecc feature
          bundleEntryIDs.push_back(ret.first->c_str());
        }
        if (xnack != ' ') {
          ret = genericDevicesSet.insert(processor + ":xnack" + xnack);
          if (ret.second) {
            // Generic target with sramecc and xnack features
            bundleEntryIDs.push_back(ret.first->c_str());
          }
        }
      }
    }
  }

  do {
    // Create Bundled dataset
    comgrStatus = amd::Comgr::create_data_set(&dataSetBundled);
    if (comgrStatus != AMD_COMGR_STATUS_SUCCESS) {
      LogPrintfError("amd::Comgr::create_data_set() failed with status 0x%xh", comgrStatus);
      hipStatus = hipErrorInvalidValue;
      break;
    }

    // CodeObject
    comgrStatus = amd::Comgr::create_data(AMD_COMGR_DATA_KIND_OBJ_BUNDLE, &dataCodeObj);
    if (comgrStatus != AMD_COMGR_STATUS_SUCCESS) {
      LogPrintfError(
          "amd::Comgr::create_data(AMD_COMGR_DATA_KIND_OBJ_BUNDLE) failed with status 0x%xh",
          comgrStatus);
      hipStatus = hipErrorInvalidValue;
      break;
    }

    comgrStatus = amd::Comgr::set_data(dataCodeObj, size, static_cast<const char*>(data));
    if (comgrStatus != AMD_COMGR_STATUS_SUCCESS) {
      LogPrintfError("amd::Comgr::set_data(size=%zu, data=%p) failed with status 0x%xh", size, data,
                     comgrStatus);
      hipStatus = hipErrorInvalidValue;
      break;
    }

    comgrStatus = amd::Comgr::set_data_name(dataCodeObj, kHipFatBinName);
    if (comgrStatus != AMD_COMGR_STATUS_SUCCESS) {
      LogPrintfError(
          "amd::Comgr::set_data_name("
          ") failed with status 0x%xh",
          comgrStatus);
      hipStatus = hipErrorInvalidValue;
      break;
    }
    comgrStatus = amd::Comgr::data_set_add(dataSetBundled, dataCodeObj);
    if (comgrStatus != AMD_COMGR_STATUS_SUCCESS) {
      LogPrintfError("amd::Comgr::data_set_add() failed with status 0x%xh", comgrStatus);
      hipStatus = hipErrorInvalidValue;
      break;
    }
    // Set up ActionInfo
    comgrStatus = amd::Comgr::create_action_info(&actionInfoUnbundle);
    if (comgrStatus != AMD_COMGR_STATUS_SUCCESS) {
      LogPrintfError("amd::Comgr::create_action_info() failed with status 0x%xh", comgrStatus);
      hipStatus = hipErrorInvalidValue;
      break;
    }

    comgrStatus = amd::Comgr::action_info_set_language(actionInfoUnbundle, AMD_COMGR_LANGUAGE_HIP);
    if (comgrStatus != AMD_COMGR_STATUS_SUCCESS) {
      LogPrintfError("amd::Comgr::action_info_set_language(HIP) failed with status 0x%xh",
                     comgrStatus);
      hipStatus = hipErrorInvalidValue;
      break;
    }

    comgrStatus = amd::Comgr::action_info_set_bundle_entry_ids(
        actionInfoUnbundle, bundleEntryIDs.data(), bundleEntryIDs.size());
    if (comgrStatus != AMD_COMGR_STATUS_SUCCESS) {
      LogPrintfError(
          "amd::Comgr::action_info_set_bundle_entry_ids(%p, %zu) failed with status 0x%xh",
          bundleEntryIDs.data(), bundleEntryIDs.size(), comgrStatus);
      hipStatus = hipErrorInvalidValue;
      break;
    }

    // Unbundle
    comgrStatus = amd::Comgr::create_data_set(&dataSetUnbundled);
    if (comgrStatus != AMD_COMGR_STATUS_SUCCESS) {
      LogPrintfError("amd::Comgr::create_data_set(&dataSetUnbundled) failed with status 0x%xh",
                     comgrStatus);
      hipStatus = hipErrorInvalidValue;
      break;
    }
    comgrStatus = amd::Comgr::do_action(AMD_COMGR_ACTION_UNBUNDLE, actionInfoUnbundle,
                                        dataSetBundled, dataSetUnbundled);
    if (comgrStatus != AMD_COMGR_STATUS_SUCCESS) {
      LogPrintfError("amd::Comgr::do_action(AMD_COMGR_ACTION_UNBUNDLE) failed with status 0x%xh",
                     comgrStatus);
      hipStatus = hipErrorInvalidValue;
      break;
    }

    // Check CodeObject count
    size_t count = 0;
    comgrStatus =
        amd::Comgr::action_data_count(dataSetUnbundled, AMD_COMGR_DATA_KIND_EXECUTABLE, &count);
    if (comgrStatus != AMD_COMGR_STATUS_SUCCESS) {
      LogPrintfError("amd::Comgr::action_data_count() failed with status 0x%xh", comgrStatus);
      hipStatus = hipErrorInvalidValue;
      break;
    }

    // Initialize Code objects
    code_objs.reserve(num_code_objs);
    for (size_t i = 0; i < num_code_objs; i++) {
      code_objs.push_back(std::make_pair(nullptr, 0));
    }

    for (size_t i = 0; i < count; i++) {
      if (num_code_objs == 0) break;

      size_t itemSize = 0;
      comgrStatus = amd::Comgr::action_data_get_data(dataSetUnbundled,
                                                     AMD_COMGR_DATA_KIND_EXECUTABLE, i, &item);
      if (comgrStatus != AMD_COMGR_STATUS_SUCCESS) {
        LogPrintfError("amd::Comgr::action_data_get_data(%zu/%zu) failed with 0x%xh", i, count,
                       comgrStatus);
        hipStatus = hipErrorInvalidValue;
        break;
      }

      comgrStatus = amd::Comgr::get_data_name(item, &itemSize, nullptr);
      if (comgrStatus != AMD_COMGR_STATUS_SUCCESS) {
        LogPrintfError("amd::Comgr::get_data_name(%zu/%zu) failed with 0x%xh", i, count,
                       comgrStatus);
        hipStatus = hipErrorInvalidValue;
        break;
      }
      std::string bundleEntryId(itemSize, 0);
      comgrStatus = amd::Comgr::get_data_name(item, &itemSize, bundleEntryId.data());
      if (comgrStatus != AMD_COMGR_STATUS_SUCCESS) {
        LogPrintfError("amd::Comgr::get_data_name(%zu/%zu, %d) failed with 0x%xh", i, count,
                       itemSize, comgrStatus);
        hipStatus = hipErrorInvalidValue;
        break;
      }
      ClPrint(amd::LOG_DEBUG, amd::LOG_COMGR, "Found bundleEntryId=%s", bundleEntryId.c_str());

      // Remove bundleEntryId_
      if (!consume(bundleEntryId, kOffloadHipV4FatBinName_)) {
        // This is behavour in comgr unbundling which is subject to change.
        // So just give info.
        ClPrint(amd::LOG_INFO, amd::LOG_COMGR,
                "bundleEntryId=%s isn't prefixed with %s", bundleEntryId.c_str(),
                kOffloadHipV4FatBinName_);
      }
      trimNameTail(bundleEntryId, '.');  // Remove .fileExtention

      // Currently we only support EF_AMDGPU_GENERIC_VERSION_MIN on generic target
      uint32_t genericVersion =
          bundleEntryId.find("generic") != bundleEntryId.npos ? EF_AMDGPU_GENERIC_VERSION_MIN : 0;
      char* itemData = nullptr;
      for (size_t dev = 0; dev < num_devices; ++dev) {
        if (code_objs[dev].first != nullptr) {
          if (!isGenericTarget(code_objs[dev].first)) {
            continue;  // Specific target already found
          } else if (genericVersion >= EF_AMDGPU_GENERIC_VERSION_MIN) {
            continue;  // Generic target already found, no need to check another generic
          }
        }
        ClPrint(amd::LOG_DEBUG, amd::LOG_COMGR, "agent_triple_target_ids[%zu]=%s, bundleEntryId=%s",
                dev, agent_triple_target_ids[dev].c_str(), bundleEntryId.c_str());
        if (isCodeObjectCompatibleWithDevice(bundleEntryId, agent_triple_target_ids[dev],
                                             genericVersion)) {
          if (itemData == nullptr) {
            itemSize = 0;
            comgrStatus = amd::Comgr::get_data(item, &itemSize, nullptr);
            if (comgrStatus != AMD_COMGR_STATUS_SUCCESS) {
              LogPrintfError("amd::Comgr::get_data(%zu/%zu) failed with 0x%xh", i, count,
                             comgrStatus);
              hipStatus = hipErrorInvalidValue;
              break;
            }
            if (itemSize == 0) {
              // If there isn't a code object for this device,
              // amd::Comgr::do_action(AMD_COMGR_ACTION_UNBUNDLE) still returns item with
              // valid name but no data. We need continue searching for other devices
              ClPrint(amd::LOG_INFO, amd::LOG_COMGR,
                  "amd::Comgr::get_data() return 0 size for agent_triple_target_ids[%zu]=%s", dev,
                  agent_triple_target_ids[dev].c_str());
              break;
            }
            // itemData should be deleted in fatbin's destructor
            itemData = new char[itemSize];
            if (itemData == nullptr) {
              LogError("no enough memory");
              hipStatus = hipErrorOutOfMemory;
              break;
            }
            comgrStatus = amd::Comgr::get_data(item, &itemSize, itemData);
            if (comgrStatus != AMD_COMGR_STATUS_SUCCESS) {
              LogPrintfError("amd::Comgr::get_data(%zu/%zu, %d) failed with 0x%xh", i, count,
                             itemSize, comgrStatus);
              hipStatus = hipErrorInvalidValue;
              delete[] itemData;
              itemData = nullptr;
              break;
            }
          }
          if (code_objs[dev].first != nullptr) {
            // This must be data of generic target
            bool used  = false; // Still used by other devices?
            for (size_t i = 0; i < num_devices; ++i) {
              if (dev != i && code_objs[dev].first == code_objs[i].first) {
                used = true;
                break;
              }
            }
            if (!used) {
              delete[] reinterpret_cast<const char*>(code_objs[dev].first);
            }
          } else {
            --num_code_objs;
          }
          code_objs[dev] = std::make_pair(reinterpret_cast<const void*>(itemData), itemSize);
          ClPrint(amd::LOG_DEBUG, amd::LOG_COMGR,
              "Found agent_triple_target_ids[%zu]=%s: item: Data=%p(%s, %s), "
              "Size=%zu, num_code_objs=%zu",
              dev, agent_triple_target_ids[dev].c_str(), itemData,
              isCompressed ? "compressed" : "uncompressed",
              genericVersion >= EF_AMDGPU_GENERIC_VERSION_MIN ? "generic" : "non-generic",
              itemSize, num_code_objs);
        }
      }

      comgrStatus = amd::Comgr::release_data(item);
      item.handle = 0;
      if (comgrStatus != AMD_COMGR_STATUS_SUCCESS) {
        LogPrintfError("amd::Comgr::release_data(item) failed with status 0x%xh", comgrStatus);
        hipStatus = hipErrorInvalidValue;
      }
      if (hipStatus != hipSuccess) break;
    }
  } while (0);

  if (hipStatus == hipSuccess && num_code_objs != 0) {
    hipStatus = hipErrorNoBinaryForGpu;

    // Leave it for debug purpose in uncompressed mode.
    if (!isCompressed) {
      LogPrintfError("%s",
                     "hipErrorNoBinaryForGpu: Unable to find code object for all current devices!");
      LogPrintfError("%s", "  Devices:");
      for (size_t i = 0; i < agent_triple_target_ids.size(); i++) {
        LogPrintfError("    %s - [%s]", agent_triple_target_ids[i].c_str(),
                       ((code_objs[i].first) ? "Found" : "Not Found"));
      }
      const auto obheader = reinterpret_cast<const __ClangOffloadBundleUncompressedHeader*>(data);
      const auto* desc = &obheader->desc[0];
      LogPrintfError("%s", "  Bundled Code Objects:");
      for (uint64_t i = 0; i < obheader->numOfCodeObjects; ++i,
                    desc = reinterpret_cast<const __ClangOffloadBundleInfo*>(
                        reinterpret_cast<uintptr_t>(&desc->bundleEntryId[0]) +
                        desc->bundleEntryIdSize)) {
        std::string bundleEntryId{desc->bundleEntryId, desc->bundleEntryIdSize};
        const void* image =
            reinterpret_cast<const void*>(reinterpret_cast<uintptr_t>(obheader) + desc->offset);

        std::string co_triple_target_id;
        bool valid_co = getTripleTargetID(bundleEntryId, image, co_triple_target_id);
        if (valid_co) {
          LogPrintfError("    %s - [Code object targetID is %s]", bundleEntryId.c_str(),
                         co_triple_target_id.c_str());
        } else {
          LogPrintfError("    %s - [Unsupported]", bundleEntryId.c_str());
        }
      }
    }
  }

  // Cleanup
  if (actionInfoUnbundle.handle) {
    comgrStatus = amd::Comgr::destroy_action_info(actionInfoUnbundle);
    if (comgrStatus != AMD_COMGR_STATUS_SUCCESS) {
      LogPrintfError("amd::Comgr::destroy_action_info(actionInfoUnbundle) failed with status 0x%xh",
                     comgrStatus);
      hipStatus = hipErrorInvalidValue;
    }
  }
  if (dataSetBundled.handle) {
    comgrStatus = amd::Comgr::destroy_data_set(dataSetBundled);
    if (comgrStatus != AMD_COMGR_STATUS_SUCCESS) {
      LogPrintfError("amd::Comgr::destroy_data_set(dataSetBundled) failed with status 0x%xh",
                     comgrStatus);
      hipStatus = hipErrorInvalidValue;
    }
  }

  if (dataSetUnbundled.handle) {
    comgrStatus = amd::Comgr::destroy_data_set(dataSetUnbundled);
    if (comgrStatus != AMD_COMGR_STATUS_SUCCESS) {
      LogPrintfError("amd::Comgr::destroy_data_set(dataSetUnbundled) failed with status 0x%xh",
                     comgrStatus);
      hipStatus = hipErrorInvalidValue;
    }
  }

  if (dataCodeObj.handle) {
    comgrStatus = amd::Comgr::release_data(dataCodeObj);
    if (comgrStatus != AMD_COMGR_STATUS_SUCCESS) {
      LogPrintfError("amd::Comgr::release_data(dataCodeObj) failed with status 0x%xh", comgrStatus);
      hipStatus = hipErrorInvalidValue;
    }
  }

  if (item.handle) {
    comgrStatus = amd::Comgr::release_data(item);
    if (comgrStatus != AMD_COMGR_STATUS_SUCCESS) {
      LogPrintfError("amd::Comgr::release_data(item) failed with status 0x%xh", comgrStatus);
      hipStatus = hipErrorInvalidValue;
    }
  }

  return hipStatus;
}

hipError_t DynCO::loadCodeObject(const char* fname, const void* image) {
  amd::ScopedLock lock(dclock_);

  // Number of devices = 1 in dynamic code object
  fb_info_ = new FatBinaryInfo(fname, image);
  std::vector<hip::Device*> devices = {g_devices[ihipGetDevice()]};
  IHIP_RETURN_ONFAIL(fb_info_->ExtractFatBinary(devices));

  // No Lazy loading for DynCO
  IHIP_RETURN_ONFAIL(fb_info_->BuildProgram(ihipGetDevice()));

  module_ = fb_info_->Module(device_id_);

  // Define Global variables
  IHIP_RETURN_ONFAIL(populateDynGlobalVars());

  // Define Global functions
  IHIP_RETURN_ONFAIL(populateDynGlobalFuncs());

  return hipSuccess;
}

// Dynamic Code Object
DynCO::~DynCO() {
  amd::ScopedLock lock(dclock_);

  for (auto& elem : vars_) {
    if (elem.second->getVarKind() == Var::DVK_Managed) {
      hipError_t err = ihipFree(elem.second->getManagedVarPtr());
      assert(err == hipSuccess);
    }

    if (elem.second->getVarKind() == Var::DVK_Variable) {
      for (auto dev : g_devices) {
        DeviceVar* dvar = nullptr;
        hipError_t err = elem.second->getDeviceVarPtr(&dvar, dev->deviceId());
        assert(err == hipSuccess);
        if (dvar != nullptr) {
          // free also deletes the device ptr
          err = ihipFree(dvar->device_ptr());
          assert(err == hipSuccess);
        }
      }
    }
    delete elem.second;
  }
  vars_.clear();

  for (auto& elem : functions_) {
    delete elem.second;
  }
  functions_.clear();

  delete fb_info_;
}

hipError_t DynCO::getDeviceVar(DeviceVar** dvar, std::string var_name) {
  amd::ScopedLock lock(dclock_);

  auto it = vars_.find(var_name);
  if (it == vars_.end()) {
    LogPrintfError("Cannot find the Var: %s ", var_name.c_str());
    return hipErrorNotFound;
  }

  hipError_t err = it->second->getDeviceVar(dvar, device_id_, module_);
  return err;
}

hipError_t DynCO::getDynFunc(hipFunction_t* hfunc, std::string func_name) {
  amd::ScopedLock lock(dclock_);

  if (hfunc == nullptr) {
    return hipErrorInvalidValue;
  }

  auto it = functions_.find(func_name);
  if (it == functions_.end()) {
    LogPrintfError("Cannot find the function: %s ", func_name.c_str());
    return hipErrorNotFound;
  }

  /* See if this could be solved */
  return it->second->getDynFunc(hfunc, module_);
}

bool DynCO::isValidDynFunc(const void* hfunc) {
  amd::ScopedLock lock(dclock_);
  return std::any_of(functions_.begin(), functions_.end(),
                     [&](auto& it) { return it.second->isValidDynFunc(hfunc); });
}

hipError_t DynCO::initDynManagedVars(const std::string& managedVar) {
  amd::ScopedLock lock(dclock_);
  DeviceVar* dvar;
  void* pointer = nullptr;
  hipError_t status = hipSuccess;
  // To get size of the managed variable
  status = getDeviceVar(&dvar, managedVar + ".managed");
  if (status != hipSuccess) {
    ClPrint(amd::LOG_ERROR, amd::LOG_API, "Status %d, failed to get .managed device variable:%s",
            status, managedVar.c_str());
    return status;
  }
  // Allocate managed memory for these symbols
  status = ihipMallocManaged(&pointer, dvar->size());
  guarantee(status == hipSuccess, "Status %d, failed to allocate managed memory", status);

  // update as manager variable and set managed memory pointer and size
  auto it = vars_.find(managedVar);
  it->second->setManagedVarInfo(pointer, dvar->size());

  // copy initial value to the managed variable to the managed memory allocated
  hip::Stream* stream = hip::getNullStream();
  if (stream != nullptr) {
    status = ihipMemcpy(pointer, reinterpret_cast<address>(dvar->device_ptr()), dvar->size(),
                        hipMemcpyDeviceToDevice, *stream);
    if (status != hipSuccess) {
      ClPrint(amd::LOG_ERROR, amd::LOG_API, "Status %d, failed to copy device ptr:%s", status,
              managedVar.c_str());
      return status;
    }
  } else {
    ClPrint(amd::LOG_ERROR, amd::LOG_API, "Host Queue is NULL");
    return hipErrorInvalidResourceHandle;
  }

  // Get deivce ptr to initialize with managed memory pointer
  status = getDeviceVar(&dvar, managedVar);
  if (status != hipSuccess) {
    ClPrint(amd::LOG_ERROR, amd::LOG_API, "Status %d, failed to get managed device variable:%s",
            status, managedVar.c_str());
    return status;
  }
  // copy managed memory pointer to the managed device variable
  status = ihipMemcpy(reinterpret_cast<address>(dvar->device_ptr()), &pointer, dvar->size(),
                      hipMemcpyHostToDevice, *stream);
  if (status != hipSuccess) {
    ClPrint(amd::LOG_ERROR, amd::LOG_API, "Status %d, failed to copy device ptr:%s", status,
            managedVar.c_str());
    return status;
  }
  return status;
}

hipError_t DynCO::populateDynGlobalVars() {
  amd::ScopedLock lock(dclock_);
  hipError_t err = hipSuccess;
  std::vector<std::string> var_names;
  std::string managedVarExt = ".managed";
  // For Dynamic Modules there is only one hipFatBinaryDevInfo_
  device::Program* dev_program = fb_info_->GetProgram(ihipGetDevice())
                                     ->getDeviceProgram(*hip::getCurrentDevice()->devices()[0]);

  if (!dev_program->getGlobalVarFromCodeObj(&var_names)) {
    LogPrintfError("Could not get Global vars from Code Obj for Module: 0x%x", module_);
    return hipErrorSharedObjectSymbolNotFound;
  }

  for (auto& elem : var_names) {
    vars_.insert(
        std::make_pair(elem, new Var(elem, Var::DeviceVarKind::DVK_Variable, 0, 0, 0, nullptr)));
  }

  for (auto& elem : var_names) {
    if (elem.find(managedVarExt) != std::string::npos) {
      std::string managedVar = elem;
      managedVar.erase(managedVar.length() - managedVarExt.length(), managedVarExt.length());
      err = initDynManagedVars(managedVar);
    }
  }
  return err;
}

hipError_t DynCO::populateDynGlobalFuncs() {
  amd::ScopedLock lock(dclock_);

  std::vector<std::string> func_names;
  device::Program* dev_program = fb_info_->GetProgram(ihipGetDevice())
                                     ->getDeviceProgram(*hip::getCurrentDevice()->devices()[0]);

  // Get all the global func names from COMGR
  if (!dev_program->getGlobalFuncFromCodeObj(&func_names)) {
    LogPrintfError("Could not get Global Funcs from Code Obj for Module: 0x%x", module_);
    return hipErrorSharedObjectSymbolNotFound;
  }

  for (auto& elem : func_names) {
    functions_.insert(std::make_pair(elem, new Function(elem)));
  }

  return hipSuccess;
}

// Static Code Object
StatCO::StatCO() {}

StatCO::~StatCO() {
  amd::ScopedLock lock(sclock_);

  for (auto& elem : functions_) {
    delete elem.second;
  }
  functions_.clear();

  for (auto& elem : vars_) {
    delete elem.second;
  }
  vars_.clear();
}

hipError_t StatCO::digestFatBinary(const void* data, FatBinaryInfo*& programs) {
  amd::ScopedLock lock(sclock_);

  if (programs != nullptr) {
    return hipSuccess;
  }

  // Create a new fat binary object and extract the fat binary for all devices.
  FatBinaryInfo* fatBinaryInfo = new FatBinaryInfo(nullptr, data);
  hipError_t err = fatBinaryInfo->ExtractFatBinary(g_devices);
  programs = fatBinaryInfo;
  return err;
}

FatBinaryInfo** StatCO::addFatBinary(const void* data, bool initialized, bool& success) {
  amd::ScopedLock lock(sclock_);
  module_to_hostModule_.insert(std::make_pair(&modules_[data], data));

  if (initialized == false) {
    success = true;
    return &modules_[data];
  }

  hipError_t err = digestFatBinary(data, modules_[data]);

  success = (err == hipSuccess);
  return &modules_[data];
}

hipError_t StatCO::removeFatBinary(FatBinaryInfo** module) {
  amd::ScopedLock lock(sclock_);

  auto hostVarsIter = module_to_hostVars_.find(module);
  if (hostVarsIter != module_to_hostVars_.end()) {
    for (auto& hostVar : hostVarsIter->second) {
      auto varIter = vars_.find(hostVar);
      if (varIter == vars_.end()) {
        LogPrintfError(
          "removeFatBinary: Unable to find module 0x%x hostVar 0x%x",
          module, hostVar);
      } else {
        delete varIter->second;
        vars_.erase(varIter);
      }
    }
    module_to_hostVars_.erase(hostVarsIter);
  }

  auto managedVarsIter = managedVars_.find(module);
  if (managedVarsIter != managedVars_.end()) {
    for (auto& managedVar : managedVarsIter->second) {
      hipError_t err;
      for (auto dev : g_devices) {
        DeviceVar* dvar = nullptr;
        IHIP_RETURN_ONFAIL(managedVar->getDeviceVarPtr(&dvar, dev->deviceId()));
        if (dvar != nullptr) {
          // free also deletes the device ptr
          err = ihipFree(dvar->device_ptr());
          assert(err == hipSuccess);
        }
      }
      err = ihipFree(*(static_cast<void**>(managedVar->getManagedVarPtr())));
      assert(err == hipSuccess);
      delete managedVar;
    }
    managedVars_.erase(managedVarsIter);
  }

  auto hostFuncsIter = module_to_hostFunctions_.find(module);
  if (hostFuncsIter != module_to_hostFunctions_.end()) {
    for (auto& hostFunc : hostFuncsIter->second) {
      auto funcIter = functions_.find(hostFunc);
      if (funcIter == functions_.end()) {
        LogPrintfError("removeFatBinary: Unable to find module 0x%x hostFunc 0x%x",
                       module, hostFunc);
      } else {
        delete funcIter->second;
        functions_.erase(funcIter);
      }
    }
    module_to_hostFunctions_.erase(hostFuncsIter);
  }

  auto hostModuleIter = module_to_hostModule_.find(module);
  if (hostModuleIter != module_to_hostModule_.end()) {
    auto hostModule = hostModuleIter->second;
    auto moduleIter = modules_.find(hostModule);
    if (moduleIter != modules_.end()) {
      delete moduleIter->second;
      modules_.erase(moduleIter);
    } else {
      LogPrintfError("removeFatBinary: Unable to find module 0x%x via hostModule 0x%x",
                    module, hostModule);
    }
    module_to_hostModule_.erase(hostModuleIter);
  }

  return hipSuccess;
}

hipError_t StatCO::registerStatFunction(const void* hostFunction, Function* func) {
  amd::ScopedLock lock(sclock_);

  if (functions_.find(hostFunction) != functions_.end()) {
    DevLogPrintfError("hostFunctionPtr: 0x%x already exists", hostFunction);
    delete func;
  } else {
    functions_.insert(std::make_pair(hostFunction, func));
    module_to_hostFunctions_[func->moduleInfo()].push_back(hostFunction);
  }

  return hipSuccess;
}

const char* StatCO::getStatFuncName(const void* hostFunction) {
  amd::ScopedLock lock(sclock_);

  const auto it = functions_.find(hostFunction);
  if (it == functions_.end()) {
    return nullptr;
  }
  return it->second->name().c_str();
}

hipError_t StatCO::getStatFunc(hipFunction_t* hfunc, const void* hostFunction, int deviceId) {
  const auto it = functions_.find(hostFunction);
  if (it == functions_.end()) {
    return hipErrorInvalidSymbol;
  }

  // Lazy load
  FatBinaryInfo **module = it->second->moduleInfo();
  if (*(module) == nullptr) {
    amd::ScopedLock lock(sclock_);
    if (*(module) == nullptr) {
      hipError_t err = digestFatBinary(module_to_hostModule_[module], *module);
      assert(err == hipSuccess);
    }
  }

  return it->second->getStatFunc(hfunc, deviceId);
}

hipError_t StatCO::getStatFuncAttr(hipFuncAttributes* func_attr, const void* hostFunction,
                                   int deviceId) {
  amd::ScopedLock lock(sclock_);

  const auto it = functions_.find(hostFunction);
  if (it == functions_.end()) {
    return hipErrorInvalidSymbol;
  }

  // Lazy load
  FatBinaryInfo **module = it->second->moduleInfo();
  if (*(module) == nullptr) {
    hipError_t err = digestFatBinary(module_to_hostModule_[module], *module);
    assert(err == hipSuccess);
  }

  return it->second->getStatFuncAttr(func_attr, deviceId);
}

hipError_t StatCO::registerStatGlobalVar(const void* hostVar, Var* var) {
  amd::ScopedLock lock(sclock_);

  auto var_it = vars_.find(hostVar);
  if ((var_it != vars_.end()) && (var_it->second->getName() != var->getName())) {
    return hipErrorInvalidSymbol;
  }

  vars_.insert(std::make_pair(hostVar, var));
  module_to_hostVars_[var->moduleInfo()].push_back(hostVar);
  return hipSuccess;
}

hipError_t StatCO::getStatGlobalVar(const void* hostVar, int deviceId, hipDeviceptr_t* dev_ptr,
                                    size_t* size_ptr) {
  amd::ScopedLock lock(sclock_);

  const auto it = vars_.find(hostVar);
  if (it == vars_.end()) {
    return hipErrorInvalidSymbol;
  }

  // Lazy load
  FatBinaryInfo **module = it->second->moduleInfo();
  if (*(module) == nullptr) {
    hipError_t err = digestFatBinary(module_to_hostModule_[module], *module);
    assert(err == hipSuccess);
  }

  DeviceVar* dvar = nullptr;
  IHIP_RETURN_ONFAIL(it->second->getStatDeviceVar(&dvar, deviceId));

  *dev_ptr = dvar->device_ptr();
  *size_ptr = dvar->size();
  return hipSuccess;
}

hipError_t StatCO::registerStatManagedVar(Var* var) {
  managedVars_[var->moduleInfo()].push_back(var);
  return hipSuccess;
}

hipError_t StatCO::initStatManagedVarDevicePtr(int deviceId) {
  amd::ScopedLock lock(sclock_);
  hipError_t err = hipSuccess;
  if (managedVarsDevicePtrInitalized_.find(deviceId) == managedVarsDevicePtrInitalized_.end() ||
      !managedVarsDevicePtrInitalized_[deviceId]) {
    for (auto& vecIter : managedVars_) {
      for (auto& var : vecIter.second) {
        // Lazy load
        FatBinaryInfo **module = var->moduleInfo();
        if (*(module) == nullptr) {
          hipError_t err = digestFatBinary(module_to_hostModule_[module], *module);
          assert(err == hipSuccess);
        }

        DeviceVar* dvar = nullptr;
        IHIP_RETURN_ONFAIL(var->getStatDeviceVar(&dvar, deviceId));

        hip::Stream* stream = g_devices.at(deviceId)->NullStream();
        if (stream != nullptr) {
          err = ihipMemcpy(reinterpret_cast<address>(dvar->device_ptr()), var->getManagedVarPtr(),
                           dvar->size(), hipMemcpyHostToDevice, *stream);
        } else {
          ClPrint(amd::LOG_ERROR, amd::LOG_API, "Host Queue is NULL");
          return hipErrorInvalidResourceHandle;
        }
      }
    }
    managedVarsDevicePtrInitalized_[deviceId] = true;
  }
  return err;
}
}  // namespace hip
