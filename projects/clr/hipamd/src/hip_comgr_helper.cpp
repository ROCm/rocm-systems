/*
Copyright (c) 2022 - Present Advanced Micro Devices, Inc. All rights reserved.

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
#include "hip_comgr_helper.hpp"
#if defined(_WIN32)
#include <io.h>
#endif
#include "../src/amd_hsa_elf.hpp"

namespace hip {
std::unordered_set<LinkProgram*> LinkProgram::linker_set_;

namespace helpers {

size_t constexpr strLiteralLength(char const* str) {
  return *str ? 1 + strLiteralLength(str + 1) : 0;
}

constexpr char const* CLANG_OFFLOAD_BUNDLER_MAGIC_STR = "__CLANG_OFFLOAD_BUNDLE__";
constexpr char const* OFFLOAD_KIND_HIP = "hip";
constexpr char const* OFFLOAD_KIND_HIPV4 = "hipv4";
constexpr char const* OFFLOAD_KIND_HCC = "hcc";
constexpr char const* AMDGCN_TARGET_TRIPLE = "amdgcn-amd-amdhsa-";
constexpr char const* SPIRV_BUNDLE_ENTRY_ID = "hip-spirv64-amd-amdhsa-unknown-amdgcnspirv";

static constexpr size_t bundle_magic_string_size =
    strLiteralLength(CLANG_OFFLOAD_BUNDLER_MAGIC_STR);

struct __ClangOffloadBundleInfo {
  uint64_t offset;
  uint64_t size;
  uint64_t bundleEntryIdSize;
  const char bundleEntryId[1];
};

struct __ClangOffloadBundleHeader {
  const char magic[bundle_magic_string_size - 1];
  uint64_t numOfCodeObjects;
  __ClangOffloadBundleInfo desc[1];
};

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
// input will become sram-ecc+.
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

bool isCodeObjectCompatibleWithDevice(std::string co_triple_target_id,
                                      std::string agent_triple_target_id,
                                      unsigned& genericVersion) {
  // Primitive Check
  if (co_triple_target_id == agent_triple_target_id) return true;

  // Parse code object triple target id
  if (!consume(co_triple_target_id,
               std::string(OFFLOAD_KIND_HIP) + "-" + std::string(AMDGCN_TARGET_TRIPLE))) {
    return false;
  }

  std::string co_processor;
  char co_sram_ecc, co_xnack;
  if (!getTargetIDValue(co_triple_target_id, co_processor, co_sram_ecc, co_xnack)) {
    return false;
  }

  if (!co_triple_target_id.empty()) return false;

  // Parse agent isa triple target id
  if (!consume(agent_triple_target_id, std::string(AMDGCN_TARGET_TRIPLE) + '-')) {
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
    if (!IsCompatibleWithGenericTarget(co_processor, agent_isa_processor)) return false;
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

static inline unsigned int getGenericVersion(const void* image) {
  const Elf64_Ehdr* ehdr = reinterpret_cast<const Elf64_Ehdr*>(image);
  return ehdr->e_ident[EI_ABIVERSION] == ELFABIVERSION_AMDGPU_HSA_V6
             ? ((ehdr->e_flags & EF_AMDGPU_GENERIC_VERSION) >> EF_AMDGPU_GENERIC_VERSION_OFFSET)
             : 0;
}

static inline bool isGenericTarget(const void* image) {
  return getGenericVersion(image) >= EF_AMDGPU_GENERIC_VERSION_MIN;
}

bool UnbundleBitCode(const std::vector<char>& bundled_llvm_bitcode, const std::string& isa,
                     size_t& co_offset, size_t& co_size) {
  std::string magic(bundled_llvm_bitcode.begin(),
                    bundled_llvm_bitcode.begin() + bundle_magic_string_size);
  if (magic.compare(CLANG_OFFLOAD_BUNDLER_MAGIC_STR)) {
    // Handle case where the whole file is unbundled
    return true;
  }

  std::string bundled_llvm_bitcode_s(bundled_llvm_bitcode.begin(),
                                     bundled_llvm_bitcode.begin() + bundled_llvm_bitcode.size());
  const void* data = reinterpret_cast<const void*>(bundled_llvm_bitcode_s.c_str());
  const auto obheader = reinterpret_cast<const __ClangOffloadBundleHeader*>(data);
  const auto* desc = &obheader->desc[0];
  for (uint64_t idx = 0; idx < obheader->numOfCodeObjects;
       ++idx, desc = reinterpret_cast<const __ClangOffloadBundleInfo*>(
                  reinterpret_cast<uintptr_t>(&desc->bundleEntryId[0]) + desc->bundleEntryIdSize)) {
    const void* image =
        reinterpret_cast<const void*>(reinterpret_cast<uintptr_t>(obheader) + desc->offset);
    const size_t image_size = desc->size;
    std::string bundleEntryId{desc->bundleEntryId, desc->bundleEntryIdSize};

    // Check if the device id and code object id are compatible
    unsigned genericVersion = getGenericVersion(image);
    if (isCodeObjectCompatibleWithDevice(bundleEntryId, isa, genericVersion)) {
      co_offset = (reinterpret_cast<uintptr_t>(image) - reinterpret_cast<uintptr_t>(data));
      co_size = image_size;
      break;
    }
  }
  return true;
}

bool addCodeObjData(amd_comgr_data_set_t& input, const std::vector<char>& source,
                    const std::string& name, const amd_comgr_data_kind_t type) {
  amd_comgr_data_t data;

  if (auto res = amd::Comgr::create_data(type, &data); res != AMD_COMGR_STATUS_SUCCESS) {
    return false;
  }

  if (auto res = amd::Comgr::set_data(data, source.size(), source.data());
      res != AMD_COMGR_STATUS_SUCCESS) {
    amd::Comgr::release_data(data);
    return false;
  }

  if (auto res = amd::Comgr::set_data_name(data, name.c_str()); res != AMD_COMGR_STATUS_SUCCESS) {
    amd::Comgr::release_data(data);
    return false;
  }

  if (auto res = amd::Comgr::data_set_add(input, data); res != AMD_COMGR_STATUS_SUCCESS) {
    amd::Comgr::release_data(data);
    return false;
  }
  amd::Comgr::release_data(data);  // Release from our end after setting the input

  return true;
}

bool extractBuildLog(amd_comgr_data_set_t dataSet, std::string& buildLog) {
  size_t count;
  if (auto res = amd::Comgr::action_data_count(dataSet, AMD_COMGR_DATA_KIND_LOG, &count);
      res != AMD_COMGR_STATUS_SUCCESS) {
    return false;
  }

  std::vector<char> log;
  if (count > 0) {
    if (!extractByteCodeBinary(dataSet, AMD_COMGR_DATA_KIND_LOG, log)) return false;
    buildLog.insert(buildLog.end(), log.data(), log.data() + log.size());
  }
  return true;
}

bool extractByteCodeBinary(const amd_comgr_data_set_t inDataSet,
                           const amd_comgr_data_kind_t dataKind, std::vector<char>& bin) {
  amd_comgr_data_t binaryData;

  if (auto res = amd::Comgr::action_data_get_data(inDataSet, dataKind, 0, &binaryData);
      res != AMD_COMGR_STATUS_SUCCESS) {
    return false;
  }

  size_t binarySize = 0;
  if (auto res = amd::Comgr::get_data(binaryData, &binarySize, NULL);
      res != AMD_COMGR_STATUS_SUCCESS) {
    amd::Comgr::release_data(binaryData);
    return false;
  }

  size_t bufSize = (dataKind == AMD_COMGR_DATA_KIND_LOG) ? binarySize + 1 : binarySize;

  char* binary = new char[bufSize];
  if (binary == nullptr) {
    amd::Comgr::release_data(binaryData);
    return false;
  }


  if (auto res = amd::Comgr::get_data(binaryData, &binarySize, binary);
      res != AMD_COMGR_STATUS_SUCCESS) {
    delete[] binary;
    amd::Comgr::release_data(binaryData);
    return false;
  }

  if (dataKind == AMD_COMGR_DATA_KIND_LOG) {
    binary[binarySize] = '\0';
  }

  amd::Comgr::release_data(binaryData);

  std::vector<char> temp_bin;
  temp_bin.assign(binary, binary + binarySize);
  bin = temp_bin;
  delete[] binary;

  return true;
}

bool createAction(amd_comgr_action_info_t& action, std::vector<std::string>& options,
                  const std::string& isa, const amd_comgr_language_t lang) {
  if (auto res = amd::Comgr::create_action_info(&action); res != AMD_COMGR_STATUS_SUCCESS) {
    return false;
  }

  if (lang != AMD_COMGR_LANGUAGE_NONE) {
    if (auto res = amd::Comgr::action_info_set_language(action, lang);
        res != AMD_COMGR_STATUS_SUCCESS) {
      amd::Comgr::destroy_action_info(action);
      return false;
    }
  }

  if (auto res = amd::Comgr::action_info_set_isa_name(action, isa.c_str());
      res != AMD_COMGR_STATUS_SUCCESS) {
    amd::Comgr::destroy_action_info(action);
    return false;
  }

  std::vector<const char*> optionsArgv;
  optionsArgv.reserve(options.size());
  for (auto& option : options) {
    optionsArgv.push_back(option.c_str());
  }

  if (auto res =
          amd::Comgr::action_info_set_option_list(action, optionsArgv.data(), optionsArgv.size());
      res != AMD_COMGR_STATUS_SUCCESS) {
    amd::Comgr::destroy_action_info(action);
    return false;
  }

  if (auto res = amd::Comgr::action_info_set_logging(action, true);
      res != AMD_COMGR_STATUS_SUCCESS) {
    amd::Comgr::destroy_action_info(action);
    return false;
  }

  return true;
}

bool compileToExecutable(const amd_comgr_data_set_t compileInputs, const std::string& isa,
                         std::vector<std::string>& compileOptions,
                         std::vector<std::string>& linkOptions, std::string& buildLog,
                         std::vector<char>& exe) {
  amd_comgr_language_t lang = AMD_COMGR_LANGUAGE_HIP;
  amd_comgr_action_info_t action;
  amd_comgr_data_set_t reloc;
  amd_comgr_data_set_t output;
  amd_comgr_data_set_t input = compileInputs;

  if (!createAction(action, compileOptions, isa, lang)) {
    return false;
  }

  if (auto res = amd::Comgr::create_data_set(&reloc); res != AMD_COMGR_STATUS_SUCCESS) {
    amd::Comgr::destroy_action_info(action);
    return false;
  }

  if (auto res = amd::Comgr::create_data_set(&output); res != AMD_COMGR_STATUS_SUCCESS) {
    amd::Comgr::destroy_action_info(action);
    amd::Comgr::destroy_data_set(reloc);
    return false;
  }

  if (auto res = amd::Comgr::do_action(AMD_COMGR_ACTION_COMPILE_SOURCE_TO_RELOCATABLE, action,
                                       input, reloc);
      res != AMD_COMGR_STATUS_SUCCESS) {
    extractBuildLog(reloc, buildLog);
    amd::Comgr::destroy_action_info(action);
    amd::Comgr::destroy_data_set(reloc);
    amd::Comgr::destroy_data_set(output);
    return false;
  }

  if (!extractBuildLog(reloc, buildLog)) {
    amd::Comgr::destroy_action_info(action);
    amd::Comgr::destroy_data_set(reloc);
    amd::Comgr::destroy_data_set(output);
    return false;
  }

  amd::Comgr::destroy_action_info(action);
  if (!createAction(action, linkOptions, isa, lang)) {
    amd::Comgr::destroy_action_info(action);
    amd::Comgr::destroy_data_set(reloc);
    amd::Comgr::destroy_data_set(output);
    return false;
  }

  if (auto res = amd::Comgr::do_action(AMD_COMGR_ACTION_LINK_RELOCATABLE_TO_EXECUTABLE, action,
                                       reloc, output);
      res != AMD_COMGR_STATUS_SUCCESS) {
    extractBuildLog(output, buildLog);
    amd::Comgr::destroy_action_info(action);
    amd::Comgr::destroy_data_set(output);
    amd::Comgr::destroy_data_set(reloc);
    return false;
  }

  if (!extractBuildLog(output, buildLog)) {
    amd::Comgr::destroy_action_info(action);
    amd::Comgr::destroy_data_set(output);
    amd::Comgr::destroy_data_set(reloc);
    return false;
  }

  if (!extractByteCodeBinary(output, AMD_COMGR_DATA_KIND_EXECUTABLE, exe)) {
    amd::Comgr::destroy_action_info(action);
    amd::Comgr::destroy_data_set(output);
    amd::Comgr::destroy_data_set(reloc);
    return false;
  }

  // Clean up
  amd::Comgr::destroy_action_info(action);
  amd::Comgr::destroy_data_set(output);
  amd::Comgr::destroy_data_set(reloc);
  return true;
}

bool compileToBitCode(const amd_comgr_data_set_t compileInputs, const std::string& isa,
                      std::vector<std::string>& compileOptions, std::string& buildLog,
                      std::vector<char>& LLVMBitcode) {
  amd_comgr_language_t lang = AMD_COMGR_LANGUAGE_HIP;
  amd_comgr_action_info_t action;
  amd_comgr_data_set_t output;
  amd_comgr_data_set_t input = compileInputs;

  if (!createAction(action, compileOptions, isa, lang)) {
    return false;
  }

  if (auto res = amd::Comgr::create_data_set(&output); res != AMD_COMGR_STATUS_SUCCESS) {
    amd::Comgr::destroy_action_info(action);
    return false;
  }

  if (auto res = amd::Comgr::do_action(AMD_COMGR_ACTION_COMPILE_SOURCE_WITH_DEVICE_LIBS_TO_BC,
                                       action, input, output);
      res != AMD_COMGR_STATUS_SUCCESS) {
    extractBuildLog(output, buildLog);
    amd::Comgr::destroy_action_info(action);
    amd::Comgr::destroy_data_set(output);
    return false;
  }

  if (!extractBuildLog(output, buildLog)) {
    amd::Comgr::destroy_action_info(action);
    amd::Comgr::destroy_data_set(output);
    return false;
  }

  if (!extractByteCodeBinary(output, AMD_COMGR_DATA_KIND_BC, LLVMBitcode)) {
    amd::Comgr::destroy_action_info(action);
    amd::Comgr::destroy_data_set(output);
    return false;
  }

  // Clean up
  amd::Comgr::destroy_action_info(action);
  amd::Comgr::destroy_data_set(output);
  return true;
}

bool CheckIfBundled(std::vector<char>& llvm_bitcode) {
  std::string magic(llvm_bitcode.begin(), llvm_bitcode.begin() + bundle_magic_string_size);

  if (magic.compare(CLANG_OFFLOAD_BUNDLER_MAGIC_STR) == 0) {
    return true;
  }
  // File is not bundled
  return false;
}
// Unbundle Bitcode using COMGR action
// Supports only 1 Bundle Entry ID for now
bool UnbundleUsingComgr(std::vector<char>& source, const std::string& isa,
                        std::vector<std::string>& linkOptions, std::string& buildLog,
                        std::vector<char>& unbundled_bitcode, const char* bundleEntryIDs[],
                        size_t bundleEntryIDsCount) {
  amd_comgr_data_set_t linkinput;
  if (amd::Comgr::create_data_set(&linkinput) != AMD_COMGR_STATUS_SUCCESS) {
    return false;
  }
  std::string name = "UnbundleCode.bc";
  if (!helpers::addCodeObjData(linkinput, source, name, AMD_COMGR_DATA_KIND_BC_BUNDLE)) {
    return false;
  }

  amd_comgr_action_info_t action;
  if (!createAction(action, linkOptions, isa, AMD_COMGR_LANGUAGE_NONE)) {
    return false;
  }

  if (bundleEntryIDsCount > 1) {
    LogError("Error in hip Linker : bundleEntryID count > 1");
    return false;
  }

  if (amd::Comgr::action_info_set_bundle_entry_ids(action, bundleEntryIDs, bundleEntryIDsCount) !=
      AMD_COMGR_STATUS_SUCCESS) {
    amd::Comgr::destroy_action_info(action);
    return false;
  }

  amd_comgr_data_set_t output;
  if (amd::Comgr::create_data_set(&output) != AMD_COMGR_STATUS_SUCCESS) {
    amd::Comgr::destroy_action_info(action);
    return false;
  }

  if (auto res = amd::Comgr::do_action(AMD_COMGR_ACTION_UNBUNDLE, action, linkinput, output);
      res != AMD_COMGR_STATUS_SUCCESS) {
    amd::Comgr::destroy_action_info(action);
    amd::Comgr::destroy_data_set(output);
    return false;
  }

  if (!extractBuildLog(output, buildLog)) {
    amd::Comgr::destroy_action_info(action);
    amd::Comgr::destroy_data_set(output);
    return false;
  }

  if (!extractByteCodeBinary(output, AMD_COMGR_DATA_KIND_BC, unbundled_bitcode)) {
    amd::Comgr::destroy_action_info(action);
    amd::Comgr::destroy_data_set(output);
    return false;
  }

  amd::Comgr::destroy_action_info(action);
  amd::Comgr::destroy_data_set(output);
  amd::Comgr::destroy_data_set(linkinput);
  return true;
}

bool linkLLVMBitcode(const amd_comgr_data_set_t linkInputs, const std::string& isa,
                     std::vector<std::string>& linkOptions, std::string& buildLog,
                     std::vector<char>& LinkedLLVMBitcode) {
  const amd_comgr_language_t lang = AMD_COMGR_LANGUAGE_HIP;
  amd_comgr_action_info_t action;

  if (!createAction(action, linkOptions, isa, lang)) {
    return false;
  }

  amd_comgr_data_set_t output;
  if (auto res = amd::Comgr::create_data_set(&output); res != AMD_COMGR_STATUS_SUCCESS) {
    amd::Comgr::destroy_action_info(action);
    return false;
  }

  if (auto res = amd::Comgr::do_action(AMD_COMGR_ACTION_LINK_BC_TO_BC, action, linkInputs, output);
      res != AMD_COMGR_STATUS_SUCCESS) {
    amd::Comgr::destroy_action_info(action);
    amd::Comgr::destroy_data_set(output);
    return false;
  }

  if (!extractBuildLog(output, buildLog)) {
    amd::Comgr::destroy_action_info(action);
    amd::Comgr::destroy_data_set(output);
    return false;
  }

  if (!extractByteCodeBinary(output, AMD_COMGR_DATA_KIND_BC, LinkedLLVMBitcode)) {
    amd::Comgr::destroy_action_info(action);
    amd::Comgr::destroy_data_set(output);
    return false;
  }

  amd::Comgr::destroy_action_info(action);
  amd::Comgr::destroy_data_set(output);
  return true;
}

bool convertSPIRVToLLVMBC(const amd_comgr_data_set_t linkInputs, const std::string& isa,
                          std::vector<std::string>& linkOptions, std::string& buildLog,
                          std::vector<char>& LinkedLLVMBitcode) {
  amd_comgr_action_info_t action;

  if (!createAction(action, linkOptions, isa, AMD_COMGR_LANGUAGE_NONE)) {
    return false;
  }

  amd_comgr_data_set_t output;
  if (auto res = amd::Comgr::create_data_set(&output); res != AMD_COMGR_STATUS_SUCCESS) {
    amd::Comgr::destroy_action_info(action);
    return false;
  }

  if (auto res =
          amd::Comgr::do_action(AMD_COMGR_ACTION_TRANSLATE_SPIRV_TO_BC, action, linkInputs, output);
      res != AMD_COMGR_STATUS_SUCCESS) {
    amd::Comgr::destroy_action_info(action);
    amd::Comgr::destroy_data_set(output);
    return false;
  }

  if (!extractBuildLog(output, buildLog)) {
    amd::Comgr::destroy_action_info(action);
    amd::Comgr::destroy_data_set(output);
    return false;
  }

  if (!extractByteCodeBinary(output, AMD_COMGR_DATA_KIND_BC, LinkedLLVMBitcode)) {
    amd::Comgr::destroy_action_info(action);
    amd::Comgr::destroy_data_set(output);
    return false;
  }

  amd::Comgr::destroy_action_info(action);
  amd::Comgr::destroy_data_set(output);
  return true;
}

bool createExecutable(const amd_comgr_data_set_t linkInputs, const std::string& isa,
                      std::vector<std::string>& exeOptions, std::string& buildLog,
                      std::vector<char>& executable, bool spirv_bc /* default false */) {
  amd_comgr_action_info_t action;

  if (!createAction(action, exeOptions, isa)) {
    return false;
  }

  // If SPIRV bitcode was processed, make sure we link device libs to it
  if (spirv_bc) {
    if (auto res = amd::Comgr::action_info_set_device_lib_linking(action, true);
        res != AMD_COMGR_STATUS_SUCCESS) {
      LogError("Can not link device libs to action");
      amd::Comgr::destroy_action_info(action);
      return false;
    }
  }

  amd_comgr_data_set_t relocatableData;
  if (auto res = amd::Comgr::create_data_set(&relocatableData); res != AMD_COMGR_STATUS_SUCCESS) {
    amd::Comgr::destroy_action_info(action);
    return false;
  }

  if (auto res = amd::Comgr::do_action(AMD_COMGR_ACTION_CODEGEN_BC_TO_RELOCATABLE, action,
                                       linkInputs, relocatableData);
      res != AMD_COMGR_STATUS_SUCCESS) {
    extractBuildLog(relocatableData, buildLog);
    amd::Comgr::destroy_action_info(action);
    amd::Comgr::destroy_data_set(relocatableData);
    return false;
  }

  if (!extractBuildLog(relocatableData, buildLog)) {
    amd::Comgr::destroy_action_info(action);
    amd::Comgr::destroy_data_set(relocatableData);
    return false;
  }


  amd::Comgr::destroy_action_info(action);
  std::vector<std::string> emptyOpt;
  if (!createAction(action, emptyOpt, isa)) {
    amd::Comgr::destroy_data_set(relocatableData);
    return false;
  }

  amd_comgr_data_set_t output;
  if (auto res = amd::Comgr::create_data_set(&output); res != AMD_COMGR_STATUS_SUCCESS) {
    amd::Comgr::destroy_action_info(action);
    amd::Comgr::destroy_data_set(relocatableData);
    return false;
  }

  if (auto res = amd::Comgr::do_action(AMD_COMGR_ACTION_LINK_RELOCATABLE_TO_EXECUTABLE, action,
                                       relocatableData, output);
      res != AMD_COMGR_STATUS_SUCCESS) {
    extractBuildLog(output, buildLog);
    amd::Comgr::destroy_action_info(action);
    amd::Comgr::destroy_data_set(output);
    amd::Comgr::destroy_data_set(relocatableData);
    return false;
  }

  if (!extractBuildLog(output, buildLog)) {
    amd::Comgr::destroy_action_info(action);
    amd::Comgr::destroy_data_set(output);
    amd::Comgr::destroy_data_set(relocatableData);
    return false;
  }

  if (!extractByteCodeBinary(output, AMD_COMGR_DATA_KIND_EXECUTABLE, executable)) {
    amd::Comgr::destroy_action_info(action);
    amd::Comgr::destroy_data_set(output);
    amd::Comgr::destroy_data_set(relocatableData);
    return false;
  }

  amd::Comgr::destroy_action_info(action);
  amd::Comgr::destroy_data_set(output);
  amd::Comgr::destroy_data_set(relocatableData);

  return true;
}

void GenerateUniqueFileName(std::string& name) {
#if !defined(_WIN32)
  char* name_template = const_cast<char*>(name.c_str());
  int temp_fd = mkstemp(name_template);
#else
  char* name_template = new char[name.length() + 1];
  strcpy_s(name_template, name.length() + 1, name.data());
  int sizeinchars = strnlen(name_template, 20) + 1;
  _mktemp_s(name_template, sizeinchars);
#endif
  name = name_template;
#if !defined(_WIN32)
  unlink(name_template);
  close(temp_fd);
#endif
}

bool demangleName(const std::string& mangledName, std::string& demangledName) {
  amd_comgr_data_t mangled_data;
  amd_comgr_data_t demangled_data;

  if (AMD_COMGR_STATUS_SUCCESS != amd::Comgr::create_data(AMD_COMGR_DATA_KIND_BYTES, &mangled_data))
    return false;

  if (AMD_COMGR_STATUS_SUCCESS !=
      amd::Comgr::set_data(mangled_data, mangledName.size(), mangledName.c_str())) {
    amd::Comgr::release_data(mangled_data);
    return false;
  }

  if (AMD_COMGR_STATUS_SUCCESS != amd::Comgr::demangle_symbol_name(mangled_data, &demangled_data)) {
    amd::Comgr::release_data(mangled_data);
    return false;
  }

  size_t demangled_size = 0;
  if (AMD_COMGR_STATUS_SUCCESS != amd::Comgr::get_data(demangled_data, &demangled_size, NULL)) {
    amd::Comgr::release_data(mangled_data);
    amd::Comgr::release_data(demangled_data);
    return false;
  }

  demangledName.resize(demangled_size);

  if (AMD_COMGR_STATUS_SUCCESS != amd::Comgr::get_data(demangled_data, &demangled_size,
                                                       const_cast<char*>(demangledName.data()))) {
    amd::Comgr::release_data(mangled_data);
    amd::Comgr::release_data(demangled_data);
    return false;
  }

  amd::Comgr::release_data(mangled_data);
  amd::Comgr::release_data(demangled_data);
  return true;
}

std::string handleMangledName(std::string loweredName) {
  if (loweredName.empty()) {
    return loweredName;
  }

  if (loweredName.find(".kd") != std::string::npos) {
    return {};
  }

  if (loweredName.find("void ") == 0) {
    loweredName.erase(0, strlen("void "));
  }

  auto dx{loweredName.find_first_of("(<")};

  if (dx == std::string::npos) {
    return loweredName;
  }

  if (loweredName[dx] == '<') {
    uint32_t count = 1;
    do {
      ++dx;
      count += (loweredName[dx] == '<') ? 1 : ((loweredName[dx] == '>') ? -1 : 0);
    } while (count);

    loweredName.erase(++dx);
  } else {
    loweredName.erase(dx);
  }

  return loweredName;
}

bool fillMangledNames(std::vector<char>& dataVec, std::map<std::string, std::string>& mangledNames,
                      bool isBitcode) {
  amd_comgr_data_t dataObject;
  if (auto res = amd::Comgr::create_data(
          isBitcode ? AMD_COMGR_DATA_KIND_BC : AMD_COMGR_DATA_KIND_EXECUTABLE, &dataObject);
      res != AMD_COMGR_STATUS_SUCCESS) {
    return false;
  }

  if (auto res = amd::Comgr::set_data(dataObject, dataVec.size(), dataVec.data())) {
    amd::Comgr::release_data(dataObject);
    return false;
  }

  size_t Count;
  if (auto res = amd::Comgr::populate_name_expression_map(dataObject, &Count)) {
    amd::Comgr::release_data(dataObject);
    return false;
  }

  for (auto& it : mangledNames) {
    size_t Size;
    char* data = const_cast<char*>(it.first.data());

    if (auto res = amd::Comgr::map_name_expression_to_symbol_name(dataObject, &Size, data, NULL)) {
      amd::Comgr::release_data(dataObject);
      return false;
    }

    std::unique_ptr<char[]> mName(new char[Size]());
    if (auto res =
            amd::Comgr::map_name_expression_to_symbol_name(dataObject, &Size, data, mName.get())) {
      amd::Comgr::release_data(dataObject);
      return false;
    }

    it.second = std::string(mName.get());
  }

  amd::Comgr::release_data(dataObject);
  return true;
}

const std::map<std::string, std::string>& GenericTargetMapping() {
  // The map is subject to change per removing policy
  static const std::map<std::string, std::string> genericTargetMap{
      // "gfx9-generic"
      {"gfx900", "gfx9-generic"},
      {"gfx902", "gfx9-generic"},
      {"gfx904", "gfx9-generic"},
      {"gfx906", "gfx9-generic"},
      {"gfx909", "gfx9-generic"},
      {"gfx90c", "gfx9-generic"},
      // "gfx9-4-generic"
      {"gfx942", "gfx9-4-generic"},
      {"gfx950", "gfx9-4-generic"},
      // "gfx10-1-generic"
      {"gfx1010", "gfx10-1-generic"},
      {"gfx1011", "gfx10-1-generic"},
      {"gfx1012", "gfx10-1-generic"},
      {"gfx1013", "gfx10-1-generic"},
      // "gfx10-3-generic"
      {"gfx1030", "gfx10-3-generic"},
      {"gfx1031", "gfx10-3-generic"},
      {"gfx1032", "gfx10-3-generic"},
      {"gfx1033", "gfx10-3-generic"},
      {"gfx1034", "gfx10-3-generic"},
      {"gfx1035", "gfx10-3-generic"},
      {"gfx1036", "gfx10-3-generic"},
      // "gfx11-generic"
      {"gfx1100", "gfx11-generic"},
      {"gfx1101", "gfx11-generic"},
      {"gfx1102", "gfx11-generic"},
      {"gfx1103", "gfx11-generic"},
      {"gfx1150", "gfx11-generic"},
      {"gfx1151", "gfx11-generic"},
      // "gfx12-generic"
      {"gfx1200", "gfx12-generic"},
      {"gfx1201", "gfx12-generic"},
  };
  return genericTargetMap;
}

bool IsCompatibleWithGenericTarget(const std::string& coTarget, const std::string& agentTarget) {
  auto& map = GenericTargetMapping();
  auto search = map.find(agentTarget);
  return search != map.end() && coTarget == search->second;
}
}  // namespace helpers

std::vector<std::string> getLinkOptions(const LinkArguments& args) {
  std::vector<std::string> res;

  { // process optimization level
    std::string opt("-O");
    opt += std::to_string(args.optimization_level_);
    res.push_back(opt);
  }

  const auto irArgCount = args.linker_ir2isa_args_count_;
  if (irArgCount > 0) {
    res.reserve(irArgCount);
    const auto irArg = args.linker_ir2isa_args_;
    for (size_t i = 0; i < irArgCount; i++) {
      res.emplace_back(std::string(irArg[i]));
    }
  }
  return res;
}

// RTC Program Member Functions
RTCProgram::RTCProgram(std::string name) : name_(name) {
  constexpr bool kComgrVersioned = true;
  std::call_once(amd::Comgr::initialized, amd::Comgr::LoadLib, kComgrVersioned);
  if (amd::Comgr::create_data_set(&exec_input_) != AMD_COMGR_STATUS_SUCCESS) {
    guarantee(false, "Failed to allocate internal hiprtc structure");
  }
}

bool RTCProgram::findIsa() {
#ifdef BUILD_SHARED_LIBS
  const char* libName;
#ifdef _WIN32
  std::string dll_name = std::string("amdhip64_" + std::to_string(HIP_VERSION_MAJOR) + ".dll");
  libName = dll_name.c_str();
#else
  libName = "libamdhip64.so.7";
#endif

  void* handle = amd::Os::loadLibrary(libName);

  if (!handle) {
    LogInfo("hip runtime failed to load using dlopen");
    build_log_ +=
        "hip runtime failed to load.\n"
        "Error: Please provide architecture for which code is to be "
        "generated.\n";
    return false;
  }

  void* sym_hipGetDevice = amd::Os::getSymbol(handle, "hipGetDevice");
  void* sym_hipGetDeviceProperties =
      amd::Os::getSymbol(handle, "hipGetDevicePropertiesR0600");  // Try to find the new symbol
  if (sym_hipGetDeviceProperties == nullptr) {
    sym_hipGetDeviceProperties =
        amd::Os::getSymbol(handle, "hipGetDeviceProperties");  // Fall back to old one
  }

  if (sym_hipGetDevice == nullptr || sym_hipGetDeviceProperties == nullptr) {
    LogInfo("ISA cannot be found to dlsym failure");
    build_log_ +=
        "ISA cannot be found from hip runtime.\n"
        "Error: Please provide architecture for which code is to be "
        "generated.\n";
    return false;
  }

  hipError_t (*dyn_hipGetDevice)(int*) = reinterpret_cast<hipError_t (*)(int*)>(sym_hipGetDevice);

  hipError_t (*dyn_hipGetDeviceProperties)(hipDeviceProp_t*, int) =
      reinterpret_cast<hipError_t (*)(hipDeviceProp_t*, int)>(sym_hipGetDeviceProperties);

  int device;
  hipError_t status = dyn_hipGetDevice(&device);
  if (status != hipSuccess) {
    return false;
  }
  hipDeviceProp_t props;
  status = dyn_hipGetDeviceProperties(&props, device);
  if (status != hipSuccess) {
    return false;
  }
  isa_ = "amdgcn-amd-amdhsa--";
  isa_.append(props.gcnArchName);

  amd::Os::unloadLibrary(handle);
  return true;

#else
  int device;
  hipError_t status = hipGetDevice(&device);
  if (status != hipSuccess) {
    return false;
  }
  hipDeviceProp_t props;
  status = hipGetDeviceProperties(&props, device);
  if (status != hipSuccess) {
    return false;
  }
  isa_ = "amdgcn-amd-amdhsa--";
  isa_.append(props.gcnArchName);

  return true;
#endif
}

// RTC Program Member Functions
void RTCProgram::AppendOptions(const std::string app_env_var, std::vector<std::string>* options) {
  if (options == nullptr) {
    LogError("Append options passed is nullptr.");
    return;
  }

  std::stringstream ss(app_env_var);
  std::istream_iterator<std::string> begin{ss}, end;
  options->insert(options->end(), begin, end);
}

// HIPRTC Program lock
amd::Monitor RTCProgram::lock_(true);

LinkProgram::LinkProgram(std::string name) : RTCProgram(name) {
  if (amd::Comgr::create_data_set(&link_input_) != AMD_COMGR_STATUS_SUCCESS) {
    guarantee(false, "Failed to allocate internal comgr structure");
  }
  amd::ScopedLock lock(lock_);
  linker_set_.insert(this);
}

bool LinkProgram::isLinkerValid(LinkProgram* link_program) {
  amd::ScopedLock lock(lock_);
  if (linker_set_.find(link_program) == linker_set_.end()) {
    return false;
  }
  return true;
}

bool LinkProgram::AddLinkerOptions(unsigned int num_options, hipJitOption* options_ptr,
                                   void** options_vals_ptr) {
  for (size_t opt_idx = 0; opt_idx < num_options; ++opt_idx) {
    if (options_vals_ptr[opt_idx] == nullptr) {
      LogError("Options value can not be nullptr");
      return false;
    }
    switch (options_ptr[opt_idx]) {
      case hipJitOptionMaxRegisters:
        link_args_.max_registers_ = *(reinterpret_cast<uint64_t*>(&options_vals_ptr[opt_idx]));
        break;
      case hipJitOptionThreadsPerBlock:
        link_args_.threads_per_block_ =
            *(reinterpret_cast<uint64_t*>(&options_vals_ptr[opt_idx]));
        break;
      case hipJitOptionWallTime:
        link_args_.wall_time_ = *(reinterpret_cast<float*>(options_vals_ptr[opt_idx]));
        break;
      case hipJitOptionInfoLogBuffer: {
        link_args_.info_log_ = (reinterpret_cast<char*>(options_vals_ptr[opt_idx]));
        break;
      }
      case hipJitOptionInfoLogBufferSizeBytes:
        link_args_.info_log_size_ = (reinterpret_cast<uint64_t>(options_vals_ptr[opt_idx]));
        break;
      case hipJitOptionErrorLogBuffer: {
        link_args_.error_log_ = reinterpret_cast<char*>(options_vals_ptr[opt_idx]);
        break;
      }
      case hipJitOptionErrorLogBufferSizeBytes:
        link_args_.error_log_size_ = (reinterpret_cast<uint64_t>(options_vals_ptr[opt_idx]));
        break;
      case hipJitOptionOptimizationLevel:
        link_args_.optimization_level_ =
            *(reinterpret_cast<uint64_t*>(&options_vals_ptr[opt_idx]));
      break;
      case hipJitOptionTargetFromContext:
        link_args_.target_from_hip_context_ =
            *(reinterpret_cast<uint64_t*>(&options_vals_ptr[opt_idx]));
        break;
      case hipJitOptionTarget:
        link_args_.jit_target_ = *(reinterpret_cast<uint64_t*>(&options_vals_ptr[opt_idx]));
        break;
      case hipJitOptionFallbackStrategy:
        link_args_.fallback_strategy_ =
            *(reinterpret_cast<uint64_t*>(&options_vals_ptr[opt_idx]));
        break;
      case hipJitOptionGenerateDebugInfo:
        link_args_.generate_debug_info_ = *(reinterpret_cast<uint32_t*>(&options_vals_ptr[opt_idx]));
        break;
      case hipJitOptionLogVerbose:
        link_args_.log_verbose_ = reinterpret_cast<uint64_t>(options_vals_ptr[opt_idx]);
        break;
      case hipJitOptionGenerateLineInfo:
        link_args_.generate_line_info_ = *(reinterpret_cast<uint32_t*>(&options_vals_ptr[opt_idx]));
        break;
      case hipJitOptionCacheMode:
        link_args_.cache_mode_ = *(reinterpret_cast<uint64_t*>(&options_vals_ptr[opt_idx]));
        break;
      case hipJitOptionSm3xOpt:
        link_args_.sm3x_opt_ = *(reinterpret_cast<bool*>(&options_vals_ptr[opt_idx]));
        break;
      case hipJitOptionFastCompile:
        link_args_.fast_compile_ = *(reinterpret_cast<bool*>(&options_vals_ptr[opt_idx]));
        break;
      case hipJitOptionGlobalSymbolNames: {
        link_args_.global_symbol_names_ = reinterpret_cast<const char**>(options_vals_ptr[opt_idx]);
        break;
      }
      case hipJitOptionGlobalSymbolAddresses: {
        link_args_.global_symbol_addresses_ = reinterpret_cast<void**>(options_vals_ptr[opt_idx]);
        break;
      }
      case hipJitOptionGlobalSymbolCount:
        link_args_.global_symbol_count_ =
            *(reinterpret_cast<uint64_t*>(&options_vals_ptr[opt_idx]));
        break;
      case hipJitOptionLto:
        link_args_.lto_ = *(reinterpret_cast<uint32_t*>(&options_vals_ptr[opt_idx]));
        break;
      case hipJitOptionFtz:
        link_args_.ftz_ = *(reinterpret_cast<uint32_t*>(&options_vals_ptr[opt_idx]));
        break;
      case hipJitOptionPrecDiv:
        link_args_.prec_div_ = *(reinterpret_cast<uint32_t*>(&options_vals_ptr[opt_idx]));
        break;
      case hipJitOptionPrecSqrt:
        link_args_.prec_sqrt_ = *(reinterpret_cast<uint32_t*>(&options_vals_ptr[opt_idx]));
        break;
      case hipJitOptionFma:
        link_args_.fma_ = *(reinterpret_cast<uint32_t*>(&options_vals_ptr[opt_idx]));
        break;
      case hipJitOptionPositionIndependentCode:
        link_args_.pic_ = *(reinterpret_cast<uint32_t*>(&options_vals_ptr[opt_idx]));
        break;
      case hipJitOptionMinCTAPerSM:
        link_args_.min_cta_per_sm_ = *(reinterpret_cast<uint32_t*>(&options_vals_ptr[opt_idx]));
        break;
      case hipJitOptionMaxThreadsPerBlock:
        link_args_.max_threads_per_block_ = *(reinterpret_cast<uint32_t*>(&options_vals_ptr[opt_idx]));
        break;
      case hipJitOptionOverrideDirectiveValues:
        link_args_.override_directive_values_ = *(reinterpret_cast<uint32_t*>(&options_vals_ptr[opt_idx]));
        break;
      case hipJitOptionIRtoISAOptExt: {
        link_args_.linker_ir2isa_args_ = reinterpret_cast<const char**>(options_vals_ptr[opt_idx]);
        break;
      }
      case hipJitOptionIRtoISAOptCountExt:
        link_args_.linker_ir2isa_args_count_ = reinterpret_cast<uint64_t>(options_vals_ptr[opt_idx]);
        break;
      default:
        break;
    }
  }

  return true;
}


amd_comgr_data_kind_t LinkProgram::GetCOMGRDataKind(hipJitInputType input_type) {
  amd_comgr_data_kind_t data_kind = AMD_COMGR_DATA_KIND_UNDEF;

  // Map the hiprtc input type to comgr data kind
  switch (input_type) {
    case hipJitInputLLVMBitcode:
      data_kind = AMD_COMGR_DATA_KIND_BC;
      break;
    case hipJitInputLLVMBundledBitcode:
      data_kind =
          HIPRTC_USE_RUNTIME_UNBUNDLER ? AMD_COMGR_DATA_KIND_BC : AMD_COMGR_DATA_KIND_BC_BUNDLE;
      break;
    case hipJitInputLLVMArchivesOfBundledBitcode:
      data_kind = AMD_COMGR_DATA_KIND_AR_BUNDLE;
      break;
    case hipJitInputSpirv:
      data_kind = AMD_COMGR_DATA_KIND_SPIRV;
      break;
    default:
      LogError("hip link : Cannot find the corresponding comgr data kind");
      break;
  }

  return data_kind;
}


bool LinkProgram::AddLinkerDataImpl(std::vector<char>& link_data, hipJitInputType input_type,
                                    std::string& link_file_name) {
  std::vector<char> llvm_code_object;
  is_bundled_ = helpers::CheckIfBundled(link_data);

  if (HIPRTC_USE_RUNTIME_UNBUNDLER && input_type == hipJitInputLLVMBundledBitcode) {
    if (!findIsa()) {
      return false;
    }

    size_t co_offset = 0;
    size_t co_size = 0;
    if (!helpers::UnbundleBitCode(link_data, isa_, co_offset, co_size)) {
      LogError("Error in hip Linker: unable to unbundle the llvm bitcode");
      return false;
    }

    llvm_code_object.assign(link_data.begin() + co_offset, link_data.begin() + co_offset + co_size);
  } else if (is_bundled_ && input_type == hipJitInputSpirv) {
    const char* bundleEntryIDs[] = {helpers::SPIRV_BUNDLE_ENTRY_ID};
    size_t bundleEntryIDsCount = sizeof(bundleEntryIDs) / sizeof(bundleEntryIDs[0]);
    if (!helpers::UnbundleUsingComgr(link_data, isa_, link_options_, build_log_, llvm_code_object,
                                     bundleEntryIDs, bundleEntryIDsCount)) {
      LogError("Error in hip Linker: Unable to unbundle SPIRV Bitcode");
      return false;
    }
  } else {
    llvm_code_object.assign(link_data.begin(), link_data.end());
  }

  if ((data_kind_ = GetCOMGRDataKind(input_type)) == AMD_COMGR_DATA_KIND_UNDEF) {
    LogError("Cannot find the correct COMGR data kind");
    return false;
  }

  if (!helpers::addCodeObjData(link_input_, llvm_code_object, link_file_name, data_kind_)) {
    LogError("Error in hip Linker: unable to add linked code object");
    return false;
  }

  return true;
}


bool LinkProgram::AddLinkerFile(std::string file_path, hipJitInputType input_type) {
  std::ifstream file_stream{file_path, std::ios_base::in | std::ios_base::binary};
  if (!file_stream.good()) {
    return false;
  }

  file_stream.seekg(0, std::ios::end);
  std::streampos file_size = file_stream.tellg();
  file_stream.seekg(0, std::ios::beg);

  // Read the file contents
  std::vector<char> link_file_info(file_size);
  file_stream.read(link_file_info.data(), file_size);

  file_stream.close();

  std::string link_file_name("LinkerProgram");

  return AddLinkerDataImpl(link_file_info, input_type, link_file_name);
}

bool LinkProgram::AddLinkerData(void* image_ptr, size_t image_size, std::string link_file_name,
                                hipJitInputType input_type) {
  char* image_char_buf = reinterpret_cast<char*>(image_ptr);
  std::vector<char> llvm_code_object(image_char_buf, image_char_buf + image_size);

  return AddLinkerDataImpl(llvm_code_object, input_type, link_file_name);
}

bool LinkProgram::LinkComplete(void** bin_out, size_t* size_out) {
  if (!findIsa()) {
    return false;
  }

  // If the data kind is SPIRV, convert it beforehand and pass it on to subsequent machinery
  // TODO I think this can be simplified a bit, we are basically reading and writing into comgr data
  // structures, do we need to do that? This might cause some errors, so adding this to come back to
  // it.
  amd_comgr_data_set_t link_input = link_input_;
  if (data_kind_ == AMD_COMGR_DATA_KIND_SPIRV) {
    // Convert SPIRV Unbundled code object to LLVM Bitcode
    std::vector<char> llvmbc_from_spirv;
    if (!helpers::convertSPIRVToLLVMBC(link_input_, isa_, link_options_, build_log_,
                                       llvmbc_from_spirv)) {
      LogError("Error in hip Linker: unable to convert SPIRV to BC");
      return false;
    }

    std::string linkedFileName = "LLVMBitcodeFromSPIRV.bc";
    if (!helpers::addCodeObjData(link_input, llvmbc_from_spirv, linkedFileName,
                                 AMD_COMGR_DATA_KIND_BC)) {
      LogError("Error in hip Linker: unable to add linked LLVM bitcode");
      return false;
    }
  }

  std::vector<char> llvm_bitcode;
  if (!helpers::linkLLVMBitcode(link_input, isa_, link_options_, build_log_, llvm_bitcode)) {
    LogError("Error in hip linker: unable to add device libs to linked bitcode");
    return false;
  }

  std::string linkedFileName = "LLVMBitcode.bc";
  if (!helpers::addCodeObjData(exec_input_, llvm_bitcode, linkedFileName, AMD_COMGR_DATA_KIND_BC)) {
    LogError("Error in hip linker: unable to add linked bitcode");
    return false;
  }

  std::vector<std::string> exe_options = getLinkOptions(link_args_);
  LogPrintfInfo("Exe options forwarded to compiler: %s",
                [&]() {
                  std::string ret;
                  for (const auto& i : exe_options) {
                    ret += i;
                    ret += " ";
                  }
                  return ret;
                }()
                    .c_str());
  if (!helpers::createExecutable(exec_input_, isa_, exe_options, build_log_, executable_,
                                 data_kind_ == AMD_COMGR_DATA_KIND_SPIRV)) {
    LogPrintfInfo("Error in hip linker: unable to create exectuable: %s", build_log_.c_str());
    return false;
  }

  *size_out = executable_.size();
  *bin_out = executable_.data();

  return true;
}


}  // namespace hip
