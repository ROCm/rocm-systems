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

#pragma once

#include <vector>
#include <string>


#include "vdi_common.hpp"
#include "rocclr/utils/debug.hpp"
#include "rocclr/utils/flags.hpp"
#include "device/comgrctx.hpp"
#include "hip/hip_runtime_api.h"

namespace hip {
namespace helpers {
bool UnbundleBitCode(const std::vector<char>& bundled_bit_code, const std::string& isa,
                     size_t& co_offset, size_t& co_size);
bool addCodeObjData(amd_comgr_data_set_t& input, const std::vector<char>& source,
                    const std::string& name, const amd_comgr_data_kind_t type);
bool extractBuildLog(amd_comgr_data_set_t dataSet, std::string& buildLog);
bool extractByteCodeBinary(const amd_comgr_data_set_t inDataSet,
                           const amd_comgr_data_kind_t dataKind, std::vector<char>& bin);
bool createAction(amd_comgr_action_info_t& action, std::vector<std::string>& options,
                  const std::string& isa,
                  const amd_comgr_language_t lang = AMD_COMGR_LANGUAGE_NONE);
bool compileToExecutable(const amd_comgr_data_set_t compileInputs, const std::string& isa,
                         std::vector<std::string>& compileOptions,
                         std::vector<std::string>& linkOptions, std::string& buildLog,
                         std::vector<char>& exe);
bool compileToBitCode(const amd_comgr_data_set_t compileInputs, const std::string& isa,
                      std::vector<std::string>& compileOptions, std::string& buildLog,
                      std::vector<char>& LLVMBitcode);
bool linkLLVMBitcode(const amd_comgr_data_set_t linkInputs, const std::string& isa,
                     std::vector<std::string>& linkOptions, std::string& buildLog,
                     std::vector<char>& LinkedLLVMBitcode);
bool createExecutable(const amd_comgr_data_set_t linkInputs, const std::string& isa,
                      std::vector<std::string>& exeOptions, std::string& buildLog,
                      std::vector<char>& executable, bool spirv_bc = false);
bool convertSPIRVToLLVMBC(const amd_comgr_data_set_t linkInputs, const std::string& isa,
                          std::vector<std::string>& linkOptions, std::string& buildLog,
                          std::vector<char>& linkedSPIRVBitcode);
bool demangleName(const std::string& mangledName, std::string& demangledName);
std::string handleMangledName(std::string loweredName);
bool fillMangledNames(std::vector<char>& executable,
                      std::map<std::string, std::string>& mangledNames, bool isBitcode);
void GenerateUniqueFileName(std::string& name);

bool CheckIfBundled(std::vector<char>& llvm_bitcode);

bool UnbundleUsingComgr(std::vector<char>& source, const std::string& isa,
                        std::vector<std::string>& linkOptions, std::string& buildLog,
                        std::vector<char>& unbundled_spirv_bitcode, const char* bundleEntryIDs,
                        size_t bundleEntryIDsCount);

// Mapping from targets to generic targets
const std::map<std::string, std::string>& GenericTargetMapping();

// Return true if agent target compatible with generic code object target, false otherwise.
// Both targets should not have any feature.
bool IsCompatibleWithGenericTarget(const std::string& coTarget, const std::string& agentTarget);
}  // namespace helpers

struct LinkArguments {
  const char** linker_ir2isa_args_;
  size_t linker_ir2isa_args_count_;

  LinkArguments() : linker_ir2isa_args_{nullptr}, linker_ir2isa_args_count_{0} {}
};

class RTCProgram {
 protected:
  // Lock and control variables
  static amd::Monitor lock_;
  static std::once_flag initialized_;

  RTCProgram(std::string name);
  ~RTCProgram() { amd::Comgr::destroy_data_set(exec_input_); }

  // Member Functions
  bool findIsa();
  static void AppendOptions(std::string app_env_var, std::vector<std::string>* options);

  // Data Members
  std::string name_;
  std::string isa_;
  std::string build_log_;
  std::vector<char> executable_;

  amd_comgr_data_set_t exec_input_;
};

class LinkProgram : public RTCProgram {
  // Private Member Functions (forbid these function calls)
  LinkProgram() = delete;
  LinkProgram(LinkProgram&) = delete;
  LinkProgram& operator=(LinkProgram&) = delete;

  amd_comgr_data_kind_t data_kind_;
  amd_comgr_data_kind_t GetCOMGRDataKind(hipJitInputType input_type);

  // Linker Argumenets at hipLinkCreate
  LinkArguments link_args_;

  // Spirv is bundled
  bool is_bundled_ = false;

  // Private Data Members
  amd_comgr_data_set_t link_input_;
  std::vector<std::string> link_options_;
  static std::unordered_set<LinkProgram*> linker_set_;

  bool AddLinkerDataImpl(std::vector<char>& link_data, hipJitInputType input_type,
                         std::string& link_file_name);

 public:
  LinkProgram(std::string name);
  ~LinkProgram() {
    amd::ScopedLock lock(lock_);
    linker_set_.erase(this);
    amd::Comgr::destroy_data_set(link_input_);
  }
  // Public Member Functions
  bool AddLinkerOptions(unsigned int num_options, hipJitOption* options_ptr,
                        void** options_vals_ptr);
  bool AddLinkerFile(std::string file_path, hipJitInputType input_type);
  bool AddLinkerData(void* image_ptr, size_t image_size, std::string link_file_name,
                     hipJitInputType input_type);
  bool LinkComplete(void** bin_out, size_t* size_out);
  void AppendLinkerOptions() { AppendOptions(HIPRTC_LINK_OPTIONS_APPEND, &link_options_); }
  static bool isLinkerValid(LinkProgram* link_program);
};


}  // namespace hip
