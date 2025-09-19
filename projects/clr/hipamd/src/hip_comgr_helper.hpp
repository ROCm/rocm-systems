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
/**
 * HIPRTC linker options
 */
struct LinkArguments {
  uint64_t max_registers_ = 0;                    ///< Maximum registers that a thread may a use
  uint64_t threads_per_block_ = 0;                ///< Minimum No. of threads per block
  float wall_time_ = 0.0f;                        ///< Value for total wall clock time
  char* info_log_ = nullptr;                      ///< Pointer to a buffer to print log information
  uint64_t info_log_size_ = 0;                    ///< Size of the buffer in bytes for logged info
  char* error_log_ = nullptr;                     ///< Pointer to a buffer to print log errors
  uint64_t error_log_size_ = 0;                   ///< Size of the buffer in bytes for logged errors
  uint64_t optimization_level_ = 3;               ///< Value of the optimization level for generated code
                                                  ///< acceptable options -O0, -O1, -O2, -O3
  uint64_t target_from_hip_context_ = 0;          ///< Determines the target, based on the current context
  uint64_t jit_target_= 0;                        ///< CUDA Only JIT target
  uint64_t fallback_strategy_ = 0;                ///< CUDA Only Choice of fallback strategy
  uint32_t generate_debug_info_ = 0;              ///< Create debug information in output -g, if set
  uint64_t log_verbose_ = 0;                      ///< Generate verbose log messages
  uint32_t generate_line_info_ = 0;               ///< Generate line number information
  uint64_t cache_mode_ = 0;                       ///< CUDA Only Enables caching explicitly
  bool sm3x_opt_ = false;                         ///< CUDA Only New SM3X option
  bool fast_compile_ = false;                     ///< CUDA Only Set fast compile
  const char** global_symbol_names_ = nullptr;    ///< Array of device symbol names to be relocated
                                                  ///< to the host
  void** global_symbol_addresses_ = nullptr;      ///< Array of host addresses to be relocated to the
                                                  ///< device
  uint64_t global_symbol_count_ = 0;              ///< Number of symbol count
  int32_t lto_ = 0;                               ///< Enable link time optimization for device code
  int32_t ftz_ = 0;                               ///< Set single-precision denormals
  int32_t prec_div_ = 1;                          ///< Set single-precision floating-point division
                                                  ///< and reciprocals
  int32_t prec_sqrt_ = 1;                         ///< Set single-precision floating-point square root
  int32_t fma_ = 1;                               ///< Enable floating-point multiplies and
                                                  ///< adds/subtracts operations
  int32_t pic_ = 0;                               ///< Generates Position Independent code
  int32_t min_cta_per_sm_ = 0;                    ///< Hints to JIT compiler the minimum number of
                                                  ///< CTAs from kernel's grid to be mapped to SM
  int32_t max_threads_per_block_ = 0;             ///< Maximum number of threads in a thread block
  int32_t override_directive_values_ = 0;         ///< Override Directive values
  const char** linker_ir2isa_args_ = nullptr;     ///< Hip Only Linker options to be passed on
                                                  ///< to compiler
  uint64_t linker_ir2isa_args_count_ = 0;         ///< Hip Only Count of linker options to be passed
                                                  ///< on to compiler
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

  // Linker Arguments at hipLinkCreate
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
