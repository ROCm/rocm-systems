/*
Copyright (c) 2022 - 2023 Advanced Micro Devices, Inc. All rights reserved.

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

#include "hiprtcInternal.hpp"

#include <fstream>
#include <streambuf>
#include <vector>

#include <sys/stat.h>

#include "vdi_common.hpp"
#include "rocclr/utils/flags.hpp"

#include "../hip_comgr_helper.hpp"

namespace hiprtc {

// RTC Compile Program Member Functions
RTCCompileProgram::RTCCompileProgram(std::string name_) : hip::RTCProgram(name_), fgpu_rdc_(false) {
  if ((amd::Comgr::create_data_set(&compile_input_) != AMD_COMGR_STATUS_SUCCESS) ||
      (amd::Comgr::create_data_set(&link_input_) != AMD_COMGR_STATUS_SUCCESS)) {
    crashWithMessage("Failed to allocate internal hiprtc structure");
  }
  // Add internal header
  if (!addBuiltinHeader()) {
    crashWithMessage("Unable to add internal header");
  }

  // Add compile options
  const std::string hipVerOpt{"--hip-version=" + std::to_string(HIP_VERSION_MAJOR) + '.' +
                              std::to_string(HIP_VERSION_MINOR) + '.' +
                              std::to_string(HIP_VERSION_PATCH)};
  const std::string hipVerMajor{"-DHIP_VERSION_MAJOR=" + std::to_string(HIP_VERSION_MAJOR)};
  const std::string hipVerMinor{"-DHIP_VERSION_MINOR=" + std::to_string(HIP_VERSION_MINOR)};
  const std::string hipVerPatch{"-DHIP_VERSION_PATCH=" + std::to_string(HIP_VERSION_PATCH)};

  compile_options_.reserve(20);  // count of options below
  compile_options_.push_back("-O3");

  if (!(GPU_ENABLE_WGP_MODE)) {
    compile_options_.push_back("-mcumode");
  }

  compile_options_.push_back(hipVerOpt);
  compile_options_.push_back(hipVerMajor);
  compile_options_.push_back(hipVerMinor);
  compile_options_.push_back(hipVerPatch);
  compile_options_.push_back("-Wno-gnu-line-marker");
  compile_options_.push_back("-Wno-missing-prototypes");
#ifdef _WIN32
  compile_options_.push_back("-target");
  compile_options_.push_back("x86_64-pc-windows-msvc");
  compile_options_.push_back("-fms-extensions");
  compile_options_.push_back("-fms-compatibility");
#endif
  AppendCompileOptions();
}

bool RTCCompileProgram::addSource(const std::string& source, const std::string& name) {
  if (source.size() == 0 || name.size() == 0) {
    LogError("Error in hiprtc: source or name is of size 0 in addSource");
    return false;
  }
  source_code_ += source;
  source_name_ = name;
  return true;
}

// addSource_impl is a different function because we need to add source when we track mangled
// objects
bool RTCCompileProgram::addSource_impl() {
  std::vector<char> vsource(source_code_.begin(), source_code_.end());
  if (!hip::helpers::addCodeObjData(compile_input_, vsource, source_name_,
                                    AMD_COMGR_DATA_KIND_SOURCE)) {
    return false;
  }
  return true;
}

bool RTCCompileProgram::addHeader(const std::string& source, const std::string& name) {
  if (source.size() == 0 || name.size() == 0) {
    LogError("Error in hiprtc: source or name is of size 0 in addHeader");
    return false;
  }
  std::vector<char> vsource(source.begin(), source.end());
  if (!hip::helpers::addCodeObjData(compile_input_, vsource, name, AMD_COMGR_DATA_KIND_INCLUDE)) {
    return false;
  }
  return true;
}

bool RTCCompileProgram::addBuiltinHeader() {
  std::vector<char> source(__hipRTC_header, __hipRTC_header + __hipRTC_header_size);
  std::string name{"hiprtc_runtime.h"};
  if (!hip::helpers::addCodeObjData(compile_input_, source, name, AMD_COMGR_DATA_KIND_INCLUDE)) {
    return false;
  }
  return true;
}

bool RTCCompileProgram::findExeOptions(const std::vector<std::string>& options,
                                       std::vector<std::string>& exe_options) {
  for (size_t i = 0; i < options.size(); ++i) {
    // -mllvm options passed by the app such as "-mllvm" "-amdgpu-early-inline-all=true"
    if (options[i] == "-mllvm") {
      if (options.size() == (i + 1)) {
        LogInfo(
            "-mllvm option passed by the app, it comes as a pair but there is no option after "
            "this");
        return false;
      }
      exe_options.push_back(options[i]);
      exe_options.push_back(options[i + 1]);
    }
    // Options like -Rpass=inline
    if (options[i].find("-Rpass=") == 0) {
      exe_options.push_back(options[i]);
    }
  }
  return true;
}

bool RTCCompileProgram::transformOptions(std::vector<std::string>& compile_options) {
  auto getValueOf = [](const std::string& option) {
    std::string res;
    auto f = std::find(option.begin(), option.end(), '=');
    if (f != option.end()) res = std::string(f + 1, option.end());
    return res;
  };

  for (auto& i : compile_options) {
    if (i == "-hip-pch") {
      LogInfo(
          "-hip-pch is deprecated option, has no impact on execution of new hiprtc programs, it "
          "can be removed");
      i.clear();
      continue;
    }
    // Some rtc samples use --gpu-architecture
    if (i.rfind("--gpu-architecture=", 0) == 0) {
      LogInfo("--gpu-architecture is nvcc option, transforming it to --offload-arch option");
      auto val = getValueOf(i);
      i = "--offload-arch=" + val;
      continue;
    }
  }

  // Removed consumed options
  compile_options.erase(
      std::remove(compile_options.begin(), compile_options.end(), std::string("")),
      compile_options.end());

  if (auto res = std::find_if(
          compile_options.begin(), compile_options.end(),
          [](const std::string& str) { return str.find("--offload-arch=") != std::string::npos; });
      res != compile_options.end()) {
    auto isaName = getValueOf(*res);
    isa_ = "amdgcn-amd-amdhsa--" + isaName;
    settings_.offloadArchProvided = true;
    return true;
  }
  // App has not provided the gpu archiecture, need to find it
  return findIsa();
}

bool RTCCompileProgram::compile(const std::vector<std::string>& options, bool fgpu_rdc) {
  if (!addSource_impl()) {
    LogError("Error in hiprtc: unable to add source code");
    return false;
  }

  fgpu_rdc_ = fgpu_rdc;

  // Append compile options
  std::vector<std::string> compileOpts(compile_options_);
  compileOpts.reserve(compile_options_.size() + options.size() + 2);
  compileOpts.insert(compileOpts.end(), options.begin(), options.end());

  if (!transformOptions(compileOpts)) {
    LogError("Error in hiprtc: unable to transform options");
    return false;
  }

  if (fgpu_rdc_) {
    if (!hip::helpers::compileToBitCode(compile_input_, isa_, compileOpts, build_log_,
                                        LLVMBitcode_)) {
      LogError("Error in hiprtc: unable to compile source to bitcode");
      return false;
    }
  } else {
    LogInfo("Using the new path of comgr");
    if (!hip::helpers::compileToExecutable(compile_input_, isa_, compileOpts, link_options_,
                                           build_log_, executable_)) {
      LogError("Failing to compile to realloc");
      return false;
    }
  }

  if (!mangled_names_.empty()) {
    auto& compile_step_output = fgpu_rdc_ ? LLVMBitcode_ : executable_;
    if (!hip::helpers::fillMangledNames(compile_step_output, mangled_names_, fgpu_rdc_)) {
      LogError("Error in hiprtc: unable to fill mangled names");
      return false;
    }
  }

  return true;
}


void RTCCompileProgram::stripNamedExpression(std::string& strippedName) {
  if (strippedName.back() == ')') {
    strippedName.pop_back();
    strippedName.erase(0, strippedName.find('('));
  }
  if (strippedName.front() == '&') {
    strippedName.erase(0, 1);
  }
}

bool RTCCompileProgram::trackMangledName(std::string& name) {
  amd::ScopedLock lock(lock_);

  if (name.size() == 0) return false;

  std::string strippedName = name;
  stripNamedExpression(strippedName);

  mangled_names_.insert(std::pair<std::string, std::string>(strippedName, ""));

  std::string gcn_expr = "__amdgcn_name_expr_";
  std::string size = std::to_string(mangled_names_.size());
  const auto var1{"\n static __device__ const void* " + gcn_expr + size + "[]= {\"" + strippedName +
                  "\", (void*)&" + strippedName + "};"};
  const auto var2{"\n static auto __amdgcn_name_expr_stub_" + size + " = " + gcn_expr + size +
                  ";\n"};
  const auto code{var1 + var2};

  source_code_ += code;
  return true;
}

bool RTCCompileProgram::getMangledName(const char* name_expression, const char** loweredName) {
  std::string strippedName = name_expression;
  stripNamedExpression(strippedName);

  if (auto dres = mangled_names_.find(strippedName); dres != mangled_names_.end()) {
    if (dres->second.size() != 0) {
      *loweredName = dres->second.c_str();
      return true;
    } else
      return false;
  }
  return false;
}

bool RTCCompileProgram::GetBitcode(char* bitcode) {
  if (!fgpu_rdc_ || LLVMBitcode_.size() <= 0) {
    return false;
  }

  std::copy(LLVMBitcode_.begin(), LLVMBitcode_.end(), bitcode);
  return true;
}

bool RTCCompileProgram::GetBitcodeSize(size_t* bitcode_size) {
  if (!fgpu_rdc_ || LLVMBitcode_.size() <= 0) {
    return false;
  }

  *bitcode_size = LLVMBitcode_.size();
  return true;
}
}  // namespace hiprtc
