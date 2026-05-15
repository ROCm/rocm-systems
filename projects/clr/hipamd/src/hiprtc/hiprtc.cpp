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

#include "../hip_comgr_helper.hpp"
#include <hip/hiprtc.h>
#include "hiprtcInternal.hpp"

namespace hiprtc {
thread_local TlsAggregator tls;
}

const char* hiprtcGetErrorString(hiprtcResult x) {
  switch (x) {
    case HIPRTC_SUCCESS:
      return "HIPRTC_SUCCESS";
    case HIPRTC_ERROR_OUT_OF_MEMORY:
      return "HIPRTC_ERROR_OUT_OF_MEMORY";
    case HIPRTC_ERROR_PROGRAM_CREATION_FAILURE:
      return "HIPRTC_ERROR_PROGRAM_CREATION_FAILURE";
    case HIPRTC_ERROR_INVALID_INPUT:
      return "HIPRTC_ERROR_INVALID_INPUT";
    case HIPRTC_ERROR_INVALID_PROGRAM:
      return "HIPRTC_ERROR_INVALID_PROGRAM";
    case HIPRTC_ERROR_INVALID_OPTION:
      return "HIPRTC_ERROR_INVALID_OPTION";
    case HIPRTC_ERROR_COMPILATION:
      return "HIPRTC_ERROR_COMPILATION";
    case HIPRTC_ERROR_BUILTIN_OPERATION_FAILURE:
      return "HIPRTC_ERROR_BUILTIN_OPERATION_FAILURE";
    case HIPRTC_ERROR_NO_NAME_EXPRESSIONS_AFTER_COMPILATION:
      return "HIPRTC_ERROR_NO_NAME_EXPRESSIONS_AFTER_COMPILATION";
    case HIPRTC_ERROR_NO_LOWERED_NAMES_BEFORE_COMPILATION:
      return "HIPRTC_ERROR_NO_LOWERED_NAMES_BEFORE_COMPILATION";
    case HIPRTC_ERROR_NAME_EXPRESSION_NOT_VALID:
      return "HIPRTC_ERROR_NAME_EXPRESSION_NOT_VALID";
    case HIPRTC_ERROR_INTERNAL_ERROR:
      return "HIPRTC_ERROR_INTERNAL_ERROR";
    case HIPRTC_ERROR_LINKING:
      return "HIPRTC_ERROR_LINKING";
    default:
      LogPrintfError("Invalid HIPRTC error code: %d", x);
      return nullptr;
  };

  return nullptr;
}


hiprtcResult hiprtcCreateProgram(hiprtcProgram* prog, const char* src, const char* name,
                                 int numHeaders, const char* const* headers,
                                 const char* const* headerNames) {
  HIPRTC_INIT_API(prog, src, name, numHeaders, headers, headerNames);

  if (prog == nullptr) {
    HIPRTC_RETURN(HIPRTC_ERROR_INVALID_PROGRAM);
  }
  if (numHeaders < 0) {
    HIPRTC_RETURN(HIPRTC_ERROR_INVALID_INPUT);
  }
  if (numHeaders && (headers == nullptr || headerNames == nullptr)) {
    HIPRTC_RETURN(HIPRTC_ERROR_INVALID_INPUT);
  }
  std::string progName;

  if (name) {
    progName = name;
  }

  auto* rtcProgram = new hiprtc::RTCCompileProgram(progName);
  if (rtcProgram == nullptr) {
    HIPRTC_RETURN(HIPRTC_ERROR_PROGRAM_CREATION_FAILURE);
  }

  if (name == nullptr || strlen(name) == 0) {
    progName = "CompileSourceXXXXXX";
    hip::helpers::GenerateUniqueFileName(progName);
  }

  if (!rtcProgram->addSource(std::string(src), progName)) {
    delete rtcProgram;
    HIPRTC_RETURN(HIPRTC_ERROR_INVALID_INPUT);
  }

  for (int i = 0; i < numHeaders; i++) {
    if (!rtcProgram->addHeader(std::string(headers[i]), std::string(headerNames[i]))) {
      delete rtcProgram;
      HIPRTC_RETURN(HIPRTC_ERROR_INVALID_INPUT);
    }
  }

  *prog = hiprtc::RTCCompileProgram::as_hiprtcProgram(rtcProgram);

  HIPRTC_RETURN(HIPRTC_SUCCESS);
}

hiprtcResult hiprtcCompileProgram(hiprtcProgram prog, int numOptions, const char* const* options) {
  HIPRTC_INIT_API(prog, numOptions, options);

  auto* rtcProgram = hiprtc::RTCCompileProgram::as_RTCCompileProgram(prog);

  bool fgpu_rdc = false;
  bool no_builtin_header = false;
  std::vector<std::string> opt, compile_options;
  opt.reserve(numOptions);
  compile_options.reserve(numOptions + 4);
  for (int i = 0; i < numOptions; i++) {
    if (std::string(options[i]) == std::string("-fgpu-rdc")) {
      fgpu_rdc = true;
    }

    if (std::string(options[i]) != std::string("--hiprtc-no-builtin-header")) {
      opt.push_back(std::string(options[i]));
    } else {
      no_builtin_header = true;
    }
  }

  // Do not include the default hiprtc header if the app passes --hiprtc-no-builtin-header option.
  // This is to avoid conflicts with std type traits defined in hiprtc header. Once the actual fix
  // to move type traits to __hip_internal namespace is made in 7.0, this option can be removed.
  if (!no_builtin_header) {
    compile_options.push_back("-D__HIPCC_RTC__");
    compile_options.push_back("-nogpuinc");
    compile_options.push_back("-include");
    compile_options.push_back("hiprtc_runtime.h");
  }

  compile_options.insert(std::end(compile_options), std::begin(opt), std::end(opt));

  if (!rtcProgram->compile(compile_options, fgpu_rdc)) {
    HIPRTC_RETURN(HIPRTC_ERROR_COMPILATION);
  }

  HIPRTC_RETURN(HIPRTC_SUCCESS);
}

hiprtcResult hiprtcAddNameExpression(hiprtcProgram prog, const char* name_expression) {
  HIPRTC_INIT_API(prog, name_expression);

  if (name_expression == nullptr) {
    HIPRTC_RETURN(HIPRTC_ERROR_INVALID_INPUT);
  }
  auto* rtcProgram = hiprtc::RTCCompileProgram::as_RTCCompileProgram(prog);
  std::string name = name_expression;
  if (!rtcProgram->trackMangledName(name)) {
    HIPRTC_RETURN(HIPRTC_ERROR_INVALID_INPUT);
  }

  HIPRTC_RETURN(HIPRTC_SUCCESS);
}

hiprtcResult hiprtcGetLoweredName(hiprtcProgram prog, const char* name_expression,
                                  const char** loweredName) {
  HIPRTC_INIT_API(prog, name_expression, loweredName);

  if (name_expression == nullptr || loweredName == nullptr) {
    HIPRTC_RETURN(HIPRTC_ERROR_INVALID_INPUT);
  }

  auto* rtcProgram = hiprtc::RTCCompileProgram::as_RTCCompileProgram(prog);

  if (!rtcProgram->getMangledName(name_expression, loweredName)) {
    return HIPRTC_RETURN(HIPRTC_ERROR_NAME_EXPRESSION_NOT_VALID);
  }

  HIPRTC_RETURN(HIPRTC_SUCCESS);
}

hiprtcResult hiprtcDestroyProgram(hiprtcProgram* prog) {
  HIPRTC_INIT_API(prog);

  if (prog == nullptr) {
    HIPRTC_RETURN(HIPRTC_ERROR_INVALID_INPUT);
  }
  auto* rtcProgram = hiprtc::RTCCompileProgram::as_RTCCompileProgram(*prog);
  delete rtcProgram;

  HIPRTC_RETURN(HIPRTC_SUCCESS);
}

hiprtcResult hiprtcGetCodeSize(hiprtcProgram prog, size_t* binarySizeRet) {
  HIPRTC_INIT_API(prog, binarySizeRet);

  if (binarySizeRet == nullptr) {
    HIPRTC_RETURN(HIPRTC_ERROR_INVALID_INPUT);
  }
  auto* rtcProgram = hiprtc::RTCCompileProgram::as_RTCCompileProgram(prog);

  *binarySizeRet = rtcProgram->getExecSize();

  HIPRTC_RETURN(HIPRTC_SUCCESS);
}

hiprtcResult hiprtcGetCode(hiprtcProgram prog, char* binaryMem) {
  HIPRTC_INIT_API(prog, binaryMem);

  if (binaryMem == nullptr) {
    HIPRTC_RETURN(HIPRTC_ERROR_INVALID_INPUT);
  }
  auto* rtcProgram = hiprtc::RTCCompileProgram::as_RTCCompileProgram(prog);
  auto binary = rtcProgram->getExec();
  ::memcpy(binaryMem, binary.data(), binary.size());

  HIPRTC_RETURN(HIPRTC_SUCCESS);
}

hiprtcResult hiprtcGetProgramLog(hiprtcProgram prog, char* dst) {
  HIPRTC_INIT_API(prog, dst);
  if (dst == nullptr) {
    HIPRTC_RETURN(HIPRTC_ERROR_INVALID_INPUT);
  }
  auto* rtcProgram = hiprtc::RTCCompileProgram::as_RTCCompileProgram(prog);
  auto log = rtcProgram->getLog();
  ::memcpy(dst, log.data(), log.size());

  HIPRTC_RETURN(HIPRTC_SUCCESS);
}

hiprtcResult hiprtcGetProgramLogSize(hiprtcProgram prog, size_t* logSizeRet) {
  HIPRTC_INIT_API(prog, logSizeRet);
  if (logSizeRet == nullptr) {
    HIPRTC_RETURN(HIPRTC_ERROR_INVALID_INPUT);
  }
  auto* rtcProgram = hiprtc::RTCCompileProgram::as_RTCCompileProgram(prog);

  *logSizeRet = rtcProgram->getLogSize();

  HIPRTC_RETURN(HIPRTC_SUCCESS);
}

hiprtcResult hiprtcVersion(int* major, int* minor) {
  HIPRTC_INIT_API(major, minor);

  if (major == nullptr || minor == nullptr) {
    HIPRTC_RETURN(HIPRTC_ERROR_INVALID_INPUT);
  }

  // TODO add actual version, what do these numbers mean
  *major = 9;
  *minor = 0;

  HIPRTC_RETURN(HIPRTC_SUCCESS);
}

hiprtcResult hiprtcGetBitcode(hiprtcProgram prog, char* bitcode) {
  HIPRTC_INIT_API(prog, bitcode);

  if (bitcode == nullptr) {
    HIPRTC_RETURN(HIPRTC_ERROR_INVALID_INPUT);
  }

  auto* rtcProgram = hiprtc::RTCCompileProgram::as_RTCCompileProgram(prog);
  if (!rtcProgram->GetBitcode(bitcode)) {
    HIPRTC_RETURN(HIPRTC_ERROR_INVALID_PROGRAM);
  }

  HIPRTC_RETURN(HIPRTC_SUCCESS);
}

hiprtcResult hiprtcGetBitcodeSize(hiprtcProgram prog, size_t* bitcode_size) {
  HIPRTC_INIT_API(prog, bitcode_size);

  if (bitcode_size == nullptr) {
    HIPRTC_RETURN(HIPRTC_ERROR_INVALID_INPUT);
  }

  auto* rtcProgram = hiprtc::RTCCompileProgram::as_RTCCompileProgram(prog);
  if (!rtcProgram->GetBitcodeSize(bitcode_size)) {
    HIPRTC_RETURN(HIPRTC_ERROR_INVALID_PROGRAM);
  }

  HIPRTC_RETURN(HIPRTC_SUCCESS);
}

hiprtcResult hiprtcLinkCreate(unsigned int num_options, hiprtcJIT_option* options_ptr,
                              void** options_vals_pptr, hiprtcLinkState* hip_link_state_ptr) {
  HIPRTC_INIT_API(num_options, options_ptr, options_vals_pptr, hip_link_state_ptr);

  if (hip_link_state_ptr == nullptr) {
    HIPRTC_RETURN(HIPRTC_ERROR_INVALID_INPUT);
  }

  if (num_options != 0) {
    for (int i = 0; i < num_options; i++) {
      if (options_ptr == nullptr || options_vals_pptr == nullptr) {
        HIPRTC_RETURN(HIPRTC_ERROR_INVALID_INPUT);
      }
    }
  }

  std::string name("LinkerProgram");
  hip::LinkProgram* rtc_link_prog_ptr = new hip::LinkProgram(name);
  if (!rtc_link_prog_ptr->AddLinkerOptions(num_options, options_ptr, options_vals_pptr)) {
    HIPRTC_RETURN(HIPRTC_ERROR_INVALID_OPTION);
  }

  *hip_link_state_ptr = reinterpret_cast<hiprtcLinkState>(rtc_link_prog_ptr);

  HIPRTC_RETURN(HIPRTC_SUCCESS);
}

hiprtcResult hiprtcLinkAddFile(hiprtcLinkState hip_link_state, hiprtcJITInputType input_type,
                               const char* file_path, unsigned int num_options,
                               hiprtcJIT_option* options_ptr, void** option_values) {
  HIPRTC_INIT_API(hip_link_state, input_type, file_path, num_options, options_ptr, option_values);

  if (hip_link_state == nullptr) {
    HIPRTC_RETURN(HIPRTC_ERROR_INVALID_INPUT);
  }

  if (input_type == HIPRTC_JIT_INPUT_CUBIN || input_type == HIPRTC_JIT_INPUT_PTX ||
      input_type == HIPRTC_JIT_INPUT_FATBINARY || input_type == HIPRTC_JIT_INPUT_OBJECT ||
      input_type == HIPRTC_JIT_INPUT_LIBRARY || input_type == HIPRTC_JIT_INPUT_NVVM ||
      input_type == HIPRTC_JIT_INPUT_SPIRV) {
    HIPRTC_RETURN(HIPRTC_ERROR_INVALID_INPUT);
  }

  hip::LinkProgram* rtc_link_prog_ptr = reinterpret_cast<hip::LinkProgram*>(hip_link_state);

  if (!hip::LinkProgram::isLinkerValid(rtc_link_prog_ptr)) {
    HIPRTC_RETURN(HIPRTC_ERROR_INVALID_INPUT);
  }

  if (!rtc_link_prog_ptr->AddLinkerFile(std::string(file_path), input_type)) {
    HIPRTC_RETURN(HIPRTC_ERROR_PROGRAM_CREATION_FAILURE);
  }

  HIPRTC_RETURN(HIPRTC_SUCCESS);
}

hiprtcResult hiprtcLinkAddData(hiprtcLinkState hip_link_state, hiprtcJITInputType input_type,
                               void* image, size_t image_size, const char* name,
                               unsigned int num_options, hiprtcJIT_option* options_ptr,
                               void** option_values) {
  HIPRTC_INIT_API(hip_link_state, image, image_size, name, num_options, options_ptr, option_values);

  if (image == nullptr || image_size <= 0) {
    HIPRTC_RETURN(HIPRTC_ERROR_INVALID_INPUT);
  }

  if (input_type == HIPRTC_JIT_INPUT_CUBIN || input_type == HIPRTC_JIT_INPUT_PTX ||
      input_type == HIPRTC_JIT_INPUT_FATBINARY || input_type == HIPRTC_JIT_INPUT_OBJECT ||
      input_type == HIPRTC_JIT_INPUT_LIBRARY || input_type == HIPRTC_JIT_INPUT_NVVM ||
      input_type == HIPRTC_JIT_INPUT_SPIRV) {
    HIPRTC_RETURN(HIPRTC_ERROR_INVALID_INPUT);
  }

  std::string input_name;
  if (name) {
    input_name = name;
  }

  hip::LinkProgram* rtc_link_prog_ptr = reinterpret_cast<hip::LinkProgram*>(hip_link_state);

  if (!hip::LinkProgram::isLinkerValid(rtc_link_prog_ptr)) {
    HIPRTC_RETURN(HIPRTC_ERROR_INVALID_INPUT);
  }

  if (!rtc_link_prog_ptr->AddLinkerData(image, image_size, input_name, input_type)) {
    HIPRTC_RETURN(HIPRTC_ERROR_PROGRAM_CREATION_FAILURE);
  }

  HIPRTC_RETURN(HIPRTC_SUCCESS);
}

hiprtcResult hiprtcLinkComplete(hiprtcLinkState hip_link_state, void** bin_out, size_t* size_out) {
  HIPRTC_INIT_API(hip_link_state, bin_out, size_out);

  if (bin_out == nullptr || size_out == nullptr) {
    HIPRTC_RETURN(HIPRTC_ERROR_INVALID_INPUT);
  }

  hip::LinkProgram* rtc_link_prog_ptr = reinterpret_cast<hip::LinkProgram*>(hip_link_state);

  if (!hip::LinkProgram::isLinkerValid(rtc_link_prog_ptr)) {
    HIPRTC_RETURN(HIPRTC_ERROR_INVALID_INPUT);
  }

  if (!rtc_link_prog_ptr->LinkComplete(bin_out, size_out)) {
    HIPRTC_RETURN(HIPRTC_ERROR_LINKING);
  }

  HIPRTC_RETURN(HIPRTC_SUCCESS);
}

hiprtcResult hiprtcLinkDestroy(hiprtcLinkState hip_link_state) {
  HIPRTC_INIT_API(hip_link_state);

  hip::LinkProgram* rtc_link_prog_ptr = reinterpret_cast<hip::LinkProgram*>(hip_link_state);

  if (!hip::LinkProgram::isLinkerValid(rtc_link_prog_ptr)) {
    HIPRTC_RETURN(HIPRTC_ERROR_INVALID_INPUT);
  }

  delete rtc_link_prog_ptr;
  HIPRTC_RETURN(HIPRTC_SUCCESS);
}
