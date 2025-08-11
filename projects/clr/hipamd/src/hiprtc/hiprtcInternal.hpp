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

#include <hip/hip_runtime.h>
#include <hip/hiprtc.h>
#include <hip/hip_version.h>


#ifdef HIPRTC_USE_EXCEPTIONS
#include <exception>
#endif
#include <atomic>
#include <map>
#include <mutex>
#include <string>

#include "top.hpp"
#include "rocclr/utils/debug.hpp"
#include "rocclr/utils/flags.hpp"
#include "rocclr/utils/macros.hpp"
#include "vdi_common.hpp"
#include "device/comgrctx.hpp"
#include "../hip_comgr_helper.hpp"


#ifdef __HIP_ENABLE_RTC
extern "C" {
extern const char __hipRTC_header[];
extern unsigned __hipRTC_header_size;
}
#endif

namespace hiprtc {
namespace internal {
template <typename T> inline std::string ToString(T v) {
  std::ostringstream ss;
  ss << v;
  return ss.str();
}

template <typename T> inline std::string ToString(T* v) {
  std::ostringstream ss;
  if (v == nullptr) {
    ss << "<null>";
  } else {
    ss << v;
  }
  return ss.str();
};

inline std::string ToString() { return (""); }

template <typename T, typename... Args> inline std::string ToString(T first, Args... args) {
  return ToString(first) + ", " + ToString(args...);
}
}  // namespace internal
}  // namespace hiprtc

// hiprtcInit lock
static amd::Monitor g_hiprtcInitlock{};
#define HIPRTC_INIT_API_INTERNAL(...)                                                              \
  amd::ScopedLock lock(g_hiprtcInitlock);                                                          \
  if (!amd::Flag::init()) {                                                                        \
    HIPRTC_RETURN(HIPRTC_ERROR_INTERNAL_ERROR);                                                    \
  }

#define HIPRTC_INIT_API(...)                                                                       \
  HIPRTC_INIT_API_INTERNAL(0, __VA_ARGS__)                                                         \
  ClPrint(amd::LOG_INFO, amd::LOG_API, "%s ( %s )", __func__,                                      \
          hiprtc::internal::ToString(__VA_ARGS__).c_str());

#define HIPRTC_RETURN(ret)                                                                         \
  hiprtc::tls.last_rtc_error_ = (ret);                                                             \
  ClPrint(amd::LOG_INFO, amd::LOG_API, "%s: Returned %s", __func__,                                \
          hiprtcGetErrorString(hiprtc::tls.last_rtc_error_));                                      \
  return hiprtc::tls.last_rtc_error_;

namespace hiprtc {

static void crashWithMessage(std::string message) {
#ifdef HIPRTC_USE_EXCEPTIONS
  throw std::runtime_error(message);
#else
  guarantee(false, message.c_str());
#endif
}

struct Settings {
  bool offloadArchProvided{false};
};

class RTCCompileProgram : public hip::RTCProgram {
  // Private Data Members
  Settings settings_;

  std::string source_code_;
  std::string source_name_;
  std::map<std::string, std::string> mangled_names_;

  std::vector<std::string> compile_options_;
  std::vector<std::string> link_options_;

  amd_comgr_data_set_t compile_input_;
  amd_comgr_data_set_t link_input_;

  bool fgpu_rdc_;
  std::vector<char> LLVMBitcode_;

  // Private Member functions
  bool addSource_impl();
  bool addBuiltinHeader();
  bool transformOptions(std::vector<std::string>& compile_options);
  bool findExeOptions(const std::vector<std::string>& options,
                      std::vector<std::string>& exe_options);
  void AppendCompileOptions() { AppendOptions(HIPRTC_COMPILE_OPTIONS_APPEND, &compile_options_); }

  RTCCompileProgram() = delete;
  RTCCompileProgram(RTCCompileProgram&) = delete;
  RTCCompileProgram& operator=(RTCCompileProgram&) = delete;

 public:
  RTCCompileProgram(std::string);
  ~RTCCompileProgram() {
    amd::Comgr::destroy_data_set(compile_input_);
    amd::Comgr::destroy_data_set(link_input_);
  }

  // Converters
  inline static hiprtcProgram as_hiprtcProgram(RTCCompileProgram* p) {
    return reinterpret_cast<hiprtcProgram>(p);
  }
  inline static RTCCompileProgram* as_RTCCompileProgram(hiprtcProgram& p) {
    return reinterpret_cast<RTCCompileProgram*>(p);
  }

  // Public Member Functions
  bool addSource(const std::string& source, const std::string& name);
  bool addHeader(const std::string& source, const std::string& name);
  bool compile(const std::vector<std::string>& options, bool fgpu_rdc);
  bool getMangledName(const char* name_expression, const char** loweredName);
  bool trackMangledName(std::string& name);
  void stripNamedExpression(std::string& named_expression);

  bool GetBitcode(char* bitcode);
  bool GetBitcodeSize(size_t* bitcode_size);
  // Public Getter/Setters
  const std::vector<char>& getExec() const { return executable_; }
  size_t getExecSize() const { return executable_.size(); }
  const std::string& getLog() const { return build_log_; }
  size_t getLogSize() const { return build_log_.size(); }
};

// Thread Local Storage Variables Aggregator Class
class TlsAggregator {
 public:
  hiprtcResult last_rtc_error_;

  TlsAggregator() : last_rtc_error_(HIPRTC_SUCCESS) {}
  ~TlsAggregator() {}
};
extern thread_local TlsAggregator tls;
}  // namespace hiprtc
