/*
 * Copyright (C) Advanced Micro Devices, Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included
 * in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
 * OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE COPYRIGHT HOLDER(S) BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN
 * AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
 * CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */

#include <cstdlib>
#include <hip_test_common.hh>
#include <picojson.h>
#include <fstream>
#include <sstream>
#include <regex>
#include "hip_test_context.hh"
#include "hip_test_filesystem.hh"
#include "hip_test_features.hh"

void TestContext::detectOS() {
#if (HT_WIN == 1)
  p_windows = true;
#elif (HT_LINUX == 1)
  p_linux = true;
#endif
}

void TestContext::detectPlatform() {
#if (HT_AMD == 1)
  amd = true;
#elif (HT_NVIDIA == 1)
  nvidia = true;
#endif
}

TestContext::TestContext() {
  detectOS();
  detectPlatform();
}

bool TestContext::isWindows() const { return p_windows; }
bool TestContext::isLinux() const { return p_linux; }

bool TestContext::isNvidia() const { return nvidia; }
bool TestContext::isAmd() const { return amd; }

std::string TestContext::currentPath() const { return fs::current_path().string(); }

void TestContext::cleanContext() {
  for (auto& pair : compiledKernels) {
    hipError_t error = hipModuleUnload(pair.second.module);
    if (error != hipSuccess) {
      throw std::runtime_error("Unable to unload rtc module");
    }
  }
}

void TestContext::trackRtcState(std::string kernelNameExpression, hipModule_t loadedModule,
                                hipFunction_t kernelFunction) {
  rtcState state{loadedModule, kernelFunction};
  compiledKernels[kernelNameExpression] = state;
}

hipFunction_t TestContext::getFunction(const std::string kernelNameExpression) {
  auto it{compiledKernels.find(kernelNameExpression)};

  if (it != compiledKernels.end()) {
    return it->second.kernelFunction;
  } else {
    return nullptr;
  }
}

void TestContext::addResults(HCResult r) {
  std::unique_lock<std::mutex> lock(resultMutex);
  results.push_back(r);
  if ((!r.conditionsResult) ||
      ((r.result != hipSuccess) && (r.result != hipErrorPeerAccessAlreadyEnabled))) {
    hasErrorOccured_.store(true);
  }
}

void TestContext::finalizeResults() {
  std::unique_lock<std::mutex> lock(resultMutex);
  // clear the results whatever happens
  std::shared_ptr<void> emptyVec(nullptr, [this](auto) { results.clear(); });

  for (const auto& i : results) {
    INFO("HIP API Result check\n    File:: "
         << i.file << "\n    Line:: " << i.line << "\n    API:: " << i.call
         << "\n    Result:: " << i.result << "\n    Result Str:: " << hipGetErrorString(i.result));
    REQUIRE(((i.result == hipSuccess) || (i.result == hipErrorPeerAccessAlreadyEnabled) ||
             (i.result == hipErrorNotSupported)));
    REQUIRE(i.conditionsResult);
  }
  hasErrorOccured_.store(false);  // Clear the flag
}

bool TestContext::hasErrorOccured() { return hasErrorOccured_.load(); }

TestContext::~TestContext() {
  // Show this message when there are unchecked results
  if (results.size() != 0) {
    std::cerr << "HIP_CHECK_THREAD_FINALIZE() has not been called after HIP_CHECK_THREAD\n"
              << "Please call HIP_CHECK_THREAD_FINALIZE after joining threads\n"
              << "There is/are " << results.size() << " unchecked results from threads."
              << std::endl;
    std::abort();  // Crash to bring users attention to this message and avoid accidental passing of
                   // tests without checking for errors
  }
}
