/*
Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
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

#include <hip_test_common.hh>
#include <hip_test_filesystem.hh>

#include <windows.h>

/**
 * Test Description
 * ------------------------
 *  - Test for: "Don't wait on command completion if worker thread
 *    is destroyed" (rocm-systems PR#3790).
 *  - Spawns a subprocess that creates a HIP stream, queues async GPU work,
 *    then kills the stream's internal worker thread via TerminateThread before
 *    process exit. Validates that the atexit cleanup path doesn't hang when
 *    finish() is called on a stream whose worker thread is already dead.
 * Test source
 * ------------------------
 *  - unit/stream/hipGracefulExit.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 6.3
 */

static constexpr DWORD kTimeoutMs = 10000;

HIP_TEST_CASE(Unit_hipStream_GracefulExitWithPendingWork) {
  auto exePath = fs::path(TestContext::get().currentPath()) / "hipGracefulExit_exe";
  std::string exeStr = exePath.string();
  if (exePath.extension().empty()) exeStr += ".exe";

  STARTUPINFOA si{};
  PROCESS_INFORMATION pi{};
  si.cb = sizeof(si);

  REQUIRE(CreateProcessA(exeStr.c_str(), nullptr, nullptr, nullptr,
                         FALSE, 0, nullptr, nullptr, &si, &pi));

  DWORD waitResult = WaitForSingleObject(pi.hProcess, kTimeoutMs);
  if (waitResult == WAIT_TIMEOUT) {
    TerminateProcess(pi.hProcess, 1);
    CloseHandle(pi.hProcess);
    CloseHandle(pi.hThread);
    FAIL("hipGracefulExit_exe timed out after " << (kTimeoutMs / 1000) << " seconds");
  }

  DWORD exitCode = 0;
  REQUIRE(GetExitCodeProcess(pi.hProcess, &exitCode));
  CloseHandle(pi.hProcess);
  CloseHandle(pi.hThread);
  REQUIRE(exitCode == 0);
}
