/* Copyright (c) 2008 - 2022 Advanced Micro Devices, Inc.

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
 FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 THE SOFTWARE. */

#include <windows.h>
#include <iostream>
#include <hip/hip_runtime.h>

#include "thread/thread.hpp"
#include "hip_platform.hpp"

namespace hip {
void ihipDestroyDevice();
}

#ifdef DEBUG
static int reportHook(int reportType, char* message, int* returnValue) {
  if (returnValue) {
    *returnValue = 1;
  }
  std::cerr << message;
  ::exit(3);
  return TRUE;
}
#endif  // DEBUG

extern "C" BOOL WINAPI DllMain(HINSTANCE hinst, DWORD reason, LPVOID reserved) {
  switch (reason) {
    case DLL_PROCESS_ATTACH:
#ifdef DEBUG
      if (!::getenv("AMD_OCL_ENABLE_MESSAGE_BOX")) {
        _CrtSetReportHook(reportHook);
        _set_error_mode(_OUT_TO_STDERR);
      }
#endif  // DEBUG
      hip::PlatformState::instance().setDynamicLibraryHandle(static_cast<void*>(hinst));
      break;
    case DLL_PROCESS_DETACH:
      hip::ihipDestroyDevice();
      break;
    case DLL_THREAD_DETACH:
      break;
    default:
      break;
  }
  return true;
}
