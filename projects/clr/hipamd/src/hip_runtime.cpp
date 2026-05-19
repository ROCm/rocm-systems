/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

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
      hip::PlatformState::Instance().SetDynamicLibraryHandle(static_cast<void*>(hinst));
      break;
    case DLL_PROCESS_DETACH:
      if (GPU_ENABLE_PAL != 0) {
        hip::ihipDestroyDevice();
      }
      break;
    case DLL_THREAD_DETACH:
      break;
    default:
      break;
  }
  return true;
}
