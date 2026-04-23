/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include "thread/thread.hpp"
#include "platform/runtime.hpp"

#include <windows.h>
#include <iostream>

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
      break;
    case DLL_PROCESS_DETACH:
      amd::Runtime::setLibraryDetached();
      break;
    case DLL_THREAD_DETACH:
      break;
    default:
      break;
  }
  return true;
}
