/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

// Windows-specific ThunkLoader implementation

#include "core/runtime/thunk_loader_platform.h"
#include "hsakmt/hsakmt.h"
#include <core/util/os.h>

namespace rocr {
namespace core {

const char* GetDtifLibraryName() {
  return "dtif64a.dll";
}

const char* GetDxgLibraryName() {
  // On Windows, DXG is the default path (no separate library)
  return "";
}

bool DetectDxgDriver() {
  // On Windows, always use DXG path
  return true;
}

bool LoadPlatformDynamicApis(ThunkLoader* loader, void* thunk_handle) {
  loader->HSAKMT_PFN(hsaKmtGetMemoryHandle) =
      (ThunkLoader::HSAKMT_DEF(hsaKmtGetMemoryHandle)*)rocr::os::GetExportAddress(
          thunk_handle, "hsaKmtGetMemoryHandle");
  if (loader->HSAKMT_PFN(hsaKmtGetMemoryHandle) == nullptr) return false;

  return true;
}

void BindPlatformStaticApis(ThunkLoader* loader) {
  loader->HSAKMT_PFN(hsaKmtQueueRingDoorbell) =
      (ThunkLoader::HSAKMT_DEF(hsaKmtQueueRingDoorbell)*)(&hsaKmtQueueRingDoorbell);
  loader->HSAKMT_PFN(hsaKmtGetMemoryHandle) =
      (ThunkLoader::HSAKMT_DEF(hsaKmtGetMemoryHandle)*)(&hsaKmtGetMemoryHandle);
}

}  // namespace core
}  // namespace rocr
