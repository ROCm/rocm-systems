/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

// Linux-specific ThunkLoader implementation

#include "core/runtime/thunk_loader_platform.h"
#include "core/inc/runtime.h"
#include <core/util/os.h>
#include <fcntl.h>
#include <unistd.h>

namespace rocr {
namespace core {

const char* GetDtifLibraryName() {
  return "libdtif.so";
}

const char* GetDxgLibraryName() {
  return "librocdxg.so";
}

bool DetectDxgDriver() {
  // Only check for DXG if detection is enabled
  if (!core::Runtime::runtime_singleton_->flag().enable_dxg_detection()) {
    return false;
  }
  // Check for /dev/dxg (WSL2 DXG driver)
  int fd = open("/dev/dxg", O_RDWR);
  if (fd >= 0) {
    close(fd);
    return true;
  }
  return false;
}

bool LoadPlatformDynamicApis(ThunkLoader* loader, void* thunk_handle) {
  // No Windows-only APIs to load on Linux
  // drmCommandWriteRead is loaded in the common path
  return true;
}

void BindPlatformStaticApis(ThunkLoader* loader) {
  loader->DRM_PFN(drmCommandWriteRead) =
      (ThunkLoader::DRM_DEF(drmCommandWriteRead)*)(&drmCommandWriteRead);
}

}  // namespace core
}  // namespace rocr
