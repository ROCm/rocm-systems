/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

// Platform-specific helpers for ThunkLoader
// Implemented in thunk_loader_win.cpp and thunk_loader_linux.cpp

#ifndef HSA_RUNTIME_CORE_RUNTIME_THUNK_LOADER_PLATFORM_H
#define HSA_RUNTIME_CORE_RUNTIME_THUNK_LOADER_PLATFORM_H

#include "core/inc/thunk_loader.h"

namespace rocr {
namespace core {

// Returns the platform-specific DTIF library name
const char* GetDtifLibraryName();

// Returns the platform-specific DXG library name (empty if not applicable)
const char* GetDxgLibraryName();

// Checks if DXG driver is available
// On Windows: always returns true
// On Linux: checks for /dev/dxg (WSL2)
bool DetectDxgDriver();

// Load platform-specific APIs from dynamic library
// Called when IsSharedLibraryLoaded() is true
bool LoadPlatformDynamicApis(ThunkLoader* loader, void* thunk_handle);

// Bind platform-specific APIs to static library functions
// Called when IsSharedLibraryLoaded() is false
void BindPlatformStaticApis(ThunkLoader* loader);

}  // namespace core
}  // namespace rocr

#endif  // HSA_RUNTIME_CORE_RUNTIME_THUNK_LOADER_PLATFORM_H
