/*
 * Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
 */

#include "dxcore_loader.h"
#include "librocdxg.h"
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <ntstatus.h>
#include <util/os.h>

namespace wsl {
namespace thunk {
namespace dxcore {

DxcoreLoader::DxcoreLoader()
    : dxcore_handle_(nullptr)
    , init_flag_()
    , pfn_D3DKMTCreateAllocation2(nullptr)
    , pfn_D3DKMTDestroyAllocation2(nullptr)
    , pfn_D3DKMTMapGpuVirtualAddress(nullptr)
    , pfn_D3DKMTReserveGpuVirtualAddress(nullptr)
    , pfn_D3DKMTFreeGpuVirtualAddress(nullptr)
    , pfn_D3DKMTCreateDevice(nullptr)
    , pfn_D3DKMTDestroyDevice(nullptr)
    , pfn_D3DKMTEnumAdapters2(nullptr)
    , pfn_D3DKMTQueryAdapterInfo(nullptr)
    , pfn_D3DKMTCreateContextVirtual(nullptr)
    , pfn_D3DKMTDestroyContext(nullptr)
    , pfn_D3DKMTSubmitCommand(nullptr)
    , pfn_D3DKMTCreateSynchronizationObject2(nullptr)
    , pfn_D3DKMTDestroySynchronizationObject(nullptr)
    , pfn_D3DKMTQueryStatistics(nullptr)
    , pfn_D3DKMTEscape(nullptr)
    , pfn_D3DKMTLock2(nullptr)
    , pfn_D3DKMTUnlock2(nullptr)
    , pfn_D3DKMTCreatePagingQueue(nullptr)
    , pfn_D3DKMTDestroyPagingQueue(nullptr)
    , pfn_D3DKMTWaitForSynchronizationObjectFromGpu(nullptr)
    , pfn_D3DKMTSignalSynchronizationObjectFromGpu(nullptr)
    , pfn_D3DKMTWaitForSynchronizationObjectFromCpu(nullptr)
    , pfn_D3DKMTQueryClockCalibration(nullptr)
    , pfn_D3DKMTMakeResident(nullptr)
    , pfn_D3DKMTEvict(nullptr)
    , pfn_D3DKMTShareObjects(nullptr)
    , pfn_D3DKMTQueryResourceInfoFromNtHandle(nullptr)
    , pfn_D3DKMTOpenResourceFromNtHandle(nullptr)
    , pfn_D3DKMTCreateHwQueue(nullptr)
    , pfn_D3DKMTDestroyHwQueue(nullptr)
    , pfn_D3DKMTSubmitCommandToHwQueue(nullptr)
    , pfn_D3DKMTEnumAdapters3(nullptr)
    , pfn_D3DKMTQueryResourceInfo(nullptr)
    , pfn_D3DKMTOpenResource(nullptr) {
}

DxcoreLoader::~DxcoreLoader() {
    Shutdown();
}

bool DxcoreLoader::Initialize() {
    std::ignore = rocr::os::DlError(); // Clear error
#if defined(__linux__)
    constexpr std::string_view dxcore_lib_name = "libdxcore.so";
#else
    constexpr std::string_view dxcore_lib_name = "Gdi32.dll";
#endif
    dxcore_handle_ = rocr::os::LoadLib(dxcore_lib_name.data());
    if (!dxcore_handle_) {
        pr_err("[DxcoreLoader] Cannot load libdxcore.so: %s\n", rocr::os::DlError());
        return false;
    }

    pr_info("[DxcoreLoader] libdxcore.so loaded successfully\n");
    if (!LoadDxcoreApis()) {
        // If API loading failed, close the handle to indicate failure
        rocr::os::CloseLib(dxcore_handle_);
        dxcore_handle_ = nullptr;
        return false;
    }

    return IsLoaded();
}

void DxcoreLoader::Shutdown() {
    if (dxcore_handle_) {
        if (rocr::os::CloseLib(dxcore_handle_) != 0) {
            pr_err("[DxcoreLoader] Cannot unload libdxcore.so: %s\n", rocr::os::DlError());
        } else {
            pr_info("[DxcoreLoader] libdxcore.so unloaded successfully\n");
        }
        dxcore_handle_ = nullptr;
    }
}

bool DxcoreLoader::LoadDxcoreApis() {
    if (!dxcore_handle_) {
        pr_err("[DxcoreLoader] Error: dxcore_handle_ is null\n");
        return false;
    }

    std::ignore = rocr::os::DlError(); // Clear error

    // Load all D3DKMT functions
    #define LOAD_DXCORE_API(func_name) \
        DXCORE_PFN(func_name) = (DXCORE_DEF(func_name)*)rocr::os::GetExportAddress(dxcore_handle_, #func_name); \
        if (!DXCORE_PFN(func_name)) { \
            pr_err("[DxcoreLoader] Failed to load " #func_name ": %s\n", rocr::os::DlError()); \
            goto ERROR_LOAD; \
        }

    LOAD_DXCORE_API(D3DKMTCreateAllocation2);
    LOAD_DXCORE_API(D3DKMTDestroyAllocation2);
    LOAD_DXCORE_API(D3DKMTMapGpuVirtualAddress);
    LOAD_DXCORE_API(D3DKMTReserveGpuVirtualAddress);
    LOAD_DXCORE_API(D3DKMTFreeGpuVirtualAddress);
    LOAD_DXCORE_API(D3DKMTCreateDevice);
    LOAD_DXCORE_API(D3DKMTDestroyDevice);
    LOAD_DXCORE_API(D3DKMTEnumAdapters2);
    LOAD_DXCORE_API(D3DKMTEnumAdapters3);
    LOAD_DXCORE_API(D3DKMTQueryAdapterInfo);
    LOAD_DXCORE_API(D3DKMTCreateContextVirtual);
    LOAD_DXCORE_API(D3DKMTDestroyContext);
    LOAD_DXCORE_API(D3DKMTSubmitCommand);
    LOAD_DXCORE_API(D3DKMTCreateSynchronizationObject2);
    LOAD_DXCORE_API(D3DKMTDestroySynchronizationObject);
    LOAD_DXCORE_API(D3DKMTQueryStatistics);
    LOAD_DXCORE_API(D3DKMTEscape);
    LOAD_DXCORE_API(D3DKMTLock2);
    LOAD_DXCORE_API(D3DKMTUnlock2);
    LOAD_DXCORE_API(D3DKMTCreatePagingQueue);
    LOAD_DXCORE_API(D3DKMTDestroyPagingQueue);
    LOAD_DXCORE_API(D3DKMTWaitForSynchronizationObjectFromGpu);
    LOAD_DXCORE_API(D3DKMTSignalSynchronizationObjectFromGpu);
    LOAD_DXCORE_API(D3DKMTWaitForSynchronizationObjectFromCpu);
    LOAD_DXCORE_API(D3DKMTQueryClockCalibration);
    LOAD_DXCORE_API(D3DKMTMakeResident);
    LOAD_DXCORE_API(D3DKMTEvict);
    LOAD_DXCORE_API(D3DKMTShareObjects);
    LOAD_DXCORE_API(D3DKMTQueryResourceInfoFromNtHandle);
    LOAD_DXCORE_API(D3DKMTQueryResourceInfo);
    LOAD_DXCORE_API(D3DKMTOpenResourceFromNtHandle);
    LOAD_DXCORE_API(D3DKMTOpenResource);
    LOAD_DXCORE_API(D3DKMTCreateHwQueue);
    LOAD_DXCORE_API(D3DKMTDestroyHwQueue);
    LOAD_DXCORE_API(D3DKMTSubmitCommandToHwQueue);

    #undef LOAD_DXCORE_API

    pr_info("[DxcoreLoader] All DXCore APIs loaded successfully\n");
    return true;
ERROR_LOAD:
    pr_err("[DxcoreLoader] Failed to load DXCore APIs\n");
    return false;
}

} // namespace dxcore
} // namespace thunk
} // namespace wsl
