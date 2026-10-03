// MIT License
//
// Copyright (c) 2023-2026 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

// Self-containment translation unit for a PREBUILT wkmi.lib linked into
// rocprofiler-sdk OUTSIDE the libhsakmt/hsa-runtime build (the prebuilt-fallback
// branch in this directory's CMakeLists.txt). It is NOT compiled when wkmi is
// built from source or when the wkmi bridge degrades to its D3DKMT-only stub.
//
// The prebuilt wkmi.lib's only undefined non-CRT external is the private default
// constructor of wsl::thunk::dxcore::DxcoreLoader, pulled in by the inline
// DxcoreLoader::Instance() (new DxcoreLoader()) that wkmi.cpp calls. In the
// libhsakmt/hsa-runtime build, libhsakmt compiles the real dxcore_loader.cpp and
// the host runtime initializes the loader's D3DKMT thunk pointers before any
// wkmi entry point runs. Linked standalone here, neither happens: the ctor is
// unresolved (LNK2019), and even once defined, the thunks would be null when
// wkmi's ParseAdapterInfo invokes them via DXCORE_CALL -> a null-pointer call.
//
// This TU therefore (1) provides the missing ctor symbol and (2) makes that ctor
// resolve the D3DKMT thunks from gdi32.dll itself - exactly the work the real
// DxcoreLoader::Initialize()/LoadDxcoreApis() would do, which dumpbin confirms
// wkmi.lib never calls (the only DxcoreLoader references are this ctor and the
// inline Instance() statics). The class is reproduced from
// libhsakmt/src/dxg/dxcore_loader.h so the member layout and the ctor's mangled
// name match exactly; the only edit is swapping the "impl/wddm/types.h" include
// for the trivial type stand-ins it provided (all used solely as pointer/return
// types, so they affect neither layout nor mangling).

#if defined(_WIN32)

// NOLINTBEGIN: this TU deliberately reproduces a third-party class verbatim (its
// member-pointer naming, function-pointer typedefs, the DXCORE_* macros, and the
// reinterpret_casts from FARPROC are required for the mangled-name / layout match
// with the prebuilt wkmi.lib and are intentionally exempt from project linting).

#    include <cstddef>
#    include <cstdint>
#    include <mutex>

using NTSTATUS          = long;
using D3DKMT_HANDLE     = unsigned int;
using WinResourceHandle = D3DKMT_HANDLE;
struct OBJECT_ATTRIBUTES;  // only ever used as a pointer in a typedef below

// Minimal Win32 loader API declarations (avoid pulling in <windows.h>, whose
// NTSTATUS/HANDLE typedefs would collide with the stand-ins above). Resolved at
// link time from the kernel32 import lib by their extern-"C" names; the
// simplified void* return type is binary-compatible with FARPROC on x64.
extern "C" __declspec(dllimport) void* __stdcall LoadLibraryA(const char* name);
extern "C" __declspec(dllimport) void* __stdcall GetProcAddress(void* module, const char* proc);

#    define DXCORE_DEF(function_name) PFN##function_name
#    define DXCORE_PFN(function_name) pfn_##function_name
#    define DXCORE_RESOLVE(function_name)                                                          \
        DXCORE_PFN(function_name) =                                                                \
            reinterpret_cast<DXCORE_DEF(function_name)*>(::GetProcAddress(mod, #function_name))

namespace wsl
{
namespace thunk
{
namespace dxcore
{
class DxcoreLoader
{
public:
    typedef NTSTATUS(DXCORE_DEF(D3DKMTCreateAllocation2))(void* args);
    typedef NTSTATUS(DXCORE_DEF(D3DKMTDestroyAllocation2))(void* args);
    typedef NTSTATUS(DXCORE_DEF(D3DKMTMapGpuVirtualAddress))(void* args);
    typedef NTSTATUS(DXCORE_DEF(D3DKMTReserveGpuVirtualAddress))(void* args);
    typedef NTSTATUS(DXCORE_DEF(D3DKMTFreeGpuVirtualAddress))(void* args);
    typedef NTSTATUS(DXCORE_DEF(D3DKMTCreateDevice))(void* args);
    typedef NTSTATUS(DXCORE_DEF(D3DKMTDestroyDevice))(void* args);
    typedef NTSTATUS(DXCORE_DEF(D3DKMTEnumAdapters2))(void* args);
    typedef NTSTATUS(DXCORE_DEF(D3DKMTQueryAdapterInfo))(void* args);
    typedef NTSTATUS(DXCORE_DEF(D3DKMTCreateContextVirtual))(void* args);
    typedef NTSTATUS(DXCORE_DEF(D3DKMTDestroyContext))(void* args);
    typedef NTSTATUS(DXCORE_DEF(D3DKMTSubmitCommand))(void* args);
    typedef NTSTATUS(DXCORE_DEF(D3DKMTCreateSynchronizationObject2))(void* args);
    typedef NTSTATUS(DXCORE_DEF(D3DKMTDestroySynchronizationObject))(void* args);
    typedef NTSTATUS(DXCORE_DEF(D3DKMTQueryStatistics))(void* args);
    typedef NTSTATUS(DXCORE_DEF(D3DKMTEscape))(void* args);
    typedef NTSTATUS(DXCORE_DEF(D3DKMTLock2))(void* args);
    typedef NTSTATUS(DXCORE_DEF(D3DKMTUnlock2))(void* args);
    typedef NTSTATUS(DXCORE_DEF(D3DKMTCreatePagingQueue))(void* args);
    typedef NTSTATUS(DXCORE_DEF(D3DKMTDestroyPagingQueue))(void* args);
    typedef NTSTATUS(DXCORE_DEF(D3DKMTWaitForSynchronizationObjectFromGpu))(void* args);
    typedef NTSTATUS(DXCORE_DEF(D3DKMTSignalSynchronizationObjectFromGpu))(void* args);
    typedef NTSTATUS(DXCORE_DEF(D3DKMTWaitForSynchronizationObjectFromCpu))(void* args);
    typedef NTSTATUS(DXCORE_DEF(D3DKMTQueryClockCalibration))(void* args);
    typedef NTSTATUS(DXCORE_DEF(D3DKMTMakeResident))(void* args);
    typedef NTSTATUS(DXCORE_DEF(D3DKMTEvict))(void* args);
    typedef NTSTATUS(DXCORE_DEF(D3DKMTShareObjects))(size_t             num_allocations,
                                                     WinResourceHandle* resource,
                                                     OBJECT_ATTRIBUTES* obj_attr,
                                                     uint32_t           flags,
                                                     void**             nt_handle);
    typedef NTSTATUS(DXCORE_DEF(D3DKMTQueryResourceInfoFromNtHandle))(void* args);
    typedef NTSTATUS(DXCORE_DEF(D3DKMTOpenResourceFromNtHandle))(void* args);
    typedef NTSTATUS(DXCORE_DEF(D3DKMTCreateHwQueue))(void* args);
    typedef NTSTATUS(DXCORE_DEF(D3DKMTDestroyHwQueue))(void* args);
    typedef NTSTATUS(DXCORE_DEF(D3DKMTSubmitCommandToHwQueue))(void* args);
    typedef NTSTATUS(DXCORE_DEF(D3DKMTEnumAdapters3))(void* args);
    typedef NTSTATUS(DXCORE_DEF(D3DKMTQueryResourceInfo))(void* args);
    typedef NTSTATUS(DXCORE_DEF(D3DKMTOpenResource))(void* args);

    static DxcoreLoader& Instance()
    {
        static DxcoreLoader* instance = new DxcoreLoader();
        return (*instance);
    }

    bool Initialize();
    void Shutdown();
    bool IsLoaded() const { return dxcore_handle_ != nullptr; }

    DXCORE_DEF(D3DKMTCreateAllocation2) * DXCORE_PFN(D3DKMTCreateAllocation2);
    DXCORE_DEF(D3DKMTDestroyAllocation2) * DXCORE_PFN(D3DKMTDestroyAllocation2);
    DXCORE_DEF(D3DKMTMapGpuVirtualAddress) * DXCORE_PFN(D3DKMTMapGpuVirtualAddress);
    DXCORE_DEF(D3DKMTReserveGpuVirtualAddress) * DXCORE_PFN(D3DKMTReserveGpuVirtualAddress);
    DXCORE_DEF(D3DKMTFreeGpuVirtualAddress) * DXCORE_PFN(D3DKMTFreeGpuVirtualAddress);
    DXCORE_DEF(D3DKMTCreateDevice) * DXCORE_PFN(D3DKMTCreateDevice);
    DXCORE_DEF(D3DKMTDestroyDevice) * DXCORE_PFN(D3DKMTDestroyDevice);
    DXCORE_DEF(D3DKMTEnumAdapters2) * DXCORE_PFN(D3DKMTEnumAdapters2);
    DXCORE_DEF(D3DKMTQueryAdapterInfo) * DXCORE_PFN(D3DKMTQueryAdapterInfo);
    DXCORE_DEF(D3DKMTCreateContextVirtual) * DXCORE_PFN(D3DKMTCreateContextVirtual);
    DXCORE_DEF(D3DKMTDestroyContext) * DXCORE_PFN(D3DKMTDestroyContext);
    DXCORE_DEF(D3DKMTSubmitCommand) * DXCORE_PFN(D3DKMTSubmitCommand);
    DXCORE_DEF(D3DKMTCreateSynchronizationObject2) * DXCORE_PFN(D3DKMTCreateSynchronizationObject2);
    DXCORE_DEF(D3DKMTDestroySynchronizationObject) * DXCORE_PFN(D3DKMTDestroySynchronizationObject);
    DXCORE_DEF(D3DKMTQueryStatistics) * DXCORE_PFN(D3DKMTQueryStatistics);
    DXCORE_DEF(D3DKMTEscape) * DXCORE_PFN(D3DKMTEscape);
    DXCORE_DEF(D3DKMTLock2) * DXCORE_PFN(D3DKMTLock2);
    DXCORE_DEF(D3DKMTUnlock2) * DXCORE_PFN(D3DKMTUnlock2);
    DXCORE_DEF(D3DKMTCreatePagingQueue) * DXCORE_PFN(D3DKMTCreatePagingQueue);
    DXCORE_DEF(D3DKMTDestroyPagingQueue) * DXCORE_PFN(D3DKMTDestroyPagingQueue);
    DXCORE_DEF(D3DKMTWaitForSynchronizationObjectFromGpu) *
        DXCORE_PFN(D3DKMTWaitForSynchronizationObjectFromGpu);
    DXCORE_DEF(D3DKMTSignalSynchronizationObjectFromGpu) *
        DXCORE_PFN(D3DKMTSignalSynchronizationObjectFromGpu);
    DXCORE_DEF(D3DKMTWaitForSynchronizationObjectFromCpu) *
        DXCORE_PFN(D3DKMTWaitForSynchronizationObjectFromCpu);
    DXCORE_DEF(D3DKMTQueryClockCalibration) * DXCORE_PFN(D3DKMTQueryClockCalibration);
    DXCORE_DEF(D3DKMTMakeResident) * DXCORE_PFN(D3DKMTMakeResident);
    DXCORE_DEF(D3DKMTEvict) * DXCORE_PFN(D3DKMTEvict);
    DXCORE_DEF(D3DKMTShareObjects) * DXCORE_PFN(D3DKMTShareObjects);
    DXCORE_DEF(D3DKMTQueryResourceInfoFromNtHandle) *
        DXCORE_PFN(D3DKMTQueryResourceInfoFromNtHandle);
    DXCORE_DEF(D3DKMTOpenResourceFromNtHandle) * DXCORE_PFN(D3DKMTOpenResourceFromNtHandle);
    DXCORE_DEF(D3DKMTCreateHwQueue) * DXCORE_PFN(D3DKMTCreateHwQueue);
    DXCORE_DEF(D3DKMTDestroyHwQueue) * DXCORE_PFN(D3DKMTDestroyHwQueue);
    DXCORE_DEF(D3DKMTSubmitCommandToHwQueue) * DXCORE_PFN(D3DKMTSubmitCommandToHwQueue);
    DXCORE_DEF(D3DKMTEnumAdapters3) * DXCORE_PFN(D3DKMTEnumAdapters3);
    DXCORE_DEF(D3DKMTQueryResourceInfo) * DXCORE_PFN(D3DKMTQueryResourceInfo);
    DXCORE_DEF(D3DKMTOpenResource) * DXCORE_PFN(D3DKMTOpenResource);

private:
    DxcoreLoader();
    ~DxcoreLoader();

    bool LoadDxcoreApis();

    void*          dxcore_handle_;
    std::once_flag init_flag_;

    DxcoreLoader(const DxcoreLoader&) = delete;
    DxcoreLoader& operator=(const DxcoreLoader&) = delete;
};

// Members are null-initialized first (declaration order avoids -Wreorder), then
// overwritten with the resolved gdi32 thunks; any export gdi32 lacks stays null,
// which is fine because wkmi's adapter parsing only touches the query thunks.
DxcoreLoader::DxcoreLoader()
: pfn_D3DKMTCreateAllocation2(nullptr)
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
, pfn_D3DKMTOpenResource(nullptr)
, dxcore_handle_(nullptr)
, init_flag_()
{
    void* mod = ::LoadLibraryA("gdi32.dll");
    if(mod == nullptr) return;

    DXCORE_RESOLVE(D3DKMTCreateAllocation2);
    DXCORE_RESOLVE(D3DKMTDestroyAllocation2);
    DXCORE_RESOLVE(D3DKMTMapGpuVirtualAddress);
    DXCORE_RESOLVE(D3DKMTReserveGpuVirtualAddress);
    DXCORE_RESOLVE(D3DKMTFreeGpuVirtualAddress);
    DXCORE_RESOLVE(D3DKMTCreateDevice);
    DXCORE_RESOLVE(D3DKMTDestroyDevice);
    DXCORE_RESOLVE(D3DKMTEnumAdapters2);
    DXCORE_RESOLVE(D3DKMTQueryAdapterInfo);
    DXCORE_RESOLVE(D3DKMTCreateContextVirtual);
    DXCORE_RESOLVE(D3DKMTDestroyContext);
    DXCORE_RESOLVE(D3DKMTSubmitCommand);
    DXCORE_RESOLVE(D3DKMTCreateSynchronizationObject2);
    DXCORE_RESOLVE(D3DKMTDestroySynchronizationObject);
    DXCORE_RESOLVE(D3DKMTQueryStatistics);
    DXCORE_RESOLVE(D3DKMTEscape);
    DXCORE_RESOLVE(D3DKMTLock2);
    DXCORE_RESOLVE(D3DKMTUnlock2);
    DXCORE_RESOLVE(D3DKMTCreatePagingQueue);
    DXCORE_RESOLVE(D3DKMTDestroyPagingQueue);
    DXCORE_RESOLVE(D3DKMTWaitForSynchronizationObjectFromGpu);
    DXCORE_RESOLVE(D3DKMTSignalSynchronizationObjectFromGpu);
    DXCORE_RESOLVE(D3DKMTWaitForSynchronizationObjectFromCpu);
    DXCORE_RESOLVE(D3DKMTQueryClockCalibration);
    DXCORE_RESOLVE(D3DKMTMakeResident);
    DXCORE_RESOLVE(D3DKMTEvict);
    DXCORE_RESOLVE(D3DKMTShareObjects);
    DXCORE_RESOLVE(D3DKMTQueryResourceInfoFromNtHandle);
    DXCORE_RESOLVE(D3DKMTOpenResourceFromNtHandle);
    DXCORE_RESOLVE(D3DKMTCreateHwQueue);
    DXCORE_RESOLVE(D3DKMTDestroyHwQueue);
    DXCORE_RESOLVE(D3DKMTSubmitCommandToHwQueue);
    DXCORE_RESOLVE(D3DKMTEnumAdapters3);
    DXCORE_RESOLVE(D3DKMTQueryResourceInfo);
    DXCORE_RESOLVE(D3DKMTOpenResource);

    dxcore_handle_ = mod;
}

}  // namespace dxcore
}  // namespace thunk
}  // namespace wsl

// NOLINTEND

#endif  // defined(_WIN32)
