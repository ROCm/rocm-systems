// Windows HIP initialization. ROCr and Linux DMA-BUF are not used on Win32.
#include "nccl.h"
#include "debug.h"
#include "rocmwrap.h"
#include "param.h"
#include "rocm_smi_wrap.h"
#include <cstdlib>
#include <mutex>

PFN_hsa_amd_portable_export_dmabuf pfn_hsa_amd_portable_export_dmabuf = nullptr;
int ncclCudaDriverVersionCache = -1;
bool ncclCudaLaunchBlocking = false;
CUmemAllocationHandleType ncclCuMemHandleType = CU_MEM_HANDLE_TYPE_WIN32;
NCCL_PARAM(DmaBufEnable, "DMABUF_ENABLE", 0);
RCCL_PARAM(ForceEnableDMABUF, "FORCE_ENABLE_DMABUF", 0);

// VMM sharing/host allocation still need a Windows implementation. The existing
// allocation paths fall back to HIP malloc and HIP IPC when these return false.
int ncclIsCuMemSupported() { return 0; }
int ncclCuMemRuntimeSupported() { return 0; }
int ncclCuMemEnable() { return 0; }
int ncclCuMemHostEnable() { return 0; }

ncclResult_t rocmLibraryInit() {
  static std::once_flag once;
  static ncclResult_t result = ncclSystemError;
  std::call_once(once, [] {
    int count = 0;
    if (hipGetDeviceCount(&count) != hipSuccess || count == 0) return;
    if (hipDriverGetVersion(&ncclCudaDriverVersionCache) != hipSuccess) return;
    const char* blocking = std::getenv("HIP_LAUNCH_BLOCKING");
    ncclCudaLaunchBlocking = blocking && blocking[0] && blocking[0] != '0';
    result = ncclSuccess;
  });
  return result;
}

// The Windows SDK has no SMI. Device identity is available through HIP;
// leave XGMI/fabric links undiscovered rather than inventing their properties.
ncclResult_t rocm_smi_init() { return rocmLibraryInit(); }
ncclResult_t rocm_smi_getNumDevice(uint32_t* count) {
  int devices;
  if (hipGetDeviceCount(&devices) != hipSuccess) return ncclSystemError;
  *count = devices;
  return ncclSuccess;
}
ncclResult_t rocm_smi_getDevicePciBusIdString(uint32_t device, char* bus, size_t length) {
  return hipDeviceGetPCIBusId(bus, static_cast<int>(length), device) == hipSuccess ? ncclSuccess : ncclSystemError;
}
ncclResult_t rocm_smi_getDeviceIndexByPciBusId(const char* bus, uint32_t* device) {
  int ordinal;
  if (hipDeviceGetByPCIBusId(&ordinal, bus) != hipSuccess) return ncclSystemError;
  *device = ordinal;
  return ncclSuccess;
}
ncclResult_t rocm_smi_getLinkInfo(int, int, RSMI_IO_LINK_TYPE* type, int* hops, int* count) {
  *type = RSMI_IOLINK_TYPE_UNDEFINED;
  *hops = *count = 0;
  return ncclInternalError;
}
