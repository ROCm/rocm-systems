/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Implementation of the src/misc/rocmwrap.cc fakes. See rocmwrap_fakes.h.

#include "rocmwrap_fakes.h"

#include <fcntl.h>
#include <unistd.h>

#include "hip_fakes.h"
#include "nccl.h"
#include "nccl_fakes.h"  // g_loadParam
#include "rocmwrap.h"

int g_rocmLibraryInitCalls = 0;
ncclResult_t rocmLibraryInit(void) {
  g_rocmLibraryInitCalls++;
  return ncclSuccess;
}

// dmaBufSupported gate: NULL -> unsupported.
PFN_hsa_amd_portable_export_dmabuf pfn_hsa_amd_portable_export_dmabuf = nullptr;

std::vector<int> g_exportedFds;
int g_nextExportFd = 200;

static int FreshDevNullFd() {
  int devNull = open("/dev/null", O_RDONLY);
  if (devNull == -1) return -1;
  int fd = fcntl(devNull, F_DUPFD, g_nextExportFd);
  close(devNull);
  if (fd != -1) g_nextExportFd = fd + 1;
  return fd;
}

hsa_status_t FakeExportDmaBuf(const void*, size_t, int* fd, uint64_t* offset) {
  *fd = FreshDevNullFd();
  if (*fd == -1) return HSA_STATUS_ERROR;
  *offset = 0x40;
  g_exportedFds.push_back(*fd);
  return HSA_STATUS_SUCCESS;
}

hsa_status_t FailExportDmaBuf(const void*, size_t, int*, uint64_t*) { return HSA_STATUS_ERROR; }

bool FdOpen(int fd) { return fcntl(fd, F_GETFD) != -1; }

void CuMemExportRecorder::Install() {
#if defined(RCCL_CUMEM_DMABUF_EXPORT_SUPPORTED)
  g_hipMemGetHandleForAddressRange = [this](void* handle, hipDeviceptr_t ptr, size_t size, hipMemRangeHandleType,
                                             unsigned long long flags) {
    if (result != hipSuccess) return result;
    int fd = FreshDevNullFd();
    if (fd == -1) return hipErrorInvalidValue;
    *static_cast<int*>(handle) = fd;
    push_back({ptr, size, flags, fd});
    return hipSuccess;
  };
#else
  result = hipErrorNotSupported;
#endif
}

void CuMemExportRecorder::CloseOpen() {
  for (const auto& e : *this) {
    if (FdOpen(e.fd)) close(e.fd);
  }
}

int64_t rcclParamForceEnableDMABUF() { return g_loadParam("RCCL_FORCE_ENABLE_DMABUF", 0); }

// hsa-runtime64, not rocmwrap.cc: the entry rocmwrap.h's HSACHECK reports failures through.
hsa_status_t hsa_status_string(hsa_status_t, const char** status_string) {
  if (status_string != nullptr) *status_string = "fake HSA error";
  return HSA_STATUS_SUCCESS;
}

void ResetRocmWrapFakes() {
  pfn_hsa_amd_portable_export_dmabuf = nullptr;
  for (int fd : g_exportedFds) {
    if (FdOpen(fd)) close(fd);
  }
  g_exportedFds.clear();
  g_nextExportFd = 200;
  g_rocmLibraryInitCalls = 0;
}
