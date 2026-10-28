/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Fakes for the symbols defined by src/misc/rocmwrap.cc. Deliberately does NOT
// include rocmwrap.h: init.cc already declares pfn_hsa_amd_portable_export_dmabuf
// through it, and pulling checks.h in here would reach the non-hipified
// recorder.h alongside the hipified copy init.cc includes.

#ifndef RCCL_TEST_HOST_ROCMWRAP_FAKES_H_
#define RCCL_TEST_HOST_ROCMWRAP_FAKES_H_

#include <cstddef>
#include <cstdint>
#include <vector>

#include <hip/hip_runtime_api.h>
#include <hsa/hsa.h>

extern std::vector<int> g_exportedFds;
extern int g_nextExportFd;

hsa_status_t FakeExportDmaBuf(const void* ptr, size_t size, int* fd, uint64_t* offset);
hsa_status_t FailExportDmaBuf(const void* ptr, size_t size, int* fd, uint64_t* offset);
bool FdOpen(int fd);

struct CuMemExport {
  void* ptr;
  size_t size;
  unsigned long long flags;
  int fd;
};

struct CuMemExportRecorder : std::vector<CuMemExport> {
  hipError_t result = hipSuccess;
  void Install();
  void CloseOpen();
};

// Resets the dmaBufSupported gate (pfn_hsa_amd_portable_export_dmabuf) to NULL, i.e. unsupported.
void ResetRocmWrapFakes();

#endif  // RCCL_TEST_HOST_ROCMWRAP_FAKES_H_
