/* Copyright © Advanced Micro Devices, Inc., or its affiliates. */

// Real OS identities for production sources compiled directly into a test binary.
// Keep this small so tests do not have to link the rest of the private OS layer.
#include "os.h"

#if defined(NCCL_OS_LINUX)
#include <sys/syscall.h>
#include <unistd.h>
#endif

uint64_t ncclOsGetPid() {
#if defined(NCCL_OS_WINDOWS)
  return static_cast<uint64_t>(GetCurrentProcessId());
#elif defined(NCCL_OS_LINUX)
  return static_cast<uint64_t>(getpid());
#else
#error "OS identity helpers require a supported RCCL OS platform"
#endif
}

uint64_t ncclOsGetTid() {
#if defined(NCCL_OS_WINDOWS)
  return static_cast<uint64_t>(GetCurrentThreadId());
#elif defined(NCCL_OS_LINUX)
  return static_cast<uint64_t>(syscall(SYS_gettid));
#else
#error "OS identity helpers require a supported RCCL OS platform"
#endif
}
