// RCCL_COMPAT_STUB: aggregated stubs for NCCL upstream symbols not yet
// implemented in RCCL. Each function/object below is tagged with
// RCCL_COMPAT_STUB on its closing brace for traceability.

#include "os.h"

#include <cstdio>
#include <dlfcn.h>
#include <sched.h>

int ncclOsCpuCount(const cpu_set_t& mask) { return CPU_COUNT(&mask); } // RCCL_COMPAT_STUB
bool ncclOsCpuIsSet(const cpu_set_t& mask, int cpu) { return CPU_ISSET(cpu, &mask); } // RCCL_COMPAT_STUB
const char* ncclVersionToString(int version, char* buf, size_t bufLen) {
  int major = version / 10000;
  int minor = (version / 100) % 100;
  int patch = version % 100;
  snprintf(buf, bufLen, "%d.%d.%d", major, minor, patch);
  return buf;
} // RCCL_COMPAT_STUB
void* ncclOsDlsym(void* handle, const char* symbol) { return ::dlsym(handle, symbol); } // RCCL_COMPAT_STUB
ncclResult_t ncclOsSetAffinity(const cpu_set_t&) { return ncclSuccess; } // RCCL_COMPAT_STUB
ncclResult_t ncclOsSetFilesLimit() { return ncclSuccess; } // RCCL_COMPAT_STUB
