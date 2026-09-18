/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Fakes for the src/os/linux.cc entry points, shared by the host-only microtest
// binaries. The allocation shims are real, not stubs: the microtests want
// working aligned allocation, and posix_memalign is exactly what production uses.

#include "os_fakes.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sched.h>

void* ncclOsAlignedAlloc(size_t alignment, size_t size) {
  void* p = nullptr;
  if (posix_memalign(&p, alignment, size) != 0) return nullptr;
  return p;
}

void ncclOsAlignedFree(void* ptr) { free(ptr); }

ncclResult_t ncclOsInitialize() { return ncclSuccess; }
uint64_t ncclOsGetPid() { return 4321; }
size_t ncclOsGetPageSize() { return 4096; }

// Controllable (was fail-loud). Records the mask too: :1608 and exit::2403 both call this, and without the
// recorder either call site could be handed the wrong affinity (affinitySave instead of comm->cpuAffinity)
// with nothing noticing. Default 0 keeps exit::2404 from calling ncclOsSetAffinity unless a test asks for it.
int g_ncclOsCpuCountValue = 0;
int g_ncclOsCpuCountCalls = 0;
std::vector<ncclAffinity> g_ncclOsCpuCountMasks;
int ncclOsCpuCount(const ncclAffinity& affinity) {
  g_ncclOsCpuCountCalls++;
  g_ncclOsCpuCountMasks.push_back(affinity);
  return g_ncclOsCpuCountValue;
}

// Controllable (was fail-loud). A std::function because :1609 writes through the pointer -- though nothing
// ever reads affinitySave back, which is what the AffinitySaveIsNeverRestored test pins.
// Writes an EMPTY mask by default, so ncclOsCpuCount's 0 default stays consistent and :1609-1610 are skipped.
std::function<ncclResult_t(ncclAffinity*)> g_ncclOsGetAffinity =
    [](ncclAffinity* a) { CPU_ZERO(a); return ncclSuccess; };
ncclResult_t ncclOsGetAffinity(ncclAffinity* affinity) { return g_ncclOsGetAffinity(affinity); }

// Controllable (was fail-loud). Records the affinity it was handed: without that, exit::2404 forwarding
// comm->cpuAffinity vs any other mask is unobservable -- a fake that drops an argument untests it.
// Keeps EVERY mask, not just the latest: :1610 and exit::2404 both call this, so a single "last"
// slot lets the exit: write mask what :1610 forwarded -- which left a mutant swapping :1610 to
// affinitySave alive. Tests index the call site they mean.
ncclResult_t g_ncclOsSetAffinityResult = ncclSuccess;
std::vector<ncclAffinity> g_ncclOsSetAffinityMasks;
ncclResult_t ncclOsSetAffinity(const ncclAffinity& affinity) {
  g_ncclOsSetAffinityMasks.push_back(affinity);
  return g_ncclOsSetAffinityResult;
}

// Must be non-empty and multi-token: ncclInit() strstr()s the strtok_r() of this, and strtok_r("") returns NULL.
// Also not "1" and not the Hyper-V BIOS string, so numa_balancing / bios_version stay on their benign arms.
ncclResult_t g_ncclOsTopoGetStrFromSysResult = ncclSuccess;
int g_ncclOsTopoGetStrFromSysCalls = 0;
ncclResult_t ncclOsTopoGetStrFromSys(const char* path, const char* fileName, char* strValue, int maxLen)
{
    ++g_ncclOsTopoGetStrFromSysCalls;
    if (g_ncclOsTopoGetStrFromSysResult != ncclSuccess) return g_ncclOsTopoGetStrFromSysResult;
    if (strValue && maxLen > 0) {
        std::snprintf(strValue, maxLen, "Linux version 6.8.0-microtest");
    }
    return ncclSuccess;
}

// fillInfo MLOPart PCI-function fallback (init.cc ~1093). The default models what sysfs actually
// reports for a GPU's own BDF -- an accelerator class -- rather than an empty string. An empty
// default is what let the mloPart=0 stamp on non-partitioned GPUs ship green: with it, isGpu is 0
// for every test that does not set comm->busId, so the fn==0 arm of the fallback was unreachable and
// InitTransportsRank_NoPeerWithMloPart_LeavesHasMloPartUnset could not see the stamp on its own rank.
// A test that wants the HIP-alias shape (BDF absent from sysfs) must now ask for "" explicitly.
// Duplicates PCI_ACCELERATOR_CLASS (src/graph/xml.h), which this TU does not include.
static const char* const kDefaultPciDeviceClass = "0x120000";
std::string g_pciDeviceClass = kDefaultPciDeviceClass;
int g_pciDeviceClassCalls = 0;
std::string g_lastPciDeviceClassBusId;
ncclResult_t ncclOsGetPciDeviceClassByBusId(const char* busId, char* deviceClass, size_t maxLen) {
  ++g_pciDeviceClassCalls;
  g_lastPciDeviceClassBusId = busId ? busId : "";
  if (deviceClass && maxLen > 0) {
    std::strncpy(deviceClass, g_pciDeviceClass.c_str(), maxLen - 1);
    deviceClass[maxLen - 1] = '\0';
  }
  return ncclSuccess;
}

// fillInfo's compute-partition probe. "SPX" is the unpartitioned default, so the class-probe
// fallback stays on the path it had before partition detection existed.
static const char* const kDefaultComputePartition = "SPX";
std::string g_pciComputePartition = kDefaultComputePartition;
int g_pciComputePartitionCalls = 0;
std::string g_lastPciComputePartitionBusId;
ncclResult_t ncclOsGetPciDeviceComputePartitionByBusId(const char* busId, char* partition, size_t maxLen) {
  ++g_pciComputePartitionCalls;
  g_lastPciComputePartitionBusId = busId ? busId : "";
  if (partition && maxLen > 0) {
    std::strncpy(partition, g_pciComputePartition.c_str(), maxLen - 1);
    partition[maxLen - 1] = '\0';
  }
  return ncclSuccess;
}

void ResetOsFakes() {
  g_ncclOsCpuCountValue = 0;
  g_ncclOsCpuCountCalls = 0;
  g_ncclOsCpuCountMasks.clear();
  g_ncclOsGetAffinity = [](ncclAffinity* a) { CPU_ZERO(a); return ncclSuccess; };
  g_ncclOsSetAffinityResult = ncclSuccess;
  g_ncclOsSetAffinityMasks.clear();
  g_ncclOsTopoGetStrFromSysResult = ncclSuccess;
  g_ncclOsTopoGetStrFromSysCalls = 0;
  g_pciDeviceClass = kDefaultPciDeviceClass;
  g_pciDeviceClassCalls = 0;
  g_lastPciDeviceClassBusId.clear();
  g_pciComputePartition = kDefaultComputePartition;
  g_pciComputePartitionCalls = 0;
  g_lastPciComputePartitionBusId.clear();
}
