/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Host-only microtests for src/misc/rocmwrap.cc. Non-HIP calls are seamed inside rocmwrap.cc only
// (see the #defines before ROCMWRAP_CC_PATH). On ROCm 7.1/7.2 the cuMem tests pin the
// out-of-window arms of ncclCuMemEnable/ncclCuMemHostEnable.

#include <gtest/gtest.h>

#include <sys/utsname.h>
#include <unistd.h>

#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <functional>
#include <map>
#include <mutex>
#include <string>
#include <vector>

#include "fakes/hip_fakes.h"

#include "nccl.h"
#include "debug.h"
#include "rocmwrap.h"
#include "archinfo.h"
#include "kernel_config.h"
#include "hsa/hsa.h"
#include "bootstrap.h"

// --- Logging and params: this binary links no shared nccl fakes.
uint32_t ncclDebugLevelMask = NCCL_DEBUG_LEVEL_MASK_UNINITIALIZED;  // log everything
uint64_t ncclDebugMask = ~uint64_t(0);

namespace {
struct LoggedMessage {
  ncclDebugLogLevel level;
  std::string text;
};
std::vector<LoggedMessage> g_logged;
std::map<std::string, int64_t> g_params;
}  // namespace

void ncclDebugLog(ncclDebugLogLevel level, unsigned long /*flags*/, const char* /*filefunc*/, int /*line*/,
                  const char* fmt, ...) {
  char buf[512];
  va_list ap;
  va_start(ap, fmt);
  std::vsnprintf(buf, sizeof(buf), fmt, ap);
  va_end(ap);
  g_logged.push_back({level, buf});
}

std::function<int64_t(const char*, int64_t)> g_loadParam = [](const char* env, int64_t deftVal) {
  auto it = g_params.find(env);
  return it == g_params.end() ? deftVal : it->second;
};

#include "fakes/param_redirect.h"

// --- HSA: the four ROCr entry points rocmwrap.cc calls.
namespace {
std::function<hsa_status_t(hsa_system_info_t, void*)> g_hsaSystemGetInfo;
int g_hsaInitCalls = 0;
}  // namespace

extern "C" hsa_status_t hsa_system_get_info(hsa_system_info_t attribute, void* value) {
  return g_hsaSystemGetInfo(attribute, value);
}
extern "C" hsa_status_t hsa_init() {
  ++g_hsaInitCalls;
  return HSA_STATUS_SUCCESS;
}
extern "C" hsa_status_t hsa_shut_down() { return HSA_STATUS_SUCCESS; }

// --- Seams scoped to rocmwrap.cc's own function bodies.
// Every header rocmwrap.cc includes is pulled in above, so their include guards
// keep the renames below from rewriting any declaration.

namespace {
std::function<int(struct utsname*)> g_uname;
std::function<FILE*(const char*, const char*)> g_fopen;
std::function<hipError_t(int*)> g_driverGetVersion;
std::function<hipError_t(hipDevice_t*, int)> g_deviceGet;
std::function<bool(std::string*, char*, size_t)> g_readKernelConfig;
PFN_hsa_amd_portable_export_dmabuf g_exportDmabuf = nullptr;
}  // namespace

static int RocmWrapTestUname(struct utsname* u) { return g_uname(u); }
static FILE* RocmWrapTestFopen(const char* path, const char* mode) { return g_fopen(path, mode); }
static hipError_t RocmWrapTestDriverGetVersion(int* v) { return g_driverGetVersion(v); }
static hipError_t RocmWrapTestDeviceGet(hipDevice_t* dev, int ordinal) { return g_deviceGet(dev, ordinal); }
static bool RocmWrapTestReadKernelConfig(std::string* content, char* pathOut, size_t pathOutLen) {
  return g_readKernelConfig(content, pathOut, pathOutLen);
}

#define uname RocmWrapTestUname
#define fopen RocmWrapTestFopen
#define hipDriverGetVersion RocmWrapTestDriverGetVersion
#define hipDeviceGet RocmWrapTestDeviceGet
#define ncclKernelConfigReadFirstAvailable RocmWrapTestReadKernelConfig
#define hsa_amd_portable_export_dmabuf g_exportDmabuf
#include ROCMWRAP_CC_PATH
#undef hsa_amd_portable_export_dmabuf
#undef ncclKernelConfigReadFirstAvailable
#undef hipDeviceGet
#undef hipDriverGetVersion
#undef fopen
#undef uname

namespace {

constexpr int kKernel_6_8 = KERNEL_VERSION_CODE(6, 8);
constexpr int kCuMemDriverVersion = ROCM_VER_7_12_60540;  // inside NCCL_CUMEM_VERSION_SUPPORTED
constexpr int kOldDriverVersion = 70002000;               // 7.0.2, below the backport window

hsa_status_t DmaBufExportStub(const void*, size_t, int*, uint64_t*) { return HSA_STATUS_SUCCESS; }

// Records each VMM call the functional probe makes, in order.
struct VmmLog {
  std::vector<std::string> calls;
  hipMemAllocationProp createProp{};
  hipMemAccessDesc accessDesc{};
  size_t createSize = 0;
  std::string failAt;  // the call that returns an error; empty means all succeed
  size_t granularity = 4096;
};

class RocmwrapMicrotest : public ::testing::Test {
protected:
  void SetUp() override {
    const char* saved = std::getenv("CUDA_LAUNCH_BLOCKING");
    hadLaunchBlocking_ = saved != nullptr;
    if (saved) savedLaunchBlocking_ = saved;
    Reset();
  }

  void TearDown() override {
    if (hadLaunchBlocking_) setenv("CUDA_LAUNCH_BLOCKING", savedLaunchBlocking_.c_str(), 1);
    else unsetenv("CUDA_LAUNCH_BLOCKING");
    ResetHipFakes();
  }

  // Every seam back to a working gfx1250 platform on kernel 6.8 with a cuMem-capable driver,
  // and rocmwrap.cc's file-scope state back to its initial values. Loops call it per case.
  void Reset() {
    ResetHipFakes();
    g_logged.clear();
    g_params.clear();
    g_hsaInitCalls = 0;
    fopenPaths_.clear();
    unsetenv("CUDA_LAUNCH_BLOCKING");

    SetKernelRelease("6.8.0-generic");
    g_fopen = [this](const char* path, const char*) -> FILE* {
      fopenPaths_.push_back(path);
      return nullptr;
    };
    g_driverGetVersion = [](int* v) {
      *v = kCuMemDriverVersion;
      return hipSuccess;
    };
    g_deviceGet = [](hipDevice_t* dev, int ordinal) {
      *dev = ordinal;
      return hipSuccess;
    };
    g_readKernelConfig = [](std::string*, char*, size_t) { return false; };
    g_exportDmabuf = DmaBufExportStub;
    g_hsaSystemGetInfo = [](hsa_system_info_t attribute, void* value) {
      if (attribute == HSA_SYSTEM_INFO_VERSION_MAJOR) *static_cast<uint16_t*>(value) = 1;
      else if (attribute == HSA_SYSTEM_INFO_VERSION_MINOR) *static_cast<uint16_t*>(value) = 14;
      else *static_cast<bool*>(value) = true;  // 0x204: DMA-BUF supported
      return HSA_STATUS_SUCCESS;
    };
    SetArch("gfx1250");
    SetVmmAttribute(1);
    defaultVmm_ = VmmLog{};
    InstallVmm(defaultVmm_);

    // rocmwrap.cc's file-scope state.
    version_major = version_minor = 0;
    ncclCudaLaunchBlocking = false;
    ncclCuMemSupported = 0;
    ncclCumemHostEnable = -1;
    initResult = ncclSuccess;
    pfn_hsa_amd_portable_export_dmabuf = nullptr;
  }

  static void SetKernelRelease(std::string release) {
    g_uname = [release](struct utsname* u) {
      std::memset(u, 0, sizeof(*u));
      std::snprintf(u->release, sizeof(u->release), "%s", release.c_str());
      return 0;
    };
  }

  static void SetArch(std::string arch) {
    g_hipGetDeviceProperties = [arch](hipDeviceProp_t* prop, int) {
      *prop = hipDeviceProp_t{};
      std::snprintf(prop->gcnArchName, sizeof(prop->gcnArchName), "%s:sramecc+:xnack-", arch.c_str());
      return hipSuccess;
    };
  }

  static void SetVmmAttribute(int value) {
    g_hipDeviceGetAttribute = [value](int* pi, hipDeviceAttribute_t attr, int) {
      if (attr != hipDeviceAttributeVirtualMemoryManagementSupported) return hipErrorInvalidValue;
      *pi = value;
      return hipSuccess;
    };
  }

  // Installs hooks for every call the functional probe makes, recording each into `log`.
  static void InstallVmm(VmmLog& log) {
    auto step = [&log](const char* name) {
      log.calls.push_back(name);
      return log.failAt == name ? hipErrorInvalidValue : hipSuccess;
    };
    g_hipMemGetAllocationGranularity = [&log, step](size_t* g, const hipMemAllocationProp* prop,
                                                    hipMemAllocationGranularity_flags) {
      hipError_t e = step("granularity");
      log.createProp = *prop;
      *g = log.granularity;
      return e;
    };
    g_hipMemCreate = [&log, step](hipMemGenericAllocationHandle_t* h, size_t size, const hipMemAllocationProp*,
                                  unsigned long long) {
      log.createSize = size;
      *h = reinterpret_cast<hipMemGenericAllocationHandle_t>(0x1);
      return step("create");
    };
    g_hipMemAddressReserve = [step](void** ptr, size_t, size_t, void*, unsigned long long) {
      *ptr = reinterpret_cast<void*>(0x10000);
      return step("reserve");
    };
    g_hipMemMap = [step](void*, size_t, size_t, hipMemGenericAllocationHandle_t, unsigned long long) {
      return step("map");
    };
    g_hipMemSetAccess = [&log, step](void*, size_t, const hipMemAccessDesc* desc, size_t) {
      log.accessDesc = *desc;
      return step("setAccess");
    };
    g_hipMemGetAddressRange = [step](hipDeviceptr_t* base, size_t* size, hipDeviceptr_t ptr) {
      *base = ptr;
      *size = 4096;
      return step("addressRange");
    };
    g_hipPointerGetAttribute = [step](void*, hipPointer_attribute, hipDeviceptr_t) { return step("pointerAttr"); };
    g_hipMemUnmap = [step](void*, size_t) { return step("unmap"); };
    g_hipMemAddressFree = [step](void*, size_t) { return step("addressFree"); };
    g_hipMemRelease = [step](hipMemGenericAllocationHandle_t) { return step("release"); };
  }

  static bool Logged(ncclDebugLogLevel level, const char* needle) {
    for (const auto& m : g_logged) {
      if (m.level == level && m.text.find(needle) != std::string::npos) return true;
    }
    return false;
  }

  // A kernel config that has (or lacks) the two options DMA-BUF needs.
  static void SetKernelConfig(bool moveNotify, bool p2pdma) {
    g_readKernelConfig = [moveNotify, p2pdma](std::string* content, char* pathOut, size_t pathOutLen) {
      *content = "CONFIG_FOO=y\n";
      if (moveNotify) *content += "CONFIG_DMABUF_MOVE_NOTIFY=y\n";
      if (p2pdma) *content += "CONFIG_PCI_P2PDMA=y\n";
      std::snprintf(pathOut, pathOutLen, "/boot/config-test");
      return true;
    };
  }

  // /proc/kallsyms served from an in-memory buffer.
  void SetKallsyms(std::string content) {
    kallsyms_ = std::move(content);
    g_fopen = [this](const char* path, const char* mode) -> FILE* {
      fopenPaths_.push_back(path);
      return fmemopen(kallsyms_.data(), kallsyms_.size(), mode);
    };
  }

  std::string kallsyms_;
  std::vector<std::string> fopenPaths_;
  VmmLog defaultVmm_;

private:
  bool hadLaunchBlocking_ = false;
  std::string savedLaunchBlocking_;
};

}  // namespace

// ncclGetKernelVersionCode

TEST_F(RocmwrapMicrotest, KernelVersionCode_PacksMajorAndMinorFromTheRelease) {
  SetKernelRelease("6.11.4-45-generic");
  EXPECT_EQ(ncclGetKernelVersionCode(), (6 << 16) | (11 << 8));
  EXPECT_TRUE(Logged(NCCL_LOG_INFO, "Kernel version 6.11"));
}

TEST_F(RocmwrapMicrotest, KernelVersionCode_UnameFailureReturnsMinusOne) {
  g_uname = [](struct utsname*) { return -1; };
  EXPECT_EQ(ncclGetKernelVersionCode(), -1);
}

// sscanf matches nothing, so major and minor keep their zero initialisers.
TEST_F(RocmwrapMicrotest, KernelVersionCode_UnparseableReleaseIsZero) {
  SetKernelRelease("custom-kernel");
  EXPECT_EQ(ncclGetKernelVersionCode(), 0);
}

// ncclCuMemFunctionalProbe

TEST_F(RocmwrapMicrotest, FunctionalProbe_AllStepsSucceed_ReturnsOneAndUndoesEveryStep) {
  VmmLog log;
  InstallVmm(log);
  EXPECT_EQ(ncclCuMemFunctionalProbe(/*dev=*/0, /*devOrdinal=*/3), 1);
  std::vector<std::string> expected = {"granularity", "create",       "reserve",     "map",        "setAccess",
                                       "addressRange", "pointerAttr", "unmap",       "addressFree", "release"};
#if HIP_VERSION >= ROCM_VER_7_12_60540
  expected.insert(expected.begin() + 7, "pointerAttr");  // the legacy-IPC capability query
#endif
  EXPECT_EQ(log.calls, expected);
  // One granularity unit of pinned device memory on the probed ordinal.
  EXPECT_EQ(log.createSize, 4096u);
  EXPECT_EQ(log.createProp.type, hipMemAllocationTypePinned);
  EXPECT_EQ(log.createProp.location.type, hipMemLocationTypeDevice);
  EXPECT_EQ(log.createProp.location.id, 3);
  EXPECT_EQ(log.createProp.requestedHandleTypes, ncclCuMemHandleType);
  EXPECT_EQ(log.accessDesc.location.type, hipMemLocationTypeDevice);
  EXPECT_EQ(log.accessDesc.location.id, 3);
  EXPECT_EQ(log.accessDesc.flags, hipMemAccessFlagsProtReadWrite);
  EXPECT_FALSE(Logged(NCCL_LOG_INFO, "functional probe failed"));
}

TEST_F(RocmwrapMicrotest, FunctionalProbe_ZeroGranularityFailsBeforeCreating) {
  VmmLog log;
  log.granularity = 0;
  InstallVmm(log);
  EXPECT_EQ(ncclCuMemFunctionalProbe(0, 0), 0);
  EXPECT_EQ(log.calls, std::vector<std::string>({"granularity"}));
  EXPECT_TRUE(Logged(NCCL_LOG_INFO, "cuMem functional probe failed on device 0"));
}

// Each failing step undoes exactly the steps that succeeded before it, in reverse order.
TEST_F(RocmwrapMicrotest, FunctionalProbe_EachFailureUndoesOnlyTheCompletedSteps) {
  struct Case {
    const char* failAt;
    std::vector<std::string> calls;
  };
  const std::vector<Case> cases = {
    {"granularity", {"granularity"}},
    {"create", {"granularity", "create"}},
    {"reserve", {"granularity", "create", "reserve", "release"}},
    {"map", {"granularity", "create", "reserve", "map", "addressFree", "release"}},
    {"setAccess", {"granularity", "create", "reserve", "map", "setAccess", "unmap", "addressFree", "release"}},
    {"addressRange",
     {"granularity", "create", "reserve", "map", "setAccess", "addressRange", "unmap", "addressFree", "release"}},
    {"pointerAttr",
     {"granularity", "create", "reserve", "map", "setAccess", "addressRange", "pointerAttr", "unmap", "addressFree",
      "release"}},
  };
  for (const auto& c : cases) {
    SCOPED_TRACE(c.failAt);
    g_logged.clear();
    VmmLog log;
    log.failAt = c.failAt;
    InstallVmm(log);
    EXPECT_EQ(ncclCuMemFunctionalProbe(0, 1), 0);
    EXPECT_EQ(log.calls, c.calls);
    EXPECT_TRUE(Logged(NCCL_LOG_INFO, "cuMem functional probe failed on device 1"));
  }
}

// A failing cleanup call is only reported; the probe still answers from the steps before it.
TEST_F(RocmwrapMicrotest, FunctionalProbe_FailingCleanupIsReportedButDoesNotChangeTheAnswer) {
  VmmLog log;
  log.failAt = "unmap";
  InstallVmm(log);
  EXPECT_EQ(ncclCuMemFunctionalProbe(0, 0), 1);
  EXPECT_EQ(log.calls.back(), "release");
  EXPECT_TRUE(Logged(NCCL_LOG_INFO, "HIP failure"));
}

// ncclCuMemCapabilityCheck / ncclIsCuMemSupported

TEST_F(RocmwrapMicrotest, CapabilityCheck_AllGatesPassAndProbeSucceeds_ReturnsOne) {
  VmmLog log;
  InstallVmm(log);
  EXPECT_EQ(ncclCuMemCapabilityCheck(/*requireGfx1250ForAutoEnable=*/1), 1);
  EXPECT_FALSE(log.calls.empty());
  EXPECT_FALSE(Logged(NCCL_LOG_WARN, "cuMem support requires"));
}

TEST_F(RocmwrapMicrotest, CapabilityCheck_AutoEnableOutsideGfx1250_ReturnsZeroWithoutProbing) {
  SetArch("gfx942");
  VmmLog log;
  InstallVmm(log);
  bool unameCalled = false;
  g_uname = [&unameCalled](struct utsname*) {
    unameCalled = true;
    return -1;
  };
  EXPECT_EQ(ncclCuMemCapabilityCheck(1), 0);
  EXPECT_TRUE(Logged(NCCL_LOG_INFO, "limited to gfx1250 (detected gfx942)"));
  EXPECT_FALSE(unameCalled);
  EXPECT_TRUE(log.calls.empty());
}

TEST_F(RocmwrapMicrotest, CapabilityCheck_ArchQueryFailureUnderAutoEnable_ReturnsZero) {
  g_hipGetDeviceProperties = [](hipDeviceProp_t*, int) { return hipErrorInvalidDevice; };
  EXPECT_EQ(ncclCuMemCapabilityCheck(1), 0);
  EXPECT_TRUE(Logged(NCCL_LOG_INFO, "detected unknown"));
}

// NCCL_CUMEM_ENABLE=1's path skips the arch gate entirely.
TEST_F(RocmwrapMicrotest, CapabilityCheck_WithoutTheGfx1250Gate_AcceptsOtherArchs) {
  SetArch("gfx942");
  bool archQueried = false;
  g_hipGetDeviceProperties = [&archQueried](hipDeviceProp_t* prop, int) {
    archQueried = true;
    *prop = hipDeviceProp_t{};
    std::snprintf(prop->gcnArchName, sizeof(prop->gcnArchName), "gfx942:sramecc+:xnack-");
    return hipSuccess;
  };
  VmmLog log;
  InstallVmm(log);
  EXPECT_EQ(ncclCuMemCapabilityCheck(/*requireGfx1250ForAutoEnable=*/0), 1);
  EXPECT_FALSE(archQueried);
}

// Every cheap gate is evaluated (and WARNs) even after an earlier one failed, but the probe is skipped.
TEST_F(RocmwrapMicrotest, CapabilityCheck_EachCheapGateFailureReturnsZeroWithoutProbing) {
  struct Case {
    const char* name;
    std::function<void()> arrange;
    const char* warn;
  };
  const std::vector<Case> cases = {
    {"kernel 6.7", [] { SetKernelRelease("6.7.12"); }, "requires Linux kernel >= 6.8"},
    {"driver 7.0.2",
     [] {
       g_driverGetVersion = [](int* v) {
         *v = kOldDriverVersion;
         return hipSuccess;
       };
     },
     "requires ROCm >= 7.12 or the 7.0.2.x backport"},
    {"no VMM", [] { SetVmmAttribute(0); }, "requires VMM RDMA support"},
  };
  for (const auto& c : cases) {
    SCOPED_TRACE(c.name);
    Reset();
    c.arrange();
    VmmLog log;
    InstallVmm(log);
    EXPECT_EQ(ncclCuMemCapabilityCheck(1), 0);
    EXPECT_TRUE(Logged(NCCL_LOG_WARN, c.warn));
    EXPECT_TRUE(log.calls.empty());
  }
}

// The kernel boundary itself: 6.8 passes, so the result rests on the probe alone.
TEST_F(RocmwrapMicrotest, CapabilityCheck_KernelExactly6_8Passes) {
  SetKernelRelease("6.8");
  VmmLog log;
  InstallVmm(log);
  EXPECT_EQ(ncclCuMemCapabilityCheck(1), 1);
  EXPECT_FALSE(Logged(NCCL_LOG_WARN, "kernel"));
  static_assert(kKernel_6_8 == ((6 << 16) | (8 << 8)), "KERNEL_VERSION_CODE layout");
}

TEST_F(RocmwrapMicrotest, CapabilityCheck_BackportDriverWindowIsAccepted) {
  g_driverGetVersion = [](int* v) {
    *v = ROCM_VER_7_0_2_2;
    return hipSuccess;
  };
  VmmLog log;
  InstallVmm(log);
  EXPECT_EQ(ncclCuMemCapabilityCheck(1), 1);
}

TEST_F(RocmwrapMicrotest, CapabilityCheck_ProbeFailureReturnsZero) {
  VmmLog log;
  log.failAt = "map";
  InstallVmm(log);
  EXPECT_EQ(ncclCuMemCapabilityCheck(1), 0);
  EXPECT_FALSE(log.calls.empty());
}

// HIP query failures jump straight to `error`, which returns 0 since ret is no longer ncclSuccess.
TEST_F(RocmwrapMicrotest, CapabilityCheck_HipQueryFailuresReturnZeroWithoutProbing) {
  struct Case {
    const char* name;
    std::function<void()> arrange;
  };
  const std::vector<Case> cases = {
    {"hipGetDevice", [] { g_hipGetDevice = [](int*) { return hipErrorNoDevice; }; }},
    {"hipDriverGetVersion", [] { g_driverGetVersion = [](int*) { return hipErrorNotInitialized; }; }},
    {"hipDeviceGet", [] { g_deviceGet = [](hipDevice_t*, int) { return hipErrorInvalidDevice; }; }},
    {"hipDeviceGetAttribute",
     [] { g_hipDeviceGetAttribute = [](int*, hipDeviceAttribute_t, int) { return hipErrorInvalidValue; }; }},
  };
  for (const auto& c : cases) {
    SCOPED_TRACE(c.name);
    Reset();
    c.arrange();
    VmmLog log;
    InstallVmm(log);
    EXPECT_EQ(ncclCuMemCapabilityCheck(0), 0);
    EXPECT_TRUE(log.calls.empty());
  }
}

TEST_F(RocmwrapMicrotest, IsCuMemSupported_AppliesTheGfx1250Gate) {
  VmmLog log;
  InstallVmm(log);
  EXPECT_EQ(ncclIsCuMemSupported(), 1);
  SetArch("gfx950");
  EXPECT_EQ(ncclIsCuMemSupported(), 0);
}

// The memo is process-wide (a function-local once_flag), so check it in a forked child.
TEST_F(RocmwrapMicrotest, RuntimeSupported_SkipsTheArchGateAndRunsTheCheckOnce) {
  EXPECT_EXIT(
    {
      SetArch("gfx942");
      VmmLog log;
      InstallVmm(log);
      int first = ncclCuMemRuntimeSupported();
      size_t probeCalls = log.calls.size();
      log.failAt = "create";  // would change the answer if the check ran again
      int second = ncclCuMemRuntimeSupported();
      bool ok = first == 1 && second == 1 && probeCalls > 0 && log.calls.size() == probeCalls;
      std::exit(ok ? 0 : 1);
    },
    ::testing::ExitedWithCode(0), "");
}

// ncclCuMemEnable / ncclCuMemHostEnable (out-of-window HIP_VERSION arm)

#if !NCCL_CUMEM_VERSION_SUPPORTED(HIP_VERSION)
TEST_F(RocmwrapMicrotest, CuMemEnable_OutsideTheCompileTimeWindowIsAlwaysZero) {
  for (int64_t param : {int64_t{-2}, int64_t{0}}) {
    g_params["CUMEM_ENABLE"] = param;
    g_logged.clear();
    EXPECT_EQ(ncclCuMemEnable(), 0) << param;
    EXPECT_FALSE(Logged(NCCL_LOG_WARN, "NCCL_CUMEM_ENABLE=1")) << param;
  }
  g_params["CUMEM_ENABLE"] = 1;
  EXPECT_EQ(ncclCuMemEnable(), 0);
  EXPECT_TRUE(Logged(NCCL_LOG_WARN, "NCCL_CUMEM_ENABLE=1 is set but cuMem VMM APIs are unavailable"));
}
#endif

#if !NCCL_CUMEM_HOST_VERSION_SUPPORTED(HIP_VERSION)
TEST_F(RocmwrapMicrotest, CuMemHostEnable_OutsideTheCompileTimeWindowCachesZero) {
  EXPECT_EQ(ncclCuMemHostEnable(), 0);
  EXPECT_EQ(ncclCumemHostEnable, 0);
}
#endif

// A cached answer is returned as-is, without re-deciding.
TEST_F(RocmwrapMicrotest, CuMemHostEnable_ReturnsTheCachedValue) {
  ncclCumemHostEnable = 1;
  EXPECT_EQ(ncclCuMemHostEnable(), 1);
}

// initOnceFunc / rocmLibraryInit

// CUDA_LAUNCH_BLOCKING is on for any non-empty value other than exactly "0".
TEST_F(RocmwrapMicrotest, Init_ParsesCudaLaunchBlocking) {
  struct Case {
    const char* value;  // nullptr: unset
    bool expected;
  };
  for (const Case& c : {Case{nullptr, false}, Case{"", false}, Case{"0", false}, Case{"1", true}, Case{"00", true},
                        Case{"10", true}}) {
    SCOPED_TRACE(c.value ? c.value : "<unset>");
    if (c.value) setenv("CUDA_LAUNCH_BLOCKING", c.value, 1);
    else unsetenv("CUDA_LAUNCH_BLOCKING");
    ncclCudaLaunchBlocking = !c.expected;
    initOnceFunc();
    EXPECT_EQ(ncclCudaLaunchBlocking, c.expected);
  }
}

TEST_F(RocmwrapMicrotest, Init_ReadsTheRocrVersion) {
  SetKernelConfig(true, true);
  initOnceFunc();
  EXPECT_EQ(version_major, 1);
  EXPECT_EQ(version_minor, 14);
  EXPECT_TRUE(Logged(NCCL_LOG_INFO, "ROCr version 1.14"));
}

TEST_F(RocmwrapMicrotest, Init_RocrVersionQueryFailureIsASystemErrorBeforeHsaInit) {
  for (hsa_system_info_t failing : {HSA_SYSTEM_INFO_VERSION_MAJOR, HSA_SYSTEM_INFO_VERSION_MINOR}) {
    SCOPED_TRACE(failing);
    Reset();
    g_hsaSystemGetInfo = [failing](hsa_system_info_t attribute, void* value) {
      if (attribute == failing) return HSA_STATUS_ERROR;
      *static_cast<uint16_t*>(value) = 1;
      return HSA_STATUS_SUCCESS;
    };
    initOnceFunc();
    EXPECT_EQ(initResult, ncclSystemError);
    EXPECT_TRUE(Logged(NCCL_LOG_WARN, "hsa_system_get_info failed"));
    EXPECT_EQ(g_hsaInitCalls, 0);
  }
}

// The gfx1250 auto-enable decision is taken once, here, and stored for ncclCuMemEnable.
TEST_F(RocmwrapMicrotest, Init_RecordsTheCuMemCapability) {
  SetKernelConfig(true, true);
  VmmLog log;
  InstallVmm(log);
  initOnceFunc();
  EXPECT_EQ(ncclCuMemSupported, 1);

  SetArch("gfx942");
  initOnceFunc();
  EXPECT_EQ(ncclCuMemSupported, 0);
}

TEST_F(RocmwrapMicrotest, Init_DmaBufDisabledByParamStopsBeforeHsaInit) {
  g_params["DMABUF_ENABLE"] = 0;
  // Pin cuMem off: in-window builds would otherwise auto-enable it on this gfx1250 fixture, and
  // ncclCuMemEnable() force-enables DMA-BUF ahead of the NCCL_DMABUF_ENABLE=0 arm under test.
  g_params["CUMEM_ENABLE"] = 0;
  initOnceFunc();
  EXPECT_EQ(initResult, ncclSystemError);
  EXPECT_TRUE(Logged(NCCL_LOG_INFO, "Dmabuf feature disabled"));
  EXPECT_EQ(g_hsaInitCalls, 0);
  EXPECT_EQ(pfn_hsa_amd_portable_export_dmabuf, nullptr);
}

TEST_F(RocmwrapMicrotest, Init_RocrWithoutDmaBufSupportStopsBeforeHsaInit) {
  struct Case {
    const char* name;
    hsa_status_t status;
    bool supported;
  };
  for (const Case& c :
       {Case{"query fails", HSA_STATUS_ERROR, true}, Case{"reports false", HSA_STATUS_SUCCESS, false}}) {
    SCOPED_TRACE(c.name);
    Reset();
    g_hsaSystemGetInfo = [c](hsa_system_info_t attribute, void* value) {
      if (attribute == HSA_SYSTEM_INFO_VERSION_MAJOR || attribute == HSA_SYSTEM_INFO_VERSION_MINOR) {
        *static_cast<uint16_t*>(value) = 1;
        return HSA_STATUS_SUCCESS;
      }
      EXPECT_EQ(attribute, static_cast<hsa_system_info_t>(0x204));
      *static_cast<bool*>(value) = c.supported;
      return c.status;
    };
    initOnceFunc();
    EXPECT_EQ(initResult, ncclSystemError);
    EXPECT_TRUE(Logged(NCCL_LOG_INFO, "does not support dmabuf"));
    EXPECT_EQ(g_hsaInitCalls, 0);
    // net.cc and coll_net.cc gate on the pointer itself, not on initResult.
    EXPECT_EQ(pfn_hsa_amd_portable_export_dmabuf, nullptr);
  }
}

// ROCr advertises DMA-BUF but its export entry did not resolve: disabled cleanly, gate left unarmed.
TEST_F(RocmwrapMicrotest, Init_MissingExportEntryLeavesTheGateUnarmed) {
  g_exportDmabuf = nullptr;
  initOnceFunc();
  EXPECT_EQ(initResult, ncclSystemError);
  EXPECT_TRUE(Logged(NCCL_LOG_INFO, "does not export hsa_amd_portable_export_dmabuf"));
  EXPECT_EQ(pfn_hsa_amd_portable_export_dmabuf, nullptr);
  EXPECT_EQ(g_hsaInitCalls, 0);
}

TEST_F(RocmwrapMicrotest, Init_KernelConfigWithBothOptionsArmsTheGateAndInitsHsa) {
  SetKernelConfig(true, true);
  initOnceFunc();
  EXPECT_EQ(initResult, ncclSuccess);
  EXPECT_EQ(pfn_hsa_amd_portable_export_dmabuf, &DmaBufExportStub);
  EXPECT_EQ(g_hsaInitCalls, 1);
  EXPECT_TRUE(Logged(NCCL_LOG_INFO, "CONFIG_DMABUF_MOVE_NOTIFY=y in /boot/config-test"));
  EXPECT_TRUE(Logged(NCCL_LOG_INFO, "CONFIG_PCI_P2PDMA=y in /boot/config-test"));
  EXPECT_TRUE(Logged(NCCL_LOG_INFO, "DMA_BUF Support Enabled"));
  EXPECT_TRUE(fopenPaths_.empty());
}

// The pfn gate is armed before the kernel check, so it stays armed on this failure; a fix flips this pin.
TEST_F(RocmwrapMicrotest, Init_KnownBug_KernelConfigFailureLeavesTheExportGateArmed) {
  for (int missing = 0; missing < 2; ++missing) {
    SCOPED_TRACE(missing == 0 ? "no MOVE_NOTIFY" : "no P2PDMA");
    Reset();
    SetKernelConfig(missing != 0, missing != 1);
    initOnceFunc();
    EXPECT_EQ(initResult, ncclSystemError);
    EXPECT_TRUE(Logged(NCCL_LOG_INFO, "Failed due to OS kernel support"));
    EXPECT_EQ(g_hsaInitCalls, 0);
    EXPECT_EQ(pfn_hsa_amd_portable_export_dmabuf, &DmaBufExportStub);
  }
}

// Either order works: each symbol is matched once, whichever line it appears on.
TEST_F(RocmwrapMicrotest, Init_KallsymsFallbackWithBothSymbolsSucceeds) {
  for (const char* symbols : {"ffffffff81000000 T _stext\n"
                              "ffffffff81234560 T dma_buf_move_notify\n"
                              "ffffffff81234570 T pci_p2pdma_add_resource\n",
                              "ffffffff81234570 T pci_p2pdma_add_resource\n"
                              "ffffffff81234580 T pci_p2pdma_distance\n"
                              "ffffffff81234560 T dma_buf_move_notify\n"}) {
    SCOPED_TRACE(symbols);
    Reset();
    SetKallsyms(symbols);
    initOnceFunc();
    EXPECT_EQ(fopenPaths_, std::vector<std::string>({"/proc/kallsyms"}));
    EXPECT_EQ(initResult, ncclSuccess);
    EXPECT_EQ(g_hsaInitCalls, 1);
    EXPECT_TRUE(Logged(NCCL_LOG_INFO, "via /proc/kallsyms"));
  }
}

TEST_F(RocmwrapMicrotest, Init_KallsymsFallbackMissingEitherSymbolIsASystemError) {
  for (const char* symbols :
       {"ffffffff81234560 T dma_buf_move_notify\n", "ffffffff81234570 T pci_p2pdma_add_resource\n"}) {
    SCOPED_TRACE(symbols);
    Reset();
    SetKallsyms(symbols);
    initOnceFunc();
    EXPECT_EQ(initResult, ncclSystemError);
    EXPECT_TRUE(Logged(NCCL_LOG_INFO, "missing kernel symbols in /proc/kallsyms"));
    EXPECT_EQ(g_hsaInitCalls, 0);
  }
}

// An unreadable /proc/kallsyms is only logged, and init still succeeds; a fix flips this pin.
TEST_F(RocmwrapMicrotest, Init_KnownBug_UnreadableKallsymsStillSucceeds) {
  initOnceFunc();
  EXPECT_TRUE(Logged(NCCL_LOG_INFO, "Could not open /proc/kallsyms"));
  EXPECT_EQ(initResult, ncclSuccess);
  EXPECT_EQ(g_hsaInitCalls, 1);
}

// RCCL_FORCE_ENABLE_DMABUF=1 overrides NCCL_DMABUF_ENABLE=0 and skips the kernel checks.
TEST_F(RocmwrapMicrotest, Init_ForceEnableSkipsTheParamAndKernelChecks) {
  g_params["RCCL_FORCE_ENABLE_DMABUF"] = 1;
  g_params["DMABUF_ENABLE"] = 0;
  bool configRead = false;
  g_readKernelConfig = [&configRead](std::string*, char*, size_t) {
    configRead = true;
    return false;
  };
  initOnceFunc();
  EXPECT_EQ(initResult, ncclSuccess);
  EXPECT_TRUE(Logged(NCCL_LOG_WARN, "DMA_BUF Support is force enabled"));
  EXPECT_FALSE(configRead);
  EXPECT_TRUE(fopenPaths_.empty());
  EXPECT_EQ(g_hsaInitCalls, 1);
}

// pthread_once state is process-wide, so check the once-only behaviour in a forked child.
TEST_F(RocmwrapMicrotest, LibraryInit_RunsInitOnceAndReturnsItsResult) {
  EXPECT_EXIT(
    {
      int majorQueries = 0;
      g_hsaSystemGetInfo = [&majorQueries](hsa_system_info_t attribute, void* value) {
        if (attribute == HSA_SYSTEM_INFO_VERSION_MAJOR) ++majorQueries;
        *static_cast<uint16_t*>(value) = 1;
        return attribute == HSA_SYSTEM_INFO_VERSION_MINOR ? HSA_STATUS_ERROR : HSA_STATUS_SUCCESS;
      };
      ncclResult_t first = rocmLibraryInit();
      ncclResult_t second = rocmLibraryInit();
      bool ok = first == ncclSystemError && second == ncclSystemError && majorQueries == 1;
      std::exit(ok ? 0 : 1);
    },
    ::testing::ExitedWithCode(0), "");
}
