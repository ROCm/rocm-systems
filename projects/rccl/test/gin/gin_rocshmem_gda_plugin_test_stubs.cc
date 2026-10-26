/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Test doubles for gin_plugin_rocshmem_gda.cc (bootstrap, rocSHMEM GIN QP factory, debug,
// and a HIP allocation ledger).

#include "gin_rocshmem_gda_plugin_test_stubs.h"

#include "gin/gin_rocshmem_gda_factory.h"

#include "bootstrap.h"
#include "debug.h"
#include <dlfcn.h>
#include <hip/hip_runtime.h>
#include <set>

namespace GinRocshmemGdaPluginStubs {

struct State {
  int probeResult = 1;
  int bootstrapNranks = 1;
  int bootstrapRank = 0;
  bool createQpsFail = false;
  bool regMrFail = false;
  std::vector<RegMrCall> regMrCalls;
  std::set<void*> liveMrs;
  int destroyQpsCalls = 0;
  std::set<const void*> liveHipAllocs;
};

static State g;
static char g_qpSetToken;
static char g_gpuQpsToken;

void Reset() {
  for (void* mr : g.liveMrs) delete static_cast<RegMrCall*>(mr);
  g = State{};
}

void SetProbeResult(int result) { g.probeResult = result; }
void SetBootstrapLayout(int nranks, int rank) {
  g.bootstrapNranks = nranks;
  g.bootstrapRank = rank;
}
void SetCreateQpsFail(bool fail) { g.createQpsFail = fail; }
void SetRegMrFail(bool fail) { g.regMrFail = fail; }

uint32_t PeerRkey(int pe) { return 0x5100u + static_cast<uint32_t>(pe); }
uintptr_t PeerAddr(int pe) { return 0x7f0000000000ULL + static_cast<uintptr_t>(pe) * 0x10000000ULL; }
void** FakeGpuQps() { return reinterpret_cast<void**>(&g_gpuQpsToken); }

const std::vector<RegMrCall>& GetRegMrCalls() { return g.regMrCalls; }
int GetLiveMrCount() { return static_cast<int>(g.liveMrs.size()); }
int GetDestroyQpsCalls() { return g.destroyQpsCalls; }
bool IsLiveHipAllocation(const void* ptr) { return g.liveHipAllocs.count(ptr) == 1; }
int GetLiveHipAllocationCount() { return static_cast<int>(g.liveHipAllocs.size()); }

static int regMr(void* addr, size_t size, int atomic, void** out_mr, uint32_t* out_lkey, uint32_t* out_rkey) {
  if (g.regMrFail) return -1;
  const uint32_t key = 0xA000u + static_cast<uint32_t>(g.regMrCalls.size());
  auto* mr = new RegMrCall{addr, size, atomic, key};
  g.regMrCalls.push_back(*mr);
  g.liveMrs.insert(mr);
  *out_mr = mr;
  *out_lkey = key;
  *out_rkey = key;
  return 0;
}

}  // namespace GinRocshmemGdaPluginStubs

namespace Stubs = GinRocshmemGdaPluginStubs;

uint32_t ncclDebugLevelMask = (1u << NCCL_LOG_ERROR) | (1u << NCCL_LOG_VERSION);
uint64_t ncclDebugMask = NCCL_INIT;
thread_local int ncclDebugNoWarn = 0;

void ncclDebugLog(ncclDebugLogLevel level, unsigned long flags, const char* filefunc, int line,
                  const char* fmt, ...) {
  (void)level;
  (void)flags;
  (void)filefunc;
  (void)line;
  (void)fmt;
}

// Real definitions live in transport/net_ib/, which this target does not link.
// -1 is the NCCL_PARAM default and leaves the plugin on its comm->config rung.
int64_t ncclParamGinIbTc() { return -1; }
int64_t ncclParamIbTc() { return -1; }

extern "C" hipError_t hipMalloc(void** ptr, size_t size) {
  using Fn = hipError_t (*)(void**, size_t);
  static Fn real = reinterpret_cast<Fn>(dlsym(RTLD_NEXT, "hipMalloc"));
  const hipError_t err = real(ptr, size);
  if (err == hipSuccess) Stubs::g.liveHipAllocs.insert(*ptr);
  return err;
}

extern "C" hipError_t hipExtMallocWithFlags(void** ptr, size_t sizeBytes, unsigned int flags) {
  using Fn = hipError_t (*)(void**, size_t, unsigned int);
  static Fn real = reinterpret_cast<Fn>(dlsym(RTLD_NEXT, "hipExtMallocWithFlags"));
  const hipError_t err = real(ptr, sizeBytes, flags);
  if (err == hipSuccess) Stubs::g.liveHipAllocs.insert(*ptr);
  return err;
}

extern "C" hipError_t hipFree(void* ptr) {
  using Fn = hipError_t (*)(void*);
  static Fn real = reinterpret_cast<Fn>(dlsym(RTLD_NEXT, "hipFree"));
  Stubs::g.liveHipAllocs.erase(ptr);
  return real(ptr);
}

ncclResult_t bootstrapAllGather(void* commState, void* allData, int size) {
  (void)commState;
  for (int pe = 0; pe < Stubs::g.bootstrapNranks; ++pe) {
    if (pe == Stubs::g.bootstrapRank) continue;
    if (size == static_cast<int>(sizeof(uint32_t))) {
      static_cast<uint32_t*>(allData)[pe] = Stubs::PeerRkey(pe);
    } else if (size == static_cast<int>(sizeof(uintptr_t))) {
      static_cast<uintptr_t*>(allData)[pe] = Stubs::PeerAddr(pe);
    }
  }
  return ncclSuccess;
}

extern "C" int rocshmem_gin_probe_devices(void) { return Stubs::g.probeResult; }

extern "C" int rocshmem_gin_create_qps(int nRanks, int myRank, int (*allgather)(void*, void*, size_t),
                                       void* allgather_ctx, int traffic_class,
                                       rocshmem_gin_qp_set_t* out_qp_set, void*** out_gpu_qps) {
  (void)allgather_ctx;
  (void)traffic_class;
  if (Stubs::g.createQpsFail || !allgather || nRanks < 1 || myRank < 0 || myRank >= nRanks) return -1;
  *out_qp_set = reinterpret_cast<rocshmem_gin_qp_set_t>(&Stubs::g_qpSetToken);
  *out_gpu_qps = Stubs::FakeGpuQps();
  return 0;
}

extern "C" void rocshmem_gin_destroy_qps(rocshmem_gin_qp_set_t qp_set) {
  (void)qp_set;
  ++Stubs::g.destroyQpsCalls;
}

extern "C" int rocshmem_gin_reg_mr(rocshmem_gin_qp_set_t qp_set, void* addr, size_t size, int atomic, void** out_mr,
                                   uint32_t* out_lkey, uint32_t* out_rkey) {
  (void)qp_set;
  return Stubs::regMr(addr, size, atomic, out_mr, out_lkey, out_rkey);
}

extern "C" int rocshmem_gin_reg_mr_vmm(rocshmem_gin_qp_set_t qp_set, void* addr, size_t size, int atomic,
                                       void** out_mr, uint32_t* out_lkey, uint32_t* out_rkey) {
  (void)qp_set;
  return Stubs::regMr(addr, size, atomic, out_mr, out_lkey, out_rkey);
}

extern "C" void rocshmem_gin_dereg_mr(void* mr) {
  if (Stubs::g.liveMrs.erase(mr) == 1) delete static_cast<Stubs::RegMrCall*>(mr);
}

extern "C" int rocshmem_gin_get_provider(rocshmem_gin_qp_set_t qp_set) { return qp_set ? 0 : -1; }

extern "C" void rocshmem_gin_init_constmem(int provider, int rank) {
  (void)provider;
  (void)rank;
}
