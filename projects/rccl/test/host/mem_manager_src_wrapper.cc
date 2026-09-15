// Compiles the real mem_manager.cc (hipified) with hipcc --offload-host-only.
#include "mem_manager.cc"

// Stubs that need real RCCL class layouts for correct C++ name mangling.
// These are compiled with hipcc so they see the real headers.

#include "roctx.h"
roctx_scoped_range_in::roctx_scoped_range_in(const char*) noexcept {}
roctx_scoped_range_in::~roctx_scoped_range_in() noexcept {}

#include "utils.h"
void* ncclMemoryStack::allocateSpilled(ncclMemoryStack*, size_t, size_t) { return nullptr; }

// ncclCommMemStats public API -> _impl dispatch (bypasses api_trace.cc)
extern "C" ncclResult_t ncclCommMemStats(ncclComm_t comm, ncclCommMemStat_t stat, uint64_t* value) {
  return ncclCommMemStats_impl(comm, stat, value);
}

