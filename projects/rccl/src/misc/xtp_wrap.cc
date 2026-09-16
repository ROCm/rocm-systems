// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.

#include "xtp_wrap.h"

#include <dlfcn.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <unordered_map>

#include "comm.h"
#include "debug.h"
#include "graph/topo.h"

namespace {

/*************************************************************************
 * The XTP reader's C ABI
 *
 * Transcribed from the reader's own header rather than included, so that RCCL
 * builds whether or not the reader is present. That makes the layouts below a
 * contract we assert rather than one the compiler checks for us, so the loader
 * refuses any reader whose reported ABI is not the one these declarations were
 * written against.
 ************************************************************************/
constexpr int kAbiMajor = 0;
constexpr int kAbiMinor = 1;

// The reader's filename, which is the one name here that follows the
// implementation rather than the XTP contract. Kept in a single place so it is
// the only line to touch if the reader ships under a different name.
constexpr const char* kReaderLibrary = "libintellitune.so";

enum it_tag { IT_INT, IT_BOOL, IT_DOUBLE, IT_STR, IT_RANGE };

struct it_value {
  it_tag tag;
  union {
    int64_t i;
    bool b;
    double d;
    const char* s;
    struct {
      int64_t lo, hi;
      bool lo_inf, hi_inf;
    } range;
  };
};

struct it_kv {
  const char* key;
  it_value value;
};

struct it_descriptor {
  const char* arch;
  int32_t n_nodes;
  int32_t ranks_per_node;
  int32_t n_ranks;
  int32_t n_domains;
  int32_t ranks_per_domain;
  const char* nic_type;
  int32_t nic_count;
  const char* hash;
};

struct it_ctx;
struct it_comm;
typedef bool (*it_resolver)(void* user, void* launch, const char* key, it_value* out);

constexpr int kItOk = 0;

struct XtpApi {
  it_ctx* (*open)(const char* bundlePath, char** errReason);
  void (*close)(it_ctx* ctx);
  it_comm* (*bindComm)(it_ctx* ctx, const it_descriptor* desc, it_resolver resolver, void* user);
  void (*unbindComm)(it_comm* comm);
  int (*classA)(it_comm* comm, const it_kv** out, size_t* nOut);
  const char* (*version)();
};

XtpApi api;
it_ctx* itCtx = nullptr;
std::mutex itLock;
std::unordered_map<const ncclComm*, it_comm*> itComms;

// Parse "ABI <major>.<minor>" out of the library's version string. Its exact
// wording is not contractual, so a string we cannot read is treated the same
// as a mismatch: refuse to use the library rather than read its structs blind.
bool abiMatches(const char* version) {
  if (version == nullptr) return false;
  const char* abi = strstr(version, "ABI ");
  int major = -1, minor = -1;
  if (abi == nullptr || sscanf(abi, "ABI %d.%d", &major, &minor) != 2) return false;
  return major == kAbiMajor && minor == kAbiMinor;
}

template <typename Fn>
bool resolveSymbol(void* handle, const char* name, Fn& slot) {
  slot = reinterpret_cast<Fn>(dlsym(handle, name));
  if (slot == nullptr) {
    WARN("XTP: %s is missing symbol %s; profile ignored", kReaderLibrary, name);
    return false;
  }
  return true;
}

// Load the library and open the bundle once per process. RCCL_XTP
// holds the bundle directory and is the activation gate: unset means the whole
// mechanism costs nothing beyond this check.
void loadOnce() {
  static std::once_flag once;
  std::call_once(once, [] {
    const char* bundle = getenv("RCCL_XTP");
    if (bundle == nullptr || bundle[0] == '\0') return;

    void* handle = dlopen(kReaderLibrary, RTLD_NOW | RTLD_LOCAL);
    if (handle == nullptr) {
      INFO(NCCL_INIT, "XTP: RCCL_XTP is set but %s could not be loaded (%s)", kReaderLibrary, dlerror());
      return;
    }

    if (!resolveSymbol(handle, "it_open", api.open) || !resolveSymbol(handle, "it_close", api.close) ||
        !resolveSymbol(handle, "it_bind_comm", api.bindComm) ||
        !resolveSymbol(handle, "it_unbind_comm", api.unbindComm) ||
        !resolveSymbol(handle, "it_class_a", api.classA) || !resolveSymbol(handle, "it_version", api.version)) {
      return;
    }

    const char* version = api.version();
    if (!abiMatches(version)) {
      WARN("XTP: %s reports \"%s\", which is not ABI %d.%d; profile ignored", kReaderLibrary,
           version ? version : "(null)", kAbiMajor, kAbiMinor);
      return;
    }

    char* err = nullptr;
    itCtx = api.open(bundle, &err);
    if (itCtx == nullptr) {
      // Fail-closed: a bad bundle is not an init failure, it just leaves the
      // built-in tuning in charge.
      WARN("XTP: could not load bundle %s (%s); using built-in tuning", bundle, err ? err : "unknown");
      return;
    }
    INFO(NCCL_INIT, "XTP: loaded bundle %s (%s)", bundle, version);
  });
}

// Bind on first use so callers do not have to place an explicit bind in the
// init sequence. Every fact below is already settled by the time the earliest
// Class A consumer (PXN, during path computation) runs.
it_comm* commHandle(struct ncclComm* comm) {
  loadOnce();
  if (itCtx == nullptr || comm == nullptr || comm->topo == nullptr) return nullptr;

  std::lock_guard<std::mutex> guard(itLock);
  auto found = itComms.find(comm);
  if (found != itComms.end()) return found->second;

  it_descriptor desc = {};
  desc.arch = comm->topo->nodes[GPU].nodes[0].gpu.gcn;
  desc.n_nodes = comm->nNodes;
  desc.ranks_per_node = comm->localRanks;
  // Carried explicitly because a comm need not be a uniform grid, and both PXN
  // and P2P_NET_CHUNKSIZE switch on the true rank count.
  desc.n_ranks = comm->nRanks;

  // No resolver: Class A matches on topology alone. One is needed only once
  // Class B rules start carrying constraints.
  it_comm* bound = api.bindComm(itCtx, &desc, nullptr, comm);
  itComms[comm] = bound; // cache misses too, so an unmatched comm is asked once
  if (bound == nullptr) {
    INFO(NCCL_INIT, "XTP: no profile matches arch=%s nodes=%d ranks=%d; using built-in tuning", desc.arch,
         desc.n_nodes, desc.n_ranks);
  }
  return bound;
}

const it_value* lookup(struct ncclComm* comm, const char* key) {
  it_comm* bound = commHandle(comm);
  if (bound == nullptr) return nullptr;

  const it_kv* kvs = nullptr;
  size_t n = 0;
  if (api.classA(bound, &kvs, &n) != kItOk || kvs == nullptr) return nullptr;
  for (size_t i = 0; i < n; i++) {
    if (strcmp(kvs[i].key, key) == 0) return &kvs[i].value;
  }
  return nullptr;
}

} // namespace

bool rcclXtpClassAInt(struct ncclComm* comm, const char* key, int64_t* out) {
  const it_value* value = lookup(comm, key);
  if (value == nullptr) return false;
  if (value->tag != IT_INT) {
    WARN("XTP: %s is not an integer in this profile; using built-in value", key);
    return false;
  }
  *out = value->i;
  return true;
}

bool rcclXtpClassAStr(struct ncclComm* comm, const char* key, const char** out) {
  const it_value* value = lookup(comm, key);
  if (value == nullptr) return false;
  if (value->tag != IT_STR) {
    WARN("XTP: %s is not a string in this profile; using built-in value", key);
    return false;
  }
  // Library-owned and stable until the comm is unbound.
  *out = value->s;
  return true;
}

void rcclXtpCommFree(struct ncclComm* comm) {
  if (itCtx == nullptr) return;
  std::lock_guard<std::mutex> guard(itLock);
  auto found = itComms.find(comm);
  if (found == itComms.end()) return;
  if (found->second != nullptr) api.unbindComm(found->second);
  itComms.erase(found);
}
