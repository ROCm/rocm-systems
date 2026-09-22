// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.

#include "xtp_wrap.h"

#include <dlfcn.h>
#include <cstddef>
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

// Must match the reader's declaration exactly: filled here, read there, and
// no compiler sees both. The static_asserts below turn a drifted copy into a
// build failure instead of wrong fields at runtime.
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

static_assert(offsetof(it_descriptor, arch) == 0, "XTP descriptor: arch moved");
static_assert(offsetof(it_descriptor, n_nodes) == 8, "XTP descriptor: n_nodes moved");
static_assert(offsetof(it_descriptor, ranks_per_node) == 12, "XTP descriptor: ranks_per_node moved");
static_assert(offsetof(it_descriptor, n_ranks) == 16, "XTP descriptor: n_ranks moved");
static_assert(offsetof(it_descriptor, n_domains) == 20, "XTP descriptor: n_domains moved");
static_assert(offsetof(it_descriptor, ranks_per_domain) == 24, "XTP descriptor: ranks_per_domain moved");
static_assert(offsetof(it_descriptor, nic_type) == 32, "XTP descriptor: nic_type moved");
static_assert(offsetof(it_descriptor, nic_count) == 40, "XTP descriptor: nic_count moved");
static_assert(offsetof(it_descriptor, hash) == 48, "XTP descriptor: hash moved");
static_assert(sizeof(it_descriptor) == 56, "XTP descriptor: size changed");

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

// Kept with the descriptor it was built from, which is not stable across a
// comm's init: see commHandle().
struct Binding {
  it_descriptor desc;
  it_comm* comm; // null when no profile matched that descriptor
};
std::unordered_map<const ncclComm*, Binding> itComms;

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

// Bind on first use, and re-bind whenever the descriptor changes, because the
// topology facts do not all arrive at once. PXN is consulted from
// ncclTopoComputePaths -- before nNodes and localRanks are derived from the
// cross-rank host-hash exchange -- so it sees only arch and nRanks, while the
// knobs read later see the full topology. Caching the first descriptor for the
// comm's lifetime would pin the incomplete one and starve every later lookup.
it_comm* commHandle(struct ncclComm* comm) {
  loadOnce();
  if (itCtx == nullptr || comm == nullptr || comm->topo == nullptr) return nullptr;

  it_descriptor desc = {};
  desc.arch = comm->topo->nodes[GPU].nodes[0].gpu.gcn;
  desc.n_nodes = comm->nNodes;
  desc.ranks_per_node = comm->localRanks;
  // Always valid, unlike the two above: nRanks is an argument to
  // ncclCommInitRank, so it is set before any knob is read.
  desc.n_ranks = comm->nRanks;

  std::lock_guard<std::mutex> guard(itLock);
  auto found = itComms.find(comm);
  if (found != itComms.end()) {
    // Same facts as last time; a remembered non-match counts, so an uncovered
    // comm is asked once rather than on every lookup.
    if (memcmp(&found->second.desc, &desc, sizeof(desc)) == 0) return found->second.comm;
    if (found->second.comm != nullptr) api.unbindComm(found->second.comm);
    itComms.erase(found);
  }

  // No resolver: Class A matches on topology alone. One is needed only once
  // Class B rules start carrying constraints.
  it_comm* bound = api.bindComm(itCtx, &desc, nullptr, comm);
  itComms[comm] = Binding{desc, bound};
  if (bound == nullptr) {
    INFO(NCCL_INIT, "XTP: no profile matches arch=%s nodes=%d ranks_per_node=%d nranks=%d; using built-in tuning",
         desc.arch, desc.n_nodes, desc.ranks_per_node, desc.n_ranks);
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
  if (found->second.comm != nullptr) api.unbindComm(found->second.comm);
  itComms.erase(found);
}
