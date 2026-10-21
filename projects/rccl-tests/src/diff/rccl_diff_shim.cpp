/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * In-process loader for two librccl.so copies. The *_diff binaries link this
 * instead of -lrccl. Stock *_perf binaries do not.
 *
 * Symbols are hidden so a dlopen'd librccl binds its own nccl* calls to
 * itself. Communicators, registrations, windows, and custom redops created
 * by the test are wrappers that own one real object per library.
 ************************************************************************/

#include "diff/diff_ctrl.h"

#include <rccl/rccl.h>
#include <nccl_device.h>

#include <dlfcn.h>
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include <mutex>
#include <string>
#include <vector>

namespace {

constexpr int kLibs = 2;
constexpr uint32_t kCommMagic = 0xD1FFC0A1u;
constexpr uint32_t kRegMagic = 0xD1FFB0A2u;
constexpr uint32_t kWinMagic = 0xD1FFB0A3u;
constexpr int kRedOpBase = 100000;

struct Api {
  void* handle;
  int version;
  char label[8];
  char path[1024];
  ncclResult_t (*GetVersion)(int*);
  ncclResult_t (*GetUniqueId)(ncclUniqueId*);
  ncclResult_t (*CommInitRank)(ncclComm_t*, int, ncclUniqueId, int);
  ncclResult_t (*CommInitRankConfig)(ncclComm_t*, int, ncclUniqueId, int, ncclConfig_t*);
  ncclResult_t (*CommDestroy)(ncclComm_t);
  ncclResult_t (*CommAbort)(ncclComm_t);
  ncclResult_t (*CommGetAsyncError)(ncclComm_t, ncclResult_t*);
  ncclResult_t (*CommRegister)(const ncclComm_t, void*, size_t, void**);
  ncclResult_t (*CommDeregister)(const ncclComm_t, void*);
  ncclResult_t (*CommWindowRegister)(ncclComm_t, void*, size_t, ncclWindow_t*, int);
  ncclResult_t (*CommWindowDeregister)(ncclComm_t, ncclWindow_t);
  ncclResult_t (*GroupStart)();
  ncclResult_t (*GroupEnd)();
  ncclResult_t (*AllReduce)(const void*, void*, size_t, ncclDataType_t, ncclRedOp_t, ncclComm_t, hipStream_t);
  ncclResult_t (*RedOpCreatePreMulSum)(ncclRedOp_t*, void*, ncclDataType_t, ncclScalarResidence_t, ncclComm_t);
  ncclResult_t (*RedOpDestroy)(ncclRedOp_t, ncclComm_t);
  ncclResult_t (*MemAlloc)(void**, size_t);
  ncclResult_t (*MemFree)(void*);
  const char* (*GetErrorString)(ncclResult_t);
  const char* (*GetLastError)(ncclComm_t);
  // Present when the library was built with the host device-API queries.
  // The diff driver rejects -D, but all_reduce.cu still references these.
  ncclResult_t (*CommQueryProperties)(ncclComm_t, ncclCommProperties_t*);
  ncclTeam_t (*TeamLsa)(ncclComm_t);
};

struct DiffComm {
  uint32_t magic;
  ncclComm_t real[kLibs];
};

struct DiffReg {
  uint32_t magic;
  void* real[kLibs];
};

struct DiffWin {
  uint32_t magic;
  ncclWindow_t real[kLibs];
};

struct DiffRedOp {
  ncclRedOp_t real[kLibs];
  bool live;
};

Api g_api[kLibs];
int g_ready = 0;
// Group state and the selected library are per thread. ncclGroupStart is
// thread-local in RCCL, and each test thread switches libraries on its own.
thread_local int g_active = 0;
thread_local int g_depth = 0;
thread_local int g_fwd[kLibs] = {0, 0};

std::mutex g_mu;
std::vector<DiffRedOp> g_redops;
ncclUniqueId g_ids[kLibs];
char g_idKey[NCCL_UNIQUE_ID_BYTES];
bool g_idsReady = false;
char g_idPath[108];

void fail(const char* msg) {
  fprintf(stderr, "rccl-diff: %s\n", msg);
  _exit(1);
}

void requireReady() {
  if (!g_ready) fail("libraries are not loaded; pass --diff_libs");
}

DiffComm* asComm(ncclComm_t comm) {
  if (!comm) return nullptr;
  auto* d = reinterpret_cast<DiffComm*>(comm);
  if (d->magic != kCommMagic) return nullptr;
  return d;
}

void touch(int lib) {
  while (g_fwd[lib] < g_depth) {
    ncclResult_t r = g_api[lib].GroupStart();
    if (r != ncclSuccess) {
      fprintf(stderr, "rccl-diff: ncclGroupStart failed on %s: %s\n",
          g_api[lib].label, g_api[lib].GetErrorString ? g_api[lib].GetErrorString(r) : "?");
      _exit(1);
    }
    g_fwd[lib]++;
  }
}

ncclRedOp_t mapOp(ncclRedOp_t op, int lib) {
  if ((int)op < kRedOpBase) return op;
  int idx = (int)op - kRedOpBase;
  std::lock_guard<std::mutex> lock(g_mu);
  if (idx < 0 || idx >= (int)g_redops.size() || !g_redops[idx].live) {
    fprintf(stderr, "rccl-diff: unknown redop %d\n", (int)op);
    return op;
  }
  return g_redops[idx].real[lib];
}

int copyFile(const char* src, const char* dst) {
  FILE* in = fopen(src, "rb");
  if (!in) return -1;
  FILE* out = fopen(dst, "wb");
  if (!out) {
    fclose(in);
    return -1;
  }
  char buf[1 << 16];
  size_t n;
  int rc = 0;
  while ((n = fread(buf, 1, sizeof(buf), in)) > 0) {
    if (fwrite(buf, 1, n, out) != n) {
      rc = -1;
      break;
    }
  }
  if (ferror(in)) rc = -1;
  fclose(in);
  if (fclose(out) != 0) rc = -1;
  return rc;
}

void* openLib(const char* path, int flags, char* err, int errLen) {
  void* h = dlopen(path, flags);
  if (h) return h;
  snprintf(err, errLen, "dlopen(%s) failed: %s", path, dlerror());
  return nullptr;
}

bool loadOne(Api* api, const char* label, const char* path, int flags, char* err, int errLen) {
  memset(api, 0, sizeof(*api));
  snprintf(api->label, sizeof(api->label), "%s", label);
  snprintf(api->path, sizeof(api->path), "%s", path);
  api->handle = openLib(path, flags, err, errLen);
  if (!api->handle) return false;

#define LOAD(field, sym)                                                         \
  do {                                                                           \
    api->field = (decltype(api->field))dlsym(api->handle, sym);                 \
    if (!api->field) {                                                           \
      snprintf(err, errLen, "%s is missing %s (%s)", label, sym, path);         \
      return false;                                                              \
    }                                                                            \
  } while (0)

  LOAD(GetVersion, "ncclGetVersion");
  LOAD(GetUniqueId, "ncclGetUniqueId");
  LOAD(CommInitRank, "ncclCommInitRank");
  LOAD(CommInitRankConfig, "ncclCommInitRankConfig");
  LOAD(CommDestroy, "ncclCommDestroy");
  LOAD(CommAbort, "ncclCommAbort");
  LOAD(CommGetAsyncError, "ncclCommGetAsyncError");
  LOAD(CommRegister, "ncclCommRegister");
  LOAD(CommDeregister, "ncclCommDeregister");
  LOAD(CommWindowRegister, "ncclCommWindowRegister");
  LOAD(CommWindowDeregister, "ncclCommWindowDeregister");
  LOAD(GroupStart, "ncclGroupStart");
  LOAD(GroupEnd, "ncclGroupEnd");
  LOAD(AllReduce, "ncclAllReduce");
  LOAD(RedOpCreatePreMulSum, "ncclRedOpCreatePreMulSum");
  LOAD(RedOpDestroy, "ncclRedOpDestroy");
  LOAD(MemAlloc, "ncclMemAlloc");
  LOAD(MemFree, "ncclMemFree");
  LOAD(GetErrorString, "ncclGetErrorString");
  LOAD(GetLastError, "ncclGetLastError");
#undef LOAD
  api->CommQueryProperties = (decltype(api->CommQueryProperties))dlsym(api->handle, "ncclCommQueryProperties");
  api->TeamLsa = (decltype(api->TeamLsa))dlsym(api->handle, "ncclTeamLsa");

  if (api->GetVersion(&api->version) != ncclSuccess) api->version = -1;
  return true;
}

void publishIds(ncclUniqueId* outToken) {
  char tmpl[] = "/tmp/rccl-diff-id-XXXXXX";
  int fd = mkstemp(tmpl);
  if (fd < 0) fail("mkstemp for unique-id exchange failed");
  const size_t nbytes = sizeof(g_ids);
  if (write(fd, g_ids, nbytes) != (ssize_t)nbytes) fail("writing unique-id file failed");
  close(fd);
  snprintf(g_idPath, sizeof(g_idPath), "%s", tmpl);

  memset(outToken, 0, sizeof(*outToken));
  outToken->internal[0] = 'D';
  outToken->internal[1] = 'F';
  memcpy(outToken->internal + 2, g_idPath, strlen(g_idPath));
  memcpy(g_idKey, outToken->internal, NCCL_UNIQUE_ID_BYTES);
  g_idsReady = true;
}

void ensureIds(const ncclUniqueId& token) {
  if (g_idsReady && memcmp(g_idKey, token.internal, NCCL_UNIQUE_ID_BYTES) == 0) return;
  if (token.internal[0] != 'D' || token.internal[1] != 'F') {
    fail("unique id was not produced by the diff shim");
  }
  char path[108];
  memset(path, 0, sizeof(path));
  memcpy(path, token.internal + 2, sizeof(path) - 1);
  if (strncmp(path, "/tmp/rccl-diff-id-", 18) != 0) fail("refusing unique-id path");
  FILE* f = fopen(path, "rb");
  if (!f) {
    fprintf(stderr, "rccl-diff: cannot read unique-id file %s: %s\n", path, strerror(errno));
    _exit(1);
  }
  if (fread(g_ids, sizeof(g_ids), 1, f) != 1) fail("short unique-id file");
  fclose(f);
  memcpy(g_idKey, token.internal, NCCL_UNIQUE_ID_BYTES);
  g_idsReady = true;
}

const char* verStr(int v, char* buf, int n) {
  if (v < 0) {
    snprintf(buf, n, "unknown");
    return buf;
  }
  snprintf(buf, n, "%d.%d.%d (%d)", v / 10000, (v / 100) % 100, v % 100, v);
  return buf;
}

}  // namespace

int rcclDiffLoadLibraries(const char* spec, int deepbind, char* err, int errLen) {
  if (!spec || !err || errLen < 8) return -1;
  const char* comma = strchr(spec, ',');
  if (!comma || strncmp(spec, "A=", 2) != 0 || strncmp(comma + 1, "B=", 2) != 0) {
    snprintf(err, errLen, "--diff_libs must look like A=/path/librccl.so.1,B=/path/librccl.so.1");
    return -1;
  }
  std::string pathA(spec + 2, comma);
  std::string pathB(comma + 3);
  if (pathA.empty() || pathB.empty()) {
    snprintf(err, errLen, "--diff_libs paths are empty");
    return -1;
  }

  int flags = RTLD_NOW | RTLD_LOCAL;
#ifdef RTLD_DEEPBIND
  if (deepbind) flags |= RTLD_DEEPBIND;
#else
  (void)deepbind;
#endif

  // Diagnostic: one image, two communicators. Used to separate a dual-load
  // kernel-registration failure from a harness failure.
  const char* share = getenv("RCCL_DIFF_SHARE_IMAGE");
  if (share && share[0] == '1' && pathA == pathB) {
    if (!loadOne(&g_api[0], "A", pathA.c_str(), flags, err, errLen)) return -1;
    g_api[1] = g_api[0];
    snprintf(g_api[1].label, sizeof(g_api[1].label), "B");
    g_ready = 1;
    fprintf(stderr, "# rccl-diff: RCCL_DIFF_SHARE_IMAGE=1, both sides use one loaded image\n");
    return 0;
  }

  if (!loadOne(&g_api[0], "A", pathA.c_str(), flags, err, errLen)) return -1;
  if (!loadOne(&g_api[1], "B", pathB.c_str(), flags, err, errLen)) return -1;

  // dlopen returns the same handle for the same inode, and some loaders also
  // collapse two files that share a SONAME. A private copy forces a second image.
  if (g_api[0].handle == g_api[1].handle) {
    dlclose(g_api[1].handle);
    char dir[] = "/tmp/rccl-diff-lib-XXXXXX";
    if (!mkdtemp(dir)) {
      snprintf(err, errLen, "mkdtemp failed: %s", strerror(errno));
      return -1;
    }
    std::string copy = std::string(dir) + "/librccl.so.1";
    if (copyFile(pathB.c_str(), copy.c_str()) != 0) {
      snprintf(err, errLen, "failed to copy %s to %s", pathB.c_str(), copy.c_str());
      return -1;
    }
    if (!loadOne(&g_api[1], "B", copy.c_str(), flags, err, errLen)) return -1;
    // Keep the original path in the report; the copy is an implementation detail.
    snprintf(g_api[1].path, sizeof(g_api[1].path), "%s", pathB.c_str());
    if (g_api[0].handle == g_api[1].handle) {
      snprintf(err, errLen, "both libraries resolved to the same dlopen handle");
      return -1;
    }
    fprintf(stderr, "# rccl-diff: B was copied aside so it loads as its own image\n");
  }

  g_ready = 1;
  char va[64], vb[64];
  fprintf(stderr, "# rccl-diff: A %s  %s\n", verStr(g_api[0].version, va, sizeof(va)), g_api[0].path);
  fprintf(stderr, "# rccl-diff: B %s  %s\n", verStr(g_api[1].version, vb, sizeof(vb)), g_api[1].path);
  fprintf(stderr, "# rccl-diff: AllReduce pointers %p %p\n",
      (void*)g_api[0].AllReduce, (void*)g_api[1].AllReduce);
  if (g_api[0].version != g_api[1].version) {
    fprintf(stderr, "# rccl-diff: library versions differ; the binary was built against one rccl.h\n");
  }
  return 0;
}

void rcclDiffSetActive(int lib) {
  if (lib < 0 || lib >= kLibs) fail("active library out of range");
  g_active = lib;
}

int rcclDiffActiveLib() { return g_active; }

void rcclDiffGetLibInfo(int index, const char** label, const char** path, int* version) {
  if (index < 0 || index >= kLibs || !g_ready) {
    if (label) *label = "?";
    if (path) *path = "";
    if (version) *version = -1;
    return;
  }
  if (label) *label = g_api[index].label;
  if (path) *path = g_api[index].path;
  if (version) *version = g_api[index].version;
}

extern "C" {

ncclResult_t ncclGetVersion(int* version) {
  requireReady();
  return g_api[0].GetVersion(version);
}

ncclResult_t ncclGetUniqueId(ncclUniqueId* uniqueId) {
  requireReady();
  if (!uniqueId) return ncclInvalidArgument;
  for (int i = 0; i < kLibs; i++) {
    ncclResult_t r = g_api[i].GetUniqueId(&g_ids[i]);
    if (r != ncclSuccess) return r;
  }
  publishIds(uniqueId);
  return ncclSuccess;
}

ncclResult_t ncclGroupStart() {
  requireReady();
  g_depth++;
  return ncclSuccess;
}

ncclResult_t ncclGroupEnd() {
  requireReady();
  if (g_depth <= 0) {
    fprintf(stderr, "rccl-diff: ncclGroupEnd without ncclGroupStart\n");
    return ncclInvalidArgument;
  }
  g_depth--;
  ncclResult_t r = ncclSuccess;
  for (int i = 0; i < kLibs; i++) {
    if (g_fwd[i] > g_depth) {
      ncclResult_t ri = g_api[i].GroupEnd();
      g_fwd[i]--;
      if (ri != ncclSuccess) r = ri;
    }
  }
  return r;
}

ncclResult_t ncclCommInitRankConfig(ncclComm_t* comm, int nranks, ncclUniqueId commId, int rank, ncclConfig_t* config) {
  requireReady();
  if (!comm) return ncclInvalidArgument;
  std::lock_guard<std::mutex> lock(g_mu);
  ensureIds(commId);
  auto* d = new DiffComm();
  d->magic = kCommMagic;
  for (int i = 0; i < kLibs; i++) {
    touch(i);
    ncclResult_t r = g_api[i].CommInitRankConfig(&d->real[i], nranks, g_ids[i], rank, config);
    if (r != ncclSuccess) {
      d->magic = 0;
      delete d;
      return r;
    }
  }
  *comm = reinterpret_cast<ncclComm_t>(d);
  return ncclSuccess;
}

ncclResult_t ncclCommInitRank(ncclComm_t* comm, int nranks, ncclUniqueId commId, int rank) {
  requireReady();
  if (!comm) return ncclInvalidArgument;
  std::lock_guard<std::mutex> lock(g_mu);
  ensureIds(commId);
  auto* d = new DiffComm();
  d->magic = kCommMagic;
  for (int i = 0; i < kLibs; i++) {
    touch(i);
    ncclResult_t r = g_api[i].CommInitRank(&d->real[i], nranks, g_ids[i], rank);
    if (r != ncclSuccess) {
      d->magic = 0;
      delete d;
      return r;
    }
  }
  *comm = reinterpret_cast<ncclComm_t>(d);
  return ncclSuccess;
}

ncclResult_t ncclCommDestroy(ncclComm_t comm) {
  requireReady();
  DiffComm* d = asComm(comm);
  if (!d) return ncclInvalidArgument;
  ncclResult_t r = ncclSuccess;
  for (int i = 0; i < kLibs; i++) {
    touch(i);
    ncclResult_t ri = g_api[i].CommDestroy(d->real[i]);
    if (ri != ncclSuccess) r = ri;
  }
  d->magic = 0;
  delete d;
  return r;
}

ncclResult_t ncclCommAbort(ncclComm_t comm) {
  requireReady();
  DiffComm* d = asComm(comm);
  if (!d) return ncclInvalidArgument;
  ncclResult_t r = ncclSuccess;
  for (int i = 0; i < kLibs; i++) {
    touch(i);
    ncclResult_t ri = g_api[i].CommAbort(d->real[i]);
    if (ri != ncclSuccess) r = ri;
  }
  d->magic = 0;
  delete d;
  return r;
}

ncclResult_t ncclCommGetAsyncError(ncclComm_t comm, ncclResult_t* asyncError) {
  requireReady();
  DiffComm* d = asComm(comm);
  if (!d) return ncclInvalidArgument;
  // Report the active library. Also surface a failure from the other one.
  ncclResult_t r = g_api[g_active].CommGetAsyncError(d->real[g_active], asyncError);
  if (r != ncclSuccess) return r;
  ncclResult_t other = ncclSuccess;
  int peer = 1 - g_active;
  ncclResult_t rp = g_api[peer].CommGetAsyncError(d->real[peer], &other);
  if (rp != ncclSuccess) return rp;
  if (other != ncclSuccess && asyncError && *asyncError == ncclSuccess) *asyncError = other;
  return ncclSuccess;
}

ncclResult_t ncclCommRegister(const ncclComm_t comm, void* buff, size_t size, void** handle) {
  requireReady();
  DiffComm* d = asComm(comm);
  if (!d || !handle) return ncclInvalidArgument;
  auto* h = new DiffReg();
  h->magic = kRegMagic;
  for (int i = 0; i < kLibs; i++) {
    touch(i);
    ncclResult_t r = g_api[i].CommRegister(d->real[i], buff, size, &h->real[i]);
    if (r != ncclSuccess) {
      delete h;
      return r;
    }
  }
  *handle = h;
  return ncclSuccess;
}

ncclResult_t ncclCommDeregister(const ncclComm_t comm, void* handle) {
  requireReady();
  DiffComm* d = asComm(comm);
  auto* h = reinterpret_cast<DiffReg*>(handle);
  if (!d || !h || h->magic != kRegMagic) return ncclInvalidArgument;
  ncclResult_t r = ncclSuccess;
  for (int i = 0; i < kLibs; i++) {
    touch(i);
    ncclResult_t ri = g_api[i].CommDeregister(d->real[i], h->real[i]);
    if (ri != ncclSuccess) r = ri;
  }
  h->magic = 0;
  delete h;
  return r;
}

ncclResult_t ncclCommWindowRegister(ncclComm_t comm, void* buff, size_t size, ncclWindow_t* win, int winFlags) {
  requireReady();
  DiffComm* d = asComm(comm);
  if (!d || !win) return ncclInvalidArgument;
  auto* w = new DiffWin();
  w->magic = kWinMagic;
  for (int i = 0; i < kLibs; i++) {
    touch(i);
    ncclResult_t r = g_api[i].CommWindowRegister(d->real[i], buff, size, &w->real[i], winFlags);
    if (r != ncclSuccess) {
      delete w;
      return r;
    }
  }
  *win = reinterpret_cast<ncclWindow_t>(w);
  return ncclSuccess;
}

ncclResult_t ncclCommWindowDeregister(ncclComm_t comm, ncclWindow_t win) {
  requireReady();
  DiffComm* d = asComm(comm);
  auto* w = reinterpret_cast<DiffWin*>(win);
  if (!d || !w || w->magic != kWinMagic) return ncclInvalidArgument;
  ncclResult_t r = ncclSuccess;
  for (int i = 0; i < kLibs; i++) {
    touch(i);
    ncclResult_t ri = g_api[i].CommWindowDeregister(d->real[i], w->real[i]);
    if (ri != ncclSuccess) r = ri;
  }
  w->magic = 0;
  delete w;
  return r;
}

ncclResult_t ncclRedOpCreatePreMulSum(ncclRedOp_t* op, void* scalar, ncclDataType_t datatype,
    ncclScalarResidence_t residence, ncclComm_t comm) {
  requireReady();
  DiffComm* d = asComm(comm);
  if (!d || !op) return ncclInvalidArgument;
  DiffRedOp rec;
  rec.live = true;
  for (int i = 0; i < kLibs; i++) {
    touch(i);
    ncclResult_t r = g_api[i].RedOpCreatePreMulSum(&rec.real[i], scalar, datatype, residence, d->real[i]);
    if (r != ncclSuccess) return r;
  }
  std::lock_guard<std::mutex> lock(g_mu);
  g_redops.push_back(rec);
  *op = (ncclRedOp_t)(kRedOpBase + (int)g_redops.size() - 1);
  return ncclSuccess;
}

ncclResult_t ncclRedOpDestroy(ncclRedOp_t op, ncclComm_t comm) {
  requireReady();
  DiffComm* d = asComm(comm);
  if (!d) return ncclInvalidArgument;
  if ((int)op < kRedOpBase) return ncclSuccess;
  int idx = (int)op - kRedOpBase;
  std::lock_guard<std::mutex> lock(g_mu);
  if (idx < 0 || idx >= (int)g_redops.size() || !g_redops[idx].live) return ncclInvalidArgument;
  ncclResult_t r = ncclSuccess;
  for (int i = 0; i < kLibs; i++) {
    touch(i);
    ncclResult_t ri = g_api[i].RedOpDestroy(g_redops[idx].real[i], d->real[i]);
    if (ri != ncclSuccess) r = ri;
  }
  g_redops[idx].live = false;
  return r;
}

ncclResult_t ncclMemAlloc(void** ptr, size_t size) {
  requireReady();
  // One allocator owns the buffer. Both libraries use the resulting device pointer.
  touch(0);
  return g_api[0].MemAlloc(ptr, size);
}

ncclResult_t ncclMemFree(void* ptr) {
  requireReady();
  touch(0);
  return g_api[0].MemFree(ptr);
}

const char* ncclGetErrorString(ncclResult_t result) {
  requireReady();
  return g_api[0].GetErrorString(result);
}

const char* ncclGetLastError(ncclComm_t comm) {
  requireReady();
  DiffComm* d = asComm(comm);
  if (!d) return g_api[g_active].GetLastError(nullptr);
  return g_api[g_active].GetLastError(d->real[g_active]);
}

ncclResult_t ncclCommQueryProperties(ncclComm_t comm, ncclCommProperties_t* props) {
  requireReady();
  DiffComm* d = asComm(comm);
  if (!d || !props) return ncclInvalidArgument;
  int lib = g_active;
  if (!g_api[lib].CommQueryProperties) return ncclInvalidUsage;
  touch(lib);
  return g_api[lib].CommQueryProperties(d->real[lib], props);
}

ncclTeam_t ncclTeamLsa(ncclComm_t comm) {
  requireReady();
  DiffComm* d = asComm(comm);
  ncclTeam_t none;
  memset(&none, 0, sizeof(none));
  if (!d || !g_api[g_active].TeamLsa) return none;
  touch(g_active);
  return g_api[g_active].TeamLsa(d->real[g_active]);
}

ncclResult_t ncclAllReduce(const void* sendbuff, void* recvbuff, size_t count,
    ncclDataType_t datatype, ncclRedOp_t op, ncclComm_t comm, hipStream_t stream) {
  requireReady();
  DiffComm* d = asComm(comm);
  if (!d) return ncclInvalidArgument;
  int lib = g_active;
  touch(lib);
  return g_api[lib].AllReduce(sendbuff, recvbuff, count, datatype, mapOp(op, lib), d->real[lib], stream);
}

}  // extern "C"
