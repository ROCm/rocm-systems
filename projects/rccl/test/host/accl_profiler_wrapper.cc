// Compiles accl_profiler.cc for host unit tests.
// The plugin is self-contained (no RCCL hipified headers needed).
// Thin forwarding functions expose static helpers to AcclProfilerTests.cpp.

#include "accl_profiler.cc"
#include "accl_net_counters.cc"

extern "C" {
  int test_acclDatatypeSize(const char* dt) { return acclDatatypeSize(dt); }
  double test_acclBusBwFactor(const char* func, int nRanks) {
    return acclBusBwFactor(func, nRanks);
  }
  size_t test_acclParseMinSize(const char* env, int* bad) {
    return acclParseMinSize(env, bad);
  }

  // Context lifetime accessors. The plugin recycles pool slots and hands raw
  // interior pointers to RCCL, so the tests below need to observe the refcount
  // and to drive the owner's half of a finalize that the drain also touches.
  int  test_acclRefCount(void* ctx) {
    return __atomic_load_n(&((struct acclCommContext*)ctx)->refCount, __ATOMIC_SEQ_CST);
  }
  void test_acclMarkFinalized(void* coll) {
    ((struct acclCollInfo*)coll)->finalized = 1;
  }
  void test_acclFreeColl(void* ctx, void* coll) {
    acclFreeColl((struct acclCommContext*)ctx, (struct acclCollInfo*)coll);
  }
  // Coll pool slot of a coll handle, or -1 if the handle is not in the pool.
  int test_acclCollSlot(void* ctxv, void* collv) {
    struct acclCommContext* ctx = (struct acclCommContext*)ctxv;
    int idx = (int)((struct acclCollInfo*)collv - ctx->collPool);
    return (idx >= 0 && idx < ACCL_COLL_POOL_SIZE) ? idx : -1;
  }
  // Proxy-op pool slot of an op handle, or -1 if the handle is not in the pool.
  int test_acclProxyOpSlot(void* ctxv, void* opv) {
    struct acclCommContext* ctx = (struct acclCommContext*)ctxv;
    int idx = (int)((struct acclProxyOpInfo*)opv - ctx->proxyOpPool);
    return (idx >= 0 && idx < ACCL_PROXY_OP_POOL_SIZE) ? idx : -1;
  }
  // glibc marks a destroyed mutex with __kind == -1. That is the only way to see
  // "this mutex has been destroyed" without locking it, which would be UB.
  int test_acclProxyOpMutexDestroyed(void* ctxv, int slot) {
    struct acclCommContext* ctx = (struct acclCommContext*)ctxv;
    if (slot < 0 || slot >= ACCL_PROXY_OP_POOL_SIZE) return -1;
    return ctx->proxyOpPool[slot].mutex.__data.__kind == -1 ? 1 : 0;
  }
  // Descriptor behind the output FILE*, so a test can break writes on it.
  int test_acclOutputFd(void* ctxv) {
    struct acclCommContext* ctx = (struct acclCommContext*)ctxv;
    return ctx->outputFile ? fileno(ctx->outputFile) : -1;
  }
  // Current tenancy of a proxy-op slot; a step pins this at start.
  unsigned long test_acclProxyOpGeneration(void* ctxv, int slot) {
    struct acclCommContext* ctx = (struct acclCommContext*)ctxv;
    if (slot < 0 || slot >= ACCL_PROXY_OP_POOL_SIZE) return 0;
    return (unsigned long)ctx->proxyOpPool[slot].generation;
  }
  void test_acclWriteDummyRecord(void* ctx) {
    struct acclCompletedRecord rec;
    memset(&rec, 0, sizeof(rec));
    rec.func = "AllReduce"; rec.algo = "Ring"; rec.proto = "Simple";
    rec.nRanks = 1; rec.nChannels = 1; rec.msgSizeBytes = 4096;
    rec.totalExecUs = 1.0;
    acclWriteRecord((struct acclCommContext*)ctx, &rec, NULL);
  }

  // Net-counter test seams: replace the syscalls / whole-snapshot reader.
  void test_acclNetSetHooks(const struct acclNetHooks* hooks) { gAcclNetHooks = *hooks; }
  // Drives one RDMA netlink device through open + learn read + read.
  int test_acclRdmaOpenAndRead(const char* name, uint64_t* first, uint64_t* second,
                               char names[][ACCL_NET_NAME_LEN], int* nNames) {
    static char rx[ACCL_NL_BUF_SIZE];
    struct acclNetDev d;
    memset(&d, 0, sizeof(d));
    snprintf(d.name, sizeof(d.name), "%s", name);
    d.nlFd = -1;
    d.ethFd = -1;
    int rc = acclRdmaOpen(&d, rx);
    if (rc == 0) rc = acclRdmaRead(&d, rx, first, 1);
    if (rc == 0) rc = acclRdmaRead(&d, rx, second, 0) == 0 ? 0 : 2;
    for (int i = 0; i < d.nRdma; i++) memcpy(names[i], d.rdmaNames[i], ACCL_NET_NAME_LEN);
    *nNames = d.nRdma;
    if (d.nlFd >= 0) close(d.nlFd);
    return rc;
  }
  // Drives the ethtool reader on netdev `ifname` through `nReads` reads.
  int test_acclEthReads(const char* ifname, int nReads, uint64_t out[][ACCL_NET_MAX_ETH],
                        char names[][ETH_GSTRING_LEN + 1], int* nNames) {
    struct acclNetDev d;
    memset(&d, 0, sizeof(d));
    snprintf(d.netdev, sizeof(d.netdev), "%s", ifname);
    d.ethFd = 0;
    int rc = acclEthFetchNames(&d);
    for (int r = 0; rc == 0 && r < nReads; r++) rc = acclEthRead(&d, out[r]);
    for (int i = 0; i < d.nEth; i++) memcpy(names[i], d.ethNames[i], ETH_GSTRING_LEN + 1);
    *nNames = d.nEth;
    free(d.ethBuf);
    return rc;
  }
}
