// Compiles accl_profiler.cc for host unit tests.
// The plugin is self-contained (no RCCL hipified headers needed).
// Thin forwarding functions expose static helpers to AcclProfilerTests.cpp.

#include "accl_profiler.cc"

extern "C" {
  int test_acclDatatypeSize(const char* dt) { return acclDatatypeSize(dt); }
  double test_acclBusBwFactor(const char* func, int nRanks) {
    return acclBusBwFactor(func, nRanks);
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
  void test_acclWriteDummyRecord(void* ctx) {
    struct acclCompletedRecord rec;
    memset(&rec, 0, sizeof(rec));
    rec.func = "AllReduce"; rec.algo = "Ring"; rec.proto = "Simple";
    rec.nRanks = 1; rec.nChannels = 1; rec.msgSizeBytes = 4096;
    rec.totalExecUs = 1.0;
    acclWriteRecord((struct acclCommContext*)ctx, &rec);
  }
}
