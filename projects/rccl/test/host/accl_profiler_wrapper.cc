// Compiles accl_profiler.cc for host unit tests.
// The plugin is self-contained (no RCCL hipified headers needed).
// Thin forwarding functions expose static helpers to AcclProfilerTests.cpp.

#include "accl_profiler.cc"

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
    pthread_mutex_lock(&ctx->collPool.mutex);
    int idx = acclPoolIndex(&ctx->collPool, collv);
    pthread_mutex_unlock(&ctx->collPool.mutex);
    return idx;
  }
  // Proxy-op pool slot of an op handle, or -1 if the handle is not in the pool.
  int test_acclProxyOpSlot(void* ctxv, void* opv) {
    struct acclCommContext* ctx = (struct acclCommContext*)ctxv;
    pthread_mutex_lock(&ctx->proxyOpPool.mutex);
    int idx = acclPoolIndex(&ctx->proxyOpPool, opv);
    pthread_mutex_unlock(&ctx->proxyOpPool.mutex);
    return idx;
  }
  // glibc marks a destroyed mutex with __kind == -1. That is the only way to see
  // "this mutex has been destroyed" without locking it, which would be UB.
  int test_acclProxyOpMutexDestroyed(void* ctxv, int slot) {
    struct acclCommContext* ctx = (struct acclCommContext*)ctxv;
    if (slot < 0 || slot >= acclPoolSlots(&ctx->proxyOpPool)) return -1;
    struct acclProxyOpInfo* op =
        (struct acclProxyOpInfo*)acclPoolSlot(&ctx->proxyOpPool, slot);
    return op->mutex.__data.__kind == -1 ? 1 : 0;
  }
  // Descriptor behind the output FILE*, so a test can break writes on it.
  int test_acclOutputFd(void* ctxv) {
    struct acclCommContext* ctx = (struct acclCommContext*)ctxv;
    return ctx->outputFile ? fileno(ctx->outputFile) : -1;
  }
  // Current tenancy of a proxy-op slot; a step pins this at start.
  unsigned long test_acclProxyOpGeneration(void* ctxv, int slot) {
    struct acclCommContext* ctx = (struct acclCommContext*)ctxv;
    if (slot < 0 || slot >= acclPoolSlots(&ctx->proxyOpPool)) return 0;
    return (unsigned long)((struct acclProxyOpInfo*)acclPoolSlot(&ctx->proxyOpPool, slot))
        ->generation;
  }
  // Proxy-op slots allocated so far: grows by one chunk each time the pool runs out.
  int test_acclProxyOpPoolSlots(void* ctxv) {
    struct acclCommContext* ctx = (struct acclCommContext*)ctxv;
    pthread_mutex_lock(&ctx->proxyOpPool.mutex);
    int n = acclPoolSlots(&ctx->proxyOpPool);
    pthread_mutex_unlock(&ctx->proxyOpPool.mutex);
    return n;
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
