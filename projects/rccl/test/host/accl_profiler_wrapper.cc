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
  void test_acclWriteDummyRecord(void* ctx) {
    struct acclCompletedRecord rec;
    memset(&rec, 0, sizeof(rec));
    rec.func = "AllReduce"; rec.algo = "Ring"; rec.proto = "Simple";
    rec.nRanks = 1; rec.nChannels = 1; rec.msgSizeBytes = 4096;
    rec.totalExecUs = 1.0;
    acclWriteRecord((struct acclCommContext*)ctx, &rec);
  }
}
