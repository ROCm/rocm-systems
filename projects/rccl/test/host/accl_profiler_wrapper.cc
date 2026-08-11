// Compiles accl_profiler.cc for host unit tests.
// The plugin is self-contained (no RCCL hipified headers needed).
// Thin forwarding functions expose static helpers to AcclProfilerTests.cpp.

#include "accl_profiler.cc"

extern "C" {
  int test_acclDatatypeSize(const char* dt) { return acclDatatypeSize(dt); }
  double test_acclBusBwFactor(const char* func, int nRanks) {
    return acclBusBwFactor(func, nRanks);
  }
}
