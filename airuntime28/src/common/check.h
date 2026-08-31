// AIRUNTIME-28 benchmark support: error checking.
#ifndef AIRUNTIME28_CHECK_H_
#define AIRUNTIME28_CHECK_H_

#include <hip/hip_runtime.h>

#include <cstdio>
#include <cstdlib>

#define HIP_CHECK(cmd)                                                              \
  do {                                                                              \
    hipError_t _e = (cmd);                                                           \
    if (_e != hipSuccess) {                                                          \
      std::fprintf(stderr, "HIP error %s at %s:%d -> %s\n", hipGetErrorString(_e),   \
                   __FILE__, __LINE__, #cmd);                                        \
      std::exit(1);                                                                  \
    }                                                                                \
  } while (0)

// Assertions that stop a run rather than printing a wrong number. Used for the
// invariants a reader of the report is entitled to assume held: byte-exact copies,
// the ISA a variant claims to emit, and the sign of a headline result.
#define BENCH_ASSERT(cond, ...)                                        \
  do {                                                                 \
    if (!(cond)) {                                                     \
      std::fprintf(stderr, "\nASSERTION FAILED (%s:%d): ", __FILE__, __LINE__); \
      std::fprintf(stderr, __VA_ARGS__);                               \
      std::fprintf(stderr, "\n  condition: %s\n", #cond);              \
      bench::markFailure();                                            \
    }                                                                  \
  } while (0)

namespace bench {

// Assertions record rather than abort, so one failure does not hide the rest of a
// run. main() returns non-zero if any fired.
inline int& failureCount() {
  static int n = 0;
  return n;
}
inline void markFailure() { ++failureCount(); }

inline int exitCode() {
  if (failureCount() > 0) {
    std::fprintf(stderr, "\n%d assertion(s) failed\n", failureCount());
    return 1;
  }
  return 0;
}

}  // namespace bench

#endif  // AIRUNTIME28_CHECK_H_
