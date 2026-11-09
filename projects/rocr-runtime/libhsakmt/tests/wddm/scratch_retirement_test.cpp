// SPDX-License-Identifier: NCSA
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>

#include "impl/wddm/scratch_retirement.h"

#define CHECK(expression) do { \
  if (!(expression)) { \
    std::cerr << "failed: " #expression " at line " << __LINE__ << '\n'; \
    std::abort(); \
  } \
} while (false)

struct Allocation {
  explicit Allocation(int* destroyed) : destroyed(destroyed) {}
  ~Allocation() { ++*destroyed; }
  int* destroyed;
};

int main() {
  using Retirement = wsl::thunk::ScratchRetirement<Allocation>;
  int destroyed = 0;
  Retirement retired;
  CHECK(retired.Retire(new Allocation(&destroyed), 10));
  CHECK(retired.Retire(new Allocation(&destroyed), 20));
  CHECK(retired.Count() == 2);
  retired.Reap(9, 20);
  retired.Reap(std::numeric_limits<uint64_t>::max(), 20);
  retired.Reap(21, 20);  // A value beyond the last submitted fence is invalid.
  CHECK(destroyed == 0);
  retired.Reap(10, 20);
  CHECK(destroyed == 1 && retired.Count() == 1);
  retired.Reap(20, 20);
  CHECK(destroyed == 2 && retired.Count() == 0);

  CHECK(retired.Retire(new Allocation(&destroyed), 30));
  CHECK(retired.Retire(new Allocation(&destroyed), 40));
  retired.ReleaseAfterQueueDestroy();
  CHECK(destroyed == 4 && retired.Count() == 0);

  // More than the old fixed capacity remains valid until its fences complete.
  for (uint64_t target = 1; target <= 65; ++target)
    CHECK(retired.Retire(new Allocation(&destroyed), target));
  CHECK(retired.Count() == 65 && destroyed == 4);
  retired.Reap(64, 65);
  CHECK(retired.Count() == 1 && destroyed == 68);
  retired.Reap(65, 65);
  CHECK(retired.Count() == 0 && destroyed == 69);

  Allocation* abandoned = new Allocation(&destroyed);
  CHECK(retired.Retire(abandoned, 3));
  retired.AbandonAfterFailedQueueDestroy();
  CHECK(destroyed == 69 && retired.Count() == 0);
  delete abandoned;  // Simulate eventual process cleanup for the ownership test.
  CHECK(destroyed == 70);
}
