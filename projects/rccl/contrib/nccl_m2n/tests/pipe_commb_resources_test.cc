/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 * See LICENSE.txt for more license information.
 ************************************************************************/

/* Host-only coverage: no CUDA/NCCL/GTest dependency. Build directly with
 * c++ -std=c++17 -I../src pipe_commb_resources_test.cc -o /tmp/pipe_resources_test
 * or use the Make/CMake target pipe_commb_resources_test. */
#include <cstdio>
#include <cstdlib>
#include <initializer_list>
#include <limits>

#include "reshard_pipe_resources.h"

#define CHECK(condition) do { \
  if (!(condition)) { \
    std::fprintf(stderr, "%s:%d: check failed: %s\n", __FILE__, __LINE__, #condition); \
    std::exit(1); \
  } \
} while (0)

int main() {
  ReshardPipeNamespacePool pool{};
  for (int parent = 1; parent <= 8; parent++) {
    const int id = pool.propose(parent, 8);
    CHECK(id == parent - 1);
    CHECK(pool.count == parent - 1); // proposal must not consume an ID
    CHECK(!pool.commit(parent, 8, (id + 1) % 8));
    CHECK(pool.commit(parent, 8, id));
    CHECK(pool.count == parent);
  }
  // Revisit every parent in reverse order, then revisit the first parent.
  for (int parent = 8; parent >= 1; parent--) {
    CHECK(pool.propose(parent, 8) == parent - 1);
    CHECK(pool.commit(parent, 8, parent - 1));
    CHECK(pool.count == 8);
  }
  CHECK(pool.propose(1, 8) == 0);
  CHECK(pool.propose(9, 8) == -1); // exhaustion must never wrap
  CHECK(!pool.commit(9, 8, 0));
  CHECK(pool.propose(0, 8) == -1);
  CHECK(pool.propose(1, 0) == -1);
  CHECK(pool.propose(1, 65) == -1);

  // A different generator pool owns a separate namespace registry.
  ReshardPipeNamespacePool other{};
  CHECK(other.propose(9, 8) == 0);
  CHECK(other.commit(9, 8, 0));
  CHECK(pool.propose(1, 8) == 0);

  // IDs 0 and 4 share execution lane 0 at C4, but have disjoint persistent
  // signal regions; all 32 control slots are contained in their namespace.
  const int signalStride = 32 * 256;
  for (int id = 0; id < 8; id++) {
    for (int control = 0; control < 32; control++) {
      const int base = id * signalStride + control * 256;
      CHECK(base >= id * signalStride);
      CHECK(base + 256 <= (id + 1) * signalStride);
    }
  }
  CHECK(reshardPipeResourceLane(0, 4) == reshardPipeResourceLane(4, 4));
  for (int lanes : {1, 2, 4, 8}) {
    for (int parent = 1; parent <= 8; parent++) {
      const int id = pool.propose(parent, 8);
      CHECK(reshardPipeResourceLane(id, lanes) == (parent - 1) % lanes);
    }
  }
  CHECK(reshardPipeResourceLane(-1, 4) == -1);
  CHECK(reshardPipeResourceLane(0, 0) == -1);
  CHECK(0 * signalStride != 4 * signalStride);

  ReshardCommBResourcePlan c1{}, c2{}, c4{}, c8{};
  CHECK(reshardBuildPipeCommBResourcePlan(8, 1, 32, 256, 128, 4, &c1));
  CHECK(reshardBuildPipeCommBResourcePlan(8, 2, 32, 256, 128, 4, &c2));
  CHECK(reshardBuildPipeCommBResourcePlan(8, 4, 32, 256, 128, 4, &c4));
  CHECK(reshardBuildPipeCommBResourcePlan(8, 8, 32, 256, 128, 4, &c8));
  CHECK(c4.signals == 65536 && c4.counters == 32768 && c4.contexts == 16 && c4.barriers == 0);
  CHECK(c1.signals == c8.signals && c1.counters == c8.counters);
  CHECK(c1.contexts == 4 && c2.contexts == 8 && c8.contexts == 32);
  CHECK(!(c4 == c8));
  ReshardCommBResourcePlan same{};
  CHECK(reshardBuildPipeCommBResourcePlan(8, 4, 32, 256, 128, 4, &same));
  CHECK(c4 == same);
  CHECK(!reshardBuildPipeCommBResourcePlan(0, 4, 32, 256, 128, 4, &same));
  CHECK(!reshardBuildPipeCommBResourcePlan(65, 4, 32, 256, 128, 4, &same));
  CHECK(!reshardBuildPipeCommBResourcePlan(8, 0, 32, 256, 128, 4, &same));
  CHECK(!reshardBuildPipeCommBResourcePlan(8, 4, 32, std::numeric_limits<int>::max(), 128, 4, &same));
  CHECK(c4 == same); // rejected plans do not mutate the output
  int product = -1;
  CHECK(reshardCheckedResourceProduct(0, 8, &product) && product == 0);
  CHECK(!reshardCheckedResourceProduct(-1, 8, &product));
  CHECK(!reshardCheckedResourceProduct(std::numeric_limits<int>::max(), 2, &product));

  std::puts("PIPE commB resources: PASS (stable namespaces, exhaustion, control ranges, C1/C2/C4/C8 lanes, overflow)");
  return 0;
}
