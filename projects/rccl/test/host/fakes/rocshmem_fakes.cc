/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Implementation of the rocSHMEM host API fakes. See rocshmem_fakes.h.

#include "rocshmem_fakes.h"

namespace rocshmem {

std::function<int()> g_rocshmemGetUniqueId = [] { return ROCSHMEM_SUCCESS; };
std::function<int()> g_rocshmemInitAttr = [] { return ROCSHMEM_SUCCESS; };

// Two allocations, never freed and never read, so one static block per call
// site is enough to keep them distinguishable.
static uint64_t g_heap[2];
void* g_rocshmemMallocReturn = nullptr;

int rocshmem_get_uniqueid(rocshmem_uniqueid_t*) { return g_rocshmemGetUniqueId(); }
int rocshmem_set_attr_uniqueid_args(int, int, rocshmem_uniqueid_t*,
                                    rocshmem_init_attr_t*) {
  return ROCSHMEM_SUCCESS;
}
int rocshmem_init_attr(unsigned int, rocshmem_init_attr_t*) { return g_rocshmemInitAttr(); }
void rocshmem_finalize() {}

void* rocshmem_malloc(size_t) {
  static int next = 0;
  g_rocshmemMallocReturn = &g_heap[next % 2];
  ++next;
  return g_rocshmemMallocReturn;
}
void rocshmem_free(void*) {}

int rocshmem_team_split_strided(rocshmem_team_t, int, int, int,
                                const rocshmem_team_config_t*, long,
                                rocshmem_team_t* new_team) {
  // init.cc keeps the handle and destroys it later, so it must not stay the
  // ROCSHMEM_TEAM_INVALID the caller pre-set it to.
  if (new_team) *new_team = host::ROCSHMEM_TEAM_WORLD;
  return ROCSHMEM_SUCCESS;
}
void rocshmem_team_destroy(rocshmem_team_t) {}

namespace host {
rocshmem_team_t ROCSHMEM_TEAM_WORLD = nullptr;
}  // namespace host

}  // namespace rocshmem

void ResetRocshmemFakes() {
  rocshmem::g_rocshmemGetUniqueId = [] { return rocshmem::ROCSHMEM_SUCCESS; };
  rocshmem::g_rocshmemInitAttr = [] { return rocshmem::ROCSHMEM_SUCCESS; };
  rocshmem::g_rocshmemMallocReturn = nullptr;
}
