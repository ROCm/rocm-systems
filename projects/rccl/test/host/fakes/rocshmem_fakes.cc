/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Implementation of the rocSHMEM host API fakes. See rocshmem_fakes.h.

#include "rocshmem_fakes.h"

#include <cstddef>
#include <cstdint>

namespace rocshmem {

static int Success() { return ROCSHMEM_SUCCESS; }
std::function<int()> g_rocshmemGetUniqueId = Success;
std::function<int()> g_rocshmemSetAttrUniqueIdArgs = Success;
std::function<int()> g_rocshmemInitAttr = Success;

int rocshmem_get_uniqueid(rocshmem_uniqueid_t*) { return g_rocshmemGetUniqueId(); }
int rocshmem_set_attr_uniqueid_args(int, int, rocshmem_uniqueid_t*, rocshmem_init_attr_t*) {
  return g_rocshmemSetAttrUniqueIdArgs();
}
int rocshmem_init_attr(unsigned int, rocshmem_init_attr_t*) { return g_rocshmemInitAttr(); }
void rocshmem_finalize() {}

// init.cc takes two heaps and keeps them apart, so hand out distinct pointers.
void* rocshmem_malloc(size_t) {
  static uint64_t heaps[2];
  static int next = 0;
  return &heaps[next++ % 2];
}
void rocshmem_free(void*) {}

int rocshmem_team_split_strided(rocshmem_team_t, int, int, int, const rocshmem_team_config_t*, long,
                                rocshmem_team_t* new_team) {
  if (new_team) *new_team = host::ROCSHMEM_TEAM_WORLD;
  return ROCSHMEM_SUCCESS;
}
void rocshmem_team_destroy(rocshmem_team_t) {}

namespace host {
rocshmem_team_t ROCSHMEM_TEAM_WORLD = nullptr;
}  // namespace host

}  // namespace rocshmem

void ResetRocshmemFakes() {
  rocshmem::g_rocshmemGetUniqueId = rocshmem::Success;
  rocshmem::g_rocshmemSetAttrUniqueIdArgs = rocshmem::Success;
  rocshmem::g_rocshmemInitAttr = rocshmem::Success;
}
