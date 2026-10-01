/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Implementation of the rocSHMEM host API fakes. See rocshmem_fakes.h.

#include "rocshmem_fakes.h"

#include <cstddef>
#include <cstdint>

#include "signature-drift.h"

namespace rocshmem {

static int DefaultGetUniqueId(rocshmem_uniqueid_t*) { return ROCSHMEM_SUCCESS; }
static int DefaultSetAttrUniqueIdArgs(int, int, rocshmem_uniqueid_t*, rocshmem_init_attr_t*) {
  return ROCSHMEM_SUCCESS;
}
static int DefaultInitAttr(unsigned int, rocshmem_init_attr_t*) { return ROCSHMEM_SUCCESS; }

std::function<int(rocshmem_uniqueid_t*)> g_rocshmemGetUniqueId = DefaultGetUniqueId;
std::function<int(int, int, rocshmem_uniqueid_t*, rocshmem_init_attr_t*)>
    g_rocshmemSetAttrUniqueIdArgs = DefaultSetAttrUniqueIdArgs;
std::function<int(unsigned int, rocshmem_init_attr_t*)> g_rocshmemInitAttr = DefaultInitAttr;

ASSERT_HOOK_MATCHES_PROD(g_rocshmemGetUniqueId, rocshmem_get_uniqueid);
ASSERT_HOOK_MATCHES_PROD(g_rocshmemSetAttrUniqueIdArgs, rocshmem_set_attr_uniqueid_args);
ASSERT_HOOK_MATCHES_PROD(g_rocshmemInitAttr, rocshmem_init_attr);

int rocshmem_get_uniqueid(rocshmem_uniqueid_t* uid) { return g_rocshmemGetUniqueId(uid); }
int rocshmem_set_attr_uniqueid_args(int rank, int nranks, rocshmem_uniqueid_t* uid,
                                    rocshmem_init_attr_t* attr) {
  return g_rocshmemSetAttrUniqueIdArgs(rank, nranks, uid, attr);
}
int rocshmem_init_attr(unsigned int flags, rocshmem_init_attr_t* attr) {
  return g_rocshmemInitAttr(flags, attr);
}
void rocshmem_finalize() {}

// init.cc takes two heaps and keeps them apart, so hand out distinct pointers.
static uint64_t g_heaps[2];
static int g_nextHeap = 0;
void* rocshmem_malloc(size_t) { return &g_heaps[g_nextHeap++ % 2]; }
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
  rocshmem::g_rocshmemGetUniqueId = rocshmem::DefaultGetUniqueId;
  rocshmem::g_rocshmemSetAttrUniqueIdArgs = rocshmem::DefaultSetAttrUniqueIdArgs;
  rocshmem::g_rocshmemInitAttr = rocshmem::DefaultInitAttr;
  rocshmem::g_nextHeap = 0;
}
