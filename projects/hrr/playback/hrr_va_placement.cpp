/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */
/* hrr_va_placement.cpp — the HIP half of capture-address placement. */

#include "hrr_va_placement.h"

#include <cstdio>

#ifndef _WIN32
#include <sys/mman.h>
#ifndef MAP_FIXED_NOREPLACE
#define MAP_FIXED_NOREPLACE 0x100000
#endif
#endif

namespace hrr {

namespace {

// Report at most this many fallbacks, and as many reserve failures, one by
// one unless --verbose. The rest are counted: past a handful, the lines stop
// helping anyone find the one that matters.
constexpr uint64_t kFallbackLines = 16;

using ull = unsigned long long;

#ifndef _WIN32
// Hold [b, e) with an inaccessible placeholder. MAP_FIXED_NOREPLACE fails
// instead of moving, and a kernel too old to know the flag treats the address
// as a hint, so the returned address is checked either way.
bool hold_exact(uint64_t b, uint64_t e) { return false; }

// Hold what can be held of [b, e). A range that collides with something
// mapped is split in half until the free pieces are found.
void hold_pieces(uint64_t b, uint64_t e, std::vector<VaRange>* out) {  }

void drop_hold(uint64_t b, uint64_t e) {  }
#else
bool hold_exact(uint64_t, uint64_t)  { return false; }
void drop_hold(uint64_t, uint64_t)  {  }
#endif

}  // namespace

std::vector<VaRange> read_proc_maps() { return {}; }

void hold_free_pieces(uint64_t b, uint64_t e, const std::vector<VaRange>& occupied,
                      std::vector<VaRange>* out) {  }

hipError_t hrr_vmm_map_into(void* va, size_t len, int device,
                            hipMemGenericAllocationHandle_t* out_handle,
                            const std::vector<int>& peers) { return hipErrorNotSupported; }

bool VaPlacement::hold(PlacementPlan plan) { return false; }

void VaPlacement::reserve_line(bool whole, uint64_t b, uint64_t e, const char* why) {  }

bool VaPlacement::reserve(int device_count, bool peer_access) { return false; }

void VaPlacement::fell_back(uint64_t rec, size_t size, const char* api,
                            const char* why) {  }

bool VaPlacement::map_at(uint64_t rec, size_t size, int device, const char* api,
                         void** live, bool capturing) { return false; }

bool VaPlacement::unmap_one(uint64_t pb, const PlacedMapping& m) { return false; }

bool VaPlacement::unmap(void* live, bool defer) { return false; }

size_t VaPlacement::drain_deferred() { return 0; }

bool VaPlacement::is_mapped(void* live) { return false; }

std::vector<uint64_t> VaPlacement::mapped_bases() { return {}; }

bool VaPlacement::release_vmm_hold(uint64_t base, size_t size) { return false; }

void VaPlacement::restore_vmm_hold(uint64_t base, size_t size) {  }

void VaPlacement::vmm_reserved(uint64_t rec, size_t size, bool held, uint64_t live) {  }

void VaPlacement::release_all() {  }

}  // namespace hrr
