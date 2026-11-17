// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "rocjitsu/code/patch/fault_injection/ordinary_acquire.h"

#include <cstdint>
#include <map>
#include <memory>
#include <vector>

namespace rocjitsu {

// Shared with later FI recipes. Only the context keeps the strong source owner;
// snapshots include load/cache/wait and the exact bookkeeping/branch proof.
struct OrdinaryAcquireRecord {
  std::weak_ptr<const AmdGpuCodeObject> source;
  FaultMutationSection section;
  FaultMutationOwner owner;
  std::vector<OrdinaryAcquireSite> sites;
  std::map<uint64_t, std::vector<uint8_t>> source_bytes;
};

} // namespace rocjitsu
