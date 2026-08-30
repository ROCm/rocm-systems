// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/vmm/vfu/bus_plan.h"

namespace rocjitsu {
namespace {

/// @brief Highest BAR index a PCI function has.
constexpr int kMaxBarIndex = 5;

/// @brief Vectors a message-interrupt capability can describe, its table-size
/// field being eleven bits holding one less than the count.
constexpr uint32_t kMaxMsixVectors = 2048;

/// @brief Units the table and pending-bit offsets are recorded in, the low
/// three bits of each field carrying the BAR index instead.
constexpr uint64_t kMsixOffsetUnit = 8;

/// @brief Largest offset those fields can express, twenty-nine bits of units.
constexpr uint64_t kMaxMsixOffset = (uint64_t{1} << 32) - kMsixOffsetUnit;

} // namespace

BarRegionPlan plan_bar_region(const simdojo::BarSpec &bar) {
  BarRegionPlan plan;
  if (bar.index < 0 || bar.index > 5 || bar.size == 0) {
    return plan;
  }

  for (const simdojo::MmapArea &area : bar.mmap_areas) {
    const bool within_region = area.offset <= bar.size && area.length <= bar.size - area.offset;
    if (area.length == 0 || !within_region) {
      return plan;
    }
  }

  if (!bar.mmap_areas.empty() && bar.backing_fd < 0) {
    return plan;
  }

  plan.valid = true;
  plan.mmap_areas = bar.mmap_areas;
  // Withholding the descriptor is what makes a window-less BAR trap: a transport
  // that has one may otherwise expose the whole region.
  plan.backing_fd = bar.mmap_areas.empty() ? -1 : bar.backing_fd;
  plan.fd_offset = plan.backing_fd < 0 ? 0 : bar.fd_offset;
  return plan;
}

InterruptPlan plan_interrupts(const simdojo::InterruptSpec &spec) {
  InterruptPlan plan;
  switch (spec.kind) {
  case simdojo::InterruptKind::None:
    plan.supported = true;
    plan.intx_count = 0;
    break;
  case simdojo::InterruptKind::IntxPin:
    plan.supported = true;
    plan.intx_count = 1;
    break;
  case simdojo::InterruptKind::MsiX:
    // A table of no vectors is not a capability: a guest would allocate from
    // it and get nothing back, which is worse than not offering it.
    if (spec.vectors == 0) {
      break;
    }
    // What a capability can say about itself, and therefore what a device is
    // allowed to declare: a table size one less than the count in eleven bits,
    // a BAR index in three, and two offsets in units of eight bytes in
    // twenty-nine. A declaration that does not fit would not be refused
    // anywhere downstream -- it would be published rounded down or wrapped
    // around, naming bytes the device does not treat as a table.
    if (spec.vectors > kMaxMsixVectors || spec.table_bar < 0 || spec.table_bar > kMaxBarIndex ||
        (spec.table_offset % kMsixOffsetUnit) != 0 ||
        (spec.pending_offset % kMsixOffsetUnit) != 0 || spec.table_offset > kMaxMsixOffset ||
        spec.pending_offset > kMaxMsixOffset) {
      break;
    }
    plan.supported = true;
    plan.msix_count = spec.vectors;
    plan.table_bar = spec.table_bar;
    plan.table_offset = spec.table_offset;
    plan.pending_offset = spec.pending_offset;
    break;
  }
  return plan;
}

} // namespace rocjitsu
