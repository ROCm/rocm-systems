// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/vmm/vfu/bus_plan.h"

namespace rocjitsu {

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
    // MSI-X needs a vector table inside a BAR, which arrives with the first
    // device that raises per-vector completions.
    break;
  }
  return plan;
}

} // namespace rocjitsu
