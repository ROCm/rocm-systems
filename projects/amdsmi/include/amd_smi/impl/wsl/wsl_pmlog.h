/*
 * Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
 */

#pragma once

#include <cstdint>

#include "amd_smi/impl/wsl/wsl_adapter.h"
#include "wkmi.h"

namespace wsl {

// A single one-shot read of the adapter's PMLog telemetry, plus the slot table
// needed to resolve Wkmi::PmlogSensorLimits (which, unlike PmlogQueryResult, is
// indexed by *slot position* rather than sensor id -- see QueryPmlogSnapshot).
// Callers should treat the members as opaque and go through PmlogValue()/
// PmlogMaxLimit() rather than indexing sensor_ids/data/limits directly.
struct PmlogSnapshot {
  uint16_t sensor_ids[Wkmi::kPmlogMaxSensors] = {};  // slot -> sensor id (0 if slot unused)
  Wkmi::PmlogQueryResult data = {};                  // indexed by sensor id (NOT slot)
  Wkmi::PmlogSensorLimits limits = {};               // indexed by slot position (NOT sensor id)
  bool has_limits = false;                           // false if QueryPMLogSensorLimits failed
};

// Populates `snapshot` via Wkmi::QueryPMLogSupport + Wkmi::QueryPMLogData (required)
// and Wkmi::QueryPMLogSensorLimits (best-effort -- see PmlogSnapshot::has_limits).
// Returns false if the required D3DKMT/Wkmi queries fail.
bool QueryPmlogSnapshot(const WslAdapterInfo& info, PmlogSnapshot* snapshot);

// Returns the current value of `id` for this snapshot, or Wkmi::kSensorUnavailable
// if the sensor isn't supported by this adapter. If a caller needs a fallback sensor
// (e.g. board power falling back to asic power when unsupported), just try each
// Wkmi::PmlogSensorId in turn and use the first one that isn't kSensorUnavailable --
// that policy is business logic for the caller, not something this wrapper encodes.
uint32_t PmlogValue(const PmlogSnapshot& snapshot, Wkmi::PmlogSensorId id);

// Returns the max limit configured for `id`, or Wkmi::kSensorUnavailable if the
// sensor isn't present in the adapter's slot table or limits weren't available
// (PmlogSnapshot::has_limits is false). Internally translates the slot-indexed
// Wkmi::PmlogSensorLimits into a value keyed by sensor id.
uint32_t PmlogMaxLimit(const PmlogSnapshot& snapshot, Wkmi::PmlogSensorId id);

// Static PCIe capability info (CWDDECI_CHIPSETIDENTIFICATION escape).
bool QueryChipsetInfo(const WslAdapterInfo& info, Wkmi::ChipsetIdInfo* out);

// VBIOS version/part-number/date (driver escape).
bool QueryVideoBios(const WslAdapterInfo& info, Wkmi::VideoBiosInfo* out);

}  // namespace wsl
