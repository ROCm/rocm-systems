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

#include "amd_smi/impl/wsl/wsl_pmlog.h"

#include <ntstatus.h>

#include <sstream>

#include "rocm_smi/rocm_smi_logger.h"

namespace wsl {

bool QueryPmlogSnapshot(const WslAdapterInfo& info, PmlogSnapshot* snapshot) {
  if (!snapshot) return false;
  *snapshot = PmlogSnapshot{};

  NTSTATUS status = Wkmi::QueryPMLogSupport(info.adapter, info.device, snapshot->sensor_ids);
  if (status != STATUS_SUCCESS) {
    std::ostringstream ss;
    ss << "[wsl_pmlog] Wkmi::QueryPMLogSupport failed: 0x" << std::hex << status;
    LOG_ERROR(ss);
    return false;
  }

  status = Wkmi::QueryPMLogData(info.adapter, info.device, &snapshot->data);
  if (status != STATUS_SUCCESS) {
    std::ostringstream ss;
    ss << "[wsl_pmlog] Wkmi::QueryPMLogData failed: 0x" << std::hex << status;
    LOG_ERROR(ss);
    return false;
  }

  // Sensor limits are used only to derive power_limit-style fields; tolerate
  // failure here rather than failing the whole snapshot.
  status = Wkmi::QueryPMLogSensorLimits(info.adapter, info.device, &snapshot->limits);
  if (status != STATUS_SUCCESS) {
    std::ostringstream ss;
    ss << "[wsl_pmlog] Wkmi::QueryPMLogSensorLimits failed: 0x" << std::hex << status
       << " (non-fatal)";
    LOG_WARN(ss);
    snapshot->has_limits = false;
  } else {
    snapshot->has_limits = true;
  }

  return true;
}

uint32_t PmlogValue(const PmlogSnapshot& snapshot, Wkmi::PmlogSensorId id) {
  // QueryPMLogData indexes PmlogQueryResult::supported/value directly by sensor id
  // (unlike PmlogSensorLimits below, which is indexed by slot position).
  uint32_t idx = static_cast<uint32_t>(id);
  if (idx >= Wkmi::kPmlogMaxSensors) return Wkmi::kSensorUnavailable;
  if (!snapshot.data.supported[idx]) return Wkmi::kSensorUnavailable;
  return snapshot.data.value[idx];
}

uint32_t PmlogMaxLimit(const PmlogSnapshot& snapshot, Wkmi::PmlogSensorId id) {
  if (!snapshot.has_limits) return Wkmi::kSensorUnavailable;

  // PmlogSensorLimits is indexed by slot position, not sensor id, so translate
  // via the slot->sensor_id map returned by QueryPMLogSupport.
  const uint16_t target = static_cast<uint16_t>(id);
  for (uint32_t slot = 0; slot < Wkmi::kPmlogMaxSensors; ++slot) {
    if (snapshot.sensor_ids[slot] == target) return snapshot.limits.limits[slot][1];
  }
  return Wkmi::kSensorUnavailable;
}

bool QueryChipsetInfo(const WslAdapterInfo& info, Wkmi::ChipsetIdInfo* out) {
  if (!out) return false;

  NTSTATUS status = Wkmi::QueryChipsetId(info.adapter, info.device, out);
  if (status != STATUS_SUCCESS) {
    std::ostringstream ss;
    ss << "[wsl_pmlog] Wkmi::QueryChipsetId failed: 0x" << std::hex << status;
    LOG_ERROR(ss);
    return false;
  }
  return true;
}

bool QueryVideoBios(const WslAdapterInfo& info, Wkmi::VideoBiosInfo* out) {
  if (!out) return false;

  NTSTATUS status = Wkmi::QueryVideoBiosInfo(info.adapter, info.device, out);
  if (status != STATUS_SUCCESS) {
    std::ostringstream ss;
    ss << "[wsl_pmlog] Wkmi::QueryVideoBiosInfo failed: 0x" << std::hex << status;
    LOG_ERROR(ss);
    return false;
  }
  return true;
}

}  // namespace wsl
