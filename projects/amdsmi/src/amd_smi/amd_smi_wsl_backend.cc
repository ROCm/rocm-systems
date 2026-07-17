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
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */

// Mock WSL/WDDM GPU backend.
//
// Stands in for the real D3DKMT (/dev/dxg) implementation so the single-backend
// seam can be exercised on a native workstation with no WSL host. Every method
// returns synthetic-but-plausible data; a real port replaces these bodies with
// wsl::thunk::Device queries and nothing in the dispatcher changes.
//
// This whole translation unit is compiled only when ENABLE_WSL_BACKEND is set.

#include "amd_smi/impl/amd_smi_wsl_backend.h"

#include <unistd.h>

#include <cstdlib>
#include <cstring>
#include <limits>

namespace amd::smi {

namespace {

// Runtime activation probe, evaluated once.
//   AMDSMI_WSL_MODE=1  -> force on (explicit test opt-in)
//   AMDSMI_WSL_MODE=0  -> force off
//   unset              -> on iff the dxgkrnl module is present (real WSL2)
bool compute_active() {
  const char* mode = std::getenv("AMDSMI_WSL_MODE");
  if (mode != nullptr) {
    return mode[0] == '1';
  }
  return ::access("/sys/module/dxgkrnl", F_OK) == 0;
}

void set_string(char* dst, const char* src) {
  std::strncpy(dst, src, AMDSMI_MAX_STRING_LENGTH - 1);
  dst[AMDSMI_MAX_STRING_LENGTH - 1] = '\0';
}

}  // namespace

bool WslBackend::active() {
  static const bool kActive = compute_active();
  return kActive;
}

WslBackend& WslBackend::instance() {
  static WslBackend backend;
  return backend;
}

amdsmi_status_t WslBackend::get_gpu_asic_info(amdsmi_processor_handle /*handle*/,
                                              amdsmi_asic_info_t* info) {
  if (info == nullptr) {
    return AMDSMI_STATUS_INVAL;
  }
  *info = {};
  set_string(info->market_name, "AMD Radeon (WSL mock)");
  info->vendor_id = 0x1002;
  set_string(info->vendor_name, "Advanced Micro Devices, Inc.");
  info->subvendor_id = 0x1002;
  info->device_id = 0x744c;
  info->rev_id = 0xc8;
  set_string(info->asic_serial, "0000000000000000");
  info->oam_id = std::numeric_limits<uint32_t>::max();
  info->num_of_compute_units = 96;
  info->target_graphics_version = 0x0000000000110000;  // gfx1100
  info->subsystem_id = 0x744c;
  info->flags = 0;
  return AMDSMI_STATUS_SUCCESS;
}

amdsmi_status_t WslBackend::get_gpu_board_info(amdsmi_processor_handle /*handle*/,
                                               amdsmi_board_info_t* info) {
  if (info == nullptr) {
    return AMDSMI_STATUS_INVAL;
  }
  *info = {};
  set_string(info->model_number, "WSL-MOCK-0000");
  set_string(info->product_serial, "N/A");
  set_string(info->fru_id, "N/A");
  set_string(info->product_name, "AMD Radeon (WSL mock)");
  set_string(info->manufacturer_name, "Advanced Micro Devices, Inc.");
  return AMDSMI_STATUS_SUCCESS;
}

amdsmi_status_t WslBackend::get_power_info(amdsmi_processor_handle /*handle*/,
                                           amdsmi_power_info_t* info) {
  if (info == nullptr) {
    return AMDSMI_STATUS_INVAL;
  }
  *info = {};
  info->current_socket_power = 42;  // W
  info->average_socket_power = 40;  // W
  info->power_limit = 241000000;    // uW (241 W)
  info->socket_power = 42;
  return AMDSMI_STATUS_SUCCESS;
}

amdsmi_status_t WslBackend::get_temp_metric(amdsmi_processor_handle /*handle*/,
                                            amdsmi_temperature_type_t /*sensor_type*/,
                                            amdsmi_temperature_metric_t metric,
                                            int64_t* temperature) {
  if (temperature == nullptr) {
    return AMDSMI_STATUS_INVAL;
  }
  // WDDM PMLog exposes a current edge temperature; other metrics are unsupported.
  if (metric != AMDSMI_TEMP_CURRENT) {
    return AMDSMI_STATUS_NOT_SUPPORTED;
  }
  *temperature = 47;  // deg C
  return AMDSMI_STATUS_SUCCESS;
}

}  // namespace amd::smi
