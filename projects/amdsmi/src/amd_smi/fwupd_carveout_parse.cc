// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Pure, D-Bus-free helpers for the fwupd UMA carveout adapter. Split out of
// fwupd_carveout.cc (which owns the dlopen()/D-Bus plumbing) so these can be
// linked and unit-tested directly, without pulling in libdbus.

#include <strings.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

#include "fwupd_carveout_internal.h"

namespace amd {
namespace smi {
namespace detail {

const BiosSetting* FindCarveout(const std::vector<BiosSetting>& settings) {
  static const char* const kCarveoutIds[] = {"com.amd-gpu.uma_carveout",
                                             "com.hp-bioscfg.Dedicated_Graphics_Memory"};
  for (const char* want : kCarveoutIds) {
    for (const auto& s : settings) {
      if (strcasecmp(s.id.c_str(), want) == 0) return &s;
    }
  }
  return nullptr;
}

amdsmi_status_t PopulateCarveoutInfo(const BiosSetting& setting, amdsmi_uma_carveout_info_t* info) {
  if (info == nullptr) return AMDSMI_STATUS_INVAL;
  if (setting.values.empty()) return AMDSMI_STATUS_NOT_SUPPORTED;
  uint32_t count = static_cast<uint32_t>(setting.values.size());
  if (count > AMDSMI_MAX_CARVEOUT_OPTIONS) count = AMDSMI_MAX_CARVEOUT_OPTIONS;
  uint32_t current = count;  // sentinel: "unknown" == num_options
  for (uint32_t i = 0; i < count; ++i) {
    info->options[i].index = i;
    std::strncpy(info->options[i].description, setting.values[i].c_str(),
                 AMDSMI_MAX_STRING_LENGTH - 1);
    info->options[i].description[AMDSMI_MAX_STRING_LENGTH - 1] = '\0';
    if (!setting.current.empty() && setting.current == setting.values[i]) current = i;
  }
  info->num_options = count;
  info->current_index = current;
  return AMDSMI_STATUS_SUCCESS;
}

amdsmi_status_t ValidateCarveoutWrite(const BiosSetting& setting, uint32_t option_index) {
  if (setting.read_only) return AMDSMI_STATUS_NO_PERM;
  // Match the getter's AMDSMI_MAX_CARVEOUT_OPTIONS clamp so a write can never
  // target an index the getter never advertised to the caller.
  const size_t max_index = std::min<size_t>(setting.values.size(), AMDSMI_MAX_CARVEOUT_OPTIONS);
  if (option_index >= max_index) return AMDSMI_STATUS_INVAL;
  if (setting.name.empty()) return AMDSMI_STATUS_NOT_SUPPORTED;
  return AMDSMI_STATUS_SUCCESS;
}

amdsmi_status_t ClassifySetReply(bool error_is_set, const std::string& error_name,
                                 const std::string& error_message, bool got_reply) {
  if (error_is_set) {
    // Classify by the canonical D-Bus error name first (stable across fwupd
    // versions and locales); the human-readable message substrings are only a
    // best-effort fallback. fwupd returns NothingToDo when the value already
    // matches -- an idempotent success for our purposes.
    if (error_name.find("NothingToDo") != std::string::npos ||
        error_message.find("already set") != std::string::npos ||
        error_message.find("no BIOS settings needed") != std::string::npos) {
      return AMDSMI_STATUS_SUCCESS;
    }
    if (error_name.find("AccessDenied") != std::string::npos ||
        error_name.find("AuthFailed") != std::string::npos ||
        error_name.find("PermissionDenied") != std::string::npos ||
        error_message.find("not authorized") != std::string::npos ||
        error_message.find("permission") != std::string::npos) {
      return AMDSMI_STATUS_NO_PERM;
    }
    // The setting was already resolved through fwupd (see the caller): any
    // other failure here is terminal, not "backend absent".
    return AMDSMI_STATUS_API_FAILED;
  }
  // No error reported but also no reply: treat as failure, not success.
  if (!got_reply) return AMDSMI_STATUS_API_FAILED;
  return AMDSMI_STATUS_SUCCESS;
}

}  // namespace detail
}  // namespace smi
}  // namespace amd
