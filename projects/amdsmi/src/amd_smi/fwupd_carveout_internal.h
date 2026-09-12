// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef AMD_SMI_FWUPD_CARVEOUT_INTERNAL_H_
#define AMD_SMI_FWUPD_CARVEOUT_INTERNAL_H_

#include <string>
#include <vector>

#include "amd_smi/amdsmi.h"

namespace amd {
namespace smi {
namespace detail {

// A single fwupd BIOS setting parsed from the daemon's GetBiosSettings a{sv}
// reply. Declared here (rather than in fwupd_carveout.cc's anonymous namespace)
// so the pure, hardware-free parsing helpers below can be unit tested without a
// live D-Bus connection.
struct BiosSetting {
  std::string id;
  std::string name;
  std::string current;
  bool read_only = false;
  std::vector<std::string> values;
};

// Select the carveout setting from a parsed list, preferring AMD's canonical
// attribute id (com.amd-gpu.uma_carveout) over HP's UEFI-HII naming. Returns
// nullptr when no known carveout id is present.
const BiosSetting* FindCarveout(const std::vector<BiosSetting>& settings);

// Map a resolved carveout setting into the public info struct. Assigns option
// indices and (truncated) descriptions, clamps the option count to
// AMDSMI_MAX_CARVEOUT_OPTIONS, and sets current_index to the matching option --
// or to num_options ("unknown") when the current value is empty/redacted.
// Returns AMDSMI_STATUS_NOT_SUPPORTED when the setting exposes no options.
amdsmi_status_t PopulateCarveoutInfo(const BiosSetting& setting, amdsmi_uma_carveout_info_t* info);

// Validate a requested write against a resolved carveout setting (pure; no
// D-Bus). option_index is bounds-checked against the same
// AMDSMI_MAX_CARVEOUT_OPTIONS clamp the getter applies, so a write can never
// target an index the getter did not also expose. Returns AMDSMI_STATUS_NO_PERM
// when the setting is read-only, AMDSMI_STATUS_INVAL when option_index is out
// of range, and AMDSMI_STATUS_NOT_SUPPORTED when the setting has no usable
// name; otherwise AMDSMI_STATUS_SUCCESS.
amdsmi_status_t ValidateCarveoutWrite(const BiosSetting& setting, uint32_t option_index);

// Classify a SetBiosSettings D-Bus outcome (pure; no D-Bus). Called only once
// the setting has already been resolved through fwupd, so every branch here is
// terminal -- never AMDSMI_STATUS_NOT_SUPPORTED -- so the caller never falls
// through to a differently-ordered sysfs option list after committing to the
// fwupd backend. error_is_set/error_name/error_message mirror a DBusError;
// got_reply is false only when no error was set but the call still returned no
// reply.
amdsmi_status_t ClassifySetReply(bool error_is_set, const std::string& error_name,
                                 const std::string& error_message, bool got_reply);

}  // namespace detail
}  // namespace smi
}  // namespace amd

#endif  // AMD_SMI_FWUPD_CARVEOUT_INTERNAL_H_
