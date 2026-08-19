/*
 * Copyright (C) Advanced Micro Devices. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy of
 * this software and associated documentation files (the "Software"), to deal in
 * the Software without restriction, including without limitation the rights to
 * use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
 * the Software, and to permit persons to whom the Software is furnished to do so,
 * subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
 * FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
 * COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
 * IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
 * CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */

#include "fwupd_carveout.h"

#ifdef AMDSMI_ENABLE_FWUPD

#include <fwupd.h>
#include <glib.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace {

// Order is significant: AMD's canonical attribute wins over HP's fallback.
constexpr const char* kCarveoutSettingIds[] = {
    "com.amd-gpu.uma_carveout",
    "com.hp-bioscfg.Dedicated_Graphics_Memory",
};

// Return the highest-priority carveout setting present, or nullptr.
FwupdBiosSetting* FindCarveout(GPtrArray* settings) {
  for (const char* wanted : kCarveoutSettingIds) {
    for (guint i = 0; i < settings->len; ++i) {
      auto* setting = static_cast<FwupdBiosSetting*>(g_ptr_array_index(settings, i));
      const char* id = fwupd_bios_setting_get_id(setting);
      if (id != nullptr && g_ascii_strcasecmp(id, wanted) == 0) {
        return setting;
      }
    }
  }
  return nullptr;
}

bool DryRun() {
  const char* value = std::getenv("AMDSMI_DRY_RUN");
  return value != nullptr && std::strcmp(value, "1") == 0;
}

}  // namespace

namespace amd {
namespace smi {

amdsmi_status_t fwupd_get_carveout_info(amdsmi_uma_carveout_info_t* info) {
  if (info == nullptr) {
    return AMDSMI_STATUS_INVAL;
  }

  g_autoptr(FwupdClient) client = fwupd_client_new();
  g_autoptr(GError) error = nullptr;
  g_autoptr(GPtrArray) settings = fwupd_client_get_bios_settings(client, nullptr, &error);
  if (settings == nullptr) {
    return AMDSMI_STATUS_NOT_SUPPORTED;
  }

  FwupdBiosSetting* setting = FindCarveout(settings);
  if (setting == nullptr) {
    return AMDSMI_STATUS_NOT_SUPPORTED;
  }

  GPtrArray* options = fwupd_bios_setting_get_possible_values(setting);
  if (options == nullptr || options->len == 0) {
    return AMDSMI_STATUS_NOT_SUPPORTED;
  }

  for (uint32_t i = 0; i < AMDSMI_MAX_CARVEOUT_OPTIONS; ++i) {
    info->options[i].index = i;
    info->options[i].description[0] = '\0';
  }

  const char* current = fwupd_bios_setting_get_current_value(setting);
  uint32_t current_index = AMDSMI_MAX_CARVEOUT_OPTIONS;  // sentinel: unknown

  guint count = options->len;
  if (count > AMDSMI_MAX_CARVEOUT_OPTIONS) {
    count = AMDSMI_MAX_CARVEOUT_OPTIONS;
  }
  for (guint i = 0; i < count; ++i) {
    const char* value = static_cast<const char*>(g_ptr_array_index(options, i));
    if (value == nullptr) {
      value = "";
    }
    info->options[i].index = i;
    std::strncpy(info->options[i].description, value, AMDSMI_MAX_STRING_LENGTH - 1);
    info->options[i].description[AMDSMI_MAX_STRING_LENGTH - 1] = '\0';
    if (current != nullptr && g_strcmp0(current, value) == 0) {
      current_index = i;
    }
  }
  info->num_options = count;

  // When the daemon does not expose the current value (reading it may require
  // privilege on some platforms), leave current_index == num_options ("unknown").
  info->current_index =
      (current_index == AMDSMI_MAX_CARVEOUT_OPTIONS) ? info->num_options : current_index;
  return AMDSMI_STATUS_SUCCESS;
}

amdsmi_status_t fwupd_set_carveout(uint32_t option_index) {
  g_autoptr(FwupdClient) client = fwupd_client_new();
  g_autoptr(GError) error = nullptr;
  g_autoptr(GPtrArray) settings = fwupd_client_get_bios_settings(client, nullptr, &error);
  if (settings == nullptr) {
    return AMDSMI_STATUS_NOT_SUPPORTED;
  }

  FwupdBiosSetting* setting = FindCarveout(settings);
  if (setting == nullptr) {
    return AMDSMI_STATUS_NOT_SUPPORTED;
  }
  if (fwupd_bios_setting_get_read_only(setting)) {
    return AMDSMI_STATUS_NO_PERM;
  }

  GPtrArray* options = fwupd_bios_setting_get_possible_values(setting);
  if (options == nullptr || option_index >= options->len) {
    return AMDSMI_STATUS_INVAL;
  }
  const char* name = fwupd_bios_setting_get_name(setting);
  const char* value = static_cast<const char*>(g_ptr_array_index(options, option_index));
  if (name == nullptr || value == nullptr) {
    return AMDSMI_STATUS_NOT_SUPPORTED;
  }

  // Setting the carveout to the value it already holds is a successful no-op.
  // Short-circuit before calling the daemon so the result stays stable across
  // fwupd versions, which otherwise surface the "nothing to change" case as an
  // error rather than success.
  const char* current = fwupd_bios_setting_get_current_value(setting);
  if (current != nullptr && g_strcmp0(current, value) == 0) {
    return AMDSMI_STATUS_SUCCESS;
  }

  if (DryRun()) {
    return AMDSMI_STATUS_SUCCESS;
  }

  g_autoptr(GHashTable) request = g_hash_table_new(g_str_hash, g_str_equal);
  g_hash_table_insert(request, const_cast<char*>(name), const_cast<char*>(value));

  g_autoptr(GError) set_error = nullptr;
  if (!fwupd_client_modify_bios_setting(client, request, nullptr, &set_error)) {
    if (set_error != nullptr) {
      // Different fwupd versions word the "value already matches" case as
      // "nothing to do" or "no BIOS settings needed to be changed".
      if (set_error->message != nullptr &&
          (std::strstr(set_error->message, "nothing to do") != nullptr ||
           std::strstr(set_error->message, "needed to be changed") != nullptr)) {
        return AMDSMI_STATUS_SUCCESS;
      }
      if (g_error_matches(set_error, FWUPD_ERROR, FWUPD_ERROR_PERMISSION_DENIED)) {
        return AMDSMI_STATUS_NO_PERM;
      }
      std::fprintf(stderr, "amd-smi: fwupd failed to set UMA carveout: %s\n", set_error->message);
    }
    return AMDSMI_STATUS_NOT_SUPPORTED;
  }
  return AMDSMI_STATUS_SUCCESS;
}

}  // namespace smi
}  // namespace amd

#else  // !AMDSMI_ENABLE_FWUPD

// Built without fwupd support: the carveout always falls back to "not supported"
// so the sysfs path remains the only interface.
namespace amd {
namespace smi {

amdsmi_status_t fwupd_get_carveout_info(amdsmi_uma_carveout_info_t* info) {
  (void)info;
  return AMDSMI_STATUS_NOT_SUPPORTED;
}

amdsmi_status_t fwupd_set_carveout(uint32_t option_index) {
  (void)option_index;
  return AMDSMI_STATUS_NOT_SUPPORTED;
}

}  // namespace smi
}  // namespace amd

#endif  // AMDSMI_ENABLE_FWUPD
