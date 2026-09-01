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

#ifndef AMD_SMI_FWUPD_CARVEOUT_H_
#define AMD_SMI_FWUPD_CARVEOUT_H_

#include "amd_smi/amdsmi.h"

namespace amd {
namespace smi {

// Read the UMA "carveout" BIOS setting exposed by the fwupd daemon over its
// D-Bus BIOS-settings interface. The setting is located by its AppStream id,
// preferring AMD's attribute (``com.amd-gpu.uma_carveout``) and falling back to
// HP's UEFI-HII naming (``com.hp-bioscfg.Dedicated_Graphics_Memory``).
//
// On success ``info`` is populated with the option list and, when the daemon
// reports it (reading the current value may require privilege on some
// platforms), the current index. When the current value is not exposed,
// ``current_index`` is set equal to ``num_options`` to signal "unknown".
//
// Returns AMDSMI_STATUS_NOT_SUPPORTED when the D-Bus client library, the fwupd
// daemon, or the setting is unavailable at runtime.
amdsmi_status_t fwupd_get_carveout_info(amdsmi_uma_carveout_info_t* info);

// Set the carveout to ``options[option_index]`` via the fwupd daemon.
// fwupd/PolicyKit enforces privilege; AMDSMI_STATUS_NO_PERM is returned when the
// request is denied. Honors AMDSMI_DRY_RUN=1 (skips the actual write).
amdsmi_status_t fwupd_set_carveout(uint32_t option_index);

}  // namespace smi
}  // namespace amd

#endif  // AMD_SMI_FWUPD_CARVEOUT_H_
