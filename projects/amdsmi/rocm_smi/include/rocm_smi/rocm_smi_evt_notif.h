// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef ROCM_SMI_INCLUDE_ROCM_SMI_ROCM_SMI_EVT_NOTIF_H_
#define ROCM_SMI_INCLUDE_ROCM_SMI_ROCM_SMI_EVT_NOTIF_H_

#include <cstdint>
#include <string>

namespace amd::smi {

// Renders the payload of one KFD SMI event line (the text after the event number) as the
// message rsmi_event_notification_get() reports. The payload layouts are the KFD_EVENT_FMT_*
// macros in the kernel's include/uapi/linux/kfd_ioctl.h.
std::string FormatEvtNotifMessage(uint32_t event, const std::string& payload);

}  // namespace amd::smi

#endif  // ROCM_SMI_INCLUDE_ROCM_SMI_ROCM_SMI_EVT_NOTIF_H_
