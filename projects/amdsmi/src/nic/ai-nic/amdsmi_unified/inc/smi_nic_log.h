// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef AMDSMI_UNIFIED_NIC_LOG_H_
#define AMDSMI_UNIFIED_NIC_LOG_H_

#include <string>

namespace amd::smi::nic::log {

/**
 * Debug-log seam for the NIC library. amdsminic cannot link the project logger
 * (it lives in libamd_smi), so the owner installs a sink at init and every read
 * chokepoint reports through it. With no sink installed the calls are no-ops.
 */
using Sink_t = void (*)(const std::string& msg);

// nullptr uninstalls. Install once at init; not meant to be swapped while reads run.
auto set_sink(Sink_t sink) -> void;

// Call sites check this before building a message so a disabled log costs nothing.
auto is_enabled() -> bool;

/**
 * Writes "<func> | <msg>", the layout the project logger uses. Prefer NIC_LOG_DEBUG,
 * which supplies the calling function's name.
 */
auto debug(const char* func, const std::string& msg) -> void;

/**
 * Keeps the last 4 characters ("****006E"); anything shorter is fully masked.
 * For serial numbers and MAC addresses, which must not reach the log file.
 */
auto mask_tail(const std::string& value) -> std::string;

}  // namespace amd::smi::nic::log

#define NIC_LOG_DEBUG(msg) amd::smi::nic::log::debug(__PRETTY_FUNCTION__, (msg))

#endif  // AMDSMI_UNIFIED_NIC_LOG_H_
