/* Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 */
#pragma once

#include "context.h"
#include "hip.h"
#include "sys.h"

#include <syslog.h>

namespace hipFile {

// Makes the given GPU current for the lifetime of the guard (device memory and signal operations act on
// the current device's context) and restores the caller's device on scope exit, including during
// exceptions.
class DeviceGuard {
public:
    explicit DeviceGuard(int device) : prev_device_{Context<Hip>::get()->hipGetDevice()}
    {
        if (device != prev_device_) {
            Context<Hip>::get()->hipSetDevice(device);
            switched_ = true;
        }
    }
    ~DeviceGuard()
    {
        if (switched_) {
            try {
                Context<Hip>::get()->hipSetDevice(prev_device_);
            }
            catch (...) {
                Context<Sys>::get()->syslog(LOG_CRIT, "Unable to restore the caller's HIP device.");
            }
        }
    }
    DeviceGuard(const DeviceGuard &)            = delete;
    DeviceGuard &operator=(const DeviceGuard &) = delete;
    DeviceGuard(DeviceGuard &&)                 = delete;
    DeviceGuard &operator=(DeviceGuard &&)      = delete;

private:
    int  prev_device_;
    bool switched_{false};
};

}
