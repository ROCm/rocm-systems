// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef IPC_PROTOCOL_H
#define IPC_PROTOCOL_H

#include "include/amd_cuid.h"
#include "src/cuid_device.h"
#include <cstdint>

#define AMDCUID_SOCKET_PATH "/var/run/amdcuid_daemon.sock"

enum class IpcMessageType : uint8_t {
    ADD_DEVICE = 1,
    REFRESH_DEVICES = 2
};

struct IpcRequest
{
    IpcMessageType type;
    char* device_path; // used for ADD_DEVICE, empty otherwise
    amdcuid_device_type_t device_type; // used for ADD_DEVICE, AMDCUID_DEVICE_TYPE_NONE otherwise
};

struct IpcResponse
{
    amdcuid_status_t status;
    amdcuid_id_t device_handle; // used for ADD_DEVICE, 0 otherwise
};

#endif // IPC_PROTOCOL_H