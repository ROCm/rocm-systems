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