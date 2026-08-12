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

#include <cstdint>
#include <cstring>
#include <string>

#include "include/amd_cuid.h"
#include "src/cuid_device.h"

#define AMDCUID_SOCKET_PATH "/var/run/amdcuid_daemon.sock"

/// Maximum device path length carried in an IpcRequest, including the
/// terminating NUL. The path must travel inside the fixed-size message body:
/// the two peers are separate processes, so a pointer field would be
/// meaningless (and dangerous) on the receiving side.
constexpr size_t AMDCUID_IPC_PATH_MAX = 4096;

enum class IpcMessageType : uint8_t { ADD_DEVICE = 1, REFRESH_DEVICES = 2 };

struct IpcRequest {
  IpcMessageType type;
  amdcuid_device_type_t device_type;  // used for ADD_DEVICE, AMDCUID_DEVICE_TYPE_NONE otherwise
  char device_path[AMDCUID_IPC_PATH_MAX];  // used for ADD_DEVICE, empty otherwise
};

/// Copy @p src into the fixed-size device_path field, always NUL-terminating.
/// Returns false when @p src does not fit, so callers reject the request
/// instead of silently operating on a truncated path.
inline bool ipc_set_device_path(IpcRequest& request, const char* src) {
  if (src == nullptr) {
    return false;
  }
  const size_t len = ::strnlen(src, AMDCUID_IPC_PATH_MAX);
  if (len >= AMDCUID_IPC_PATH_MAX) {
    return false;  // no room for the terminating NUL
  }
  std::memcpy(request.device_path, src, len);
  request.device_path[len] = '\0';
  return true;
}

/// Read the device_path field of a request received off the wire. A malicious
/// or buggy peer may send an unterminated buffer, so the length is bounded by
/// the field size rather than trusting a NUL to be present.
inline std::string ipc_get_device_path(const IpcRequest& request) {
  return std::string(request.device_path, ::strnlen(request.device_path, AMDCUID_IPC_PATH_MAX));
}

struct IpcResponse {
  amdcuid_status_t status;
  amdcuid_id_t device_handle;  // used for ADD_DEVICE, 0 otherwise
};

#endif  // IPC_PROTOCOL_H
