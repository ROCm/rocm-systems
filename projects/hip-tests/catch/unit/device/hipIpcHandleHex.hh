/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#ifndef CATCH_UNIT_DEVICE_HIPIPCHANDLEHEX_HH_
#define CATCH_UNIT_DEVICE_HIPIPCHANDLEHEX_HH_

#include <iomanip>
#include <sstream>
#include <string>

/**
 * An IPC handle is an opaque blob containing zero bytes, so it cannot be handed
 * to another process through argv or the environment as-is: both are arrays of
 * NUL-terminated strings and would truncate it. Hex text survives them
 * unchanged.
 *
 * Works for any handle whose payload is the `reserved` byte array, i.e.
 * hipIpcMemHandle_t and hipIpcEventHandle_t.
 */
template <typename IpcHandle> inline std::string ipcHandleToHex(const IpcHandle& handle) {
  std::ostringstream oss;
  oss << std::hex << std::setfill('0');
  for (size_t i = 0; i < sizeof(handle.reserved); i++) {
    oss << std::setw(2) << static_cast<unsigned>(static_cast<unsigned char>(handle.reserved[i]));
  }
  return oss.str();
}

template <typename IpcHandle> inline IpcHandle hexToIpcHandle(const std::string& hex_str) {
  IpcHandle handle = {};
  std::stringstream hex_ss(hex_str);
  for (size_t i = 0; i < sizeof(handle.reserved); i++) {
    std::string byte_str(2, '\0');
    if (!hex_ss.read(&byte_str[0], 2)) break;
    unsigned int byte_val = 0;
    std::stringstream byte_ss;
    byte_ss << std::hex << byte_str;
    byte_ss >> byte_val;
    handle.reserved[i] = static_cast<char>(static_cast<unsigned char>(byte_val));
  }
  return handle;
}

#endif  // CATCH_UNIT_DEVICE_HIPIPCHANDLEHEX_HH_
