// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file log_buffer_test_access.h
/// @brief Test-only access to LogBuffer's private factory and raw image.

#pragma once

#include "rocjitsu/code/patch/log_abi.h"
#include "rocjitsu/code/patch/log_buffer.h"

#include <cstdint>
#include <memory>
#include <span>
#include <string>

namespace rocjitsu {

// LogBuffer keeps its factory and raw header/record pointers private because no
// production consumer defines their contract yet; the tests reach those seams
// through this friend shim rather than through a public API that would freeze
// prematurely. Must live in namespace rocjitsu (not an anonymous namespace) so it
// names the same type as the friend declaration in LogBuffer, and in this one
// header so every test translation unit sees the same definition.
struct LogBufferTestAccess {
  static std::unique_ptr<LogBuffer> create(uint32_t slot_count, std::string *error_out = nullptr) {
    return LogBuffer::create(slot_count, error_out);
  }
  static RjLogBufferHeader *header(LogBuffer &buf) { return buf.header(); }
  static RjLogRecord *records(LogBuffer &buf) { return buf.records(); }
  // The whole image, header then records, as a device producer addresses it once
  // the image is copied into its memory.
  static std::span<uint8_t> bytes(LogBuffer &buf) {
    return {reinterpret_cast<uint8_t *>(buf.header()), buf.total_bytes()};
  }
};

} // namespace rocjitsu
