// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file log_buffer.h
/// @brief Host-side logging ring buffer over the logging ABI (log_abi.h).
///
/// This is CPU-only logging: it owns a single contiguous allocation
/// ([header][slot_count * record]), initializes the header, and drains written
/// records after a dispatch has completed. It has no dependency on HSA,
/// the probe machinery, or GPU hardware.

#pragma once

#include "rocjitsu/code/patch/log_abi.h"

#include <cstdint>
#include <functional>
#include <memory>
#include <string>

namespace rocjitsu {

/// @brief Callback invoked once per drained (valid) record.
using LogRecordCallback = std::function<void(const RjLogRecord &)>;

/// @brief Result of a single drain() pass.
struct DrainStats {
  uint64_t records_drained = 0; ///< Valid records passed to the callback.
  uint64_t invalid_slots = 0;   ///< Slots advertised by write_ptr but whose
                                ///< valid flag was 0 (a post-completion
                                ///< protocol error, not an expected race).
  uint64_t overflow_count = 0;  ///< Snapshot of header.overflow_count.
};

/// @brief Owns and drains a single logging ring buffer in host memory.
///
/// Ownership/lifetime: the buffer must outlive the dispatch it serves and must
/// remain alive until drain() has consumed its records. Currently supports one
/// controlled patched dispatch per buffer at a time; call reinitialize() before
/// reusing the same instance for another dispatch.
class LogBuffer final {
public:
  /// @brief Allocate a host-memory logging buffer sized for @p slot_count
  ///        records.
  /// @param slot_count  Number of record slots; must be a nonzero power of two.
  /// @param error_out   Optional; assigned a single diagnostic on failure.
  /// @returns The buffer, or nullptr if @p slot_count is zero or not a power of
  ///          two, or if allocation fails.
  [[nodiscard]] static std::unique_ptr<LogBuffer>
  create_for_host_tests(uint32_t slot_count, std::string *error_out = nullptr);

  ~LogBuffer();

  LogBuffer(const LogBuffer &) = delete;
  LogBuffer &operator=(const LogBuffer &) = delete;

  /// @brief Pointer to the control header (start of the allocation).
  [[nodiscard]] RjLogBufferHeader *header();
  [[nodiscard]] const RjLogBufferHeader *header() const;

  /// @brief Pointer to the first record slot (immediately after the header).
  [[nodiscard]] RjLogRecord *records();
  [[nodiscard]] const RjLogRecord *records() const;

  /// @brief Number of record slots (power of two).
  [[nodiscard]] uint32_t slot_count() const;

  /// @brief Current overflow counter value from the header.
  [[nodiscard]] uint64_t overflow_count() const;

  /// @brief Total byte size of the owned allocation (header + all slots).
  [[nodiscard]] size_t total_bytes() const;

  /// @brief Drain all records in [read_ptr, write_ptr) after completion.
  ///
  /// For each advertised index the live slot is `index & (slot_count-1)`. A
  /// slot with `valid == 1` is passed to @p callback, its `valid` flag is
  /// cleared, and records_drained is incremented; a slot with `valid == 0` is
  /// counted in invalid_slots and skipped. On return, read_ptr is advanced to
  /// write_ptr.
  ///
  /// @note For now this is single-threaded and intended to run only after the
  ///       producing dispatch has completed.
  DrainStats drain(const LogRecordCallback &callback);

  /// @brief Reset the buffer for reuse: zero the counters and all record
  ///        `valid` flags while leaving the immutable header control fields
  ///        intact.
  void reinitialize();

private:
  LogBuffer(void *storage, uint32_t slot_count, size_t total_bytes);

  /// @brief Custom deleter matching the aligned operator new used to allocate.
  struct AlignedStorageDeleter {
    void operator()(void *p) const noexcept;
  };

  std::unique_ptr<void, AlignedStorageDeleter> storage_;
  uint32_t slot_count_ = 0;
  size_t total_bytes_ = 0;
};

/// @brief Format a single record:
///        `rj-log record_type=<n> site=0x<hex> wave=0x<hex> lane=<n>
///        payload=0x<hex>`.
[[nodiscard]] std::string format_log_line(const RjLogRecord &record);

} // namespace rocjitsu
