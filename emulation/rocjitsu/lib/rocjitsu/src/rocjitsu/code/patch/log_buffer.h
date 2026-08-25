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
///
/// TODO: the diagnostic counters below (dropped_overrun, invalid_slots,
/// incompatible_records) are returned but not yet acted on -- the drain caller
/// that wires up post-dispatch logging does not exist. When it lands, log these
/// via Logger when nonzero so a misbehaving producer is visible instead of
/// silently discarded.
struct DrainStats {
  uint64_t records_drained = 0;      ///< Valid records passed to the callback.
  uint64_t invalid_slots = 0;        ///< Slots advertised by write_ptr but whose
                                     ///< valid flag was 0 (a post-completion
                                     ///< protocol error, not an expected race).
  uint64_t overflow_count = 0;       ///< Snapshot of header.overflow_count.
  uint64_t dropped_overrun = 0;      ///< Debug signal: nonzero only when a
                                     ///< producer broke the drop-newest contract
                                     ///< and advanced write_ptr past
                                     ///< read_ptr + slot_count. Counts the
                                     ///< clamped excess; not a recovery path
                                     ///< (see drain()).
  uint64_t incompatible_records = 0; ///< Valid records skipped because their
                                     ///< abi_version/record_size did not match
                                     ///< the compiled ABI (see drain()).
};

/// @brief Owns and drains a single logging ring buffer in host memory.
///
/// Ownership/lifetime: the buffer must outlive the dispatch it serves and must
/// remain alive until drain() has consumed its records. Currently supports one
/// controlled patched dispatch per buffer at a time; call reinitialize() before
/// reusing the same instance for another dispatch.
class LogBuffer final {
public:
  ~LogBuffer();

  LogBuffer(const LogBuffer &) = delete;
  LogBuffer &operator=(const LogBuffer &) = delete;

  /// @brief Number of record slots (power of two).
  [[nodiscard]] uint32_t slot_count() const;

  /// @brief Current overflow counter value from the header.
  [[nodiscard]] uint64_t overflow_count() const;

  /// @brief Total byte size of the owned allocation (header + all slots).
  [[nodiscard]] size_t total_bytes() const;

  /// @brief Validate the header's self-describing fields before trusting it.
  ///
  /// Checks magic, abi_version, header_size, record_size, and slot_count against
  /// the compiled-in ABI. Required precondition of drain() for any buffer whose
  /// header was written by another agent (a device or a shared allocation):
  /// drain() strides by a fixed sizeof(RjLogRecord) and cannot itself detect a
  /// header describing a different record geometry. Returns false and sets
  /// @p error_out on the first mismatch.
  [[nodiscard]] bool validate(std::string *error_out = nullptr) const;

  /// @brief Drain all records in [read_ptr, write_ptr) after completion.
  ///
  /// For each advertised index the live slot is `index & (slot_count-1)`. A
  /// slot with `valid == 1` is passed to @p callback, its `valid` flag is
  /// cleared, and records_drained is incremented; a slot with `valid == 0` is
  /// counted in invalid_slots and skipped. On return, read_ptr is advanced to
  /// write_ptr.
  ///
  /// A valid record whose per-record abi_version or record_size disagrees with
  /// the compiled ABI is cleared, counted in incompatible_records, and not
  /// delivered to @p callback. This is defense-in-depth against a single
  /// corrupted record, not the primary layout guard: drain() strides by a fixed
  /// sizeof(RjLogRecord), so a wholesale record_size mismatch desyncs the stride
  /// itself and must be caught by validate() on the header (see validate()).
  ///
  /// The producer contract is drop-newest: a full ring drops the incoming record
  /// and bumps overflow_count without advancing write_ptr, so a well-behaved
  /// producer keeps (write_ptr - read_ptr) <= slot_count. Because the counters
  /// come from an untrusted producer, drain still clamps the walk to slot_count
  /// in all builds: a contract-violating producer that overran the ring would
  /// otherwise alias physical slots and deliver the same record more than once.
  /// The clamp only prevents that double-delivery -- it does not recover the
  /// dropped records or define their order. The clamped excess is surfaced in
  /// DrainStats::dropped_overrun as a debug signal that the contract was broken.
  ///
  /// @note For now this is single-threaded and intended to run only after the
  ///       producing dispatch has completed. The invalid_slots hole-skip
  ///       (advancing read_ptr past a valid==0 slot) is correct ONLY for this
  ///       post-completion model; it would permanently drop a not-yet-published
  ///       slot once producers publish `valid` concurrently, so this must be
  ///       revisited before the host drains while a kernel is still running.
  DrainStats drain(const LogRecordCallback &callback);

  /// @brief Reset the buffer for reuse: zero the counters and all record
  ///        `valid` flags while leaving the immutable header control fields
  ///        intact.
  void reinitialize();

private:
  // Construction and raw header/record access are test-only seams: no production
  // consumer defines their contract yet, so they stay private and are reached in
  // tests through the LogBufferTestAccess friend. A real drain caller uses the
  // public validate()/drain()/reinitialize() API and never touches raw pointers.
  friend struct LogBufferTestAccess;

  /// @brief Allocate a host-memory logging buffer sized for @p slot_count
  ///        records.
  /// @param slot_count  Number of record slots; must be a nonzero power of two.
  /// @param error_out   Optional; assigned a single diagnostic on failure.
  /// @returns The buffer, or nullptr if @p slot_count is zero or not a power of
  ///          two, or if allocation fails.
  [[nodiscard]] static std::unique_ptr<LogBuffer> create(uint32_t slot_count,
                                                         std::string *error_out = nullptr);

  /// @brief Pointer to the control header (start of the allocation).
  [[nodiscard]] RjLogBufferHeader *header();
  [[nodiscard]] const RjLogBufferHeader *header() const;

  /// @brief Pointer to the first record slot (immediately after the header).
  [[nodiscard]] RjLogRecord *records();
  [[nodiscard]] const RjLogRecord *records() const;

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
