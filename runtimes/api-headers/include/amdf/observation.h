// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef AMDF_OBSERVATION_H_
#define AMDF_OBSERVATION_H_

#include "amdf/base.h"

/// First version exposing the shared profiling host clock.
#define AMDF_OBSERVATION_EXTENSION_VERSION_1 1u

/// Most recent observation extension described by this header.
#define AMDF_OBSERVATION_EXTENSION_VERSION_LATEST \
  AMDF_OBSERVATION_EXTENSION_VERSION_1

/// Representation of one observation source's timestamps.
///
/// The source and its capture lifetime identify the clock. Equal rates or a
/// shared physical device establish neither a shared epoch nor correlation
/// with a host clock. Rate describes units, not resolution or accuracy.
typedef struct amdf_observation_clock_info_t {
  /// Represented ticks per second, or zero when the rate is not established.
  uint64_t frequency_hz;
  /// Number of meaningful counter bits, or zero when unspecified.
  uint32_t counter_bit_count;
  /// Reserved; zero.
  uint32_t reserved;
} amdf_observation_clock_info_t;

/// One native device/host correlation sample, without retained calibration.
/// All primary host values use the observation extension's profiling clock.
/// The native device source is identified by the operation returning this
/// record; frequency equality alone does not identify a queue timestamp epoch.
typedef struct amdf_observation_clock_sample_t {
  /// Native device clock units, sampled or queried with this observation.
  amdf_observation_clock_info_t clock;
  /// Device counter value in clock's units.
  uint64_t timestamp;
  /// Host observations surrounding and returned by the native sampling call.
  struct {
    /// Host read before entering the native sampling call.
    uint64_t before;
    /// Host sample returned alongside the device value by the native service.
    uint64_t timestamp;
    /// Host read after the native sampling call returns.
    uint64_t after;
  } host;
  /// Additional native system-clock observation, absent when frequency is zero.
  /// Linux KFD supplies CLOCK_BOOTTIME nanoseconds. Windows KMT has no second
  /// host sample. These values are auxiliary; ordinary CPU/device correlation
  /// uses host, which is directly readable through the observation extension.
  struct {
    /// Additional native clock value, or zero when absent.
    uint64_t timestamp;
    /// Units per second for timestamp, or zero when absent.
    uint64_t frequency_hz;
  } system;
} amdf_observation_clock_sample_t;

/// Discontinuities accompanying an otherwise successful observation read.
typedef uint32_t amdf_observation_read_flags_t;
enum amdf_observation_read_flag_bits_e {
  /// Bytes were lost before this batch. Interval pairing crosses no such gap.
  AMDF_OBSERVATION_READ_FLAG_GAP = 1u << 0,
  /// Native progress restarted. Previous clock and cursor continuity is gone.
  AMDF_OBSERVATION_READ_FLAG_RESTART = 1u << 1,
  /// A copied batch failed its publication/integrity checks and was discarded.
  AMDF_OBSERVATION_READ_FLAG_DISCARDED = 1u << 2,
};

/// One read's progress and borrowed output in caller-owned host storage.
///
/// Positions and lost-byte counts use the source's native transport units;
/// decoded output bytes need not have the native record size. The caller
/// advances its own position only after accepting the result. A successful
/// empty read is a snapshot of availability, not producer retirement.
typedef struct amdf_observation_read_result_t {
  /// Position for the next read of this same capture.
  uint64_t next_position;
  /// Offset of valid output from the beginning of supplied storage.
  uint64_t byte_offset;
  /// Valid output length; zero for an empty or discarded batch.
  uint64_t byte_length;
  /// Lost native bytes, or UINT64_MAX when a restart prevents quantification.
  uint64_t lost_byte_length;
  /// Discontinuities that the consumer processes before this batch's records.
  amdf_observation_read_flags_t flags;
  /// Reserved; zero.
  uint32_t reserved;
} amdf_observation_read_result_t;

/// Portable host observations shared by CPU, GPU and XDNA profiling.
///
/// The profiling host clock is monotonic, system-wide within one boot, and
/// independent of wall-clock adjustments. Every native device clock sample's
/// primary host timestamps use this clock. Its units are queried explicitly;
/// callers subtract timestamps before scaling to avoid absolute-time rounding.
/// Capture relations are refreshed after suspend, reset or source loss.
///
/// These services need no instance or device. Negotiating the table performs
/// no initialization. Calls retain no state and allocate no storage; each is
/// thread-safe and may read the native OS clock without waiting for device
/// work.
typedef struct amdf_observation_api_t {
  /// Size in bytes of this table version.
  uint32_t structure_size;
  /// Observation extension version implemented by this table.
  uint32_t extension_version;
  /// Queries the host clock's nonzero, boot-stable frequency and counter width.
  /// Failure leaves out_info unchanged. This clock is a profiling timebase,
  /// distinct from wall time and from any application-selected host clock.
  amdf_status_t(AMDF_CALL* query_host_clock_info)(
      amdf_observation_clock_info_t* out_info);
  /// Reads the host clock in the units from query_host_clock_info.
  /// Failure leaves out_timestamp unchanged. Reads add no synchronization to
  /// application data or device work; the caller supplies those dependencies.
  amdf_status_t(AMDF_CALL* query_host_clock)(uint64_t* out_timestamp);
} amdf_observation_api_t;

#endif  // AMDF_OBSERVATION_H_
