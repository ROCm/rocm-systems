// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef AMDF_NATIVE_EVENT_H_
#define AMDF_NATIVE_EVENT_H_

#include "amdf/base.h"

/// Native wake destination supplied by the caller.
typedef uint32_t amdf_native_event_type_t;
enum amdf_native_event_type_e {
  /// No destination; not accepted by notification requests.
  AMDF_NATIVE_EVENT_TYPE_NONE = 0,
  /// Linux eventfd opened with EFD_NONBLOCK. A notification adds a counter
  /// credit; the caller drains readiness through its native event loop.
  AMDF_NATIVE_EVENT_TYPE_EVENTFD = 1,
  /// Windows event HANDLE with EVENT_MODIFY_STATE access. The caller chooses
  /// auto-reset or manual-reset semantics and owns wait/reset handling.
  AMDF_NATIVE_EVENT_TYPE_WIN32_EVENT = 2,
};

/// Native event representations supported by an activated resource.
typedef uint64_t amdf_native_event_types_t;
enum amdf_native_event_type_bits_e {
  /// Supports a nonblocking Linux eventfd.
  AMDF_NATIVE_EVENT_TYPE_BIT_EVENTFD = UINT64_C(1)
                                       << AMDF_NATIVE_EVENT_TYPE_EVENTFD,
  /// Supports a Windows event HANDLE.
  AMDF_NATIVE_EVENT_TYPE_BIT_WIN32_EVENT =
      UINT64_C(1) << AMDF_NATIVE_EVENT_TYPE_WIN32_EVENT,
};

/// Borrowed native payload; no library-owned event is constructed or retained.
typedef union amdf_native_event_payload_t {
  /// Nonnegative eventfd descriptor representable by a native int.
  int64_t file_descriptor;
  /// Non-null Windows event HANDLE, not a fence or an IOCP handle.
  void* native_handle;
} amdf_native_event_payload_t;

/// Caller-owned wake destination, independent of completion storage.
///
/// The payload must name a live object of the declared native type. Copying
/// this value does not duplicate ownership. A requesting call borrows the
/// descriptor only for the call, but the caller keeps the native event live
/// until the outstanding native notifications have been consumed. Native
/// registration may retain an OS reference; libamdf retains neither this
/// descriptor nor a subscription object. Event-loop callback cancellation
/// does not cancel those native requests or accepted device work.
typedef struct amdf_native_event_t {
  /// Native representation of the borrowed payload.
  amdf_native_event_type_t type;
  /// Reserved for future use and must be zero.
  uint32_t reserved;
  /// Caller-owned native wake destination.
  amdf_native_event_payload_t payload;
} amdf_native_event_t;

#endif  // AMDF_NATIVE_EVENT_H_
