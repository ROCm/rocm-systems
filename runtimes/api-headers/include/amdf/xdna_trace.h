// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef AMDF_XDNA_TRACE_H_
#define AMDF_XDNA_TRACE_H_

#include "amdf/base.h"
#include "amdf/observation.h"

/// An `amdf_xdna_firmware_trace_info_t` output structure.
#define AMDF_STRUCTURE_TYPE_XDNA_FIRMWARE_TRACE_INFO \
  ((amdf_structure_type_t)0x0001000Eu)

/// An `amdf_xdna_context_firmware_trace_info_t` output structure.
#define AMDF_STRUCTURE_TYPE_XDNA_CONTEXT_FIRMWARE_TRACE_INFO \
  ((amdf_structure_type_t)0x00010010u)

/// The native snapshot has no firmware slot assigned to this context.
#define AMDF_XDNA_FIRMWARE_CONTEXT_ID_UNASSIGNED UINT32_MAX

/// A context's instantaneous association with the firmware trace namespace.
/// The driver can recreate its firmware context during suspension or recovery
/// while the public context remains live. A snapshot supplies no historical
/// validity interval, incarnation, virtual-context ID, or submission identity.
typedef struct amdf_xdna_context_firmware_trace_info_t {
  /// Must be AMDF_STRUCTURE_TYPE_XDNA_CONTEXT_FIRMWARE_TRACE_INFO.
  amdf_structure_type_t type;
  /// Must be at least sizeof(amdf_xdna_context_firmware_trace_info_t).
  uint32_t structure_size;
  /// Optional output extension chain. No extensions are defined.
  void* next;
  /// Current physical firmware context ID, or
  /// AMDF_XDNA_FIRMWARE_CONTEXT_ID_UNASSIGNED when not assigned.
  uint32_t firmware_context_id;
  /// Reserved; returned as zero.
  uint32_t reserved;
} amdf_xdna_context_firmware_trace_info_t;

/// Categories selected in the device-wide firmware recorder.
typedef uint32_t amdf_xdna_firmware_trace_categories_t;
enum amdf_xdna_firmware_trace_category_bits_e {
  AMDF_XDNA_FIRMWARE_TRACE_CATEGORY_SCHEDULING = 1u << 0,
  AMDF_XDNA_FIRMWARE_TRACE_CATEGORY_MAILBOX = 1u << 1,
  AMDF_XDNA_FIRMWARE_TRACE_CATEGORY_CLOCK_POWER = 1u << 2,
  AMDF_XDNA_FIRMWARE_TRACE_CATEGORY_SUSPEND_RESUME = 1u << 3,
  AMDF_XDNA_FIRMWARE_TRACE_CATEGORY_ERRORS = 1u << 4,
  AMDF_XDNA_FIRMWARE_TRACE_CATEGORY_PDI_LOAD = 1u << 5,
  AMDF_XDNA_FIRMWARE_TRACE_CATEGORY_FAST_POWER = 1u << 6,
  AMDF_XDNA_FIRMWARE_TRACE_CATEGORY_L2 = 1u << 7,
  AMDF_XDNA_FIRMWARE_TRACE_CATEGORY_PREEMPTION = 1u << 8,
};

/// Native Strix schema with 32-byte framed records and nanosecond timestamps.
#define AMDF_XDNA_FIRMWARE_TRACE_FORMAT_2_0 UINT32_C(0x00000200)

/// Native recorder state and caller-owned read-storage requirements.
/// Query success establishes service availability, not permission to configure
/// or read it. Each operation checks its own native admission. The recorder is
/// device-wide and may already belong to another process.
typedef struct amdf_xdna_firmware_trace_info_t {
  /// Must be AMDF_STRUCTURE_TYPE_XDNA_FIRMWARE_TRACE_INFO.
  amdf_structure_type_t type;
  /// Must be at least sizeof(amdf_xdna_firmware_trace_info_t).
  uint32_t structure_size;
  /// Optional output extension chain. No extensions are defined.
  void* next;
  /// One when the native recorder is enabled, otherwise zero.
  uint32_t enabled;
  /// Enabled category mask; zero when disabled.
  amdf_xdna_firmware_trace_categories_t categories;
  /// Native payload schema identifier; zero when disabled.
  uint32_t format_version;
  /// Required power-of-two alignment of caller read storage.
  uint32_t buffer_alignment;
  /// Smallest accepted storage, including native request overhead.
  uint32_t minimum_buffer_byte_length;
  /// Storage for the native provider's ordinary full-window read.
  uint32_t recommended_buffer_byte_length;
  /// Timestamp representation for the active schema; zero when unknown.
  amdf_observation_clock_info_t clock;
} amdf_xdna_firmware_trace_info_t;

/// One decoded event. Unknown event identifiers and their arguments survive
/// unchanged, so schema consumers can retain events they do not interpret.
/// Command opcodes identify firmware operations, not caller submissions.
/// Context fields belong to the firmware's namespace, independently of the
/// opaque amdf_xdna_context_info_t.id. context_query_firmware_trace_info
/// samples their current association; attributing historical events requires
/// the association valid during those events. Recorder ownership does not
/// exclude other applications' work from the stream.
typedef struct amdf_xdna_firmware_trace_event_t {
  /// Source timestamp in the units reported by the active clock description.
  uint64_t timestamp;
  /// Native event arguments in the low 48 bits; upper bits are zero.
  uint64_t arguments;
  /// Native schema event identifier.
  uint32_t event_id;
  /// Native entry sequence, wrapping from 65535 to 1 in format 2.0.
  uint32_t sequence;
} amdf_xdna_firmware_trace_event_t;

/// Format-2 application message start event.
#define AMDF_XDNA_FIRMWARE_TRACE_EVENT_COMMAND_BEGIN 0u
/// Format-2 application message completion event.
#define AMDF_XDNA_FIRMWARE_TRACE_EVENT_COMMAND_END 1u
/// Format-2 native context creation event.
#define AMDF_XDNA_FIRMWARE_TRACE_EVENT_CONTEXT_CREATE 2u
/// Format-2 native context deletion event.
#define AMDF_XDNA_FIRMWARE_TRACE_EVENT_CONTEXT_DELETE 3u

/// Extracts the physical context from a COMMAND_BEGIN or COMMAND_END event.
static inline uint32_t amdf_xdna_firmware_trace_command_context(
    const amdf_xdna_firmware_trace_event_t* event) {
  return (uint32_t)(event->arguments & 15u);
}

/// Extracts the virtual context from a COMMAND_BEGIN or COMMAND_END event.
static inline uint32_t amdf_xdna_firmware_trace_command_virtual_context(
    const amdf_xdna_firmware_trace_event_t* event) {
  return (uint32_t)((event->arguments >> 20) & 255u);
}

/// Extracts the firmware opcode from a COMMAND_BEGIN or COMMAND_END event.
static inline uint32_t amdf_xdna_firmware_trace_command_opcode(
    const amdf_xdna_firmware_trace_event_t* event) {
  return (uint32_t)((event->arguments >> 4) & 65535u);
}

#endif  // AMDF_XDNA_TRACE_H_
