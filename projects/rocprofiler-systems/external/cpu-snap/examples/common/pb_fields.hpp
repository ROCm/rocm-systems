#pragma once

/**
 * @file pb_fields.hpp
 * @brief Perfetto-proto field-number constants grouped into
 *        sub-namespaces per submessage.
 *
 * Lifted out of perfetto_proto_writer.hpp so that the writer body
 * only carries the trace-emit logic, not 50+ raw TP_* / TD_* / PS_*
 * constants in its class scope.
 *
 * @see perfetto_trace.proto (Perfetto v34+) for the source field numbers.
 */

#include <cstdint>

namespace cs_examples::perfetto::pb_fields
{

// NOLINTBEGIN(readability-identifier-naming): codebase uses k_*-prefixed
//   lower_case for global constants.

/** TracePacket field numbers. */
namespace trace_packet
{
inline constexpr std::uint32_t k_timestamp        = 8;
inline constexpr std::uint32_t k_track_descriptor = 60;
inline constexpr std::uint32_t k_track_event      = 11;
inline constexpr std::uint32_t k_trusted_seq_id   = 10;
}  // namespace trace_packet

/** TrackDescriptor field numbers. */
namespace track_descriptor
{
inline constexpr std::uint32_t k_uuid        = 1;
inline constexpr std::uint32_t k_name        = 2;
inline constexpr std::uint32_t k_process     = 3;
inline constexpr std::uint32_t k_thread      = 4;
inline constexpr std::uint32_t k_parent_uuid = 5;
}  // namespace track_descriptor

/** ProcessDescriptor field numbers. */
namespace process_descriptor
{
inline constexpr std::uint32_t k_pid  = 1;
inline constexpr std::uint32_t k_name = 6;
}  // namespace process_descriptor

/** ThreadDescriptor field numbers. */
namespace thread_descriptor
{
inline constexpr std::uint32_t k_pid  = 1;
inline constexpr std::uint32_t k_tid  = 2;
inline constexpr std::uint32_t k_name = 5;
}  // namespace thread_descriptor

/** TrackEvent field numbers + type-enum values. */
namespace track_event
{
inline constexpr std::uint32_t k_type             = 9;
inline constexpr std::uint32_t k_track_uuid       = 11;
inline constexpr std::uint32_t k_name             = 23;
inline constexpr std::uint32_t k_type_slice_begin = 1;
inline constexpr std::uint32_t k_type_slice_end   = 2;
}  // namespace track_event

/** Trace field numbers (top-level wrapper message). */
namespace trace
{
inline constexpr std::uint32_t k_packet = 1;
}  // namespace trace

// NOLINTEND(readability-identifier-naming)

}  // namespace cs_examples::perfetto::pb_fields
