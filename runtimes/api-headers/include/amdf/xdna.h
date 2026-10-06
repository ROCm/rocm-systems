// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef AMDF_XDNA_H_
#define AMDF_XDNA_H_

#include <stdint.h>

#include "amdf/amdf.h"

#ifdef __cplusplus
extern "C" {
#endif

/// The first supported XDNA extension version.
#define AMDF_XDNA_EXTENSION_VERSION_1 1u

/// The most recent XDNA extension version described by this header.
#define AMDF_XDNA_EXTENSION_VERSION_LATEST AMDF_XDNA_EXTENSION_VERSION_1

/// One program-independent schedulable XDNA context.
typedef struct amdf_xdna_context_t amdf_xdna_context_t;

/// An `amdf_xdna_endpoint_info_t` output structure.
#define AMDF_STRUCTURE_TYPE_XDNA_ENDPOINT_INFO \
  ((amdf_structure_type_t)0x00010001u)

/// An `amdf_xdna_device_create_info_t` input structure.
#define AMDF_STRUCTURE_TYPE_XDNA_DEVICE_CREATE_INFO \
  ((amdf_structure_type_t)0x00010002u)

/// An `amdf_xdna_device_info_t` output structure.
#define AMDF_STRUCTURE_TYPE_XDNA_DEVICE_INFO \
  ((amdf_structure_type_t)0x00010003u)

/// An `amdf_xdna_kernel_queue_create_info_t` input structure.
#define AMDF_STRUCTURE_TYPE_XDNA_KERNEL_QUEUE_CREATE_INFO \
  ((amdf_structure_type_t)0x00010008u)

/// An `amdf_xdna_kernel_queue_submission_info_t` input structure.
#define AMDF_STRUCTURE_TYPE_XDNA_KERNEL_QUEUE_SUBMISSION_INFO \
  ((amdf_structure_type_t)0x00010009u)

/// An `amdf_xdna_context_create_info_t` input structure.
#define AMDF_STRUCTURE_TYPE_XDNA_CONTEXT_CREATE_INFO \
  ((amdf_structure_type_t)0x0001000Au)

/// An `amdf_xdna_context_info_t` output structure.
#define AMDF_STRUCTURE_TYPE_XDNA_CONTEXT_INFO \
  ((amdf_structure_type_t)0x0001000Bu)

/// An `amdf_xdna_context_placement_info_t` output structure.
#define AMDF_STRUCTURE_TYPE_XDNA_CONTEXT_PLACEMENT_INFO \
  ((amdf_structure_type_t)0x0001000Du)

/// Lets the provider select the physical origin of an XDNA context.
#define AMDF_XDNA_PHYSICAL_COLUMN_ORIGIN_ANY UINT32_MAX

/// Capacity in bytes of a NUL-terminated canonical XDNA target identifier.
#define AMDF_XDNA_TARGET_ID_CAPACITY 64u

/// Compiler-visible AIE instruction and array architecture.
typedef uint32_t amdf_xdna_architecture_t;
enum amdf_xdna_architecture_e {
  /// The architecture is unavailable or not represented by this API version.
  AMDF_XDNA_ARCHITECTURE_UNKNOWN = 0,
  /// The AIE2 architecture implemented by NPU1-family XDNA devices.
  AMDF_XDNA_ARCHITECTURE_AIE2 = 1,
  /// The AIE2P architecture implemented by Strix-family XDNA devices.
  AMDF_XDNA_ARCHITECTURE_AIE2P = 2,
  /// The AIE4 architecture implemented by newer XDNA devices.
  AMDF_XDNA_ARCHITECTURE_AIE4 = 3,
};

/// Context scheduling modes admitted by an XDNA endpoint.
typedef uint32_t amdf_xdna_scheduling_modes_t;
enum amdf_xdna_scheduling_mode_bits_e {
  /// A context can receive exclusive ownership of its physical placement.
  AMDF_XDNA_SCHEDULING_MODE_EXCLUSIVE = 1u << 0,
  /// Contexts can execute concurrently on disjoint spatial placements.
  AMDF_XDNA_SCHEDULING_MODE_SPATIAL = 1u << 1,
  /// Contexts can time-share one physical placement.
  AMDF_XDNA_SCHEDULING_MODE_TIME_SLICED = 1u << 2,
};

/// Binding physical placement contracts supported by the native provider.
/// Zero means no fixed placement is available, not an unqueried value.
typedef uint32_t amdf_xdna_placement_modes_t;
enum amdf_xdna_placement_mode_bits_e {
  /// Every admitted context has fixed backing covering the endpoint's complete
  /// physical array, including when its logical column count is smaller. The
  /// only explicit origin accepted is `array.column_origin`. ANY is also
  /// accepted and provides the same fixed backing. Fixed geometry grants
  /// neither exclusive ownership nor uninterrupted residency. Reset may discard
  /// state.
  AMDF_XDNA_PLACEMENT_MODE_FIXED_FULL_ARRAY = 1u << 0,
};

/// Public target-native byte encodings accepted by an XDNA endpoint.
typedef uint32_t amdf_xdna_binary_format_t;
enum amdf_xdna_binary_format_e {
  /// No public encoding is available for this role.
  AMDF_XDNA_BINARY_FORMAT_UNKNOWN = 0,
  /// A device-generation-specific AIE transaction stream.
  AMDF_XDNA_BINARY_FORMAT_TRANSACTION = 2,
};

/// First public contract for AIE transaction-header version 0.1.
#define AMDF_XDNA_TRANSACTION_FORMAT_VERSION_0_1 1u

/// First XDNA kernel-published instruction-range format.
#define AMDF_XDNA_QUEUE_FORMAT_VERSION_1 1u

/// One exact public target-native byte encoding.
typedef struct amdf_xdna_binary_format_info_t {
  /// Transaction stream or `AMDF_XDNA_BINARY_FORMAT_UNKNOWN`.
  amdf_xdna_binary_format_t format;
  /// Version defining the complete byte encoding.
  uint32_t version;
} amdf_xdna_binary_format_info_t;

/// Passive target identity available without activating the device.
///
/// Geometry and execution capabilities belong to `amdf_xdna_device_info_t` and
/// are obtained during explicit device creation.
typedef struct amdf_xdna_endpoint_info_t {
  /// Must be `AMDF_STRUCTURE_TYPE_XDNA_ENDPOINT_INFO`.
  amdf_structure_type_t type;
  /// Must be at least `sizeof(amdf_xdna_endpoint_info_t)`.
  uint32_t structure_size;
  /// Optional output extension chain. No extensions are currently defined.
  void* next;
  /// Compiler-visible AIE instruction and array architecture.
  amdf_xdna_architecture_t architecture;
  /// Exact NUL-terminated compiler target identifier.
  char target_id[AMDF_XDNA_TARGET_ID_CAPACITY];
} amdf_xdna_endpoint_info_t;

/// Opaque identity of one live XDNA context.
///
/// The value is meaningful only while the context and its provider instance
/// remain live. It is intended for correlation and compatibility checks, not
/// persistence or native-handle recovery.
typedef struct amdf_xdna_context_id_t {
  /// Provider-defined identity words.
  uint64_t words[2];
} amdf_xdna_context_id_t;

/// Returns true when two XDNA context identities contain the same value.
static inline bool amdf_xdna_context_id_is_equal(
    const amdf_xdna_context_id_t* lhs, const amdf_xdna_context_id_t* rhs) {
  return lhs->words[0] == rhs->words[0] && lhs->words[1] == rhs->words[1];
}

/// Parameters used to materialize one XDNA ordinary-address-domain device.
///
/// Version 1 has no device-specific fields. The structure retains its extension
/// chain so later address-domain admission inputs do not require a new entry
/// point. Context placement, scheduling, and program bytes are not device
/// creation inputs.
typedef struct amdf_xdna_device_create_info_t {
  /// Must be `AMDF_STRUCTURE_TYPE_XDNA_DEVICE_CREATE_INFO`.
  amdf_structure_type_t type;
  /// Must be at least `sizeof(amdf_xdna_device_create_info_t)`.
  uint32_t structure_size;
  /// Optional input extension chain. No extensions are currently defined.
  const void* next;
} amdf_xdna_device_create_info_t;

/// Immutable identity and effective capabilities of one live XDNA device.
/// Native array geometry is captured during activation; queries do no work in
/// the driver. Capacity limits do not reserve resources or promise
/// availability.
typedef struct amdf_xdna_device_info_t {
  /// Must be `AMDF_STRUCTURE_TYPE_XDNA_DEVICE_INFO`.
  amdf_structure_type_t type;
  /// Must be at least `sizeof(amdf_xdna_device_info_t)`.
  uint32_t structure_size;
  /// Optional output extension chain. No extensions are currently defined.
  void* next;
  /// Opaque identity of this live materialized device.
  amdf_device_id_t id;
  /// Monotonic provider epoch invalidating state after a device reset.
  uint64_t reset_epoch;
  /// Effective binding placement contracts for new contexts on this device.
  /// Zero means explicit origins and placement queries are unsupported.
  /// ANY-origin execution does not require fixed placement support.
  amdf_xdna_placement_modes_t placement_modes;
  /// Physical array geometry and addressing.
  struct {
    /// Physical index of the first addressable array column.
    uint32_t column_origin;
    /// Number of contiguous addressable physical array columns.
    uint32_t column_count;
    /// Total physical rows, including shim, memory, and compute rows.
    uint32_t row_count;
    /// Device-address distance in bytes between adjacent columns.
    uint64_t column_stride;
  } array;
  /// Context admission through the activated native interface.
  struct {
    /// Exclusive, spatial, and time-sliced modes accepted by the device.
    /// Zero means context creation is unsupported.
    amdf_xdna_scheduling_modes_t scheduling_modes;
    /// Minimum logical column count requestable by one context.
    uint32_t minimum_column_count;
    /// Maximum logical column count requestable by one context.
    uint32_t maximum_column_count;
    /// Granularity of requestable logical column counts.
    uint32_t column_count_granularity;
  } context;
  /// Opaque target-native instruction submission contract.
  struct {
    /// Maximum bytes in one instruction range; zero without execution.
    uint64_t maximum_byte_length;
    /// Required power-of-two instruction address alignment in bytes.
    uint32_t address_alignment;
    /// Required multiple of the instruction byte length.
    uint32_t byte_length_granularity;
    /// Native encoding supplied by the caller, never parsed by libamdf.
    amdf_xdna_binary_format_info_t format;
  } instruction;
} amdf_xdna_device_info_t;

/// Parameters used to admit one program-independent XDNA context.
typedef struct amdf_xdna_context_create_info_t {
  /// Must be `AMDF_STRUCTURE_TYPE_XDNA_CONTEXT_CREATE_INFO`.
  amdf_structure_type_t type;
  /// Must be at least `sizeof(amdf_xdna_context_create_info_t)`.
  uint32_t structure_size;
  /// Optional input extension chain. No extensions are currently defined.
  const void* next;
  /// Number of logical array columns requested for the context.
  uint32_t logical_column_count;
  /// Exact physical backing origin, constrained by the device's placement
  /// modes, or `AMDF_XDNA_PHYSICAL_COLUMN_ORIGIN_ANY` to let the provider
  /// choose. An explicit origin requires a supported binding placement
  /// contract; it is never silently weakened into a preference. Unsupported
  /// requests fail with `AMDF_STATUS_CODE_UNSUPPORTED`.
  uint32_t physical_column_origin;
  /// Nonempty set of scheduling modes acceptable to the caller.
  amdf_xdna_scheduling_modes_t acceptable_scheduling_modes;
} amdf_xdna_context_create_info_t;

/// Identity and logical admission of one live XDNA context.
/// Physical backing is a separate capability-gated placement query.
typedef struct amdf_xdna_context_info_t {
  /// Must be `AMDF_STRUCTURE_TYPE_XDNA_CONTEXT_INFO`.
  amdf_structure_type_t type;
  /// Must be at least `sizeof(amdf_xdna_context_info_t)`.
  uint32_t structure_size;
  /// Optional output extension chain. No extensions are currently defined.
  void* next;
  /// Opaque identity of this live schedulable context.
  amdf_xdna_context_id_t id;
  /// Identity of the ordinary-address-domain device borrowed by this context.
  amdf_device_id_t device_id;
  /// Device reset epoch in which this context remains valid.
  uint64_t reset_epoch;
  /// Single scheduling mode selected from the acceptable input set.
  amdf_xdna_scheduling_modes_t scheduling_mode;
  /// Logical column count admitted for programs and commands.
  uint32_t logical_column_count;
  /// Number of physical rows visible within each admitted column.
  uint32_t row_count;
} amdf_xdna_context_info_t;

/// Complete fixed physical backing of one live XDNA context.
/// Geometry is immutable for the context's valid lifetime and is not a snapshot
/// of scheduler telemetry. Device reset invalidates the context's state; fixed
/// placement does not imply exclusive or uninterrupted execution.
typedef struct amdf_xdna_context_placement_info_t {
  /// Must be `AMDF_STRUCTURE_TYPE_XDNA_CONTEXT_PLACEMENT_INFO`.
  amdf_structure_type_t type;
  /// Must be at least `sizeof(amdf_xdna_context_placement_info_t)`.
  uint32_t structure_size;
  /// Optional output extension chain. No extensions are currently defined.
  void* next;
  /// Physical origin of the backing array partition.
  uint32_t column_origin;
  /// Physical width of the backing partition, including unused columns.
  uint32_t column_count;
} amdf_xdna_context_placement_info_t;

/// Parameters used to acquire one kernel-mediated XDNA queue.
typedef struct amdf_xdna_kernel_queue_create_info_t {
  /// Must be `AMDF_STRUCTURE_TYPE_XDNA_KERNEL_QUEUE_CREATE_INFO`.
  amdf_structure_type_t type;
  /// Must be at least `sizeof(amdf_xdna_kernel_queue_create_info_t)`.
  uint32_t structure_size;
  /// Optional input extension chain. No extensions are currently defined.
  const void* next;
  /// Endpoint-local XDNA family supporting kernel publication.
  uint32_t queue_family_ordinal;
  /// Reserved for compatible growth and must be zero.
  uint32_t reserved;
} amdf_xdna_kernel_queue_create_info_t;

/// One caller-owned target-native instruction range.
typedef struct amdf_xdna_kernel_command_t {
  /// Caller-owned private memory from the queue's exact context. The memory
  /// and its instruction bytes remain live and immutable through retirement.
  amdf_memory_t* memory;
  /// Consumer granting EXECUTE access and a firmware address.
  uint32_t access_ordinal;
  /// Reserved for compatible growth and must be zero.
  uint32_t reserved;
  /// Offset of opaque target-native instructions. The resulting firmware
  /// address satisfies the endpoint's instruction.address_alignment.
  uint64_t byte_offset;
  /// Nonzero instruction length satisfying the endpoint's instruction limits.
  uint64_t byte_length;
} amdf_xdna_kernel_command_t;

/// One bounded kernel-mediated XDNA submission.
typedef struct amdf_xdna_kernel_queue_submission_info_t {
  /// Must be `AMDF_STRUCTURE_TYPE_XDNA_KERNEL_QUEUE_SUBMISSION_INFO`.
  amdf_structure_type_t type;
  /// Must be at least `sizeof(amdf_xdna_kernel_queue_submission_info_t)`.
  uint32_t structure_size;
  /// Optional input extension chain. No extensions are currently defined.
  const void* next;
  /// Number of descriptors in `commands`, bounded by the queue's command limit.
  uint32_t command_count;
  /// Reserved for compatible growth and must be zero.
  uint32_t reserved;
  /// Borrowed descriptors consumed before return. Accepted instruction ranges
  /// remain borrowed through checked native retirement.
  const amdf_xdna_kernel_command_t* commands;
} amdf_xdna_kernel_queue_submission_info_t;

/// Immutable entry-point table for one negotiated XDNA extension version.
///
/// Tables grow only by appending fields. The table and every function pointer
/// covered by `structure_size` remain valid until the providing library is
/// unloaded. The per-method cost guarantees follow amdf_api_t and apply to
/// first use as well as subsequent calls.
typedef struct amdf_xdna_api_t {
  /// Size in bytes of this table version.
  uint32_t structure_size;
  /// XDNA extension version implemented by this table.
  uint32_t extension_version;

  /// Copies passive XDNA target identity for `endpoint`.
  ///
  /// The endpoint must have a recognized XDNA architecture. Passing another
  /// engine or an unrecognized XDNA candidate returns
  /// `AMDF_STATUS_CODE_UNSUPPORTED`. The operation is thread-safe and performs
  /// no system call, firmware transaction, allocation, retry, sleep, or device
  /// wait. The caller initializes `out_info` and its complete extension chain;
  /// no output is modified when validation or qualification fails.
  /// Identity is read without locking, lazy initialization or
  /// ownership-counter updates.
  amdf_status_t(AMDF_CALL* endpoint_query_info)(
      amdf_endpoint_t* endpoint, amdf_xdna_endpoint_info_t* out_info);

  /// Materializes one XDNA ordinary-address domain and allocation namespace.
  ///
  /// `endpoint` remains query-only and may create independent devices. The
  /// returned device borrows the endpoint, which must outlive it. No context
  /// placement, scheduling, executable, PDI, xclbin, transaction, or control
  /// bytes are accepted or parsed. Failure leaves `out_device` unchanged.
  /// This cold boundary may allocate and enter the driver. It establishes
  /// ordinary address and native ABI state instead of deferring them to a
  /// memory query or first submission.
  amdf_status_t(AMDF_CALL* device_create)(
      amdf_endpoint_t* endpoint,
      const amdf_xdna_device_create_info_t* create_info,
      amdf_device_t** out_device);

  /// Copies the identity, reset epoch and effective capabilities of `device`.
  ///
  /// The operation is thread-safe and performs no system call, allocation,
  /// device initialization, retry, sleep, or device wait. The caller
  /// initializes `out_info` and its complete extension chain. No output is
  /// modified when validation or engine compatibility fails.
  /// Reset observation may use atomic loads; this query takes no lock,
  /// initializes no state and updates no ownership counters.
  amdf_status_t(AMDF_CALL* device_query_info)(
      amdf_device_t* device, amdf_xdna_device_info_t* out_info);

  /// Acquires one kernel-mediated queue from an XDNA context.
  ///
  /// The returned queue borrows `context`, which must outlive it. Creation
  /// selects an advertised `XDNA + KERNEL` family and allocates all bounded
  /// submission bookkeeping before publication. It may configure and wait for
  /// provider-owned firmware bootstrap work. Failure leaves `out_queue`
  /// unchanged; any unfinished bootstrap ownership remains with the context.
  /// Native packet storage, completion resources and required bootstrap are
  /// ready before success. Their costs never move to first submission or wait.
  amdf_status_t(AMDF_CALL* kernel_queue_create)(
      amdf_xdna_context_t* context,
      const amdf_xdna_kernel_queue_create_info_t* create_info,
      amdf_kernel_queue_t** out_queue);

  /// Publishes opaque instruction ranges from caller-owned private memory.
  ///
  /// Each memory resource must come from the queue's exact context scope and
  /// grant EXECUTE access. Caller writes must be published before submission;
  /// libamdf neither reads, copies nor modifies instruction bytes. The native
  /// provider fills its preallocated transport packet with address and length.
  /// Submission performs no allocation, format parsing, lowering, relocation,
  /// binding resolution, native submission retry, sleep or host wait. Native
  /// retirement consumes the command result. The caller keeps instruction
  /// memory live until retirement; libamdf neither retains nor tracks it.
  /// Native rejection leaves `out_submission` unchanged.
  /// The caller also keeps memory reachable through opaque device addresses
  /// live; native command retirement does not prove that user-mode work
  /// scheduled by those commands has stopped accessing that memory.
  ///
  /// This hot path takes no library lock and performs no lazy initialization,
  /// mapping, pinning or indirect-buffer scan. It is thread-safe with other
  /// submissions and progress operations. Queue-slot contention returns BUSY
  /// rather than waiting. This is not a wait-free guarantee. Native publication
  /// may enter the driver and publish the queue-owned packet's cache lines, not
  /// the caller's instruction or data bytes.
  amdf_status_t(AMDF_CALL* kernel_queue_submit)(
      amdf_kernel_queue_t* queue,
      const amdf_xdna_kernel_queue_submission_info_t* submission_info,
      uint64_t* out_submission);

  /// Admits one program-independent schedulable context beneath `device`.
  ///
  /// The returned context borrows the ordinary-address-domain device, which
  /// must outlive it. Creation selects one scheduling mode and establishes the
  /// logical admission and native completion state before publication. Explicit
  /// physical placement requires a supported device placement mode and is
  /// binding on success. It consumes no program or invocation bytes. Failure
  /// leaves `out_context` unchanged and creates no caller cleanup obligation.
  /// Construction releases its unpublished state locally; a native cleanup
  /// failure is reported without transferring that state to `device`.
  /// This cold operation may allocate and enter the native driver. Mandatory
  /// queue bootstrap can occur during kernel_queue_create, but never during
  /// a metadata query or first submission.
  amdf_status_t(AMDF_CALL* context_create)(
      amdf_device_t* device, const amdf_xdna_context_create_info_t* create_info,
      amdf_xdna_context_t** out_context);

  /// Copies the immutable identity and logical admission of `context`.
  ///
  /// The operation is thread-safe and performs no system call, allocation,
  /// native initialization, retry, sleep, or device wait. The caller
  /// initializes `out_info` and its complete extension chain. No output is
  /// modified when validation fails.
  /// It takes no lock, performs no lazy initialization and updates no
  /// ownership counters.
  amdf_status_t(AMDF_CALL* context_query_info)(
      amdf_xdna_context_t* context, amdf_xdna_context_info_t* out_info);

  /// Copies the complete fixed physical backing of `context`.
  ///
  /// Returns `AMDF_STATUS_CODE_UNSUPPORTED` if the device has no supported
  /// placement mode. This applies even when context creation used ANY origin.
  /// The operation is thread-safe and performs no system call, allocation,
  /// initialization, retry or wait. The caller initializes `out_info`; every
  /// failure leaves it unchanged.
  /// It reads the established placement without locking, lazy initialization
  /// or ownership-counter updates; it does not re-query firmware placement.
  amdf_status_t(AMDF_CALL* context_query_placement_info)(
      amdf_xdna_context_t* context,
      amdf_xdna_context_placement_info_t* out_info);

  /// Enumerates borrowed private memory scopes of this live context.
  ///
  /// A private scope accepts only the context's exact device as its consumer.
  /// Its addresses are qualified by this context, not interchangeable with
  /// addresses from another context on the same device. The caller destroys
  /// all memory obtained from this scope before destroying the context; the
  /// library neither retains the context nor tracks its private allocations.
  /// Querying performs no allocation or native operation. `out_count` receives
  /// the total count on success and BUFFER_TOO_SMALL. A zero-capacity call may
  /// pass NULL for `scopes`.
  /// The scope is established by context_create. Enumeration takes no lock,
  /// initializes no state and updates no ownership counters.
  amdf_status_t(AMDF_CALL* context_enumerate_memory_scopes)(
      amdf_xdna_context_t* context, uint32_t capacity,
      amdf_memory_scope_t** scopes, uint32_t* out_count);

  /// Destroys one context after all context-local children are gone.
  ///
  /// Returns `AMDF_STATUS_CODE_BUSY` without native mutation while a queue
  /// remains live. Private-memory lifetime is a caller precondition. A native
  /// teardown failure leaves the context live so destruction can be retried.
  /// The caller must otherwise have exclusive access.
  amdf_status_t(AMDF_CALL* context_destroy)(amdf_xdna_context_t* context);
} amdf_xdna_api_t;

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // AMDF_XDNA_H_
