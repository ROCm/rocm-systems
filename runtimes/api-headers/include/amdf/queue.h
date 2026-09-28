// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef AMDF_QUEUE_H_
#define AMDF_QUEUE_H_

#include "amdf/base.h"
#include "amdf/memory.h"

#ifdef __cplusplus
extern "C" {
#endif

/// Native command representation accepted by a queue family.
typedef uint32_t amdf_queue_command_type_t;
enum amdf_queue_command_type_e {
  /// No command representation. Advertised families never use this value.
  AMDF_QUEUE_COMMAND_TYPE_UNKNOWN = 0,
  /// Native AMD GPU PM4 command streams.
  AMDF_QUEUE_COMMAND_TYPE_GPU_PM4 = 1,
  /// Native AMD GPU SDMA command streams.
  AMDF_QUEUE_COMMAND_TYPE_GPU_SDMA = 2,
  /// Native AMD GPU AQL packets reaching hardware without CPU translation.
  AMDF_QUEUE_COMMAND_TYPE_GPU_AQL = 3,
  /// Native AMD XDNA execution commands.
  AMDF_QUEUE_COMMAND_TYPE_XDNA = 4,
  /// Gfx1250 AQL metadata dispatch and barrier packets.
  AMDF_QUEUE_COMMAND_TYPE_GPU_AQL_METADATA = 5,
};

/// Encoding features within a command format and version.
///
/// Bit meanings belong to the command type and are documented by its extension.
/// These describe native packet layouts, not additional publication services.
/// A zero value reports the baseline encoding; it never means unqueried state.
typedef uint64_t amdf_queue_format_features_t;

/// Queue publication mechanisms implemented by a provider.
typedef uint32_t amdf_queue_publication_modes_t;
enum amdf_queue_publication_mode_bits_e {
  /// Commands are published directly through caller-mapped queue state.
  AMDF_QUEUE_PUBLICATION_MODE_USER = 1u << 0,
  /// Commands are accepted through a bounded provider call.
  AMDF_QUEUE_PUBLICATION_MODE_KERNEL = 1u << 1,
};

/// Semantic operations accepted by one queue family.
typedef uint64_t amdf_queue_roles_t;
enum amdf_queue_role_bits_e {
  /// Executes compute dispatches.
  AMDF_QUEUE_ROLE_COMPUTE = UINT64_C(1) << 0,
  /// Executes memory transfer operations.
  AMDF_QUEUE_ROLE_TRANSFER = UINT64_C(1) << 1,
  /// Executes device atomic operations.
  AMDF_QUEUE_ROLE_ATOMIC = UINT64_C(1) << 2,
  /// Executes engine cache ownership transitions.
  AMDF_QUEUE_ROLE_CACHE_CONTROL = UINT64_C(1) << 3,
  /// Executes virtual-address mapping operations.
  AMDF_QUEUE_ROLE_VIRTUAL_MEMORY = UINT64_C(1) << 4,
};

/// Reservation protocol used by a directly published queue.
typedef uint32_t amdf_queue_producer_mode_t;
enum amdf_queue_producer_mode_e {
  /// Exactly one producer reserves and publishes commands.
  AMDF_QUEUE_PRODUCER_MODE_SINGLE = 1,
  /// Multiple producers atomically reserve and independently publish commands.
  AMDF_QUEUE_PRODUCER_MODE_MULTI = 2,
};

/// Supported directly published queue producer modes.
typedef uint32_t amdf_queue_producer_modes_t;
enum amdf_queue_producer_mode_bits_e {
  /// Single-producer reservation is supported.
  AMDF_QUEUE_PRODUCER_MODE_BIT_SINGLE = 1u << AMDF_QUEUE_PRODUCER_MODE_SINGLE,
  /// Multi-producer reservation is supported.
  AMDF_QUEUE_PRODUCER_MODE_BIT_MULTI = 1u << AMDF_QUEUE_PRODUCER_MODE_MULTI,
};

/// Scheduling priority requested at queue creation.
typedef uint32_t amdf_queue_priority_t;
enum amdf_queue_priority_e {
  /// Lower than normal scheduling priority.
  AMDF_QUEUE_PRIORITY_LOW = 1,
  /// Normal scheduling priority.
  AMDF_QUEUE_PRIORITY_NORMAL = 2,
  /// Higher than normal scheduling priority.
  AMDF_QUEUE_PRIORITY_HIGH = 3,
};

/// Supported scheduling priorities.
typedef uint32_t amdf_queue_priority_capabilities_t;
enum amdf_queue_priority_capability_bits_e {
  /// Low scheduling priority is supported.
  AMDF_QUEUE_PRIORITY_CAPABILITY_LOW = 1u << AMDF_QUEUE_PRIORITY_LOW,
  /// Normal scheduling priority is supported.
  AMDF_QUEUE_PRIORITY_CAPABILITY_NORMAL = 1u << AMDF_QUEUE_PRIORITY_NORMAL,
  /// High scheduling priority is supported.
  AMDF_QUEUE_PRIORITY_CAPABILITY_HIGH = 1u << AMDF_QUEUE_PRIORITY_HIGH,
};

/// Direct producer operations implemented by one user queue.
typedef uint64_t amdf_user_queue_capabilities_t;
enum amdf_user_queue_capability_bits_e {
  /// The queue can be mapped into the host process.
  AMDF_USER_QUEUE_CAPABILITY_HOST_PRODUCER = UINT64_C(1) << 0,
  /// The queue can be mapped into a device producer.
  AMDF_USER_QUEUE_CAPABILITY_DEVICE_PRODUCER = UINT64_C(1) << 1,
};

/// Optional kernel-mediated queue operations.
typedef uint64_t amdf_kernel_queue_capabilities_t;
enum amdf_kernel_queue_capability_bits_e {
  /// One call can submit several native batches atomically.
  AMDF_KERNEL_QUEUE_CAPABILITY_VECTOR_SUBMIT = UINT64_C(1) << 0,
  /// The queue can apply virtual-address mapping operations.
  AMDF_KERNEL_QUEUE_CAPABILITY_VIRTUAL_MEMORY = UINT64_C(1) << 1,
  /// Native external wait and signal primitives can be attached.
  AMDF_KERNEL_QUEUE_CAPABILITY_EXTERNAL_SYNCHRONIZATION = UINT64_C(1) << 2,
};

/// Format identity of an optional queue metadata sidecar.
typedef struct amdf_queue_metadata_format_t {
  /// Sidecar command representation, or UNKNOWN when absent.
  amdf_queue_command_type_t command_type;
  /// Version defining metadata dispatch packets.
  uint32_t dispatch_version;
  /// Version defining metadata barrier packets.
  uint32_t barrier_version;
  /// Reserved for compatible growth and always zero.
  uint32_t reserved;
} amdf_queue_metadata_format_t;

/// Immutable properties of one endpoint-local native queue family.
///
/// A family identifies a command representation and every mechanism its later
/// constructors can actually create. Command types, versions, and encoding
/// features define command encoding, index units, atomic access, ring wrap,
/// and direct publication ordering. User- and kernel-specific fields are zero
/// when their corresponding publication mode is absent.
typedef struct amdf_queue_family_info_t {
  /// Must be `AMDF_STRUCTURE_TYPE_QUEUE_FAMILY_INFO`.
  amdf_structure_type_t type;
  /// Must be at least `sizeof(amdf_queue_family_info_t)`.
  uint32_t structure_size;
  /// Optional output extension chain. No extensions are defined in ABI v1.
  void* next;
  /// Dense ordinal accepted by later queue creation operations.
  uint32_t ordinal;
  /// Native command representation accepted by queues in this family.
  amdf_queue_command_type_t command_type;
  /// User- and kernel-mode publication paths implemented by the provider.
  amdf_queue_publication_modes_t publication_modes;
  /// Version defining commands and their queue publication protocol.
  uint32_t format_version;
  /// Native encoding features defined by the command type and version.
  amdf_queue_format_features_t format_features;
  /// Semantic operations accepted by queues in this family.
  amdf_queue_roles_t roles;
  /// Semantic cache operations encoded by this command representation.
  amdf_cache_operations_t cache_operations;
  /// Range and global forms available for the reported cache operations.
  amdf_cache_transition_kinds_t cache_transition_kinds;
  /// Reserved for compatible growth and always zero.
  uint32_t reserved;
  /// Atomic operations encoded by this command representation.
  amdf_atomic_capabilities_t atomic_capabilities;
  /// Direct producer operations supported by user queues.
  amdf_user_queue_capabilities_t user_queue_capabilities;
  /// Optional operations supported by kernel-mediated queues.
  amdf_kernel_queue_capabilities_t kernel_queue_capabilities;
  /// Supported direct-publication producer modes.
  amdf_queue_producer_modes_t producer_modes;
  /// Supported scheduling priorities.
  amdf_queue_priority_capabilities_t priority_capabilities;
  /// Optional sidecar metadata format.
  amdf_queue_metadata_format_t metadata;
  /// Minimum supported power-of-two primary user ring length in bytes.
  uint64_t minimum_ring_byte_length;
  /// Maximum supported power-of-two primary user ring length in bytes.
  uint64_t maximum_ring_byte_length;
  /// Required primary user ring length alignment in bytes.
  uint64_t ring_byte_length_alignment;
} amdf_queue_family_info_t;

/// An infinite timeout accepted by operations that explicitly wait.
#define AMDF_TIMEOUT_INFINITE UINT64_MAX

/// Lifecycle state shared by native user and kernel queues.
typedef uint32_t amdf_queue_state_t;
enum amdf_queue_state_e {
  /// The queue accepts new work subject to its reported capacity.
  AMDF_QUEUE_STATE_ACTIVE = 1,
  /// The queue encountered a terminal provider or firmware failure.
  AMDF_QUEUE_STATE_FAILED = 2,
  /// The queue's device can no longer execute or retire work.
  AMDF_QUEUE_STATE_DEVICE_LOST = 3,
};

/// Immutable properties of one directly published native queue.
typedef struct amdf_user_queue_info_t {
  /// Must be `AMDF_STRUCTURE_TYPE_USER_QUEUE_INFO`.
  amdf_structure_type_t type;
  /// Must be at least `sizeof(amdf_user_queue_info_t)`.
  uint32_t structure_size;
  /// Optional output extension chain. No extensions are currently defined.
  void* next;
  /// Identity of the materialized device owning this queue.
  amdf_device_id_t device_id;
  /// Stable queue identity for telemetry and failure attribution.
  amdf_queue_id_t queue_id;
  /// Device reset epoch in which this queue remains valid.
  uint64_t reset_epoch;
  /// Endpoint-local family selected when the queue was created.
  uint32_t queue_family_ordinal;
  /// Native command representation accepted by the queue.
  amdf_queue_command_type_t command_type;
  /// Version defining command, index, and publication semantics.
  uint32_t format_version;
  /// Encoding features inherited from the selected queue family.
  amdf_queue_format_features_t format_features;
  /// Reservation protocol selected for this queue.
  amdf_queue_producer_mode_t producer_mode;
  /// Scheduling priority selected for this queue.
  amdf_queue_priority_t priority;
  /// Direct producer operations achieved by this queue.
  amdf_user_queue_capabilities_t capabilities;
  /// Semantic operations achieved by this queue.
  amdf_queue_roles_t roles;
  /// Optional sidecar metadata format.
  amdf_queue_metadata_format_t metadata;
  /// Writable primary ring capacity in bytes.
  uint64_t ring_byte_length;
  /// Writable sidecar ring capacity in bytes, or zero when absent.
  uint64_t metadata_ring_byte_length;
} amdf_user_queue_info_t;

/// Producer-local addresses in one directly published queue mapping.
///
/// An all-zero `producer_device_id` identifies a host mapping. Other addresses
/// are meaningful only to the exact producer device. The queue's command type
/// and format version define the required atomic loads, stores, fences, ring
/// wrap, and doorbell publication protocol. Every mapping exposes a directly
/// writable notification doorbell; there is no mediated notification path.
typedef struct amdf_user_queue_mapping_info_t {
  /// Must be `AMDF_STRUCTURE_TYPE_USER_QUEUE_MAPPING_INFO`.
  amdf_structure_type_t type;
  /// Must be at least `sizeof(amdf_user_queue_mapping_info_t)`.
  uint32_t structure_size;
  /// Optional output extension chain. No extensions are currently defined.
  void* next;
  /// Exact producer device identity, or all-zero for the host.
  amdf_device_id_t producer_device_id;
  /// Identity of the directly published queue.
  amdf_queue_id_t queue_id;
  /// Consumer queue reset epoch in which returned addresses remain valid.
  uint64_t queue_reset_epoch;
  /// Producer reset epoch, or zero for a host mapping.
  uint64_t producer_reset_epoch;
  /// Native command representation stored in the primary ring.
  amdf_queue_command_type_t command_type;
  /// Version defining command, index, and publication semantics.
  uint32_t format_version;
  /// Encoding features inherited from the mapped queue.
  amdf_queue_format_features_t format_features;
  /// Producer-local base address of the writable primary ring.
  uint64_t ring_address;
  /// Writable primary ring capacity in bytes.
  uint64_t ring_byte_length;
  /// Producer-local address of the consumer-owned read index.
  uint64_t read_index_address;
  /// Producer-local address of the producer-owned write index.
  uint64_t write_index_address;
  /// Producer-local address of the write-only notification doorbell.
  uint64_t doorbell_address;
  /// Storage width of the read and write indices.
  uint32_t index_bits;
  /// Atomic write width of the doorbell aperture.
  uint32_t doorbell_bits;
  /// Optional sidecar metadata format.
  amdf_queue_metadata_format_t metadata;
  /// Producer-local base address of the writable metadata ring.
  uint64_t metadata_ring_address;
  /// Writable metadata ring capacity in bytes, or zero when absent.
  uint64_t metadata_ring_byte_length;
} amdf_user_queue_mapping_info_t;

/// Current progress and terminal state of one directly published queue.
typedef struct amdf_user_queue_status_t {
  /// Must be `AMDF_STRUCTURE_TYPE_USER_QUEUE_STATUS`.
  amdf_structure_type_t type;
  /// Must be at least `sizeof(amdf_user_queue_status_t)`.
  uint32_t structure_size;
  /// Optional output extension chain. No extensions are currently defined.
  void* next;
  /// Current queue lifecycle state.
  amdf_queue_state_t state;
  /// Reserved for compatible growth and always zero.
  uint32_t reserved;
  /// Device reset epoch associated with state and progress.
  uint64_t reset_epoch;
  /// Current producer-owned queue frontier.
  ///
  /// The queue format defines whether this frontier advances when storage is
  /// reserved or when commands are release-published. It is the value sampled
  /// from the mapping's write index and uses that format's index units.
  uint64_t producer_index;
  /// Greatest index completely consumed by the native queue, expanded into
  /// the producer frontier's monotonic domain when native counters wrap.
  uint64_t consumed_index;
  /// Sticky terminal failure, or `AMDF_STATUS_OK` while active.
  amdf_status_t terminal_status;
} amdf_user_queue_status_t;

/// Immutable properties of one kernel-mediated queue.
typedef struct amdf_kernel_queue_info_t {
  /// Must be `AMDF_STRUCTURE_TYPE_KERNEL_QUEUE_INFO`.
  amdf_structure_type_t type;
  /// Must be at least `sizeof(amdf_kernel_queue_info_t)`.
  uint32_t structure_size;
  /// Optional output extension chain. No extensions are currently defined.
  void* next;
  /// Identity of the materialized device owning this queue.
  amdf_device_id_t device_id;
  /// Device reset epoch in which this queue remains valid.
  uint64_t reset_epoch;
  /// Endpoint-local family selected when the queue was created.
  uint32_t queue_family_ordinal;
  /// Native command representation accepted by the queue.
  amdf_queue_command_type_t command_type;
  /// Maximum accepted submissions that may remain unretired.
  uint32_t maximum_pending_submission_count;
  /// Maximum commands accepted by one submission.
  uint32_t maximum_command_count;
} amdf_kernel_queue_info_t;

/// Current retirement and terminal state of one kernel-mediated queue.
typedef struct amdf_kernel_queue_status_t {
  /// Must be `AMDF_STRUCTURE_TYPE_KERNEL_QUEUE_STATUS`.
  amdf_structure_type_t type;
  /// Must be at least `sizeof(amdf_kernel_queue_status_t)`.
  uint32_t structure_size;
  /// Optional output extension chain. No extensions are currently defined.
  void* next;
  /// Greatest accepted submission whose native storage is no longer in use.
  uint64_t retired_submission;
  /// Current queue lifecycle state.
  amdf_queue_state_t state;
  /// Reserved for future use and always zero.
  uint32_t reserved;
  /// Sticky terminal failure, or `AMDF_STATUS_OK` while active.
  amdf_status_t terminal_status;
} amdf_kernel_queue_status_t;

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // AMDF_QUEUE_H_
