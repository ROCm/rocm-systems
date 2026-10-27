// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef AMDF_MEMORY_RESERVATION_H_
#define AMDF_MEMORY_RESERVATION_H_

#include "amdf/memory.h"

#ifdef __cplusplus
extern "C" {
#endif

/// Stable host or device addresses with independently replaceable backing.
/// The reservation borrows its live devices and owns no application backing.
typedef struct amdf_memory_reservation_t amdf_memory_reservation_t;

/// Live-domain limits for host-completed virtual-memory updates.
typedef struct amdf_memory_reservation_capabilities_t {
  /// Must be AMDF_STRUCTURE_TYPE_MEMORY_RESERVATION_CAPABILITIES.
  amdf_structure_type_t type;
  /// Must be at least sizeof(amdf_memory_reservation_capabilities_t).
  uint32_t structure_size;
  /// Optional output extension chain. No extensions are defined.
  void* next;
  /// Exact permissions accepted by mapping updates in this domain.
  amdf_memory_access_t supported_access;
  /// Permissions that every nonempty mapping must include on this domain.
  amdf_memory_access_t required_access;
  /// Consumer-local address domain described by these limits.
  uint32_t address_domain_ordinal;
  /// Reserved for compatible growth; zero.
  uint32_t reserved;
  /// Largest reservation length in bytes.
  uint64_t maximum_byte_length;
  /// Required reservation length and minimum base alignment in bytes.
  uint64_t reservation_granularity;
  /// Strongest power-of-two minimum base alignment accepted by creation.
  uint64_t maximum_alignment;
  /// Required destination offset, native source offset and length alignment.
  uint64_t mapping_granularity;
  /// Inclusive lower bound of a complete reservation's address interval.
  uint64_t minimum_address;
  /// Inclusive upper bound of a complete reservation's address interval.
  uint64_t maximum_address;
  /// Whole-backing flags required of every nonempty mapping source. Request
  /// these in memory_create.required_flags, in addition to preparing each
  /// device's MAPPING_SOURCE access and required_access permissions. Zero means
  /// no additional whole-backing requirement, not that preparation is optional.
  amdf_memory_flags_t required_source_flags;
} amdf_memory_reservation_capabilities_t;

/// One explicitly initialized consumer of a reservation.
typedef struct amdf_memory_reservation_access_t {
  /// Borrowed live device, or NULL for the process host address space.
  /// The caller keeps a live device alive through reservation destruction.
  amdf_device_t* device;
  /// Consumer-local address domain, or ADDRESS_DOMAIN_ORDINAL_NONE for host.
  uint32_t address_domain_ordinal;
  /// Maximum permissions that updates may establish for this consumer.
  /// Consumers sharing host/SVA translations require identical envelopes.
  amdf_memory_access_t access;
  /// Minimum mapping offset/length unit, or zero for provider policy.
  /// Must be a power-of-two multiple of the native mapping granularity. The
  /// final unit may be shorter, ending exactly at the native-aligned
  /// reservation length. The returned geometry also satisfies any consumers
  /// sharing this native mapping group. Independently owned native views use
  /// these units on platforms requiring whole-view removal.
  uint64_t mapping_granularity;
} amdf_memory_reservation_access_t;

/// Parameters for reserving addresses without allocating application backing.
typedef struct amdf_memory_reservation_create_info_t {
  /// Must be AMDF_STRUCTURE_TYPE_MEMORY_RESERVATION_CREATE_INFO.
  amdf_structure_type_t type;
  /// Must be at least sizeof(amdf_memory_reservation_create_info_t).
  uint32_t structure_size;
  /// Optional input extension chain. No extensions are defined.
  const void* next;
  /// Nonzero length shared by all requested consumers, satisfying their native
  /// reservation and mapping granularities. It need not fill the final unit of
  /// a larger caller-selected mapping granularity.
  uint64_t byte_length;
  /// Minimum power-of-two base alignment, or zero for provider policy.
  uint64_t minimum_alignment;
  /// Nonzero number of caller-ordered, unique host/live-device consumers.
  uint32_t access_count;
  /// Reserved for compatible growth; must be zero.
  uint32_t reserved;
  /// Consumers borrowed only for this call; live devices belong to instance.
  const amdf_memory_reservation_access_t* accesses;
} amdf_memory_reservation_create_info_t;

/// Immutable facts of one reservation's consumer projection.
typedef struct amdf_memory_reservation_access_info_t {
  /// Must be AMDF_STRUCTURE_TYPE_MEMORY_RESERVATION_ACCESS_INFO.
  amdf_structure_type_t type;
  /// Must be at least sizeof(amdf_memory_reservation_access_info_t).
  uint32_t structure_size;
  /// Optional output extension chain. No extensions are defined.
  void* next;
  /// Exact live consumer, or all-zero for the host projection.
  amdf_device_id_t device_id;
  /// Resource-local ordinal identifying this projection.
  uint32_t ordinal;
  /// First ordinal sharing this native mapping state. Updates through any of
  /// these aliases affect all of them; they are not independent address spaces.
  uint32_t mapping_access_ordinal;
  /// Consumer-local address domain of the stable address.
  uint32_t address_domain_ordinal;
  /// Maximum permissions admitted by this reservation projection.
  amdf_memory_access_t access;
  /// Interpretation of address; HOST for a process pointer.
  amdf_memory_address_kind_t address_kind;
  /// Reserved for compatible growth; zero.
  uint32_t reserved;
  /// Stable base, queryable even when none of the range is backed.
  uint64_t address;
  /// Complete reserved length in bytes.
  uint64_t byte_length;
  /// Required mapping offset alignment and full replacement unit in bytes.
  /// The final unit may be shorter and ends exactly at byte_length.
  uint64_t mapping_granularity;
  /// Device reset epoch, or zero for a host projection.
  uint64_t reset_epoch;
} amdf_memory_reservation_access_info_t;

/// One replacement or detach within a stable reservation.
/// Destination ranges in a batch must not overlap. Source ranges may repeat.
typedef struct amdf_memory_mapping_update_t {
  /// Byte offset in the destination reservation.
  uint64_t target_byte_offset;
  /// Nonzero length wholly contained by the destination and any source. A
  /// non-multiple of the mapping unit must end exactly at the reservation end.
  uint64_t byte_length;
  /// Ordinary prepared backing, or NULL to detach the destination range.
  /// The caller keeps backing alive through its last installed alias and user.
  /// Host/SVA groups require HOST_MAPPING_SOURCE backing with prepared
  /// MAPPING_SOURCE access for every device sharing their translations.
  amdf_memory_t* memory;
  /// Source access prepared for this exact live device. Must be zero for a
  /// host projection (which uses backing, not a device access) or detach.
  uint32_t memory_access_ordinal;
  /// Exact installed permissions, or zero for detach.
  amdf_memory_access_t access;
  /// Offset from the source's logical byte zero, or zero for detach.
  uint64_t source_byte_offset;
} amdf_memory_mapping_update_t;

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // AMDF_MEMORY_RESERVATION_H_
