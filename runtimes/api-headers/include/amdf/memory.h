// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef AMDF_MEMORY_H_
#define AMDF_MEMORY_H_

#include "amdf/base.h"

#ifdef __cplusplus
extern "C" {
#endif

/// Physical placement class of a memory profile or live attachment.
typedef uint32_t amdf_memory_class_t;
enum amdf_memory_class_e {
  /// No placement class. This value is never accepted by memory creation.
  AMDF_MEMORY_CLASS_UNKNOWN = 0,
  /// System memory accessible through a host mapping, including borrowed pages.
  AMDF_MEMORY_CLASS_SYSTEM = 1,
  /// Device-local physical memory that may not be host visible.
  AMDF_MEMORY_CLASS_LOCAL = 2,
  /// Native-private storage qualified by its live scope owner.
  AMDF_MEMORY_CLASS_PRIVATE = 3,
};

/// Required or achieved properties of a memory attachment.
typedef uint64_t amdf_memory_flags_t;
enum amdf_memory_flag_bits_e {
  /// The allocation can be explicitly mapped for host access.
  AMDF_MEMORY_FLAG_HOST_VISIBLE = UINT64_C(1) << 0,
  /// The physical placement is local to the attached device.
  AMDF_MEMORY_FLAG_DEVICE_LOCAL = UINT64_C(1) << 1,
  /// The physical backing can be exported and attached to another device.
  AMDF_MEMORY_FLAG_SHAREABLE = UINT64_C(1) << 2,
  /// The allocation can hold directly published user-mode queue state.
  AMDF_MEMORY_FLAG_QUEUE_STORAGE = UINT64_C(1) << 3,
  /// Host and device access requires no explicit host cache transition.
  AMDF_MEMORY_FLAG_HOST_COHERENT = UINT64_C(1) << 4,
  /// A stable device address is established before creation returns.
  AMDF_MEMORY_FLAG_DEVICE_ADDRESS = UINT64_C(1) << 5,
};

/// Exact device page-table access granted to one memory attachment.
typedef uint32_t amdf_memory_access_t;
enum amdf_memory_access_bit_e {
  /// Device loads are permitted from the attachment.
  AMDF_MEMORY_ACCESS_READ = 1u << 0,
  /// Device stores are permitted to the attachment.
  AMDF_MEMORY_ACCESS_WRITE = 1u << 1,
  /// Device instruction fetches are permitted from the attachment.
  AMDF_MEMORY_ACCESS_EXECUTE = 1u << 2,
};

/// Interface consuming a numeric memory address, not an address-space owner.
typedef uint32_t amdf_memory_address_kind_t;
enum amdf_memory_address_kind_e {
  /// Ordinary GPU loads, stores, instruction fetches, and command addresses.
  AMDF_MEMORY_ADDRESS_GPU = 0,
  /// XDNA shim DMA descriptors, with the native DRAM translation applied.
  AMDF_MEMORY_ADDRESS_XDNA_DMA = 1,
  /// XDNA firmware buffer arguments or scoped instruction-storage addresses.
  /// This is not an address to place directly in a shim DMA descriptor.
  AMDF_MEMORY_ADDRESS_XDNA_FIRMWARE = 2,
};

/// Set of address kinds, with bit `1 << amdf_memory_address_kind_t` set for
/// each supported kind. Support describes usable addresses, not ownership.
typedef uint64_t amdf_memory_address_kinds_t;

/// Host access requested for one explicit mapping.
typedef uint32_t amdf_memory_map_flags_t;
enum amdf_memory_map_flag_bits_e {
  /// Host loads are permitted from the mapped range.
  AMDF_MEMORY_MAP_FLAG_READ = 1u << 0,
  /// Host stores are permitted to the mapped range.
  AMDF_MEMORY_MAP_FLAG_WRITE = 1u << 1,
};

/// Atomic operations supported by queues and target memory.
typedef uint64_t amdf_atomic_operations_t;
enum amdf_atomic_operation_bits_e {
  /// Waits until an atomic value satisfies a supported condition.
  AMDF_ATOMIC_OPERATION_WAIT = UINT64_C(1) << 0,
  /// Atomic store.
  AMDF_ATOMIC_OPERATION_STORE = UINT64_C(1) << 1,
  /// No-result atomic addition.
  AMDF_ATOMIC_OPERATION_ADD = UINT64_C(1) << 2,
  /// No-result atomic subtraction.
  AMDF_ATOMIC_OPERATION_SUBTRACT = UINT64_C(1) << 3,
  /// Atomic bitwise AND.
  AMDF_ATOMIC_OPERATION_AND = UINT64_C(1) << 4,
  /// Atomic bitwise OR.
  AMDF_ATOMIC_OPERATION_OR = UINT64_C(1) << 5,
  /// Atomic bitwise XOR.
  AMDF_ATOMIC_OPERATION_XOR = UINT64_C(1) << 6,
};

/// Atomic wait conditions supported by one queue command representation.
typedef uint32_t amdf_atomic_wait_conditions_t;
enum amdf_atomic_wait_condition_bits_e {
  /// Wait while the masked value is not equal to the requested value.
  AMDF_ATOMIC_WAIT_CONDITION_EQUAL = 1u << 0,
  /// Wait while the masked value is equal to the requested value.
  AMDF_ATOMIC_WAIT_CONDITION_NOT_EQUAL = 1u << 1,
  /// Wait while the masked value is less than the requested unsigned value.
  AMDF_ATOMIC_WAIT_CONDITION_UNSIGNED_GREATER_EQUAL = 1u << 2,
};

/// Atomic commands encoded by one queue family.
///
/// These capabilities describe command encoding only. Callers intersect them
/// with target-memory operation support and concrete pair reach before use.
typedef struct amdf_atomic_capabilities_t {
  /// Operations accepted on naturally aligned 32-bit words.
  amdf_atomic_operations_t operations_32;
  /// Operations accepted on naturally aligned 64-bit words.
  amdf_atomic_operations_t operations_64;
  /// Wait conditions accepted on 32-bit words.
  amdf_atomic_wait_conditions_t wait_conditions_32;
  /// Wait conditions accepted on 64-bit words.
  amdf_atomic_wait_conditions_t wait_conditions_64;
  /// 32-bit operations which consume no dispatch resources.
  amdf_atomic_operations_t operations_without_dispatch_32;
  /// 64-bit operations which consume no dispatch resources.
  amdf_atomic_operations_t operations_without_dispatch_64;
} amdf_atomic_capabilities_t;

/// Largest domain over which an atomic access is mutually atomic.
typedef uint32_t amdf_atomic_scope_t;
enum amdf_atomic_scope_e {
  /// No qualified atomic scope.
  AMDF_ATOMIC_SCOPE_NONE = 0,
  /// One device address domain.
  AMDF_ATOMIC_SCOPE_DEVICE = 1,
  /// One correlated device fabric.
  AMDF_ATOMIC_SCOPE_FABRIC = 2,
  /// Host and every reported device participant.
  AMDF_ATOMIC_SCOPE_SYSTEM = 3,
};

/// Atomic-cell reach shared by two exact execution sites.
typedef struct amdf_atomic_reach_t {
  /// Largest mutually atomic scope for naturally aligned 32-bit accesses.
  amdf_atomic_scope_t scope_32;
  /// Largest mutually atomic scope for naturally aligned 64-bit accesses.
  amdf_atomic_scope_t scope_64;
} amdf_atomic_reach_t;

/// Native representation carried by one external-memory value.
typedef uint32_t amdf_external_memory_type_t;
enum amdf_external_memory_type_e {
  /// No payload. The all-zero external-memory value has this type.
  AMDF_EXTERNAL_MEMORY_TYPE_NONE = 0,
  /// Linux DMA-BUF file descriptor.
  AMDF_EXTERNAL_MEMORY_TYPE_DMA_BUF_FD = 1,
  /// File descriptor accepted only by a provider with matching provenance.
  AMDF_EXTERNAL_MEMORY_TYPE_OPAQUE_FD = 2,
  /// Owning NT handle to a committed D3D12 buffer and its implicit heap.
  /// The Windows GPU import profile accepts a same-adapter DEFAULT heap with
  /// SHARED, one uncompressed linear native allocation, and no cross-adapter
  /// flag. Logical offsets refer to buffer bytes, not allocation padding.
  /// Import establishes independent native references; export returns this
  /// same resource format. Heap handles, images, fences, legacy KMT handles
  /// and Vulkan opaque memory have different interpretations and are rejected.
  /// The caller returns the resource to COMMON and completes foreign work
  /// before native use, then completes native work and its cache release before
  /// foreign use. Import/export neither wait nor change D3D12 resource state.
  AMDF_EXTERNAL_MEMORY_TYPE_D3D12_RESOURCE = 3,
  /// Host virtual address with a caller-defined lifetime lease.
  AMDF_EXTERNAL_MEMORY_TYPE_HOST_POINTER = 4,
  /// Device virtual address with a caller-defined lifetime lease.
  AMDF_EXTERNAL_MEMORY_TYPE_DEVICE_ADDRESS = 5,
};

/// Number of external-memory types.
#define AMDF_EXTERNAL_MEMORY_TYPE_COUNT 5u

/// Opaque identity of the exact native interpretation of a transport payload.
///
/// Portable self-describing transports use an all-zero identity. Opaque file
/// descriptors and raw device addresses require a matching nonzero identity in
/// the importing memory profile. The value is meaningful only while its
/// provider instance remains live and is never a native handle.
typedef struct amdf_external_memory_provenance_t {
  /// Provider-defined identity words.
  uint64_t words[2];
} amdf_external_memory_provenance_t;

/// Returns true when an external-memory provenance identity is available.
static inline bool amdf_external_memory_provenance_is_valid(
    const amdf_external_memory_provenance_t* provenance) {
  return (provenance->words[0] | provenance->words[1]) != 0;
}

/// Returns true when two transport provenance identities contain one value.
static inline bool amdf_external_memory_provenance_is_equal(
    const amdf_external_memory_provenance_t* lhs,
    const amdf_external_memory_provenance_t* rhs) {
  return lhs->words[0] == rhs->words[0] && lhs->words[1] == rhs->words[1];
}

/// Payload of one typed external-memory value.
typedef union amdf_external_memory_payload_t {
  /// Nonnegative DMA-BUF or opaque file descriptor.
  int64_t file_descriptor;
  /// Windows NT handle.
  void* native_handle;
  /// First host byte of the exported backing.
  void* host_pointer;
  /// First device byte of the exported backing.
  uint64_t device_address;
} amdf_external_memory_payload_t;

/// Infallibly releases one external-memory payload.
///
/// The callback must be thread-safe and must not throw or unwind across the C
/// ABI. It receives the exact type and payload copied into the external value.
typedef void(AMDF_CALL* amdf_external_memory_release_fn_t)(
    void* user_data, amdf_external_memory_type_t type,
    amdf_external_memory_payload_t payload);

/// Move-owned transport for one logical range of physical memory.
///
/// The all-zero value is empty. A nonempty value with a null `release` callback
/// borrows its payload; the caller keeps the native owner live through import
/// or explicit release. Copying a nonempty value does not duplicate ownership.
typedef struct amdf_external_memory_t {
  /// Native payload representation.
  amdf_external_memory_type_t type;
  /// Reserved for future use and always zero.
  uint32_t reserved;
  /// Typed native payload.
  amdf_external_memory_payload_t payload;
  /// Exact payload interpretation, or all-zero for portable transports.
  amdf_external_memory_provenance_t provenance;
  /// Byte offset of logical byte zero in the exported physical backing.
  uint64_t source_byte_offset;
  /// Nonzero logical byte length represented by this value.
  uint64_t byte_length;
  /// Physical identity established by the exporter, when available.
  amdf_physical_memory_id_t physical_backing_id;
  /// Optional infallible payload release callback.
  amdf_external_memory_release_fn_t release;
  /// Opaque value passed to `release`.
  void* release_user_data;
} amdf_external_memory_t;

/// Operations supported by one memory profile.
typedef uint64_t amdf_memory_profile_roles_t;
enum amdf_memory_profile_role_bits_e {
  /// Creates provider-owned physical backing.
  AMDF_MEMORY_PROFILE_ROLE_CREATE = UINT64_C(1) << 0,
  /// Registers caller-owned host pages.
  AMDF_MEMORY_PROFILE_ROLE_REGISTER = UINT64_C(1) << 1,
  /// Imports typed external memory.
  AMDF_MEMORY_PROFILE_ROLE_IMPORT = UINT64_C(1) << 2,
  /// Exports typed external memory.
  AMDF_MEMORY_PROFILE_ROLE_EXPORT = UINT64_C(1) << 3,
  /// Creates explicit host mappings.
  AMDF_MEMORY_PROFILE_ROLE_HOST_MAP = UINT64_C(1) << 4,
  /// Supplies physical backing to a virtual-memory mapping operation.
  AMDF_MEMORY_PROFILE_ROLE_MAPPING_SOURCE = UINT64_C(1) << 5,
  /// Owns virtual-address reservations and mapping operations.
  AMDF_MEMORY_PROFILE_ROLE_MAPPING_TARGET = UINT64_C(1) << 6,
};

/// Capabilities of one external-memory type in a memory profile.
typedef uint32_t amdf_external_memory_support_flags_t;
enum amdf_external_memory_support_flag_bits_e {
  /// Values of this type can be imported through this profile.
  AMDF_EXTERNAL_MEMORY_SUPPORT_FLAG_IMPORT = 1u << 0,
  /// Attachments using this profile can export values of this type.
  AMDF_EXTERNAL_MEMORY_SUPPORT_FLAG_EXPORT = 1u << 1,
  /// Nonzero logical source offsets are supported.
  AMDF_EXTERNAL_MEMORY_SUPPORT_FLAG_SOURCE_OFFSET = 1u << 2,
  /// Values of this type may cross a process boundary.
  AMDF_EXTERNAL_MEMORY_SUPPORT_FLAG_CROSS_PROCESS = 1u << 3,
  /// Values created by a foreign API provider may be imported.
  AMDF_EXTERNAL_MEMORY_SUPPORT_FLAG_FOREIGN_API = 1u << 4,
};

/// Type-specific import and export limits within one memory profile.
typedef struct amdf_external_memory_support_t {
  /// External-memory representation described by this entry.
  amdf_external_memory_type_t type;
  /// Supported import, export, offset, and provenance capabilities.
  amdf_external_memory_support_flags_t flags;
  /// Required payload provenance, or all-zero for portable transports.
  amdf_external_memory_provenance_t provenance;
  /// Required source-offset alignment, or zero when offsets must be zero.
  uint64_t source_offset_alignment;
  /// Required logical byte-length alignment.
  uint64_t byte_length_alignment;
  /// Maximum logical byte length, or zero for no type-specific limit.
  uint64_t maximum_byte_length;
} amdf_external_memory_support_t;

/// Maximum external-memory support entries in one profile.
#define AMDF_MEMORY_PROFILE_EXTERNAL_SUPPORT_CAPACITY \
  AMDF_EXTERNAL_MEMORY_TYPE_COUNT

/// Indicates that an attachment has no queryable memory-profile ordinal.
#define AMDF_MEMORY_PROFILE_ORDINAL_UNKNOWN UINT32_MAX

/// Indicates that a profile or attachment has no ordinary address domain.
#define AMDF_ADDRESS_DOMAIN_ORDINAL_NONE UINT32_MAX

/// Host cache behavior of a mapped allocation.
typedef uint32_t amdf_host_cacheability_t;
enum amdf_host_cacheability_e {
  /// The provider cannot describe the mapping's cache behavior.
  AMDF_HOST_CACHEABILITY_UNKNOWN = 0,
  /// Ordinary host write-back caching. Device coherence is a separate property
  /// of the device's access, not of this CPU mapping.
  AMDF_HOST_CACHEABILITY_WRITE_BACK = 2,
  /// Host write-combined caching intended for sequential stores.
  AMDF_HOST_CACHEABILITY_WRITE_COMBINED = 3,
  /// Uncached host access.
  AMDF_HOST_CACHEABILITY_UNCACHED = 4,
};

/// Limits for one way of constructing a memory attachment.
///
/// For CREATE, the exact owned native payload extent is
/// `align_up(byte_length + native_byte_length_prefix,
///           native_byte_length_granularity)` for every admitted request.
/// Maximum logical length leaves enough uint64_t headroom for both additions.
/// Any admitted minimum alignment preserves this extent. It excludes virtual
/// address reservation slack, driver/library metadata and residency charges.
/// REGISTER and IMPORT instead describe an extent established from the source.
typedef struct amdf_memory_construction_capabilities_t {
  /// Maximum logical attachment length accepted by this operation.
  uint64_t maximum_byte_length;
  /// Required logical byte-length granularity.
  uint64_t byte_length_granularity;
  /// Required caller-owned host-pointer alignment for registration, or zero
  /// when the operation consumes no host pointer.
  uint64_t registered_host_pointer_alignment;
  /// Required cache class of caller-owned pages for REGISTER, or UNKNOWN
  /// when this operation consumes no caller pages. This is a caller-established
  /// property of the source mapping, not inferred from its pointer value.
  amdf_host_cacheability_t registered_host_cacheability;
  /// Reserved for compatible growth and always zero.
  uint32_t reserved;
  /// Power-of-two base alignment guaranteed when callers request no stronger
  /// alignment.
  uint64_t minimum_alignment;
  /// Strongest power-of-two minimum alignment callers may request.
  uint64_t maximum_alignment;
  /// Granularity of the native physical allocation or page cover in bytes.
  uint64_t native_byte_length_granularity;
  /// Fixed native payload bytes preceding logical byte zero for CREATE.
  /// Successful CREATE reports this as source_byte_offset. Zero for REGISTER
  /// and IMPORT, whose source offset is determined by the supplied storage.
  uint64_t native_byte_length_prefix;
} amdf_memory_construction_capabilities_t;

/// Numeric envelope shared by every address kind produced by one profile.
/// Bounds cover complete logical ranges, including any consumer translation;
/// they do not promise that every address inside the envelope is allocatable.
typedef struct amdf_memory_address_capabilities_t {
  /// Device-local ordinary address-domain ordinal.
  uint32_t address_domain_ordinal;
  /// Maximum significant width across produced address kinds. Nonzero for
  /// addressable profiles; zero when the profile produces no device address.
  uint32_t address_bit_count;
  /// Inclusive lower bound, or zero for profiles without device addresses.
  uint64_t minimum_address;
  /// Inclusive upper bound, or zero for profiles without device addresses.
  uint64_t maximum_address;
  /// Minimum power-of-two alignment common to all produced address kinds.
  uint64_t minimum_alignment;
} amdf_memory_address_capabilities_t;

/// Exact access requirements shared by live-device queries and construction.
typedef struct amdf_memory_access_requirements_t {
  /// Exact device permissions; construction never silently widens them.
  amdf_memory_access_t access;
  /// Reserved for compatible growth; must be zero.
  uint32_t reserved;
  /// Required access properties: QUEUE_STORAGE, HOST_COHERENT, DEVICE_ADDRESS.
  amdf_memory_flags_t flags;
  /// Interfaces whose complete logical-range addresses must be established.
  amdf_memory_address_kinds_t address_kinds;
} amdf_memory_access_requirements_t;

/// One explicitly initialized consumer of a memory resource.
typedef struct amdf_memory_device_access_t {
  /// Borrowed live device. The caller keeps it live through memory release.
  amdf_device_t* device;
  /// Access established completely before successful construction returns.
  amdf_memory_access_requirements_t requirements;
} amdf_memory_device_access_t;

/// Complete access capabilities for one live device in a selected scope
/// contract.
typedef struct amdf_memory_access_capabilities_t {
  /// Must be `AMDF_STRUCTURE_TYPE_MEMORY_ACCESS_CAPABILITIES`.
  amdf_structure_type_t type;
  /// Must be at least `sizeof(amdf_memory_access_capabilities_t)`.
  uint32_t structure_size;
  /// Optional output extension chain. No extensions are currently defined.
  void* next;
  /// Permissions present for every construction of this contract.
  amdf_memory_access_t guaranteed_access;
  /// Permissions the contract can establish.
  amdf_memory_access_t supported_access;
  /// Access properties present for every construction.
  amdf_memory_flags_t guaranteed_flags;
  /// Access properties which may be required.
  amdf_memory_flags_t supported_flags;
  /// Atomic operations on naturally aligned 32-bit words.
  amdf_atomic_operations_t atomic_operations_32;
  /// Atomic operations on naturally aligned 64-bit words.
  amdf_atomic_operations_t atomic_operations_64;
  /// Complete numeric envelope covering the reported address kinds.
  amdf_memory_address_capabilities_t device_address;
  /// Interfaces whose complete logical-range addresses can be established.
  amdf_memory_address_kinds_t address_kinds;
} amdf_memory_access_capabilities_t;

/// Range and access capabilities of explicit host mappings.
typedef struct amdf_host_mapping_capabilities_t {
  /// Maximum logical mapping length in bytes.
  uint64_t maximum_byte_length;
  /// Required mapping byte-offset granularity.
  uint64_t byte_offset_granularity;
  /// Required mapping byte-length granularity.
  uint64_t byte_length_granularity;
  /// Host read and write requests accepted by the profile.
  amdf_memory_map_flags_t supported_access;
  /// Reserved for future use and always zero.
  uint32_t reserved;
} amdf_host_mapping_capabilities_t;

/// Immutable construction and transport capabilities of one memory profile.
///
/// A profile is one valid physical placement and operation combination, not a
/// set of independently composable feature bits. A profile exposes at most one
/// of CREATE and REGISTER; IMPORT and transport roles may describe additional
/// ways to attach the same placement. Callers derive each request from one
/// profile and require only flags and access included in its supported sets.
typedef struct amdf_memory_profile_t {
  /// Must be `AMDF_STRUCTURE_TYPE_MEMORY_PROFILE`.
  amdf_structure_type_t type;
  /// Must be at least `sizeof(amdf_memory_profile_t)`.
  uint32_t structure_size;
  /// Optional output extension chain. No extensions are currently defined.
  void* next;
  /// Dense query ordinal reported by attachments and selected for construction
  /// or import.
  uint32_t ordinal;
  /// Physical placement class produced by this profile.
  amdf_memory_class_t memory_class;
  /// Creation, registration, transport, mapping, and address roles.
  amdf_memory_profile_roles_t roles;
  /// Properties present on every attachment created with this profile.
  amdf_memory_flags_t guaranteed_flags;
  /// Properties which callers may require from this profile.
  amdf_memory_flags_t supported_flags;
  /// Provider-owned physical allocation limits, or all-zero without CREATE.
  amdf_memory_construction_capabilities_t allocation;
  /// Caller-owned host registration limits, or all-zero without REGISTER.
  amdf_memory_construction_capabilities_t registration;
  /// External-memory attachment limits, or all-zero without IMPORT.
  amdf_memory_construction_capabilities_t import;
  /// Explicit host-mapping limits, or all-zero without HOST_MAP.
  amdf_host_mapping_capabilities_t host_mapping;
  /// Number of valid entries in `external_memory_support`.
  uint32_t external_memory_support_count;
  /// Reserved for future use and always zero.
  uint32_t reserved;
  /// Type-specific external-memory support records.
  amdf_external_memory_support_t
      external_memory_support[AMDF_MEMORY_PROFILE_EXTERNAL_SUPPORT_CAPACITY];
} amdf_memory_profile_t;

/// Storage locality and lifetime owner of a borrowed scope.
typedef uint32_t amdf_memory_scope_kind_t;
enum amdf_memory_scope_kind_e {
  /// System storage for a supported set of consumers, owned by the instance.
  AMDF_MEMORY_SCOPE_KIND_SYSTEM = 0,
  /// Storage local to one physical endpoint, owned by that passive endpoint.
  AMDF_MEMORY_SCOPE_KIND_LOCAL = 1,
  /// Storage qualified by a particular live native device or context owner.
  AMDF_MEMORY_SCOPE_KIND_PRIVATE = 2,
};

/// Complete immutable facts of one borrowed storage scope.
typedef struct amdf_memory_scope_info_t {
  /// Must be `AMDF_STRUCTURE_TYPE_MEMORY_SCOPE_INFO`.
  amdf_structure_type_t type;
  /// Must be at least `sizeof(amdf_memory_scope_info_t)`.
  uint32_t structure_size;
  /// Optional output extension chain. No extensions are currently defined.
  void* next;
  /// Storage and owner class.
  amdf_memory_scope_kind_t kind;
  /// Number of stable scope-local profile ordinals.
  uint32_t memory_profile_count;
  /// Physical storage endpoint for LOCAL; all zero otherwise.
  amdf_endpoint_id_t physical_endpoint_id;
} amdf_memory_scope_info_t;

/// Parameters obtaining one backing and all requested live-device accesses.
typedef struct amdf_memory_create_info_t {
  /// Must be `AMDF_STRUCTURE_TYPE_MEMORY_CREATE_INFO`.
  amdf_structure_type_t type;
  /// Must be at least `sizeof(amdf_memory_create_info_t)`.
  uint32_t structure_size;
  /// Optional input extension chain. No extensions are currently defined.
  const void* next;
  /// Scope-local memory-profile ordinal selected for construction. A CREATE
  /// profile requires `registered_host_pointer` to be `NULL`; a REGISTER
  /// profile requires it to be non-`NULL`.
  uint32_t memory_profile_ordinal;
  /// Number of entries in accesses; zero requests CPU-only system memory.
  uint32_t access_count;
  /// Required backing properties: HOST_VISIBLE, DEVICE_LOCAL and SHAREABLE.
  amdf_memory_flags_t required_flags;
  /// Logical byte length established for every requested consumer. Native
  /// allocation rounding is reported separately and grants no additional
  /// access.
  uint64_t byte_length;
  /// Minimum power-of-two allocation-base alignment in every supported address
  /// space, or zero for provider policy.
  uint64_t minimum_alignment;
  /// Borrowed host base for a REGISTER profile, otherwise `NULL`. The caller
  /// keeps this address range backed by the same live pages through
  /// `memory_destroy`. A failed native detach can leave registrations live;
  /// that error does not authorize recycling the source. Registration does
  /// not take ownership.
  void* registered_host_pointer;
  /// Caller-ordered consumers, consumed during the call and not retained.
  /// Devices must be unique and belong to the scope's instance. NULL at zero
  /// count. Each device remains a caller-enforced lifetime dependency.
  const amdf_memory_device_access_t* accesses;
  /// Cache class of registered_host_pointer, or UNKNOWN for CREATE. The caller
  /// establishes this property from the source allocation/mapping contract and
  /// keeps it unchanged while registered. A pointer alone does not establish a
  /// cache class. UNKNOWN and unsupported classes are rejected for REGISTER.
  amdf_host_cacheability_t registered_host_cacheability;
  /// Reserved for compatible growth; must be zero.
  uint32_t reserved;
} amdf_memory_create_info_t;

/// Immutable backing properties of one live memory resource.
typedef struct amdf_memory_info_t {
  /// Must be `AMDF_STRUCTURE_TYPE_MEMORY_INFO`.
  amdf_structure_type_t type;
  /// Must be at least `sizeof(amdf_memory_info_t)`.
  uint32_t structure_size;
  /// Optional output extension chain. No extensions are currently defined.
  void* next;
  /// Achieved profile ordinal, or `AMDF_MEMORY_PROFILE_ORDINAL_UNKNOWN` when
  /// the provider exposes no profile for this attachment path.
  uint32_t memory_profile_ordinal;
  /// Achieved physical placement class.
  amdf_memory_class_t memory_class;
  /// Number of immutable device accesses, indexed by resource-local ordinal.
  uint32_t access_count;
  /// Reserved for compatible growth; always zero.
  uint32_t reserved;
  /// Achieved backing properties: HOST_VISIBLE, DEVICE_LOCAL and SHAREABLE.
  amdf_memory_flags_t flags;
  /// Byte offset of logical byte zero in the physical backing.
  uint64_t source_byte_offset;
  /// Logical attachment length in bytes.
  uint64_t byte_length;
  /// Guaranteed power-of-two logical-base alignment in every supported address
  /// space.
  uint64_t alignment;
  /// Complete native payload allocation or registered source extent in bytes,
  /// including native rounding. Excludes driver and library metadata. Multiple
  /// attachments can report the same backing extent independently.
  uint64_t native_allocation_byte_length;
  /// Granularity of `native_allocation_byte_length` in bytes.
  uint64_t native_allocation_granularity;
  /// Identity shared by attachments to the same physical backing, when known.
  amdf_physical_memory_id_t physical_backing_id;
} amdf_memory_info_t;

/// Immutable facts of one established device access to a memory resource.
typedef struct amdf_memory_access_info_t {
  /// Must be `AMDF_STRUCTURE_TYPE_MEMORY_ACCESS_INFO`.
  amdf_structure_type_t type;
  /// Must be at least `sizeof(amdf_memory_access_info_t)`.
  uint32_t structure_size;
  /// Optional output extension chain. No extensions are currently defined.
  void* next;
  /// Resource-local ordinal identifying this access.
  uint32_t ordinal;
  /// Exact established device permissions.
  amdf_memory_access_t access;
  /// Identity of the required live consumer, not an allocation owner.
  amdf_device_id_t device_id;
  /// Established access properties: QUEUE_STORAGE, HOST_COHERENT and
  /// DEVICE_ADDRESS. Backing properties belong to amdf_memory_info_t.
  amdf_memory_flags_t flags;
  /// Supported atomic operations on naturally aligned 32-bit words.
  amdf_atomic_operations_t atomic_operations_32;
  /// Supported atomic operations on naturally aligned 64-bit words.
  amdf_atomic_operations_t atomic_operations_64;
  /// Consumer-local address domain, or NONE without an address.
  uint32_t address_domain_ordinal;
  /// Reserved for compatible growth; always zero.
  uint32_t reserved;
  /// Address kinds established for the complete logical range. Each set bit
  /// guarantees that memory_query_address succeeds for that kind. Zero means
  /// no device address was established; unsupported kinds have no address.
  amdf_memory_address_kinds_t address_kinds;
  /// Device reset epoch in which this access and its addresses remain valid.
  uint64_t reset_epoch;
} amdf_memory_access_info_t;

/// Parameters obtaining external backing and all requested live-device access.
typedef struct amdf_memory_import_info_t {
  /// Must be `AMDF_STRUCTURE_TYPE_MEMORY_IMPORT_INFO`.
  amdf_structure_type_t type;
  /// Must be at least `sizeof(amdf_memory_import_info_t)`.
  uint32_t structure_size;
  /// Optional input extension chain. No extensions are currently defined.
  const void* next;
  /// Scope-local profile with the IMPORT role.
  uint32_t memory_profile_ordinal;
  /// Number of entries in accesses.
  uint32_t access_count;
  /// Required backing properties: HOST_VISIBLE, DEVICE_LOCAL and SHAREABLE.
  amdf_memory_flags_t required_flags;
  /// Minimum power-of-two destination device-address alignment, or zero for
  /// profile policy. A nonzero external source offset must be divisible by
  /// this value.
  uint64_t minimum_alignment;
  /// Caller-ordered unique consumers, consumed during the call and not
  /// retained. Each device belongs to the scope's instance and outlives the
  /// memory.
  const amdf_memory_device_access_t* accesses;
} amdf_memory_import_info_t;

/// Parameters used to export one logical memory range.
typedef struct amdf_memory_export_info_t {
  /// Must be `AMDF_STRUCTURE_TYPE_MEMORY_EXPORT_INFO`.
  amdf_structure_type_t type;
  /// Must be at least `sizeof(amdf_memory_export_info_t)`.
  uint32_t structure_size;
  /// Optional input extension chain. No extensions are currently defined.
  const void* next;
  /// Required external-memory representation.
  amdf_external_memory_type_t external_memory_type;
  /// Reserved for future use and always zero.
  uint32_t reserved;
  /// Byte offset within the logical attachment.
  uint64_t byte_offset;
  /// Nonzero logical byte length to export.
  uint64_t byte_length;
} amdf_memory_export_info_t;

/// Parameters used to map a range of host-visible memory.
typedef struct amdf_memory_map_info_t {
  /// Must be `AMDF_STRUCTURE_TYPE_MEMORY_MAP_INFO`.
  amdf_structure_type_t type;
  /// Must be at least `sizeof(amdf_memory_map_info_t)`.
  uint32_t structure_size;
  /// Optional input extension chain. No extensions are currently defined.
  const void* next;
  /// Byte offset within the logical attachment.
  uint64_t byte_offset;
  /// Nonzero byte length of the mapped range.
  uint64_t byte_length;
  /// Required host read and write access.
  amdf_memory_map_flags_t flags;
} amdf_memory_map_info_t;

/// Direction of one explicit host cache ownership transition.
typedef uint32_t amdf_host_cache_operation_t;
enum amdf_host_cache_operation_e {
  /// No host cache operation.
  AMDF_HOST_CACHE_OPERATION_NONE = 0,
  /// Releases prior host writes for subsequent device reads.
  AMDF_HOST_CACHE_OPERATION_FLUSH = 1,
  /// Acquires prior device writes for subsequent host reads.
  AMDF_HOST_CACHE_OPERATION_INVALIDATE = 2,
};

/// Semantic cache operation encoded by an exact engine queue family.
typedef uint32_t amdf_cache_operation_t;
enum amdf_cache_operation_e {
  /// No engine cache operation.
  AMDF_CACHE_OPERATION_NONE = 0,
  /// Releases prior engine writes to the system visibility domain.
  AMDF_CACHE_OPERATION_RELEASE_TO_SYSTEM = 1,
  /// Acquires prior system-visible writes for subsequent engine reads.
  AMDF_CACHE_OPERATION_ACQUIRE_FROM_SYSTEM = 2,
};

/// Set of semantic engine cache operations.
typedef uint64_t amdf_cache_operations_t;
enum amdf_cache_operation_bits_e {
  /// System release operations are supported.
  AMDF_CACHE_OPERATIONS_RELEASE_TO_SYSTEM =
      UINT64_C(1) << AMDF_CACHE_OPERATION_RELEASE_TO_SYSTEM,
  /// System acquire operations are supported.
  AMDF_CACHE_OPERATIONS_ACQUIRE_FROM_SYSTEM =
      UINT64_C(1) << AMDF_CACHE_OPERATION_ACQUIRE_FROM_SYSTEM,
};

/// Granularity of one directional cache transition.
typedef uint32_t amdf_cache_transition_kind_t;
enum amdf_cache_transition_kind_e {
  /// No qualified transition is available.
  AMDF_CACHE_TRANSITION_KIND_UNKNOWN = 0,
  /// The local access requires no cache-maintenance operation.
  AMDF_CACHE_TRANSITION_KIND_NONE = 1,
  /// The transition applies to an explicit byte range.
  AMDF_CACHE_TRANSITION_KIND_RANGE = 2,
  /// The transition applies to the complete native cache domain.
  AMDF_CACHE_TRANSITION_KIND_GLOBAL = 3,
};

/// Set of cache-transition granularities implemented by a queue family.
typedef uint32_t amdf_cache_transition_kinds_t;
enum amdf_cache_transition_kind_bits_e {
  /// Range cache transitions are supported.
  AMDF_CACHE_TRANSITION_KINDS_RANGE = 1u << AMDF_CACHE_TRANSITION_KIND_RANGE,
  /// Global cache transitions are supported.
  AMDF_CACHE_TRANSITION_KINDS_GLOBAL = 1u << AMDF_CACHE_TRANSITION_KIND_GLOBAL,
};

/// Execution site responsible for one cache transition.
typedef uint32_t amdf_cache_transition_executor_t;
enum amdf_cache_transition_executor_e {
  /// No executor is available or required.
  AMDF_CACHE_TRANSITION_EXECUTOR_NONE = 0,
  /// The transition is encoded for an exact queue family.
  AMDF_CACHE_TRANSITION_EXECUTOR_QUEUE = 1,
  /// The transition is encoded in an engine program.
  AMDF_CACHE_TRANSITION_EXECUTOR_PROGRAM = 2,
  /// The host executes the reported instruction and fence directly.
  AMDF_CACHE_TRANSITION_EXECUTOR_HOST_DIRECT = 3,
  /// A provider host API performs the complete transition.
  AMDF_CACHE_TRANSITION_EXECUTOR_HOST_API = 4,
};

/// Public host instruction used by a direct cache transition.
typedef uint32_t amdf_host_cache_instruction_t;
enum amdf_host_cache_instruction_e {
  /// No host cache instruction.
  AMDF_HOST_CACHE_INSTRUCTION_NONE = 0,
  /// The x86 CLFLUSH instruction.
  AMDF_HOST_CACHE_INSTRUCTION_X86_CLFLUSH = 1,
  /// The x86 CLFLUSHOPT instruction.
  AMDF_HOST_CACHE_INSTRUCTION_X86_CLFLUSHOPT = 2,
  /// The x86 CLWB instruction.
  AMDF_HOST_CACHE_INSTRUCTION_X86_CLWB = 3,
};

/// Public host fence completing a direct cache transition.
typedef uint32_t amdf_host_cache_fence_t;
enum amdf_host_cache_fence_e {
  /// No host fence.
  AMDF_HOST_CACHE_FENCE_NONE = 0,
  /// The x86 SFENCE instruction.
  AMDF_HOST_CACHE_FENCE_X86_SFENCE = 1,
  /// The x86 MFENCE instruction.
  AMDF_HOST_CACHE_FENCE_X86_MFENCE = 2,
};

/// Exact operation required for one directional visibility transition.
typedef struct amdf_cache_transition_t {
  /// Qualified no-op, ranged, global, or unavailable transition kind.
  amdf_cache_transition_kind_t kind;
  /// Queue, program, direct-host, host-API, or no-op executor.
  amdf_cache_transition_executor_t executor;
  /// Semantic queue operation, or `AMDF_CACHE_OPERATION_NONE`.
  amdf_cache_operation_t operation;
  /// Host flush or invalidate operation for a host executor.
  amdf_host_cache_operation_t host_operation;
  /// Direct host instruction, or `AMDF_HOST_CACHE_INSTRUCTION_NONE`.
  amdf_host_cache_instruction_t host_instruction;
  /// Host fence required before the first direct instruction.
  amdf_host_cache_fence_t host_fence_before;
  /// Host fence required after the final direct instruction.
  amdf_host_cache_fence_t host_fence_after;
  /// Smallest independently transitionable range, or zero when not ranged.
  /// Host operations cover absolute cache-line boundaries, not logical buffer
  /// offsets. Unaligned ranges can affect neighboring bytes; the caller must
  /// own those boundary lines for the duration of the transition.
  uint64_t range_granularity;
} amdf_cache_transition_t;

/// Immutable properties of one live host mapping.
typedef struct amdf_host_mapping_info_t {
  /// Must be `AMDF_STRUCTURE_TYPE_HOST_MAPPING_INFO`.
  amdf_structure_type_t type;
  /// Must be at least `sizeof(amdf_host_mapping_info_t)`.
  uint32_t structure_size;
  /// Optional output extension chain. No extensions are currently defined.
  void* next;
  /// Achieved host read and write access.
  amdf_memory_map_flags_t flags;
  /// Host cache behavior of the mapped pages.
  amdf_host_cacheability_t cacheability;
  /// First mapped byte borrowed until `host_mapping_destroy` is called.
  void* pointer;
  /// Byte offset of `pointer` within the logical memory attachment.
  uint64_t memory_byte_offset;
  /// Mapped byte length.
  uint64_t byte_length;
  /// Native byte-offset granularity of mapping requests.
  uint64_t byte_offset_granularity;
  /// Native byte-length granularity of mapping requests.
  uint64_t byte_length_granularity;
  /// Host cache-line length in bytes, or zero when not applicable.
  uint32_t cache_line_size;
  /// Reserved for future use and always zero.
  uint32_t reserved;
  /// Available host flush operation, independent of any consumer's coherence.
  /// UNKNOWN means unsupported; NONE means no cache operation is necessary.
  amdf_cache_transition_t flush;
  /// Available host invalidate operation, independent of consumer coherence.
  /// UNKNOWN means unsupported; NONE means no cache operation is necessary.
  amdf_cache_transition_t invalidate;
} amdf_host_mapping_info_t;

/// Kind of concrete access participating in a directional pair query.
typedef uint32_t amdf_memory_site_kind_t;
enum amdf_memory_site_kind_e {
  /// An initialized device access and its exact queue family.
  AMDF_MEMORY_SITE_KIND_DEVICE = 0,
  /// A live host mapping, including its range and CPU cache behavior.
  AMDF_MEMORY_SITE_KIND_HOST = 1,
};

/// One concrete device access or host mapping in a pair query.
typedef struct amdf_memory_site_t {
  /// Must be `AMDF_STRUCTURE_TYPE_MEMORY_SITE`.
  amdf_structure_type_t type;
  /// Must be at least `sizeof(amdf_memory_site_t)`.
  uint32_t structure_size;
  /// Optional input extension chain. No extensions are currently defined.
  const void* next;
  /// Selects the active member of `value`.
  amdf_memory_site_kind_t kind;
  /// Reserved for future use and must be zero.
  uint32_t reserved;
  /// Borrowed access selected by `kind`; its owner must outlive the query.
  union {
    /// Exact initialized device access and execution family.
    struct {
      /// Memory containing the immutable device access.
      amdf_memory_t* memory;
      /// Resource-local access ordinal identifying the consuming device.
      uint32_t access_ordinal;
      /// Exact queue family used for access and cache transitions.
      uint32_t queue_family_ordinal;
    } device;
    /// Host mapping supplying memory identity, range and cache behavior.
    amdf_host_mapping_t* host_mapping;
  } value;
} amdf_memory_site_t;

/// One prospective access within a scope/profile construction contract.
/// DEVICE names an exact queue family; it does not qualify shader/program-side
/// cache operations. HOST names a view created through memory_map on that
/// backing.
typedef struct amdf_memory_profile_site_t {
  /// Selects the active member of value.
  amdf_memory_site_kind_t kind;
  /// Reserved for compatible growth; must be zero.
  uint32_t reserved;
  /// Local execution site or intended host-view permissions.
  union {
    /// Device access in the query's caller-ordered live consumer array.
    struct {
      /// Index into the query's accesses array.
      uint32_t access_ordinal;
      /// Exact queue family of that device's endpoint.
      uint32_t queue_family_ordinal;
    } device;
    /// Required host read and write permissions.
    amdf_memory_map_flags_t host_access;
  } value;
} amdf_memory_profile_site_t;

/// Construction facts and exact sites for a pre-allocation visibility query.
/// This borrows its inputs only for the call and creates no retained plan
/// owner.
typedef struct amdf_memory_profile_pair_query_t {
  /// Must be AMDF_STRUCTURE_TYPE_MEMORY_PROFILE_PAIR_QUERY.
  amdf_structure_type_t type;
  /// Must be at least sizeof(amdf_memory_profile_pair_query_t).
  uint32_t structure_size;
  /// Optional input extension chain. No extensions are currently defined.
  const void* next;
  /// Scope-local profile ordinal, as used by memory_create or memory_import.
  uint32_t memory_profile_ordinal;
  /// Complete number of live consumers; zero selects CPU-only system memory.
  uint32_t access_count;
  /// Required backing properties, identical to the construction request.
  amdf_memory_flags_t required_flags;
  /// Caller-ordered live consumers, as used by construction; NULL at zero
  /// count.
  const amdf_memory_device_access_t* accesses;
  /// Source mapping cache class for REGISTER; UNKNOWN for other operations.
  amdf_host_cacheability_t registered_host_cacheability;
  /// Reserved for compatible growth; must be zero.
  uint32_t reserved;
  /// Exact import transport, or zero for CREATE and REGISTER profiles.
  amdf_external_memory_type_t external_memory_type;
  /// Reserved for compatible growth; must be zero.
  uint32_t external_memory_reserved;
  /// Import protocol identity; all zero for portable transports and
  /// non-imports.
  amdf_external_memory_provenance_t external_memory_provenance;
  /// Site performing writes before the caller's ordering edge.
  amdf_memory_profile_site_t producer;
  /// Site performing reads after the caller's ordering edge.
  amdf_memory_profile_site_t consumer;
} amdf_memory_profile_pair_query_t;

/// Directional capabilities of one concrete shared-backing memory pair.
typedef uint64_t amdf_memory_pair_flags_t;
enum amdf_memory_pair_flag_bits_e {
  /// The consumer execution site can directly reach the shared backing.
  AMDF_MEMORY_PAIR_FLAG_SHARED_BACKING_REACHABLE = UINT64_C(1) << 0,
  /// The producer attachment can source consumer virtual-memory mappings.
  AMDF_MEMORY_PAIR_FLAG_MAPPING_SOURCE = UINT64_C(1) << 1,
  /// `estimated_fixed_cost_nanoseconds` is a qualified value, including zero.
  AMDF_MEMORY_PAIR_FLAG_FIXED_COST_KNOWN = UINT64_C(1) << 2,
};

/// Exact directional relation from one producer attachment to one consumer.
typedef struct amdf_memory_pair_info_t {
  /// Must be `AMDF_STRUCTURE_TYPE_MEMORY_PAIR_INFO`.
  amdf_structure_type_t type;
  /// Must be at least `sizeof(amdf_memory_pair_info_t)`.
  uint32_t structure_size;
  /// Optional output extension chain. No extensions are currently defined.
  void* next;
  /// Direct reach and mapping-source capabilities.
  amdf_memory_pair_flags_t flags;
  /// Visibility operation performed after producer writes.
  amdf_cache_transition_t release;
  /// Visibility operation performed before consumer reads.
  amdf_cache_transition_t acquire;
  /// Width-specific mutually atomic reach shared by both execution sites. This
  /// does not imply operation support; callers also intersect both the queue
  /// family and target-memory operation masks.
  amdf_atomic_reach_t atomic_reach;
  /// Informational fixed transition cost in nanoseconds. The value is valid
  /// only when `AMDF_MEMORY_PAIR_FLAG_FIXED_COST_KNOWN` is set.
  uint64_t estimated_fixed_cost_nanoseconds;
} amdf_memory_pair_info_t;

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // AMDF_MEMORY_H_
