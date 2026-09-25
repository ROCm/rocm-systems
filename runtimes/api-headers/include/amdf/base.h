// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef AMDF_BASE_H_
#define AMDF_BASE_H_

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>

/// Compile-time assertion policy, independent of the C library's `NDEBUG`.
///
/// Set `AMDF_ENABLE_ASSERTS=0` as a compiler definition to remove invariant
/// checks. Expressions remain parsed but are not evaluated when disabled.
#if !defined(AMDF_ENABLE_ASSERTS)
#define AMDF_ENABLE_ASSERTS 1
#endif

/// Terminates the process after an invariant violation.
static inline void amdf_abort(void) { abort(); }

#if AMDF_ENABLE_ASSERTS
#define amdf_assert(condition)      \
  do {                              \
    if (!(condition)) amdf_abort(); \
  } while (0)
#else
#define amdf_assert(condition)   \
  do {                           \
    (void)sizeof(!!(condition)); \
  } while (0)
#endif

/// Returns the ABI alignment of a type as a compile-time constant.
#if defined(_MSC_VER)
#define amdf_alignof(type) __alignof(type)
/// Natural maximum scalar alignment. MSVC omits C's max_align_t; long double
/// has the maximum scalar alignment in the Microsoft ABI.
#define amdf_max_align_t amdf_alignof(long double)
#else
#define amdf_alignof(type) __alignof__(type)
/// Natural maximum scalar alignment, shared by C and C++ allocation callbacks.
#define amdf_max_align_t amdf_alignof(max_align_t)
#endif

#if defined(_WIN32)
#define AMDF_CALL __cdecl
#if defined(AMDF_SHARED_LIBRARY)
#define AMDF_API __declspec(dllimport)
#else
#define AMDF_API
#endif
#else
#define AMDF_CALL
#if defined(__GNUC__) || defined(__clang__)
#define AMDF_API __attribute__((visibility("default")))
#else
#define AMDF_API
#endif
#endif

#ifdef __cplusplus
extern "C" {
#endif

/// Version number identifying compatible public API tables and structures.
typedef uint32_t amdf_abi_version_t;

/// The original libamdf ABI version.
#define AMDF_ABI_VERSION_1 ((amdf_abi_version_t)1)

/// Memory visibility qualification and explicit registered-page cacheability.
#define AMDF_ABI_VERSION_2 ((amdf_abi_version_t)2)

/// Exact prospective CREATE payload geometry in construction capabilities.
#define AMDF_ABI_VERSION_3 ((amdf_abi_version_t)3)

/// The most recent ABI version described by this header.
#define AMDF_ABI_VERSION_LATEST AMDF_ABI_VERSION_3

/// The unmangled symbol used to acquire the immutable API table.
#define AMDF_QUERY_API_SYMBOL "amdf_query_api"

/// An encoded status domain and domain-specific 32-bit code.
typedef uint64_t amdf_status_t;

/// Indicates successful completion.
#define AMDF_STATUS_OK ((amdf_status_t)0)

/// Status-code domain identifier.
typedef uint32_t amdf_status_domain_t;
enum amdf_status_domain_e {
  /// Portable status codes declared by this header.
  AMDF_STATUS_DOMAIN_API = 0,
  /// Windows NTSTATUS values.
  AMDF_STATUS_DOMAIN_NTSTATUS = 1,
  /// Device-firmware status values.
  AMDF_STATUS_DOMAIN_FIRMWARE = 2,
  /// POSIX errno values.
  AMDF_STATUS_DOMAIN_ERRNO = 3,
  /// Windows Win32 error values returned by `GetLastError`.
  AMDF_STATUS_DOMAIN_WIN32 = 4,
  /// Failed Windows COM/DXGI/D3D HRESULT values.
  AMDF_STATUS_DOMAIN_HRESULT = 5,
};

/// Portable status code used with `AMDF_STATUS_DOMAIN_API`.
typedef uint32_t amdf_status_code_t;
enum amdf_status_code_e {
  /// The operation completed successfully.
  AMDF_STATUS_CODE_OK = 0,
  /// An argument was malformed or violated a precondition of its type.
  AMDF_STATUS_CODE_INVALID_ARGUMENT = 1,
  /// A numeric argument was outside the accepted range.
  AMDF_STATUS_CODE_OUT_OF_RANGE = 2,
  /// The requested operation or capability is not implemented.
  AMDF_STATUS_CODE_UNSUPPORTED = 3,
  /// The requested object or capability was not found.
  AMDF_STATUS_CODE_NOT_FOUND = 4,
  /// A finite resource required by the operation was exhausted.
  AMDF_STATUS_CODE_RESOURCE_EXHAUSTED = 5,
  /// A resource is temporarily owned by another operation.
  AMDF_STATUS_CODE_BUSY = 6,
  /// The operation did not complete before its caller-provided deadline.
  AMDF_STATUS_CODE_DEADLINE_EXCEEDED = 7,
  /// The process lacks permission for the requested operation.
  AMDF_STATUS_CODE_PERMISSION_DENIED = 8,
  /// The device or its owning driver can no longer execute work.
  AMDF_STATUS_CODE_DEVICE_LOST = 9,
  /// The caller and library have no mutually supported ABI version.
  AMDF_STATUS_CODE_VERSION_MISMATCH = 10,
  /// Caller-provided storage is too small for the requested result.
  AMDF_STATUS_CODE_BUFFER_TOO_SMALL = 11,
  /// Valid arguments describe an operation that cannot begin in this state.
  AMDF_STATUS_CODE_FAILED_PRECONDITION = 12,
  /// An invariant failed inside the implementation.
  AMDF_STATUS_CODE_INTERNAL = 13,
};

/// Encodes one domain-specific status code. Code zero maps to
/// `AMDF_STATUS_OK` in every domain.
static inline amdf_status_t amdf_make_status(amdf_status_domain_t domain,
                                             uint32_t code) {
  return code == 0 ? AMDF_STATUS_OK : ((amdf_status_t)domain << 32) | code;
}

/// Encodes one portable libamdf status code.
static inline amdf_status_t amdf_make_api_status(amdf_status_code_t code) {
  return amdf_make_status(AMDF_STATUS_DOMAIN_API, code);
}

/// Returns true when status represents success.
static inline bool amdf_status_is_ok(amdf_status_t status) {
  return status == AMDF_STATUS_OK;
}

/// Returns the domain encoded in status.
static inline amdf_status_domain_t amdf_status_domain(amdf_status_t status) {
  return (amdf_status_domain_t)(status >> 32);
}

/// Returns the domain-specific code encoded in status.
static inline uint32_t amdf_status_code(amdf_status_t status) {
  return (uint32_t)status;
}

/// Type identifier carried by every extensible API structure.
typedef uint32_t amdf_structure_type_t;
enum amdf_structure_type_e {
  /// No structure type. This value is never accepted by an API operation.
  AMDF_STRUCTURE_TYPE_NONE = 0,
  /// An `amdf_instance_create_info_t` input structure.
  AMDF_STRUCTURE_TYPE_INSTANCE_CREATE_INFO = 1,
  /// An `amdf_endpoint_info_t` output structure.
  AMDF_STRUCTURE_TYPE_ENDPOINT_INFO = 2,
  /// An `amdf_queue_family_info_t` output structure.
  AMDF_STRUCTURE_TYPE_QUEUE_FAMILY_INFO = 3,
  /// An `amdf_memory_create_info_t` input structure.
  AMDF_STRUCTURE_TYPE_MEMORY_CREATE_INFO = 4,
  /// An `amdf_memory_info_t` output structure.
  AMDF_STRUCTURE_TYPE_MEMORY_INFO = 5,
  /// An `amdf_memory_map_info_t` input structure.
  AMDF_STRUCTURE_TYPE_MEMORY_MAP_INFO = 6,
  /// An `amdf_host_mapping_info_t` output structure.
  AMDF_STRUCTURE_TYPE_HOST_MAPPING_INFO = 7,
  /// An `amdf_kernel_queue_info_t` output structure.
  AMDF_STRUCTURE_TYPE_KERNEL_QUEUE_INFO = 8,
  /// An `amdf_kernel_queue_status_t` output structure.
  AMDF_STRUCTURE_TYPE_KERNEL_QUEUE_STATUS = 9,
  /// An `amdf_memory_profile_t` output structure.
  AMDF_STRUCTURE_TYPE_MEMORY_PROFILE = 10,
  /// An `amdf_memory_import_info_t` input structure.
  AMDF_STRUCTURE_TYPE_MEMORY_IMPORT_INFO = 11,
  /// An `amdf_memory_export_info_t` input structure.
  AMDF_STRUCTURE_TYPE_MEMORY_EXPORT_INFO = 12,
  /// An `amdf_memory_site_t` input structure.
  AMDF_STRUCTURE_TYPE_MEMORY_SITE = 13,
  /// An `amdf_memory_pair_info_t` output structure.
  AMDF_STRUCTURE_TYPE_MEMORY_PAIR_INFO = 14,
  /// An `amdf_user_queue_info_t` output structure.
  AMDF_STRUCTURE_TYPE_USER_QUEUE_INFO = 15,
  /// An `amdf_user_queue_mapping_info_t` output structure.
  AMDF_STRUCTURE_TYPE_USER_QUEUE_MAPPING_INFO = 16,
  /// An `amdf_user_queue_status_t` output structure.
  AMDF_STRUCTURE_TYPE_USER_QUEUE_STATUS = 17,
  /// An `amdf_memory_access_info_t` output structure.
  AMDF_STRUCTURE_TYPE_MEMORY_ACCESS_INFO = 18,
  /// An `amdf_memory_scope_info_t` output structure.
  AMDF_STRUCTURE_TYPE_MEMORY_SCOPE_INFO = 19,
  /// An `amdf_memory_access_capabilities_t` output structure.
  AMDF_STRUCTURE_TYPE_MEMORY_ACCESS_CAPABILITIES = 20,
  /// An `amdf_memory_profile_pair_query_t` input structure.
  AMDF_STRUCTURE_TYPE_MEMORY_PROFILE_PAIR_QUERY = 21,
};

/// Identifier of an optional API table compiled into the providing library.
typedef uint32_t amdf_extension_id_t;
enum amdf_extension_id_e {
  /// XDNA endpoint qualification and execution services.
  AMDF_EXTENSION_XDNA = 1,
  /// GPU endpoint qualification and execution services.
  AMDF_EXTENSION_GPU = 2,
};

/// Common prefix of every extensible input structure.
typedef struct amdf_input_structure_t {
  /// Type identifying the complete structure.
  amdf_structure_type_t type;
  /// Size in bytes of the complete structure supplied by the caller.
  uint32_t structure_size;
  /// Optional pointer to the next extensible input structure.
  const void* next;
} amdf_input_structure_t;

/// Common prefix of every extensible output structure.
typedef struct amdf_output_structure_t {
  /// Type identifying the complete structure.
  amdf_structure_type_t type;
  /// Size in bytes of the complete structure supplied by the caller.
  uint32_t structure_size;
  /// Optional pointer to the next extensible output structure.
  void* next;
} amdf_output_structure_t;

/// Explicit provider instance owning loaded modules and native API state.
typedef struct amdf_instance_t amdf_instance_t;

/// Query-only handle to one independently selectable execution endpoint.
typedef struct amdf_endpoint_t amdf_endpoint_t;

/// Live engine context and address domain materialized from an endpoint.
typedef struct amdf_device_t amdf_device_t;

/// Borrowed storage contract embedded in an instance, endpoint or native owner.
typedef struct amdf_memory_scope_t amdf_memory_scope_t;

/// Physical backing and its established device accesses.
typedef struct amdf_memory_t amdf_memory_t;

/// Explicit host access to one range of host-visible memory.
typedef struct amdf_host_mapping_t amdf_host_mapping_t;

/// Directly published native queue owned by one materialized device.
typedef struct amdf_user_queue_t amdf_user_queue_t;

/// Producer-local mapping of one directly published native queue.
typedef struct amdf_user_queue_mapping_t amdf_user_queue_mapping_t;

/// Kernel-mediated publication and retirement of native commands.
typedef struct amdf_kernel_queue_t amdf_kernel_queue_t;

/// Allocates one uninitialized host-memory range.
///
/// `byte_length` is nonzero. `minimum_alignment` is a power of two and at
/// least the platform's natural maximum alignment. Returns a suitably aligned
/// allocation or `NULL` when storage is unavailable. The callback must be
/// thread-safe and must not throw or unwind across the C ABI.
typedef void*(AMDF_CALL* amdf_allocator_allocate_fn_t)(
    void* user_data, uint64_t byte_length, uint64_t minimum_alignment);

/// Resizes one host-memory allocation without changing it on failure.
///
/// `allocation` was returned by the same allocator. Success preserves the
/// lesser of `old_byte_length` and `new_byte_length` bytes and returns a
/// suitably aligned allocation. Returning `NULL` leaves `allocation` live and
/// unchanged. The callback must be thread-safe and must not throw or unwind
/// across the C ABI.
typedef void*(AMDF_CALL* amdf_allocator_resize_fn_t)(
    void* user_data, void* allocation, uint64_t old_byte_length,
    uint64_t new_byte_length, uint64_t minimum_alignment);

/// Frees one host-memory allocation returned by the same allocator.
///
/// `allocation` may be `NULL`. The callback is infallible, must be thread-safe,
/// and must not throw or unwind across the C ABI.
typedef void(AMDF_CALL* amdf_allocator_free_fn_t)(void* user_data,
                                                  void* allocation);

/// Instance-scoped host-memory allocator used by libamdf-owned metadata.
///
/// An all-zero value selects the built-in system allocator. Otherwise
/// `allocate` and `free` are required. `resize` may be `NULL`, in which case
/// libamdf implements resize with allocate-copy-free. The value is copied when
/// the instance is created; `user_data` and callback code must remain valid
/// until instance destruction succeeds. Native device backing and operating
/// system allocations do not use this allocator.
typedef struct amdf_allocator_t {
  /// Opaque value passed to every callback.
  void* user_data;
  /// Required uninitialized allocation callback.
  amdf_allocator_allocate_fn_t allocate;
  /// Optional allocation-resize callback.
  amdf_allocator_resize_fn_t resize;
  /// Required infallible release callback.
  amdf_allocator_free_fn_t free;
} amdf_allocator_t;

/// Maximum lifetime permitted for native state acquired by an instance.
///
/// Resources may be released earlier. This policy neither changes public
/// handle ownership nor promises a separate virtual address space, exclusive
/// hardware access, or automatic interoperability with another native client.
typedef uint32_t amdf_native_lifetime_t;
enum amdf_native_lifetime_e {
  /// Permits native process-owned state to survive instance destruction until
  /// process exit. This is the default and enables Linux KFD host registration.
  /// KFD's primary VM binding requires coordination with other KFD clients;
  /// after instance destruction a fresh acquisition can fail with native EBUSY.
  AMDF_NATIVE_LIFETIME_PROCESS = 0,
  /// Requires acquired native state to be releasable by instance destruction.
  /// A provider that cannot meet this bound reports UNSUPPORTED when its
  /// services are queried or constructed; it never falls back to PROCESS.
  /// Linux KFD secondary contexts currently lack host registration. Windows
  /// can reclaim native objects under either policy without isolating GPUVA.
  AMDF_NATIVE_LIFETIME_INSTANCE = 1,
};

/// Parameters used to create a provider instance without activating devices.
typedef struct amdf_instance_create_info_t {
  /// Must be `AMDF_STRUCTURE_TYPE_INSTANCE_CREATE_INFO`.
  amdf_structure_type_t type;
  /// Must be at least `sizeof(amdf_instance_create_info_t)`.
  uint32_t structure_size;
  /// Optional input extension chain. No extensions are defined in ABI v1.
  const void* next;
  /// Native lifetime policy copied by the instance; zero selects PROCESS.
  amdf_native_lifetime_t native_lifetime;
  /// Reserved for compatible growth and must be zero.
  uint32_t reserved;
  /// Host allocator copied and used by this instance and all of its children.
  amdf_allocator_t host_allocator;
} amdf_instance_create_info_t;

/// Opaque provider identity used to reopen one enumerated endpoint.
///
/// The value is meaningful only on the machine and provider implementation
/// that produced it. It is not a persistent machine identifier and may become
/// stale after device removal, reset, disable/enable, or driver replacement.
typedef struct amdf_endpoint_id_t {
  /// Provider-defined identity words.
  uint64_t words[2];
} amdf_endpoint_id_t;

/// Returns true when two endpoint identities contain the same opaque value.
static inline bool amdf_endpoint_id_is_equal(const amdf_endpoint_id_t* lhs,
                                             const amdf_endpoint_id_t* rhs) {
  return lhs->words[0] == rhs->words[0] && lhs->words[1] == rhs->words[1];
}

/// Opaque identity of one live materialized device.
///
/// The value is meaningful only while the device and its provider instance
/// remain live. It is intended for correlation and compatibility checks, not
/// persistence or native-handle recovery.
typedef struct amdf_device_id_t {
  /// Provider-defined identity words.
  uint64_t words[2];
} amdf_device_id_t;

/// Returns true when two device identities contain the same opaque value.
static inline bool amdf_device_id_is_equal(const amdf_device_id_t* lhs,
                                           const amdf_device_id_t* rhs) {
  return lhs->words[0] == rhs->words[0] && lhs->words[1] == rhs->words[1];
}

/// Opaque identity of one live native queue.
///
/// The value remains meaningful through the lifetime of the queue and is used
/// only for correlation and failure attribution. It is never a native handle.
typedef struct amdf_queue_id_t {
  /// Provider-defined identity words.
  uint64_t words[2];
} amdf_queue_id_t;

/// Returns true when a queue identity is available.
static inline bool amdf_queue_id_is_valid(const amdf_queue_id_t* id) {
  return (id->words[0] | id->words[1]) != 0;
}

/// Returns true when two queue identities contain the same opaque value.
static inline bool amdf_queue_id_is_equal(const amdf_queue_id_t* lhs,
                                          const amdf_queue_id_t* rhs) {
  return lhs->words[0] == rhs->words[0] && lhs->words[1] == rhs->words[1];
}

/// Opaque identity of live physical backing within one provider instance.
///
/// Equal nonzero identities prove that two attachments name the same physical
/// backing. An all-zero identity means that the provider cannot establish
/// physical identity. The value is not persistent and is never a native
/// handle.
typedef struct amdf_physical_memory_id_t {
  /// Provider-defined identity words.
  uint64_t words[2];
} amdf_physical_memory_id_t;

/// Returns true when a physical-memory identity is available.
static inline bool amdf_physical_memory_id_is_valid(
    const amdf_physical_memory_id_t* id) {
  return (id->words[0] | id->words[1]) != 0;
}

/// Returns true when two physical-memory identities contain the same value.
static inline bool amdf_physical_memory_id_is_equal(
    const amdf_physical_memory_id_t* lhs,
    const amdf_physical_memory_id_t* rhs) {
  return lhs->words[0] == rhs->words[0] && lhs->words[1] == rhs->words[1];
}

/// Broad execution-engine class of an endpoint.
typedef uint32_t amdf_engine_kind_t;
enum amdf_engine_kind_e {
  /// The provider cannot classify the endpoint without engine qualification.
  AMDF_ENGINE_KIND_UNKNOWN = 0,
  /// An AMD GPU endpoint, including RDNA and CDNA targets.
  AMDF_ENGINE_KIND_GPU = 1,
  /// An AMD XDNA/AIE endpoint.
  AMDF_ENGINE_KIND_XDNA = 2,
};

/// Standard operating-system properties of an endpoint.
typedef uint32_t amdf_endpoint_type_flags_t;
enum amdf_endpoint_type_flag_bits_e {
  /// The endpoint can drive a display.
  AMDF_ENDPOINT_TYPE_FLAG_DISPLAY_SUPPORTED = 1u << 0,
  /// The endpoint supports graphics rendering.
  AMDF_ENDPOINT_TYPE_FLAG_RENDER_SUPPORTED = 1u << 1,
  /// The operating system reports a compute-only adapter.
  AMDF_ENDPOINT_TYPE_FLAG_COMPUTE_ONLY = 1u << 2,
  /// The endpoint is implemented entirely in software.
  AMDF_ENDPOINT_TYPE_FLAG_SOFTWARE_DEVICE = 1u << 3,
};

/// Capacity in bytes of a NUL-terminated endpoint diagnostic name.
#define AMDF_ENDPOINT_NAME_CAPACITY 128u

/// Immutable PCI identity of an endpoint, when reported by the platform.
typedef struct amdf_pci_info_t {
  /// PCI vendor identifier, or zero when unavailable.
  uint32_t vendor_id;
  /// PCI device identifier, or zero when unavailable.
  uint32_t device_id;
  /// PCI subsystem vendor identifier, or zero when unavailable.
  uint32_t subsystem_vendor_id;
  /// PCI subsystem device identifier, or zero when unavailable.
  uint32_t subsystem_device_id;
  /// PCI revision identifier, or zero when unavailable.
  uint32_t revision_id;
} amdf_pci_info_t;

/// Native namespace used to correlate an endpoint with another API provider.
typedef uint32_t amdf_endpoint_native_identity_type_t;
enum amdf_endpoint_native_identity_type_e {
  /// No native correlation identity is available.
  AMDF_ENDPOINT_NATIVE_IDENTITY_TYPE_NONE = 0,
  /// Linux character-device major/minor numbers.
  AMDF_ENDPOINT_NATIVE_IDENTITY_TYPE_LINUX_DEVICE = 1,
  /// Windows adapter LUID and physical-adapter index.
  AMDF_ENDPOINT_NATIVE_IDENTITY_TYPE_WINDOWS_ADAPTER = 2,
};

/// Passive identity for matching contemporaneous native API providers.
///
/// This value conveys neither a native handle nor ownership. It is not stable
/// across device removal or reboot. Matching identifies the native device;
/// foreign-driver, memory-transport and access compatibility still require
/// their own qualification. The opaque endpoint id remains the value used to
/// open libamdf endpoints.
typedef struct amdf_endpoint_native_identity_t {
  /// Native namespace selecting the active member of `value`.
  amdf_endpoint_native_identity_type_t type;
  /// Native identity in the namespace selected by `type`.
  union {
    /// Linux DRM render node for GPU endpoints, accelerator node for XDNA.
    struct {
      /// Character-device major number, as returned by `major(st_rdev)`.
      uint32_t major;
      /// Character-device minor number, as returned by `minor(st_rdev)`.
      uint32_t minor;
    } linux_device;
    /// Windows logical adapter and the endpoint's physical adapter within it.
    struct {
      /// LUID bits: `(uint64_t)(uint32_t)HighPart << 32 | LowPart`.
      uint64_t luid;
      /// KMT physical-adapter index within the logical adapter.
      uint32_t physical_adapter_index;
    } windows_adapter;
  } value;
} amdf_endpoint_native_identity_t;

/// Fixed-stride endpoint identity returned by `endpoint_enumerate`.
///
/// This structure is immutable for ABI v1. It contains no pointers, extension
/// chain, or provider-owned storage so arrays always have one stable stride.
typedef struct amdf_endpoint_summary_t {
  /// Opaque identity accepted by `endpoint_open`.
  amdf_endpoint_id_t id;
  /// Broad engine class, or `AMDF_ENGINE_KIND_UNKNOWN` when not yet qualified.
  amdf_engine_kind_t engine_kind;
  /// Standard endpoint properties.
  amdf_endpoint_type_flags_t type_flags;
  /// NUL-terminated UTF-8 diagnostic name. Never use this for classification.
  char name[AMDF_ENDPOINT_NAME_CAPACITY];
} amdf_endpoint_summary_t;

/// Immutable properties of one opened endpoint.
typedef struct amdf_endpoint_info_t {
  /// Must be `AMDF_STRUCTURE_TYPE_ENDPOINT_INFO`.
  amdf_structure_type_t type;
  /// Must be at least `sizeof(amdf_endpoint_info_t)`.
  uint32_t structure_size;
  /// Optional output extension chain. No extensions are defined in ABI v1.
  void* next;
  /// Opaque identity used to open this endpoint.
  amdf_endpoint_id_t id;
  /// Broad engine class, or `AMDF_ENGINE_KIND_UNKNOWN` when not yet qualified.
  amdf_engine_kind_t engine_kind;
  /// Standard endpoint properties.
  amdf_endpoint_type_flags_t type_flags;
  /// Immutable PCI identity reported by the platform.
  amdf_pci_info_t pci;
  /// NUL-terminated UTF-8 diagnostic name. Never use this for classification.
  char name[AMDF_ENDPOINT_NAME_CAPACITY];
  /// Number of immutable endpoint-local native queue families.
  uint32_t queue_family_count;
  /// Cached native identity for matching other API providers before activation.
  amdf_endpoint_native_identity_t native_identity;
} amdf_endpoint_info_t;

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // AMDF_BASE_H_
