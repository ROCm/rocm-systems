// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef AMDF_API_H_
#define AMDF_API_H_

#include "amdf/base.h"
#include "amdf/memory.h"
#include "amdf/queue.h"

#ifdef __cplusplus
extern "C" {
#endif

/// Immutable entry-point table for one negotiated ABI version.
///
/// Tables grow only by appending fields. The table and every function pointer
/// reachable from it remain valid until the providing library is unloaded.
///
/// Per-method cost guarantees apply to every call, including first use and
/// failure paths. They constrain libamdf-owned work, not OS scheduling or the
/// internals of a native driver call explicitly required by the operation.
/// Thread safety does not authorize hidden locking, allocation or lazy setup
/// on methods that exclude those costs. Resource creation owns preparation;
/// metadata access, publication and waiting retain their separate contracts.
typedef struct amdf_api_t {
  /// Size in bytes of this table version.
  uint32_t structure_size;
  /// ABI version implemented by this table.
  amdf_abi_version_t abi_version;

  /// Creates an independent provider instance.
  ///
  /// The instance owns every dependent library reference and resolved native
  /// procedure table used by its children. Creation is thread-safe and performs
  /// bounded constant work. It performs no endpoint enumeration, device or
  /// firmware initialization, worker creation, retry, sleep, or process-global
  /// initialization. Failure leaves `out_instance` unchanged.
  amdf_status_t(AMDF_CALL* instance_create)(
      const amdf_instance_create_info_t* create_info,
      amdf_instance_t** out_instance);

  /// Destroys an instance after all of its children have been closed.
  ///
  /// The caller must have exclusive access. Returns
  /// `AMDF_STATUS_CODE_BUSY` without mutation while a child remains open.
  /// Destruction performs no implicit device wait. A native teardown failure
  /// retains the instance only for another destruction attempt; some native
  /// connections may already have been released.
  amdf_status_t(AMDF_CALL* instance_destroy)(amdf_instance_t* instance);

  /// Enumerates a bounded snapshot of independently selectable AMD endpoints.
  ///
  /// A zero `capacity` queries the total count and may pass `summaries` as
  /// `NULL`. A nonzero capacity requires `summaries` to reference that many
  /// elements. When capacity is insufficient, the available prefix is written,
  /// `out_count` receives the total, and `AMDF_STATUS_CODE_BUFFER_TOO_SMALL` is
  /// returned. Every other failure leaves both outputs unchanged. The call
  /// creates no device, paging queue, address space, context, allocation,
  /// executable, or hardware queue. Arrival or removal may change the result of
  /// a later call.
  amdf_status_t(AMDF_CALL* endpoint_enumerate)(
      amdf_instance_t* instance, uint32_t capacity,
      amdf_endpoint_summary_t* summaries, uint32_t* out_count);

  /// Opens one endpoint identity without enumerating the machine again.
  ///
  /// The returned query-only endpoint borrows `instance`; the instance must
  /// outlive it. Opening may acquire a native query handle and cache immutable
  /// identity, but creates no schedulable device state. A stale identity fails
  /// rather than selecting another endpoint. Failure leaves `out_endpoint`
  /// unchanged.
  amdf_status_t(AMDF_CALL* endpoint_open)(amdf_instance_t* instance,
                                          const amdf_endpoint_id_t* id,
                                          amdf_endpoint_t** out_endpoint);

  /// Copies immutable cached properties without a system call or device wait.
  ///
  /// The operation is thread-safe and performs no allocation, locking or lazy
  /// initialization. It reads the snapshot established by endpoint_open, not
  /// live driver state. The caller initializes `out_info` and its
  /// complete extension chain before the call. No output is modified when
  /// validation fails.
  amdf_status_t(AMDF_CALL* endpoint_query_info)(amdf_endpoint_t* endpoint,
                                                amdf_endpoint_info_t* out_info);

  /// Closes a query-only endpoint after all future children are destroyed.
  ///
  /// The caller must have exclusive access. The operation performs no implicit
  /// device wait. Failure leaves the endpoint live so destruction can be
  /// retried.
  amdf_status_t(AMDF_CALL* endpoint_close)(amdf_endpoint_t* endpoint);

  /// Acquires an immutable optional API table compiled into this library.
  ///
  /// Extension availability describes the library composition and never
  /// depends on endpoint enumeration or active hardware. Hardware support is
  /// reported by operations on the returned table. `minimum_version` and
  /// `maximum_version` form an inclusive range. An unknown or excluded
  /// extension returns `AMDF_STATUS_CODE_UNSUPPORTED`; a compiled extension
  /// with no version in range returns `AMDF_STATUS_CODE_VERSION_MISMATCH`.
  /// Failure leaves `out_extension_api` unchanged when it is non-NULL.
  ///
  /// This operation is thread-safe, bounded constant time, and inert. It
  /// performs no allocation, system call, device discovery, dependent-library
  /// load, or other observable initialization. The returned table remains
  /// valid until the providing library is unloaded. No lock or one-time
  /// initialization guard is acquired, including on the first call.
  amdf_status_t(AMDF_CALL* query_extension)(amdf_extension_id_t extension_id,
                                            uint32_t minimum_version,
                                            uint32_t maximum_version,
                                            const void** out_extension_api);

  /// Copies one immutable queue-family record cached while opening `endpoint`.
  ///
  /// Records describe expected implemented services. Explicit native device
  /// creation validates the installed driver before making them usable.
  /// `queue_family_ordinal` must be less than the endpoint's reported family
  /// count. The operation is thread-safe and performs no system call,
  /// allocation, device initialization, queue creation, retry, sleep, or
  /// device wait. It acquires no lock and initializes no cached state. The
  /// caller initializes `out_info` and its complete extension
  /// chain. No output is modified on failure.
  amdf_status_t(AMDF_CALL* endpoint_query_queue_family_info)(
      amdf_endpoint_t* endpoint, uint32_t queue_family_ordinal,
      amdf_queue_family_info_t* out_info);

  /// Destroys a materialized device after its dependent resources are released.
  ///
  /// The caller must have exclusive access. Returns
  /// `AMDF_STATUS_CODE_BUSY` without native mutation while a queue or context
  /// remains live. Memory lifetimes are caller preconditions:
  /// dependent memory must already be destroyed, and libamdf neither retains
  /// the device through memory nor tracks memory to diagnose premature device
  /// destruction. Violating that precondition is undefined behavior.
  /// Caller-submitted work must already be retired before its owning resources
  /// are destroyed. A native teardown failure retains the device only for
  /// another destruction attempt; some native execution resources may already
  /// have been released.
  amdf_status_t(AMDF_CALL* device_destroy)(amdf_device_t* device);

  /// Enumerates borrowed system-storage scopes owned by the instance.
  ///
  /// This metadata-only operation activates no device. Capacity zero permits
  /// NULL storage. Success and BUFFER_TOO_SMALL publish the available prefix
  /// and full required count together; every other failure leaves outputs
  /// unchanged. A nonempty zero-capacity query returns BUFFER_TOO_SMALL.
  /// Returned scopes remain valid while the instance lives. Scope descriptors
  /// are prepared by the owner; enumeration allocates nothing and performs no
  /// locking, native query or lazy initialization.
  amdf_status_t(AMDF_CALL* instance_enumerate_memory_scopes)(
      amdf_instance_t* instance, uint32_t capacity,
      amdf_memory_scope_t** scopes, uint32_t* out_count);

  /// Enumerates physical-local storage available through a live device.
  ///
  /// The count/prefix protocol matches instance_enumerate_memory_scopes.
  /// Returned physical scopes are owned by the device's endpoint and remain
  /// valid while that endpoint lives. Allocations require their storage device
  /// in the live access set. Context-private scopes come from their extension.
  /// This reads achieved profiles without allocation, native queries, locking
  /// or lazy setup; it creates no resources.
  amdf_status_t(AMDF_CALL* device_enumerate_memory_scopes)(
      amdf_device_t* device, uint32_t capacity, amdf_memory_scope_t** scopes,
      uint32_t* out_count);

  /// Copies complete immutable facts of a borrowed scope.
  ///
  /// This thread-safe metadata query performs no native operation or
  /// allocation, locking or lazy initialization. The caller initializes the
  /// output header and extension chain. Failure leaves all output bytes
  /// unchanged.
  amdf_status_t(AMDF_CALL* memory_scope_query_info)(
      amdf_memory_scope_t* scope, amdf_memory_scope_info_t* out_info);

  /// Queries a scope contract for explicitly initialized device consumers.
  ///
  /// Devices are unique and belong to the scope's instance. The caller keeps
  /// them live during this call. Zero count permits NULL arrays and requests
  /// CPU-only storage without activating any accelerator. The caller
  /// initializes the profile and exactly access_count capability records.
  /// Success publishes complete backing facts and one capability record per
  /// input in caller order. Failure leaves all outputs unchanged. An ordinal
  /// outside the scope's profile count returns OUT_OF_RANGE; a valid profile
  /// unable to satisfy the complete access set returns UNSUPPORTED.
  ///
  /// This thread-safe cold planning query consumes native facts retained at
  /// activation. It performs no native operation, mapping or synchronization.
  /// Temporary host storage scales with the access set. The result is not a
  /// reservation: construction can fail from exhaustion or native errors and
  /// never weakens explicit requirements.
  amdf_status_t(AMDF_CALL* memory_scope_query_device_profile)(
      amdf_memory_scope_t* scope, uint32_t profile_ordinal,
      uint32_t access_count, const amdf_memory_device_access_t* accesses,
      amdf_memory_profile_t* out_profile,
      amdf_memory_access_capabilities_t* out_access_capabilities);

  /// Creates one backing with every requested live-device access established.
  ///
  /// The selected profile exposes exactly one of CREATE and REGISTER, with
  /// registered_host_pointer present exactly for REGISTER. The resource
  /// borrows the scope and all requested devices, which the caller keeps live
  /// through memory release without hidden retention or lifetime tracking.
  /// Access ordinals preserve request order. Exact permissions, required
  /// properties, mapping and residency are established before publication.
  /// No device is implicitly activated and no access is deferred to first use.
  ///
  /// Registered pages remain caller-owned and backed by the same live pages
  /// through memory release. A native release failure can leave registrations
  /// live and does not permit recycling those pages. If they came from a host
  /// mapping, that mapping and its backing obey the same requirement.
  /// Construction performs no command inspection, queue submission or
  /// device-wide synchronization. Failure leaves out_memory unchanged and
  /// creates no public cleanup owner.
  amdf_status_t(AMDF_CALL* memory_create)(
      amdf_memory_scope_t* scope, const amdf_memory_create_info_t* create_info,
      amdf_memory_t** out_memory);

  /// Acquires external backing and establishes the requested live access set.
  ///
  /// The selected scope profile exposes IMPORT and accepts the transport and
  /// all requirements. The same caller-enforced lifetime and readiness rules
  /// as memory_create apply. Complete success acquires the native backing
  /// references, consumes and zeros inout_external_memory, and publishes one
  /// memory handle. Every failure leaves both caller values byte-for-byte
  /// unchanged. This cold operation performs no queue submission or
  /// device-wide wait.
  amdf_status_t(AMDF_CALL* memory_import)(
      amdf_memory_scope_t* scope, const amdf_memory_import_info_t* import_info,
      amdf_external_memory_t* inout_external_memory,
      amdf_memory_t** out_memory);

  /// Copies immutable backing properties cached when `memory` was created.
  ///
  /// The operation is thread-safe and performs no system call, allocation,
  /// mapping mutation, retry, sleep, or device wait. The caller initializes
  /// `out_info` and its complete extension chain. No output is modified when
  /// validation fails. It reads immutable storage without locking, lazy
  /// initialization or ownership-counter updates.
  amdf_status_t(AMDF_CALL* memory_query_info)(amdf_memory_t* memory,
                                              amdf_memory_info_t* out_info);

  /// Copies immutable facts of one established device access.
  ///
  /// The ordinal is below memory info's `access_count` and preserves the
  /// construction request's consumer order. An out-of-range ordinal
  /// returns OUT_OF_RANGE. The caller initializes `out_info` and its extension
  /// chain; failure leaves it unchanged. This thread-safe metadata query
  /// performs no allocation, native query, mapping, locking, lazy
  /// initialization or ownership-counter update. The ordinal directly indexes
  /// retained facts; no search over the other consumers is required.
  amdf_status_t(AMDF_CALL* memory_query_access_info)(
      amdf_memory_t* memory, uint32_t access_ordinal,
      amdf_memory_access_info_t* out_info);

  /// Exports one logical range as a move-owned external-memory value.
  ///
  /// Success publishes a complete value whose payload remains valid until it
  /// is imported or explicitly released. The caller obtains a separate export
  /// for every independent import transaction. Export performs no queue wait,
  /// cache transition, or implicit synchronization. Failure leaves `out_value`
  /// byte-for-byte unchanged.
  amdf_status_t(AMDF_CALL* memory_export)(
      amdf_memory_t* memory, const amdf_memory_export_info_t* export_info,
      amdf_external_memory_t* out_value);

  /// Releases one move-owned external-memory value and zeros it.
  ///
  /// An empty value and a null pointer are no-ops. A nonempty value invokes its
  /// release callback exactly once when one is present. The caller must have
  /// exclusive access to the value.
  void(AMDF_CALL* external_memory_release)(amdf_external_memory_t* value);

  /// Copies the exact directional relation between two concrete access sites.
  ///
  /// Each site names either a device access and queue family or a host mapping.
  /// An out-of-range access ordinal returns OUT_OF_RANGE. Sites on one memory
  /// handle share backing directly, including CPU-only memory without a
  /// physical identity. Different memory handles must belong to one provider
  /// instance and have equal valid physical identities. A scope or identity
  /// mismatch returns
  /// `AMDF_STATUS_CODE_FAILED_PRECONDITION`; an unavailable identity returns
  /// `AMDF_STATUS_CODE_UNSUPPORTED`. Equal addresses or physical identities do
  /// not imply reach, visibility, or atomics. Common code composes
  /// independently reported local facts and gives neither implementation the
  /// other one's object. Host transitions account for the mapping's available
  /// operations and the selected peer's coherence, not other device accesses
  /// in the resource. The result describes visibility over corresponding bytes
  /// reachable by both sites; callers select those ranges and provide ordering.
  /// It does not establish synchronization or report host atomic capabilities.
  /// The operation performs no allocation, native query, import, mapping,
  /// cache transition, locking, lazy initialization or wait. It composes the
  /// two sites' retained facts without scanning other allocations or consumers.
  /// Failure leaves `out_info` byte-for-byte unchanged.
  amdf_status_t(AMDF_CALL* memory_query_pair_info)(
      const amdf_memory_site_t* producer_site,
      const amdf_memory_site_t* consumer_site,
      amdf_memory_pair_info_t* out_info);

  /// Creates an explicit host mapping of one memory range.
  ///
  /// Requested access is a minimum. The returned mapping reports the actual
  /// native access, which may include additional bits advertised by the
  /// attachment profile. The mapping borrows `memory`, which must outlive it.
  /// Mapping does not wait for device work or transfer cache ownership.
  /// Failure leaves `out_mapping` unchanged.
  amdf_status_t(AMDF_CALL* memory_map)(amdf_memory_t* memory,
                                       const amdf_memory_map_info_t* map_info,
                                       amdf_host_mapping_t** out_mapping);

  /// Copies immutable properties of one live host mapping.
  ///
  /// The copied pointer is borrowed until `host_mapping_destroy` is called. Its
  /// flush and invalidate recipes describe available CPU cache operations,
  /// not requirements relative to one attached device. A device's
  /// HOST_COHERENT access can make those operations unnecessary for that
  /// consumer without making the CPU mapping universally coherent. Recipes
  /// state whether a caller can execute cache maintenance directly or must
  /// invoke `host_mapping_cache_control`. The operation is thread-safe and
  /// performs no system call, allocation, locking, lazy initialization, cache
  /// transition or ownership-counter update. No output is modified on
  /// validation failure.
  amdf_status_t(AMDF_CALL* host_mapping_query_info)(
      amdf_host_mapping_t* mapping, amdf_host_mapping_info_t* out_info);

  /// Performs one explicit host cache ownership transition over a mapped range.
  ///
  /// `byte_offset` is relative to the mapping. The implementation may touch
  /// every cache line intersecting the range; callers externally synchronize
  /// the complete intersected lines. A non-empty flush requires write access
  /// and a non-empty invalidate requires read access; otherwise the operation
  /// returns `AMDF_STATUS_CODE_FAILED_PRECONDITION`. An empty range is a no-op.
  /// A nonempty request executes the advertised operation even when the
  /// memory's attached device is host-coherent. It never waits for device
  /// execution or supplies an execution dependency.
  /// The mapping's cache recipe and native entry points are prepared before
  /// this call. No allocation, locking or lazy initialization occurs. Work
  /// scales with the affected cache lines or native allocation spans. A recipe
  /// requiring HOST_API execution may enter the driver; this is not a blanket
  /// syscall-free operation.
  amdf_status_t(AMDF_CALL* host_mapping_cache_control)(
      amdf_host_mapping_t* mapping, amdf_host_cache_operation_t operation,
      uint64_t byte_offset, uint64_t byte_length);

  /// Destroys one mapping after all host access to its pointer has stopped.
  ///
  /// The caller must have exclusive access. A valid mapping is consumed and
  /// returns OK; NULL returns INVALID_ARGUMENT. The view borrows the memory's
  /// persistent native mapping, whose unmapping occurs at memory destruction.
  /// No execution wait or cache transition is implied.
  amdf_status_t(AMDF_CALL* host_mapping_destroy)(amdf_host_mapping_t* mapping);

  /// Destroys memory after all host mappings and future device uses are gone.
  ///
  /// The caller must have exclusive access and have completed every use,
  /// including host mappings, queue commands, scratch and opaque device work.
  /// These are caller preconditions: libamdf neither tracks memory users nor
  /// diagnoses premature destruction. Violating them is undefined behavior.
  /// Destruction performs no execution wait or cache transition, but may wait
  /// for native unmapping or paging cleanup.
  /// The memory object is consumed even when native release fails. The first
  /// native error is returned; neither this handle nor its addresses may be
  /// used again. Each independent consumer receives one release attempt, and
  /// backing is released only after every consumer succeeds. A failure can leak
  /// native resources and required backing, without a retained library object
  /// or later cleanup attempt. Borrowed scopes and devices must outlive this
  /// call; their destruction does not guarantee reclamation of native residue.
  /// Registered source storage remains caller-owned. Failed native detach does
  /// not authorize recycling it: the caller must leave that storage unreclaimed
  /// through the remaining native lifetime. Failure is observable resource
  /// loss, not a retryable memory handle or successful source release.
  amdf_status_t(AMDF_CALL* memory_destroy)(amdf_memory_t* memory);

  /// Copies immutable properties cached when `queue` was created.
  ///
  /// The operation is thread-safe and performs no system call, allocation,
  /// native progress query, retry, sleep, or device wait. No output is modified
  /// when validation fails. It acquires no lock, initializes no state and
  /// updates no ownership counters.
  amdf_status_t(AMDF_CALL* kernel_queue_query_info)(
      amdf_kernel_queue_t* queue, amdf_kernel_queue_info_t* out_info);

  /// Samples retirement and observed terminal state without waiting.
  ///
  /// The operation may retire completed submissions and consume their native
  /// results. It is thread-safe with submission and other status operations. It
  /// performs no allocation, system call, sleep, or active polling. No
  /// output is modified when validation fails. ACTIVE means no terminal failure
  /// has been observed, not that a fresh native health check was performed.
  /// Rejection and timeout errors do not themselves mark a queue failed. A
  /// terminal failure remains sticky and is not itself retirement proof.
  /// Providers without a mapped completion fence report cached progress;
  /// `kernel_queue_wait`, including a zero-time wait, refreshes that progress.
  /// This path takes no library lock and performs no lazy initialization. It
  /// may atomically claim retirement; queue-slot updates can contend, so this
  /// is not a wait-free guarantee.
  amdf_status_t(AMDF_CALL* kernel_queue_query_status)(
      amdf_kernel_queue_t* queue, amdf_kernel_queue_status_t* out_status);

  /// Waits until `submission` retires, a failure is observed, or time expires.
  ///
  /// `timeout_nanoseconds` includes host contention, active polling, and native
  /// waiting under one deadline. A zero timeout performs one nonblocking native
  /// poll when progress is not already known. `poll_duration_nanoseconds` is
  /// clipped to that timeout; zero disables active polling.
  /// `AMDF_TIMEOUT_INFINITE` requests no deadline. A
  /// timeout observes but never cancels accepted work or permits command
  /// storage reuse. The operation is thread-safe with submission and status
  /// queries.
  /// A native wait error is returned even if progress concurrently advances;
  /// callers use `kernel_queue_query_status` to determine retirement and
  /// whether a terminal failure was observed before deciding to retry.
  /// This is the explicit synchronization path. It may query clocks, poll,
  /// yield, enter native waits and serialize access to a reusable wait event.
  /// Contention consumes the same deadline. Queue creation prepares wait
  /// resources; waiting performs no library allocation or lazy resource setup.
  amdf_status_t(AMDF_CALL* kernel_queue_wait)(
      amdf_kernel_queue_t* queue, uint64_t submission,
      uint64_t timeout_nanoseconds, uint64_t poll_duration_nanoseconds);

  /// Destroys one queue after every accepted submission has retired.
  ///
  /// The caller must have exclusive access. The operation samples progress once
  /// and returns `AMDF_STATUS_CODE_BUSY` without waiting while work remains. A
  /// native teardown failure leaves the queue live so destruction can be
  /// retried.
  amdf_status_t(AMDF_CALL* kernel_queue_destroy)(amdf_kernel_queue_t* queue);

  /// Copies immutable properties cached when `queue` was created.
  ///
  /// The operation is thread-safe and performs no system call, allocation,
  /// native progress query, retry, sleep, or device wait. No output is modified
  /// when validation fails. It acquires no lock, initializes no state and
  /// updates no ownership counters.
  amdf_status_t(AMDF_CALL* user_queue_query_info)(
      amdf_user_queue_t* queue, amdf_user_queue_info_t* out_info);

  /// Maps a directly published queue into one producer.
  ///
  /// A null `producer_device` maps the queue for host publication. A non-null
  /// producer must be an exact device for which the queue advertises and can
  /// establish device publication. The returned mapping borrows the queue,
  /// which must outlive it. Failure leaves `out_mapping` unchanged.
  amdf_status_t(AMDF_CALL* user_queue_map)(
      amdf_user_queue_t* queue, amdf_device_t* producer_device,
      amdf_user_queue_mapping_t** out_mapping);

  /// Copies producer-local addresses from one live queue mapping.
  ///
  /// The operation is thread-safe and performs no system call, allocation,
  /// native progress query, retry, sleep, or device wait. No output is modified
  /// when validation fails. The mapped addresses were established by
  /// user_queue_map; this query takes no lock, initializes no state and
  /// updates no ownership counters.
  amdf_status_t(AMDF_CALL* user_queue_mapping_query_info)(
      amdf_user_queue_mapping_t* mapping,
      amdf_user_queue_mapping_info_t* out_info);

  /// Releases one queue mapping after that producer has stopped using it.
  ///
  /// The caller must have exclusive access. Failure leaves the mapping live so
  /// destruction can be retried. Queue execution is not implicitly waited.
  amdf_status_t(AMDF_CALL* user_queue_mapping_destroy)(
      amdf_user_queue_mapping_t* mapping);

  /// Samples directly published progress and observed terminal state.
  ///
  /// The operation is thread-safe with producer publication. It performs no
  /// allocation, retry, sleep, or active polling. ACTIVE means no terminal
  /// failure has been observed. A terminal failure is sticky and does not by
  /// itself prove that published commands retired. No output is modified when
  /// validation or the native observation fails.
  /// Unlike kernel_queue_query_status, this operation may query the native
  /// driver for queue or VM faults. It takes no library lock and performs no
  /// lazy initialization; terminal-state updates may use atomics.
  amdf_status_t(AMDF_CALL* user_queue_query_status)(
      amdf_user_queue_t* queue, amdf_user_queue_status_t* out_status);

  /// Waits until `published_index` is consumed, failure is observed, or time
  /// expires.
  ///
  /// The index uses the units defined by the queue format. The target must have
  /// already been release-published by a producer. `timeout_nanoseconds`
  /// includes host contention, active polling, and native waiting under one
  /// deadline. `poll_duration_nanoseconds` is clipped to that timeout; zero
  /// disables active polling. `AMDF_TIMEOUT_INFINITE` requests no deadline. A
  /// timeout observes but never cancels work.
  /// This synchronization path may query clocks, poll, yield and enter the
  /// native driver. It performs no library allocation or lazy resource setup.
  amdf_status_t(AMDF_CALL* user_queue_wait_consumed)(
      amdf_user_queue_t* queue, uint64_t published_index,
      uint64_t timeout_nanoseconds, uint64_t poll_duration_nanoseconds);

  /// Destroys one directly published queue after all mappings are gone and all
  /// published work has been consumed.
  ///
  /// The caller must have exclusive access. The operation returns BUSY without
  /// native mutation while a mapping or unconsumed publication remains. A
  /// native teardown failure leaves the queue live so destruction can be
  /// retried.
  amdf_status_t(AMDF_CALL* user_queue_destroy)(amdf_user_queue_t* queue);

  /// Returns the stable base address consumed by `kind` for `memory`.
  ///
  /// The address names logical byte zero; offsets below the memory's reported
  /// byte_length are valid. The caller retains the memory through every use.
  /// Kinds identify consuming interfaces, not independently selectable address
  /// spaces. The selected access info's address_kinds reports established
  /// kinds; zero is a valid address value, not an indication of availability.
  /// An unknown kind is INVALID_ARGUMENT; a known but unavailable kind is
  /// UNSUPPORTED. An out-of-range access ordinal returns OUT_OF_RANGE.
  /// Every failure leaves `out_address` unchanged.
  ///
  /// This thread-safe metadata query performs no allocation, native query,
  /// mapping, pinning, locking, lazy initialization or ownership-counter
  /// update. It directly indexes the access record and address kind,
  /// independent of allocation size or the number of other consumers and
  /// allocations. There is no address-to-handle lookup or first-use
  /// mapping/residency work.
  amdf_status_t(AMDF_CALL* memory_query_address)(
      amdf_memory_t* memory, uint32_t access_ordinal,
      amdf_memory_address_kind_t kind, uint64_t* out_address);

  /// Qualifies visibility before allocating backing or creating host views.
  ///
  /// Success supplies sufficient release/acquire recipes for every successful
  /// construction with the same scope, profile, backing requirements, complete
  /// live consumer set and registration cache class. Host sites require
  /// HOST_VISIBLE in required_flags. Views must provide the declared
  /// permissions. Allocation size, addresses and host-view ranges may vary
  /// within the profile; callers still satisfy each transition's range and
  /// execution requirements. This describes access to one shared backing, not
  /// transfers between distinct allocations. Ordering and completion remain
  /// separate caller obligations.
  ///
  /// Devices and the scope remain live while the result is used. No backing,
  /// mapping, handle or plan is retained. The exact queue family owns QUEUE
  /// operations; HOST_DIRECT uses each actual view's pointer and range, and
  /// HOST_API uses its actual host_mapping handle. A queue answer does not
  /// qualify PROGRAM execution. Unknown native/exporter protocols, unsupported
  /// permissions and unqualified sites return UNSUPPORTED, never a no-op.
  /// Foreign import host visibility may be unqualified even when device access
  /// and memory_map are available. Imports are not admitted to a host
  /// visibility contract by borrowing an allocation profile's answer.
  ///
  /// Import qualification additionally fixes the exact external transport type
  /// and provenance supplied at construction. Opaque payloads preserve their
  /// exporter-provided interpretation; changing the tag does not qualify a
  /// foreign payload.
  ///
  /// This thread-safe cold metadata query uses the same native selection and
  /// visibility policy as construction and concrete pair queries. It performs
  /// no native operation, mapping, synchronization or device activation.
  /// Temporary host storage scales with the live consumer set. Success is not a
  /// resource reservation. Failure leaves out_info unchanged.
  amdf_status_t(AMDF_CALL* memory_scope_query_pair_info)(
      amdf_memory_scope_t* scope, const amdf_memory_profile_pair_query_t* query,
      amdf_memory_pair_info_t* out_info);
} amdf_api_t;

/// Function type used to acquire the immutable API table.
typedef amdf_status_t(AMDF_CALL* amdf_query_api_fn_t)(
    amdf_abi_version_t minimum_version, amdf_abi_version_t maximum_version,
    const amdf_api_t** out_api);

/// Acquires the newest supported API table in the inclusive requested range.
///
/// On success, `out_api` receives a borrowed immutable table that remains valid
/// until the providing library is unloaded. Failure leaves `out_api` unchanged
/// when it is non-NULL.
///
/// This function is thread-safe, bounded constant time, and inert. It performs
/// no allocation, system call, device discovery, dependent-library load, or
/// other observable initialization. No lock or one-time initialization guard
/// is acquired, including on the first call.
AMDF_API amdf_status_t AMDF_CALL
amdf_query_api(amdf_abi_version_t minimum_version,
               amdf_abi_version_t maximum_version, const amdf_api_t** out_api);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // AMDF_API_H_
