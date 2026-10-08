// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef AMDF_GPU_H_
#define AMDF_GPU_H_

#include <stdint.h>

#include "amdf/amdf.h"

#ifdef __cplusplus
extern "C" {
#endif

/// The first supported GPU extension version.
#define AMDF_GPU_EXTENSION_VERSION_1 1u

/// The most recent GPU extension version described by this header.
#define AMDF_GPU_EXTENSION_VERSION_LATEST AMDF_GPU_EXTENSION_VERSION_1

/// An `amdf_gpu_endpoint_info_t` output structure.
#define AMDF_STRUCTURE_TYPE_GPU_ENDPOINT_INFO \
  ((amdf_structure_type_t)0x00020001u)

/// An `amdf_gpu_device_create_info_t` input structure.
#define AMDF_STRUCTURE_TYPE_GPU_DEVICE_CREATE_INFO \
  ((amdf_structure_type_t)0x00020002u)

/// An `amdf_gpu_device_info_t` output structure.
#define AMDF_STRUCTURE_TYPE_GPU_DEVICE_INFO ((amdf_structure_type_t)0x00020003u)

/// An `amdf_gpu_kernel_queue_create_info_t` input structure.
#define AMDF_STRUCTURE_TYPE_GPU_KERNEL_QUEUE_CREATE_INFO \
  ((amdf_structure_type_t)0x00020004u)

/// An `amdf_gpu_kernel_queue_submission_info_t` input structure.
#define AMDF_STRUCTURE_TYPE_GPU_KERNEL_QUEUE_SUBMISSION_INFO \
  ((amdf_structure_type_t)0x00020005u)

/// An `amdf_gpu_user_queue_create_info_t` input structure.
#define AMDF_STRUCTURE_TYPE_GPU_USER_QUEUE_CREATE_INFO \
  ((amdf_structure_type_t)0x00020007u)

/// Features available on a live device under its instance's lifetime policy.
typedef uint64_t amdf_gpu_device_features_t;
enum amdf_gpu_device_feature_bits_e {
  /// REGISTERED_HOST borrows caller pages without copying their contents.
  AMDF_GPU_DEVICE_FEATURE_HOST_REGISTRATION = UINT64_C(1) << 0,
  /// Destroying a device permits later creation within the same instance,
  /// without caller-retained native handles or process exit. This does not
  /// imply native process-context reclamation when the instance is destroyed.
  AMDF_GPU_DEVICE_FEATURE_DEVICE_RECREATION = UINT64_C(1) << 1,
  /// LOCAL allocations provide device-local physical placement.
  AMDF_GPU_DEVICE_FEATURE_LOCAL_MEMORY = UINT64_C(1) << 2,
  /// LOCAL allocations support HOST_VISIBLE together with DEVICE_LOCAL.
  AMDF_GPU_DEVICE_FEATURE_HOST_VISIBLE_LOCAL_MEMORY = UINT64_C(1) << 3,
};

/// Immutable target identity and compute topology of one GPU endpoint.
///
/// This record contains live provider-qualified hardware facts. It does not
/// select an executable format, process-level target features, memory policy,
/// or queue implementation.
typedef struct amdf_gpu_endpoint_info_t {
  /// Must be `AMDF_STRUCTURE_TYPE_GPU_ENDPOINT_INFO`.
  amdf_structure_type_t type;
  /// Must be at least `sizeof(amdf_gpu_endpoint_info_t)`.
  uint32_t structure_size;
  /// Optional output extension chain. No extensions are currently defined.
  void* next;
  /// Exact Graphics IP identity.
  struct {
    /// GFX IP major version.
    uint32_t major;
    /// GFX IP minor version.
    uint32_t minor;
    /// GFX IP stepping encoded in the canonical `gfxMmn` target name.
    uint32_t stepping;
  } gfx_ip;
  /// ASIC revision used for physical compiler-target selection.
  ///
  /// This is the HSA/KFD target revision and is distinct from the PCI
  /// config-space revision in `amdf_endpoint_info_t`. Gfx1250 uses values zero
  /// and one to distinguish A0 and B0 targets.
  uint32_t asic_revision;
  /// Active compute geometry and per-compute-unit limits.
  struct {
    /// Number of lanes in one hardware wavefront.
    uint32_t wavefront_size;
    /// Total active compute units across all XCCs after harvesting.
    uint32_t compute_unit_count;
    /// Maximum resident hardware waves per compute unit.
    uint32_t maximum_wave_count_per_compute_unit;
    /// Effective maximum scratch-backed waves per compute unit.
    uint32_t maximum_scratch_wave_count_per_compute_unit;
    /// Local data share capacity per compute unit in bytes.
    uint64_t local_data_share_byte_length;
  } compute;
  /// Multi-chiplet command-processor topology.
  struct {
    /// Nonzero number of active XCCs represented by the endpoint.
    uint32_t xcc_count;
    /// Uniform number of shader engines within each XCC.
    uint32_t shader_engine_count_per_xcc;
  } topology;
} amdf_gpu_endpoint_info_t;

/// Parameters used to materialize one program-independent GPU device.
///
/// A device owns one native GPU execution and address domain. Queue,
/// executable, and memory policy are selected by later operations. Native
/// lifetime is inherited from the endpoint's instance, not selected per device.
typedef struct amdf_gpu_device_create_info_t {
  /// Must be `AMDF_STRUCTURE_TYPE_GPU_DEVICE_CREATE_INFO`.
  amdf_structure_type_t type;
  /// Must be at least `sizeof(amdf_gpu_device_create_info_t)`.
  uint32_t structure_size;
  /// Optional input extension chain. No extensions are currently defined.
  const void* next;
  /// Reserved for compatible growth and must be zero.
  uint64_t reserved;
} amdf_gpu_device_create_info_t;

/// Immutable identity, reset state, and achieved features of one live GPU
/// device.
typedef struct amdf_gpu_device_info_t {
  /// Must be `AMDF_STRUCTURE_TYPE_GPU_DEVICE_INFO`.
  amdf_structure_type_t type;
  /// Must be at least `sizeof(amdf_gpu_device_info_t)`.
  uint32_t structure_size;
  /// Optional output extension chain. No extensions are currently defined.
  void* next;
  /// Opaque identity of this live materialized device.
  amdf_device_id_t id;
  /// Monotonic provider epoch invalidating state after a device reset.
  uint64_t reset_epoch;
  /// Features supported by this materialized device and native lifetime policy.
  /// These may refine the expected endpoint capabilities without changing them.
  amdf_gpu_device_features_t features;
} amdf_gpu_device_info_t;

/// First directly published PM4 queue format.
///
/// The primary ring contains native type-3 PM4 packets. Transfer commands use
/// six-dword COPY_DATA and WRITE_DATA with a four-dword prefix and payload.
/// GFX11.5.1 MEC also supports seven-dword DMA_DATA incrementing L2 copies
/// between owned coherent SYSTEM ranges, with a direct byte count and write
/// confirmation. The caller follows each
/// copy sequence with the zero-byte DMA_DATA drain, explicit cache work and
/// a completion marker before releasing its operands; ring consumption alone
/// does not complete a transfer. This does not admit PFP controls, other DMA
/// selectors or DMA_DATA on kernel-publication or AQL-carried PM4 transports.
/// Cache-control encoding is described by the reported PM4 format features.
/// Indices occupy naturally aligned 64-bit storage. The write index is a
/// monotonic dword count; the native read index wraps at the ring capacity
/// `ring_byte_length / 4`. The producer always leaves at least one dword free:
/// fewer than that capacity may remain unconsumed, including reserved storage.
/// Given a stable published write index W and acquired native read index R,
/// the consumed frontier is W - ((W - R) & (capacity - 1)). Status and wait
/// operations expand the native counter to this monotonic frontier; the raw
/// mapping continues to expose the native ring-relative read index.
/// After storing complete commands, the producer release-stores the new write
/// index followed by the same value to the 64-bit doorbell. Consumption proves
/// ring storage can be reused; command completion requires a separate fence.
/// Each publication ends on an eight-dword boundary, padded with type-3 NOP
/// packets when necessary; packets never straddle ring wrap.
/// Kernel publication accepts an immutable dword-aligned first-level IB.
/// A USER primary ring can call a first-level IB and resume after it. KERNEL
/// submits that IB directly; placing the ring's call inside a KERNEL buffer
/// would request unsupported compute IB2 nesting.
#define AMDF_GPU_PM4_QUEUE_FORMAT_VERSION_1 1u

/// Native PM4 encoding features reported in `format_features`.
enum amdf_gpu_pm4_format_feature_bits_e {
  /// Eight-dword ACQUIRE_MEM with GCR_CNTL in dword 7. Global cache control
  /// uses zero bases and maximum sizes; EVENT_WRITE CS_PARTIAL_FLUSH supplies
  /// preceding compute completion. This does not select a cache policy for
  /// the caller: cache_operations and cache_transition_kinds remain required.
  AMDF_GPU_PM4_FORMAT_FEATURE_ACQUIRE_MEM_GCR = UINT64_C(1) << 0,
};

/// First native SDMA command-stream format.
///
/// The primary ring contains native SDMA dwords. COPY_LINEAR uses seven dwords
/// with a byte-count-minus-one field and full 64-bit source/destination
/// addresses. FENCE uses four dwords with a 32-bit value. Read and
/// write indices are naturally aligned 64-bit monotonic byte counts; each
/// index selects storage modulo `ring_byte_length`. A producer never advances
/// more than that capacity beyond the acquired read index. After storing
/// complete packets into the ring, the producer performs a release store of
/// the new write index followed by a release store of the same value to the
/// 64-bit doorbell. An acquire load of a read index at least that value proves
/// the corresponding ring bytes are no longer in use by the queue. Packets
/// are dword-aligned and never straddle ring wrap. NOP dwords have value zero.
/// Optional commands and field layouts use the reported format features.
/// Kernel publication accepts an immutable dword-aligned command stream.
#define AMDF_GPU_SDMA_QUEUE_FORMAT_VERSION_1 1u

/// Native SDMA encoding features reported in `format_features`.
enum amdf_gpu_sdma_format_feature_bits_e {
  /// Five-dword USER_GCR, opcode 17 and suboperation 1. Whole-cache data
  /// acquire uses control 0xc3c0 and release uses 0x8040 in dword 2 bits
  /// 31:16, with every other operand zero. Dependency waits precede acquire;
  /// release follows the data commands and precedes completion. These are
  /// queue GLOBAL ACQUIRE_FROM_SYSTEM/RELEASE_TO_SYSTEM operations. This
  /// contract admits neither address ranges nor explicit VMID selection.
  AMDF_GPU_SDMA_FORMAT_FEATURE_USER_GCR = UINT64_C(1) << 0,
  /// FENCE uses a two-bit memory type at header bit 16 and an explicit system
  /// bit at bit 20. A fence to system memory sets that bit. This feature and
  /// FENCE_MEMORY_TYPE are mutually exclusive. When neither is reported,
  /// callers leave the optional memory-type and system bits zero.
  AMDF_GPU_SDMA_FORMAT_FEATURE_FENCE_SYSTEM = UINT64_C(1) << 1,
  /// COPY_LINEAR source/destination scope fields occupy bits 26/18 of dword 2;
  /// COPY_LINEAR_RECT uses those positions in dword 12 when its WIDE layout
  /// is reported. Each scope field is two bits wide.
  /// FENCE, CONSTANT_FILL and TIMESTAMP_GET_GLOBAL scope occupies header bits
  /// 25:24; WRITE_LINEAR uses dword 3 bits 27:26; POLL_REGMEM uses dword 5
  /// bits 29:28. Scope 3 denotes the system.
  /// NPD (no prior dependency) occupies COPY_LINEAR/COPY_LINEAR_RECT header
  /// bit 28 and CONSTANT_FILL header bit 29. Field availability does not
  /// establish dependencies or completion. Without this feature scope and
  /// NPD are zero.
  /// Data commands using system scope realize the site's payload visibility
  /// without a separate stream cache operation. Execution dependencies and
  /// completion remain explicit; this establishes no system atomic reach.
  AMDF_GPU_SDMA_FORMAT_FEATURE_MEMORY_SCOPE = UINT64_C(1) << 2,
  /// FENCE uses the classic three-bit memory type at header bits 18:16.
  /// Memory type 3 denotes uncached access; callers leave the system bit at
  /// bit 20 zero. This feature and FENCE_SYSTEM are mutually exclusive.
  AMDF_GPU_SDMA_FORMAT_FEATURE_FENCE_MEMORY_TYPE = UINT64_C(1) << 3,
  /// Thirteen-dword COPY_LINEAR_RECT, opcode 1 and suboperation 4. Header
  /// bits 31:29 encode log2(element bytes), from 0 through 4. Source and
  /// destination byte addresses occupy dwords 1:2 and 6:7, low word first.
  /// Bases and byte pitches are dword-aligned. X coordinates, width, row
  /// pitch and slice pitch count elements; Y/height count rows and Z/depth
  /// count slices. Coordinates are direct; positive pitches and extents
  /// are encoded minus one. Source and destination regions must not overlap.
  ///
  /// The classic layout packs source/destination X in bits 13:0 and Y in
  /// bits 29:16 of dwords 3/8. Dwords 4/9 contain Z in bits 10:0 and row
  /// pitch in bits 31:13. Dwords 5/10 contain slice pitch in bits 27:0.
  /// Dword 11 contains width in bits 13:0 and height in bits 29:16; dword 12
  /// contains depth in bits 10:0. EXTENDED_Z and WIDE modify these fields.
  /// Other fields are zero unless separately admitted by MEMORY_SCOPE.
  ///
  /// Each side addresses base + element_bytes * (x + y * row_pitch +
  /// z * slice_pitch). The caller owns every selected row and retains all
  /// accessed backing through completion. Cross-queue producer dependencies,
  /// the memory-site cache contract and explicit completion still apply.
  AMDF_GPU_SDMA_FORMAT_FEATURE_COPY_LINEAR_RECT = UINT64_C(1) << 4,
  /// Classic COPY_LINEAR_RECT with 13-bit Z coordinates and depth-minus-one
  /// in dwords 4, 9 and 12 bits 12:0. Requires COPY_LINEAR_RECT and is mutually
  /// exclusive with COPY_LINEAR_RECT_WIDE. All other classic fields remain.
  AMDF_GPU_SDMA_FORMAT_FEATURE_COPY_LINEAR_RECT_EXTENDED_Z = UINT64_C(1) << 5,
  /// Wide COPY_LINEAR_RECT geometry. Dwords 3/8 contain 16-bit X and Y at
  /// bits 15:0 and 31:16. Dwords 4/9 contain 14-bit Z at bits 13:0 and
  /// 16-bit row pitch at bits 31:16. Slice pitch uses all 32 bits of dwords
  /// 5/10, representing positive counts through 2^32 elements. Dword 11
  /// contains 16-bit width and height at bits 15:0 and 31:16; dword 12 has
  /// 14-bit depth at bits 13:0. Addresses, units and minus-one encoding are
  /// unchanged. Requires COPY_LINEAR_RECT and is mutually exclusive with
  /// COPY_LINEAR_RECT_EXTENDED_Z. MEMORY_SCOPE independently admits scopes.
  AMDF_GPU_SDMA_FORMAT_FEATURE_COPY_LINEAR_RECT_WIDE = UINT64_C(1) << 6,
};

/// First directly published AQL queue format.
///
/// The ring contains native 64-byte AMD HSA packets, initially INVALID (type
/// 1). Both 64-bit indices count packets monotonically. Producers reserve
/// through the write index (atomic fetch-add for MULTI), wait until the
/// reserved slot is below acquired read_index + ring_byte_length / 64, and
/// write the packet body before release-storing its header/setup dword. A
/// release store of the packet's reservation index to the 64-bit doorbell
/// notifies firmware. MULTI producers may notify out of order; an unpublished
/// earlier slot blocks later consumption. SINGLE producers notify
/// monotonically. Firmware invalidates retired packet slots. Consumption
/// releases ring storage, not code, kernargs, scratch, or kernel data: those
/// require native execution completion.
///
/// Dispatches use AMD kernel descriptors and caller-owned kernargs. Completion
/// and barrier dependencies use native 64-byte AMD signal blocks, not bare
/// counters or handles created by libamdf. Callers initialize USER kind (1), a
/// signed 64-bit value at byte offset 8, and zero unused fields; the packet
/// carries the block's GPU address. Firmware decrements dispatch completion
/// once after all workgroups complete. System acquire/release fence scopes
/// provide the advertised global cache transitions. This contract does not
/// enable HSA runtime services, dynamic scratch growth, or device enqueue.
///
/// AMD vendor format 1 carries a four-dword INDIRECT_BUFFER at packet byte 4,
/// remaining-dword count 10 at byte 20, zero reserved words, and the native
/// completion signal address at byte 56. The jump uses a dword-aligned 48-bit
/// byte address, a positive 20-bit dword count and VALID at bit 23. Its entire
/// extent fits the executable backing and address field. The caller keeps
/// that extent immutable until execution completes on every participating XCC.
///
/// TRANSFER permits confirmed WRITE_DATA and COPY_DATA in that vendor buffer.
/// The caller selects the target's PM4 field encodings, including transfer
/// width, address alignment, and cache policy. Memory-pair queries describe
/// the required payload visibility; SYSTEM packet fences implement the
/// advertised global actions. With multiple XCCs, a two-dword PRED_EXEC
/// prefix selects virtual XCC 0 for a body that must execute once: mask 1 at
/// bit 24 and a positive 14-bit body dword count excluding the prefix. A
/// single-XCC buffer needs no such prefix. Confirmed GPU-clock COPY_DATA
/// observes command-processor progress and does not complete shader work.
#define AMDF_GPU_AQL_QUEUE_FORMAT_VERSION_1 1u

/// Optional native AQL packets reported in `format_features`.
enum amdf_gpu_aql_format_feature_bits_e {
  /// AMD vendor format 2 BARRIER_VALUE waits until (signal_value & mask)
  /// satisfies the signed 64-bit HSA comparison against its reference value.
  /// The dependency and optional completion are native AMD signal blocks.
  /// All reserved packet fields are zero. Signal storage remains live through
  /// the last dependent execution completion. Ordinary dispatch, BARRIER_AND,
  /// and BARRIER_OR support does not imply this vendor extension.
  AMDF_GPU_AQL_FORMAT_FEATURE_BARRIER_VALUE = UINT64_C(1) << 4,
};

/// Scratch backing borrowed by one directly published compute queue.
///
/// An all-zero value disables scratch and accepts only commands whose private
/// segment is empty. Otherwise the selected access must belong to the queue's
/// device and provide read/write permission and a stable GPU address. The queue
/// borrows the memory without lifetime tracking. The caller keeps the scratch
/// backing live until queue destruction succeeds.
///
/// AQL format 1 retains fixed scratch across dispatches. Let C be
/// compute_unit_count, E be xcc_count * shader_engine_count_per_xcc, and S be
/// maximum_scratch_wave_count_per_compute_unit. Its physical slot count is
/// ceil(C / E) * E * S, including padding for asymmetric CU harvesting. Set
/// maximum_wave_count to this slot count; smaller pools requiring firmware
/// scratch reclamation are unsupported. The GPU base is 4096-byte aligned.
/// Each slot receives the requested per-workitem private byte length
/// times 64, rounded up to 1024 bytes. Backing must cover every wave; excess
/// backing does not change the configured capacity. Every submitted dispatch
/// fits the configured per-workitem private limit. Scratch is exclusive to the
/// queue until its final execution completion and successful destruction.
typedef struct amdf_gpu_queue_scratch_t {
  /// Memory resource borrowed for the queue lifetime, or NULL when disabled.
  amdf_memory_t* memory;
  /// Resource-local access for the queue's device, or zero when disabled.
  uint32_t access_ordinal;
  /// Reserved for compatible growth; always zero.
  uint32_t reserved;
  /// Byte offset from the attachment's stable device base.
  uint64_t byte_offset;
  /// Nonzero scratch backing length in bytes, or zero when disabled.
  uint64_t byte_length;
  /// Maximum private-segment bytes per workitem accepted by the queue.
  uint32_t maximum_private_segment_byte_length;
  /// Physical scratch slot capacity, including shader-engine rounding.
  uint32_t maximum_wave_count;
} amdf_gpu_queue_scratch_t;

/// Parameters used to acquire one directly published GPU queue.
typedef struct amdf_gpu_user_queue_create_info_t {
  /// Must be `AMDF_STRUCTURE_TYPE_GPU_USER_QUEUE_CREATE_INFO`.
  amdf_structure_type_t type;
  /// Must be at least `sizeof(amdf_gpu_user_queue_create_info_t)`.
  uint32_t structure_size;
  /// Optional input extension chain. No extensions are currently defined.
  const void* next;
  /// Endpoint-local PM4, SDMA, or AQL family supporting user publication.
  uint32_t queue_family_ordinal;
  /// Requested scheduling priority.
  amdf_queue_priority_t priority;
  /// Required producer reservation protocol.
  amdf_queue_producer_mode_t producer_mode;
  /// Reserved for compatible growth and must be zero.
  uint32_t reserved;
  /// Direct producer capabilities that creation must achieve.
  amdf_user_queue_capabilities_t required_capabilities;
  /// Requested power-of-two primary ring capacity, or zero for the default.
  uint64_t ring_byte_length;
  /// Compute scratch borrowed for the queue lifetime, or all-zero when absent.
  amdf_gpu_queue_scratch_t scratch;
} amdf_gpu_user_queue_create_info_t;

/// One already-materialized native GPU command stream.
///
/// The command bytes reside in executable memory with a stable device address
/// and may be device-local or non-host-visible. Submission resolves only that
/// address; it never reads, validates, copies, hashes, or transcribes the
/// command bytes.
typedef struct amdf_gpu_kernel_command_t {
  /// Memory resource borrowed until the accepted submission retires.
  amdf_memory_t* memory;
  /// Resource-local access for the queue's device.
  uint32_t access_ordinal;
  /// Reserved for compatible growth; always zero.
  uint32_t reserved;
  /// Dword-aligned byte offset from the attachment's stable device base.
  uint64_t byte_offset;
  /// Nonzero dword-aligned command length.
  uint64_t byte_length;
} amdf_gpu_kernel_command_t;

/// Default pending capacity for kernel-mediated GPU queues.
#define AMDF_GPU_KERNEL_QUEUE_DEFAULT_PENDING_SUBMISSION_COUNT 4096u

/// Parameters used to acquire one kernel-mediated GPU queue.
typedef struct amdf_gpu_kernel_queue_create_info_t {
  /// Must be `AMDF_STRUCTURE_TYPE_GPU_KERNEL_QUEUE_CREATE_INFO`.
  amdf_structure_type_t type;
  /// Must be at least `sizeof(amdf_gpu_kernel_queue_create_info_t)`.
  uint32_t structure_size;
  /// Optional input extension chain. No extensions are currently defined.
  const void* next;
  /// Endpoint-local PM4 or SDMA family supporting kernel publication.
  uint32_t queue_family_ordinal;
  /// Maximum accepted submissions that may remain unretired, or zero for
  /// AMDF_GPU_KERNEL_QUEUE_DEFAULT_PENDING_SUBMISSION_COUNT. This admission
  /// bound does not reserve native driver capacity or retain command memory.
  uint32_t maximum_pending_submission_count;
} amdf_gpu_kernel_queue_create_info_t;

/// One bounded kernel-mediated GPU submission.
typedef struct amdf_gpu_kernel_queue_submission_info_t {
  /// Must be `AMDF_STRUCTURE_TYPE_GPU_KERNEL_QUEUE_SUBMISSION_INFO`.
  amdf_structure_type_t type;
  /// Must be at least `sizeof(amdf_gpu_kernel_queue_submission_info_t)`.
  uint32_t structure_size;
  /// Optional input extension chain. No extensions are currently defined.
  const void* next;
  /// Number of immutable descriptors in `commands`.
  uint32_t command_count;
  /// Reserved for compatible growth and must be zero.
  uint32_t reserved;
  /// Borrowed descriptor array consumed before return. Each referenced memory
  /// attachment remains borrowed until the accepted submission retires.
  const amdf_gpu_kernel_command_t* commands;
} amdf_gpu_kernel_queue_submission_info_t;

/// Immutable entry-point table for one negotiated GPU extension version.
///
/// Tables grow only by appending fields. The table and every function pointer
/// covered by `structure_size` remain valid until the providing library is
/// unloaded.
typedef struct amdf_gpu_api_t {
  /// Size in bytes of this table version.
  uint32_t structure_size;
  /// GPU extension version implemented by this table.
  uint32_t extension_version;

  /// Copies immutable GPU properties cached while opening `endpoint`.
  ///
  /// The endpoint must have a provider-qualified GPU profile. Passing another
  /// engine or a GPU that the active provider could not qualify returns the
  /// cached qualification error without modifying the output. The operation is
  /// thread-safe and performs no system call, allocation, provider-library
  /// load, retry, sleep, or device wait. The caller initializes `out_info` and
  /// its complete extension chain.
  /// Cached facts are read without locking, lazy initialization or
  /// ownership-counter updates.
  amdf_status_t(AMDF_CALL* endpoint_query_info)(
      amdf_endpoint_t* endpoint, amdf_gpu_endpoint_info_t* out_info);

  /// Materializes one program-independent GPU execution and address domain.
  ///
  /// `endpoint` remains query-only. Its instance selects the native lifetime;
  /// an unsupported lifetime is rejected without fallback. The returned device
  /// borrows the endpoint, which must outlive it. No queue,
  /// executable, command stream, or public memory object is created. Failure
  /// leaves `out_device` unchanged.
  /// This cold boundary may allocate, load native dependencies, serialize
  /// shared connection setup and enter the driver. Address-domain preparation
  /// completes here instead of during a query or first submission.
  amdf_status_t(AMDF_CALL* device_create)(
      amdf_endpoint_t* endpoint,
      const amdf_gpu_device_create_info_t* create_info,
      amdf_device_t** out_device);

  /// Copies the identity and current reset epoch of `device`.
  ///
  /// The operation is thread-safe and performs no system call, allocation,
  /// device initialization, retry, sleep, or device wait. The caller
  /// initializes `out_info` and its complete extension chain. No output is
  /// modified when validation or engine compatibility fails.
  /// Reset observation may use atomic loads; this query takes no lock,
  /// initializes no state and updates no ownership counters.
  amdf_status_t(AMDF_CALL* device_query_info)(amdf_device_t* device,
                                              amdf_gpu_device_info_t* out_info);

  /// Acquires one kernel-mediated native command queue from a GPU device.
  ///
  /// The returned queue borrows `device`, which must outlive it. Creation
  /// selects an advertised `GPU_PM4 + KERNEL` or `GPU_SDMA + KERNEL` family
  /// and allocates every bounded submission resource before publication.
  /// Failure leaves `out_queue` unchanged.
  /// Native command and completion resources are ready before success; neither
  /// submission nor waiting allocates them on first use.
  amdf_status_t(AMDF_CALL* kernel_queue_create)(
      amdf_device_t* device,
      const amdf_gpu_kernel_queue_create_info_t* create_info,
      amdf_kernel_queue_t** out_queue);

  /// Publishes one bounded array of already-materialized native command
  /// streams in the representation selected when the queue was created.
  ///
  /// Every command range must belong to the queue's device and reset epoch and
  /// have execute device access and the `DEVICE_ADDRESS` memory flag. The call
  /// borrows command memory without retaining it or tracking its lifetime.
  /// The caller keeps it live until the returned submission retires. It
  /// performs no allocation, command-byte access, native-format parsing,
  /// lowering, transcription, native submission retry, sleep, or host wait.
  /// Native rejection leaves `out_submission` unchanged. Because command bytes
  /// are opaque, the caller keeps every indirectly referenced memory or native
  /// object live until the submission retires.
  /// Successful calls return increasing queue-local completion points. Callers
  /// use returned points without assuming a starting value or dense numbering.
  /// A successful wait for one point covers all earlier accepted submissions
  /// on this queue, not independently scheduled work on other queues.
  ///
  /// This hot path takes no library lock and performs no lazy initialization,
  /// mapping, pinning or indirect-buffer scan. It is thread-safe with other
  /// submissions and progress operations. Concurrent publication returns BUSY
  /// rather than waiting. The publication claim ends when the native call
  /// returns, not when execution completes. At the configured pending bound,
  /// one nonblocking native progress check reclaims completed capacity before
  /// returning BUSY if the queue remains full. No intermediate host wait is
  /// required. This is not a wait-free guarantee. Native publication may enter
  /// the driver; it does not initialize a host scheduler or translate commands.
  amdf_status_t(AMDF_CALL* kernel_queue_submit)(
      amdf_kernel_queue_t* queue,
      const amdf_gpu_kernel_queue_submission_info_t* submission_info,
      uint64_t* out_submission);

  /// Acquires one directly published native command queue from a GPU device.
  ///
  /// Creation selects an advertised `GPU_PM4 + USER`, `GPU_SDMA + USER`, or
  /// `GPU_AQL + USER` family and allocates every queue-owned ring and native
  /// sidecar before publication. The returned queue borrows `device` and any
  /// supplied scratch memory, which must outlive it. Commands are published
  /// through a mapping from the base API. Failure leaves `out_queue` unchanged.
  /// This is a cold allocation boundary. Once explicitly mapped, publication
  /// consists of caller-owned native ring and doorbell operations, without a
  /// libamdf submit call or first-publication setup.
  amdf_status_t(AMDF_CALL* user_queue_create)(
      amdf_device_t* device,
      const amdf_gpu_user_queue_create_info_t* create_info,
      amdf_user_queue_t** out_queue);
} amdf_gpu_api_t;

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // AMDF_GPU_H_
