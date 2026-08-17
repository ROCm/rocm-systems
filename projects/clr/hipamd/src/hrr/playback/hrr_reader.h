/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */
#pragma once

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>
#include <unordered_map>

// hrr_api_args.h — HRR_API_* enum values and hrr_api_names[] string table
#include "hrr_api_args.h"

// HRR Archive Reader — reads .hrr trace archives produced by the in-tree
// capture layer (HIP_HRR_CAPTURE_OUTPUT).
//
// Binary format (v3):
//   events.bin:
//     [0..7]   hrr_file_header  { HRR_MAGIC, HRR_VERSION, reserved }
//     [8..]    hrr_event_header (32 bytes) + payload bytes, repeated per event
//   blobs/<2hex>/  — content-addressed raw buffers (FNV-1a-128 hash)
//   code_objects/  — .hsaco ELFs keyed by hash
// Format constants (HRR_MAGIC, HRR_VERSION, hrr_file_header, hrr_event_header)
// are defined in hrr_api_args.h (included above).

namespace hrr {

// ---------------------------------------------------------------------------
// Typed payload structs — populated by load_archive for fast replay access.
// All handle fields are raw runtime pointers cast to uint64_t.
// ---------------------------------------------------------------------------

struct MallocEvent {
  uint64_t ptr_handle;  // device pointer returned by hipMalloc
  uint64_t size;
};

struct MemcpyEvent {
  uint64_t dst_addr;
  uint64_t src_addr;
  uint64_t size;
  int32_t  kind;        // hipMemcpyKind
  uint64_t hash_lo;     // blob hash (non-zero for H2D with captured data)
  uint64_t hash_hi;
};

struct ModuleLoadEvent {
  uint64_t hash_lo;        // code object hash
  uint64_t hash_hi;
  uint64_t module_handle;  // raw hipModule_t pointer
};

struct MallocAsyncEvent {
  uint64_t ptr_handle;     // device pointer
  uint64_t size;
  uint64_t stream_handle;  // raw hipStream_t pointer
};

struct FreeAsyncEvent {
  uint64_t ptr_handle;
  uint64_t stream_handle;
};

struct StreamCreateEvent {
  uint64_t stream_handle;  // raw hipStream_t pointer (the created stream)
  uint32_t flags;
  int32_t  priority;
};

struct EventRecordEvent {
  uint64_t event_handle;
  uint64_t stream_handle;
};

struct StreamWaitEventEvent {
  uint64_t stream_handle;
  uint64_t event_handle;
  uint32_t flags;
};

// Kernel argument (value_kind: 0=scalar, 1=gpu-pointer, 2=hidden,
// 3=scalar/struct with embedded gpu pointer(s) at ptr_offsets)
struct KernelArg {
  uint8_t  value_kind;
  uint16_t size;
  std::vector<uint8_t> data;
  std::vector<uint16_t> ptr_offsets;  // only for value_kind == 3
};

// Buffer snapshot (always empty in in-tree captures)
struct BufferSnapshot {
  uint64_t ptr_handle;
  uint64_t offset;
  uint64_t length;
  uint64_t hash_lo;
  uint64_t hash_hi;
  uint8_t  direction;
};

// Parsed kernel launch event
struct KernelLaunchEvent {
  std::string              kernel_name;
  uint64_t                 co_hash_lo   = 0;
  uint64_t                 co_hash_hi   = 0;
  uint32_t                 grid[3]      = {};
  uint32_t                 block[3]     = {};
  uint32_t                 shared_mem   = 0;
  std::vector<KernelArg>       args;
  std::vector<BufferSnapshot>  snapshots;
};

// ---------------------------------------------------------------------------
// Single event
// ---------------------------------------------------------------------------

struct Event {
  // raw_payload is the single source of truth: header(32) + fields.
  // Cast data() directly to hrr_args_* for field access.
  // Full struct bytes: header(32) + fields. Cast data() directly to hrr_args_*.
  std::vector<uint8_t> raw_payload;

  const hrr_event_header& header() const {
    return *reinterpret_cast<const hrr_event_header*>(raw_payload.data());
  }

  // Stream handle for kernel launch events (raw hipStream_t pointer)
  uint64_t stream_handle = 0;

  // Generic single-handle field: stream, event, or module depending on event_type
  uint64_t handle64 = 0;

  // Typed payload (set based on event_type)
  MallocEvent          malloc_ev{};
  MemcpyEvent          memcpy_ev{};
  ModuleLoadEvent      module_load_ev{};
  MallocAsyncEvent     malloc_async_ev{};
  FreeAsyncEvent       free_async_ev{};
  StreamCreateEvent    stream_create_ev{};
  EventRecordEvent     event_record_ev{};
  StreamWaitEventEvent stream_wait_ev{};

  // Heap-allocated for kernel launches (variable-length binary payload)
  KernelLaunchEvent* kernel_launch = nullptr;

  ~Event()                       { delete kernel_launch; }
  Event()                        = default;
  Event(Event&& o) noexcept;
  Event& operator=(Event&& o) noexcept;
  Event(const Event&)            = delete;
  Event& operator=(const Event&) = delete;
};

// ---------------------------------------------------------------------------
// Full loaded archive
// ---------------------------------------------------------------------------

struct Archive {
  std::string path;
  uint16_t    version = 0;        // format version from hrr_file_header
  std::vector<Event> events;

  // Crash-resilience status, set by load_archive:
  //   complete  — the clean-shutdown trailer (hrr_eof_record) was found, so the
  //               capturing process exited normally and the archive is whole.
  //   truncated — a torn trailing record was detected and discarded; all
  //               complete records before it were recovered. Replay still works.
  // A capture interrupted by a crash typically has complete=false; it may also
  // have truncated=true if the final record was only partially written.
  bool complete  = false;
  bool truncated = false;

  // Content-addressed blobs: hash_hex -> file path
  std::unordered_map<std::string, std::string> blobs;
  // Code objects: hash_hex -> file path
  std::unordered_map<std::string, std::string> code_objects;

  size_t event_count       = 0;
  size_t kernel_count      = 0;
  size_t blob_count        = 0;
  size_t code_object_count = 0;

  // Distinct thread IDs that appear in the archive, in first-seen order.
  // Populated during load_archive — no extra scan needed by the replayer.
  std::vector<uint64_t> threads;
};

// ---------------------------------------------------------------------------
// API
// ---------------------------------------------------------------------------

// Load an archive from disk. Returns false on error.
bool load_archive(const std::string& path, Archive& archive);

// ---------------------------------------------------------------------------
// Record-stream primitives
//
// events.bin and the external-region sidecars (regions/*.hrrr, see
// hrr_regions.h) share one on-disk shape: an 8-byte hrr_file_header followed by
// self-delimiting hrr_event_header + payload records. These primitives are that
// shared framing, so both readers get the same torn-tail recovery.
// ---------------------------------------------------------------------------

enum class RecordStatus {
  Ok,           // a complete record was read into the output buffer
  EndOfStream,  // clean EOF exactly at a record boundary
  Torn          // partial record at the tail; everything before it is intact
};

// Read one framed record into `out` (resized to the record's full length,
// header included). Only the final record of a stream can ever be Torn, because
// the writer appends whole records.
RecordStatus read_raw_record(FILE* f, std::vector<uint8_t>& out);

// Open a record stream and validate its 8-byte header. `expected_magic` selects
// the stream kind (HRR_MAGIC for events.bin, HRR_REGION_MAGIC for a region
// sidecar). On success returns the open FILE* positioned at the first record
// and stores the file's format version in *version; returns nullptr and logs on
// a missing file, bad magic, or version mismatch.
FILE* open_record_stream(const std::string& file_path, uint32_t expected_magic,
                         uint16_t expected_version, uint16_t* version);

// Sidecar region streams for a resolved archive directory: every
// <archive_dir>/regions/*.hrrr, sorted by name for a deterministic merge order.
// Returns an empty vector when the directory does not exist, which is the
// normal case for a capture with no external producer.
std::vector<std::string> find_region_streams(const std::string& archive_dir);

// Read a blob's bytes given its hash. Returns false if not found.
bool read_blob(const Archive& archive, uint64_t hash_lo, uint64_t hash_hi,
               std::vector<uint8_t>& data);

// Read a code object (.hsaco) given its hash. Returns false if not found.
bool read_code_object(const Archive& archive, uint64_t hash_lo, uint64_t hash_hi,
                      std::vector<uint8_t>& data);

// Format a 128-bit hash as a 32-character hex string.
std::string hash_hex(uint64_t lo, uint64_t hi);

// Human-readable API name for an event_type (uses hrr_api_names[]).
const char* event_type_name(uint16_t type);

}  // namespace hrr
