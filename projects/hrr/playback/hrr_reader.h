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

#include "hrr/hrr_api_args.h"
#include "hrr_replay_limits.h"  // kDefaultMaxEvents

// Binary format (v5):
//   events.bin:
//     [0..7]   hrr_file_header  { HRR_MAGIC, HRR_VERSION, reserved }
//     [8..]    hrr_event_header (32 bytes) + payload bytes, repeated per event
//   blobs/<2hex>/  — content-addressed raw buffers (FNV-1a-128 hash)
//   code_objects/  — .hsaco ELFs keyed by hash

namespace hrr {

// Handle fields are raw runtime pointers cast to uint64_t.

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

// Pinned host snapshot (v10): the blob replay writes to [offset, offset +
// length) of the pinned allocation recorded at ptr_handle before the launch.
struct BufferSnapshot {
  uint64_t ptr_handle;
  uint64_t offset;
  uint64_t length;
  uint64_t hash_lo;
  uint64_t hash_hi;
  uint8_t  direction;
};

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


struct Event {
  // raw_payload is the single source of truth: header(32) + fields.
  std::vector<uint8_t> raw_payload;

  const hrr_event_header& header() const {
    return *reinterpret_cast<const hrr_event_header*>(raw_payload.data());
  }

  uint64_t stream_handle = 0;

  // Generic single-handle field: stream, event, or module depending on event_type
  uint64_t handle64 = 0;

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


struct Archive {
  std::string path;
  uint16_t    version = 0;        // format version from hrr_file_header
  uint16_t    flags = 0;          // HRR_FILE_FLAG_* bits from hrr_file_header
  std::vector<Event> events;

  bool complete  = false;
  bool truncated = false;
  // The trailer was found and how many events it counts. trailer && !complete
  // means the count disagrees with the records before it.
  bool     trailer        = false;
  uint64_t trailer_events = 0;
  // Whole records found after the trailer. They are not replayed, and the
  // trailer's event count must match the records before it for complete=true.
  size_t skipped_after_trailer = 0;

  // Content-addressed blobs: hash_hex -> file path
  std::unordered_map<std::string, std::string> blobs;
  // Code objects: hash_hex -> file path
  std::unordered_map<std::string, std::string> code_objects;

  size_t event_count       = 0;
  size_t kernel_count      = 0;
  size_t blob_count        = 0;
  size_t code_object_count = 0;

  // Distinct thread IDs in first-seen order.
  std::vector<uint64_t> threads;
};


// Load an archive from disk. Returns false on error, including a load that
// would take `bytes` past its cap; the diagnostic names --max-host-bytes.
bool load_archive(const std::string& path, Archive& archive,
                  ByteBudget* bytes = nullptr);

// The pid in a per-process archive directory name, "pid-<decimal pid>", or 0 for any other name.
uint64_t pid_from_archive_dir_name(const std::string& name);

// Whether load_archive would find the archive in `archive_dir` complete:
// a trailer whose total_events counts the records before it.
bool has_clean_trailer(const std::string& archive_dir);


// payload_length is file-supplied. Without a ceiling, a corrupt header can ask
// resize() for ~4 GiB and OOM the process.
constexpr uint32_t kMaxRecordBytes = 64u * 1024u * 1024u;

enum class RecordStatus {
  Ok,           // a complete record was read into the output buffer
  EndOfStream,  // clean EOF exactly at a record boundary
  Torn,         // partial record at the tail; everything before it is intact
  OverBudget    // the record would not fit the budget; nothing was allocated
};

// Read one framed record into `out` (resized to its full length, header included).
// Only the final record of a stream can be Torn.
RecordStatus read_raw_record(FILE* f, std::vector<uint8_t>& out,
                             ByteBudget* budget = nullptr);

// Open a record stream and validate its 8-byte header. `expected_magic` selects the stream
// kind: HRR_MAGIC for events.bin, HRR_REGION_MAGIC for a region stream.
FILE* open_record_stream(const std::string& file_path, uint32_t expected_magic,
                         uint16_t expected_version, uint16_t* version,
                         uint16_t* flags = nullptr);

// Sidecar region streams for a resolved archive directory: every
// <archive_dir>/regions/*.hrrr, sorted by name for a deterministic merge order.
std::vector<std::string> find_region_streams(const std::string& archive_dir);

// Largest file read_file_capped() and the readers below load whole.
// A larger blob or code object is refused, not allocated.
constexpr uint64_t kDefaultMaxFileBytes = 4ull << 30;
uint64_t max_file_bytes();
void set_max_file_bytes(uint64_t bytes);

// Most events load_archive() holds. It fails at the first record past the cap,
// naming the count and the cap.
uint64_t max_events();
void set_max_events(uint64_t events);

bool read_file_capped(const std::string& path, std::vector<uint8_t>& data,
                      std::string* error = nullptr, bool* refused = nullptr);

bool read_blob(const Archive& archive, uint64_t hash_lo, uint64_t hash_hi,
               std::vector<uint8_t>& data);

bool read_code_object(const Archive& archive, uint64_t hash_lo, uint64_t hash_hi,
                      std::vector<uint8_t>& data);

std::string hash_hex(uint64_t lo, uint64_t hi);

const char* event_type_name(uint16_t type);

}  // namespace hrr
