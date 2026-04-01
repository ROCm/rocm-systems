/* Copyright (c) 2026 Advanced Micro Devices, Inc. - MIT License */
#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>
#include <unordered_map>

// HRR Archive Reader - reads .hrr trace archives produced by HIP_RECORD=1

namespace hrr {

constexpr uint32_t MAGIC   = 0x52524845;  // "HRRE"
constexpr uint16_t VERSION = 1;

// Event header (32 bytes) - matches hip_hrr.h
struct EventHeader {
  uint32_t magic;
  uint16_t version;
  uint16_t event_type;
  uint64_t sequence_id;
  uint64_t timestamp_ns;
  uint32_t stream_id;
  uint16_t device_id;
  uint16_t payload_length;
};
static_assert(sizeof(EventHeader) == 32, "EventHeader must be 32 bytes");

// Event types
enum EventType : uint16_t {
  EVENT_MALLOC        = 0x0001,
  EVENT_FREE          = 0x0002,
  EVENT_MEMCPY        = 0x0003,
  EVENT_MEMSET        = 0x0004,
  EVENT_MODULE_LOAD   = 0x0010,
  EVENT_MODULE_UNLOAD = 0x0011,
  EVENT_KERNEL_LAUNCH = 0x0020,
  EVENT_STREAM_CREATE = 0x0030,
  EVENT_STREAM_DESTROY= 0x0031,
  EVENT_STREAM_SYNC   = 0x0032,
  EVENT_DEVICE_SYNC   = 0x0050,
};

// Parsed malloc event
struct MallocEvent {
  uint64_t ptr_handle;
  uint64_t size;
  uint32_t flags;
};

// Parsed memcpy event
// NOTE: must be packed to match the trace writer's #pragma pack(1) layout.
// Writer places hash_lo at byte offset 28 (no gap after kind); natural
// alignment would insert 4 bytes of padding there.
struct __attribute__((packed)) MemcpyEvent {
  uint64_t dst_addr;
  uint64_t src_addr;
  uint64_t size;
  uint32_t kind;
  uint64_t hash_lo;
  uint64_t hash_hi;
};

// Parsed module load event
struct ModuleLoadEvent {
  uint64_t hash_lo;
  uint64_t hash_hi;
  uint64_t module_handle;
};

// Parsed kernel arg
struct KernelArg {
  uint8_t value_kind;   // 0=scalar, 1=pointer, 2=hidden
  uint16_t size;
  std::vector<uint8_t> data;
};

// Parsed buffer snapshot
struct BufferSnapshot {
  uint64_t ptr_handle;
  uint64_t offset;
  uint64_t length;
  uint64_t hash_lo;
  uint64_t hash_hi;
  uint8_t direction;  // 0=input, 1=output
};

// Parsed kernel launch event
struct KernelLaunchEvent {
  std::string kernel_name;
  uint64_t co_hash_lo = 0;  // code object hash (0,0 = unknown, search all modules)
  uint64_t co_hash_hi = 0;
  uint32_t grid[3];
  uint32_t block[3];
  uint32_t shared_mem;
  std::vector<KernelArg> args;
  std::vector<BufferSnapshot> snapshots;
};

// A single event with its header and parsed payload
struct Event {
  EventHeader header;
  // Payload stored as one of the typed structs depending on event_type
  union {
    MallocEvent malloc_ev;
    MemcpyEvent memcpy_ev;
    ModuleLoadEvent module_load_ev;
  };
  // For kernel launch (complex, heap-allocated)
  KernelLaunchEvent* kernel_launch = nullptr;
  // Raw payload for events we don't parse
  std::vector<uint8_t> raw_payload;

  ~Event() { delete kernel_launch; }
  Event() { memset(&malloc_ev, 0, sizeof(memcpy_ev)); }  // sizeof(memcpy_ev) = largest union member
  Event(Event&& o) noexcept;
  Event& operator=(Event&& o) noexcept;
  Event(const Event&) = delete;
  Event& operator=(const Event&) = delete;
};

// Full archive
struct Archive {
  std::string path;
  std::vector<Event> events;
  // Code objects: hash_hex -> file path
  std::unordered_map<std::string, std::string> code_objects;
  // Blob paths: hash_hex -> file path
  std::unordered_map<std::string, std::string> blobs;

  // Summary stats
  size_t event_count = 0;
  size_t kernel_count = 0;
  size_t blob_count = 0;
  size_t code_object_count = 0;
};

// Load an archive from disk. Returns false on error.
bool load_archive(const std::string& path, Archive& archive);

// Read a blob's contents given its hash
bool read_blob(const Archive& archive, uint64_t hash_lo, uint64_t hash_hi,
               std::vector<uint8_t>& data);

// Read a code object given its hash
bool read_code_object(const Archive& archive, uint64_t hash_lo, uint64_t hash_hi,
                      std::vector<uint8_t>& data);

// Format hash as hex string
std::string hash_hex(uint64_t lo, uint64_t hi);

// Get human-readable event type name
const char* event_type_name(uint16_t type);

}  // namespace hrr
