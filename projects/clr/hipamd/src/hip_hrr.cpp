/* Copyright (c) 2026 Advanced Micro Devices, Inc.

 Permission is hereby granted, free of charge, to any person obtaining a copy
 of this software and associated documentation files (the "Software"), to deal
 in the Software without restriction, including without limitation the rights
 to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 copies of the Software, and to permit persons to whom the Software is
 furnished to do so, subject to the following conditions:

 The above copyright notice and this permission notice shall be included in
 all copies or substantial portions of the Software.

 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 THE SOFTWARE. */

// HIP Record & Replay (HRR) - Trace writer implementation
//
// Captures HIP API calls to a .hrr trace archive when HIP_RECORD=1 is set.
// Uses content-addressed blob store with XXH3-128 for buffer deduplication.
// All writes are mutex-protected for thread safety.

#include "hip_hrr.h"

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <unordered_map>

#ifdef _WIN32
#include <windows.h>
#include <direct.h>
#define HRR_MKDIR(path) _mkdir(path)
#define HRR_PATH_SEP "\\"
#else
#include <sys/stat.h>
#include <sys/types.h>
#include <time.h>
#define HRR_MKDIR(path) mkdir(path, 0755)
#define HRR_PATH_SEP "/"
#endif

// We use a simple FNV-1a as a fast hash for the blob store since we don't
// want to pull in xxhash as a dependency in the initial skeleton.
// TODO: Replace with XXH3-128 for production (faster + better distribution).

namespace {

// --- 128-bit hash type (placeholder for XXH3-128) ---
struct Hash128 {
  uint64_t lo;
  uint64_t hi;

  bool operator==(const Hash128& o) const { return lo == o.lo && hi == o.hi; }
};

struct Hash128Hasher {
  size_t operator()(const Hash128& h) const { return h.lo ^ h.hi; }
};

Hash128 hash_buffer(const void* data, size_t len) {
  // FNV-1a 128-bit (simplified: two interleaved 64-bit FNV-1a)
  uint64_t h1 = 0xcbf29ce484222325ULL;
  uint64_t h2 = 0x100000001b3ULL;
  const uint8_t* p = static_cast<const uint8_t*>(data);
  for (size_t i = 0; i < len; i++) {
    h1 ^= p[i];
    h1 *= 0x100000001b3ULL;
    h2 ^= p[i];
    h2 *= 0xcbf29ce484222325ULL;
  }
  return {h1, h2};
}

// --- Global state ---
struct HrrState {
  bool active = false;
  hrr::RecordMode record_mode = hrr::RecordMode::Inputs;
  std::string output_dir;
  std::string kernel_filter;
  size_t max_blob_mb = 0;  // 0 = no limit
  bool compress = false;

  FILE* events_file = nullptr;
  std::mutex mu;
  std::atomic<uint64_t> seq_id{0};

  // Track allocations: device_ptr -> {size, handle}
  struct AllocInfo {
    size_t size;
    uint64_t handle;
  };
  std::unordered_map<uintptr_t, AllocInfo> allocs;
  uint64_t next_handle = 1;

  // Track which blobs we've already written (dedup)
  std::unordered_map<Hash128, bool, Hash128Hasher> written_blobs;

  // Track modules: module_ptr -> handle
  std::unordered_map<uintptr_t, uint64_t> modules;
  uint64_t next_module_handle = 1;
};

HrrState g_state;

// --- Helpers ---

uint64_t now_ns() {
#ifdef _WIN32
  LARGE_INTEGER freq, count;
  QueryPerformanceFrequency(&freq);
  QueryPerformanceCounter(&count);
  return static_cast<uint64_t>(count.QuadPart * 1000000000LL / freq.QuadPart);
#else
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return static_cast<uint64_t>(ts.tv_sec) * 1000000000ULL +
         static_cast<uint64_t>(ts.tv_nsec);
#endif
}

void make_dir(const std::string& path) {
  HRR_MKDIR(path.c_str());
}

// Format hash as hex string
std::string hash_hex(const Hash128& h) {
  char buf[33];
  snprintf(buf, sizeof(buf), "%016llx%016llx",
           (unsigned long long)h.lo, (unsigned long long)h.hi);
  return std::string(buf);
}

// Write a blob to the content-addressed store. Returns the hash.
// Deduplicates: if the blob already exists, skips the write.
Hash128 write_blob(const void* data, size_t len) {
  Hash128 h = hash_buffer(data, len);

  // Check dedup (under lock, but blob write is outside critical section)
  {
    std::lock_guard<std::mutex> lock(g_state.mu);
    if (g_state.written_blobs.count(h)) {
      return h;
    }
    g_state.written_blobs[h] = true;
  }

  // Two-level directory: blobs/ab/ab1234...
  std::string hex = hash_hex(h);
  std::string subdir = g_state.output_dir + HRR_PATH_SEP "blobs" HRR_PATH_SEP +
                        hex.substr(0, 2);
  make_dir(subdir);

  std::string path = subdir + HRR_PATH_SEP + hex + ".blob";
  FILE* f = fopen(path.c_str(), "wb");
  if (f) {
    fwrite(data, 1, len, f);
    fclose(f);
  }
  return h;
}

void write_event(hrr::EventType type, uint32_t stream_id, uint16_t device_id,
                 const void* payload, uint16_t payload_len) {
  hrr::EventHeader hdr;
  hdr.magic = hrr::HRR_MAGIC;
  hdr.version = hrr::HRR_VERSION;
  hdr.event_type = static_cast<uint16_t>(type);
  hdr.sequence_id = g_state.seq_id.fetch_add(1, std::memory_order_relaxed);
  hdr.timestamp_ns = now_ns();
  hdr.stream_id = stream_id;
  hdr.device_id = device_id;
  hdr.payload_length = payload_len;

  std::lock_guard<std::mutex> lock(g_state.mu);
  if (g_state.events_file) {
    fwrite(&hdr, sizeof(hdr), 1, g_state.events_file);
    if (payload && payload_len > 0) {
      fwrite(payload, 1, payload_len, g_state.events_file);
    }
  }
}

void write_manifest() {
  std::string path = g_state.output_dir + HRR_PATH_SEP "manifest.json";
  FILE* f = fopen(path.c_str(), "w");
  if (!f) return;

  const char* mode_str = "inputs";
  if (g_state.record_mode == hrr::RecordMode::Timeline) mode_str = "timeline";
  else if (g_state.record_mode == hrr::RecordMode::Full) mode_str = "full";

  fprintf(f,
    "{\n"
    "  \"version\": 1,\n"
    "  \"format\": \"hrr-v1\",\n"
    "  \"capture_mode\": \"%s\",\n"
    "  \"event_count\": %llu,\n"
    "  \"blob_count\": %zu\n"
    "}\n",
    mode_str,
    (unsigned long long)g_state.seq_id.load(),
    g_state.written_blobs.size());
  fclose(f);
}

bool matches_kernel_filter(const char* name) {
  if (g_state.kernel_filter.empty()) return true;
  if (!name) return true;

  const std::string& filter = g_state.kernel_filter;
  // Simple glob: only trailing * supported
  if (filter.back() == '*') {
    return strncmp(name, filter.c_str(), filter.size() - 1) == 0;
  }
  return strcmp(name, filter.c_str()) == 0;
}

}  // anonymous namespace

// ============================================================================
// Public API
// ============================================================================

namespace hrr {

void init() {
  // Check env var
  const char* record_env = getenv("HIP_RECORD");
  if (!record_env || strcmp(record_env, "1") != 0) {
    g_state.active = false;
    return;
  }

  // Output directory
  const char* output_env = getenv("HIP_RECORD_OUTPUT");
  if (output_env && output_env[0]) {
    g_state.output_dir = output_env;
  } else {
    // Default: capture_YYYYMMDD_HHMMSS.hrr in current directory
    time_t t = time(nullptr);
    struct tm tm;
#ifdef _WIN32
    localtime_s(&tm, &t);
#else
    localtime_r(&t, &tm);
#endif
    char buf[64];
    snprintf(buf, sizeof(buf), "capture_%04d%02d%02d_%02d%02d%02d.hrr",
             tm.tm_year + 1900, tm.tm_mon + 1, tm.tm_mday,
             tm.tm_hour, tm.tm_min, tm.tm_sec);
    g_state.output_dir = buf;
  }

  // Recording mode
  const char* mode_env = getenv("HIP_RECORD_MODE");
  if (mode_env) {
    if (strcmp(mode_env, "timeline") == 0)
      g_state.record_mode = RecordMode::Timeline;
    else if (strcmp(mode_env, "full") == 0)
      g_state.record_mode = RecordMode::Full;
    else
      g_state.record_mode = RecordMode::Inputs;
  }

  // Kernel filter
  const char* filter_env = getenv("HIP_RECORD_KERNEL_FILTER");
  if (filter_env) g_state.kernel_filter = filter_env;

  // Max blob size
  const char* max_blob_env = getenv("HIP_RECORD_MAX_BLOB_MB");
  if (max_blob_env) g_state.max_blob_mb = static_cast<size_t>(atol(max_blob_env));

  // Compress
  const char* compress_env = getenv("HIP_RECORD_COMPRESS");
  if (compress_env && strcmp(compress_env, "1") == 0) g_state.compress = true;

  // Create output directory structure
  make_dir(g_state.output_dir);
  make_dir(g_state.output_dir + HRR_PATH_SEP "blobs");
  make_dir(g_state.output_dir + HRR_PATH_SEP "code_objects");

  // Open events file
  std::string events_path = g_state.output_dir + HRR_PATH_SEP "events.bin";
  g_state.events_file = fopen(events_path.c_str(), "wb");
  if (!g_state.events_file) {
    fprintf(stderr, "[HRR] ERROR: Failed to open %s for writing\n",
            events_path.c_str());
    return;
  }

  g_state.active = true;
  fprintf(stderr, "[HRR] Recording to %s (mode=%s)\n",
          g_state.output_dir.c_str(),
          mode_env ? mode_env : "inputs");
}

void shutdown() {
  if (!g_state.active) return;

  // Flush and close events
  if (g_state.events_file) {
    fflush(g_state.events_file);
    fclose(g_state.events_file);
    g_state.events_file = nullptr;
  }

  // Write manifest
  write_manifest();

  fprintf(stderr, "[HRR] Recording complete: %llu events, %zu blobs\n",
          (unsigned long long)g_state.seq_id.load(),
          g_state.written_blobs.size());

  g_state.active = false;
}

bool enabled() {
  return g_state.active;
}

RecordMode mode() {
  return g_state.record_mode;
}

// --- Memory recording ---

void record_malloc(const void* ptr, size_t size, unsigned int flags) {
  if (!g_state.active) return;

  uint64_t handle;
  {
    std::lock_guard<std::mutex> lock(g_state.mu);
    handle = g_state.next_handle++;
    g_state.allocs[reinterpret_cast<uintptr_t>(ptr)] = {size, handle};
  }

  // Payload: ptr_handle(8) + size(8) + flags(4) = 20 bytes
  struct {
    uint64_t ptr_handle;
    uint64_t size;
    uint32_t flags;
  } __attribute__((packed)) payload = {handle, size, flags};

  write_event(EVENT_MALLOC, 0, 0, &payload, sizeof(payload));
}

void record_free(const void* ptr) {
  if (!g_state.active) return;

  uint64_t handle = 0;
  {
    std::lock_guard<std::mutex> lock(g_state.mu);
    auto it = g_state.allocs.find(reinterpret_cast<uintptr_t>(ptr));
    if (it != g_state.allocs.end()) {
      handle = it->second.handle;
      g_state.allocs.erase(it);
    }
  }

  write_event(EVENT_FREE, 0, 0, &handle, sizeof(handle));
}

void record_memcpy(void* dst, const void* src, size_t size_bytes,
                   unsigned int kind, const void* stream) {
  if (!g_state.active) return;
  if (g_state.record_mode == RecordMode::Timeline) {
    // Timeline mode: just log the event, no data capture
    struct {
      uint64_t dst_handle;
      uint64_t src_handle;
      uint64_t size;
      uint32_t kind;
    } __attribute__((packed)) payload = {
      reinterpret_cast<uint64_t>(dst),
      reinterpret_cast<uint64_t>(src),
      size_bytes,
      kind
    };
    write_event(EVENT_MEMCPY, static_cast<uint32_t>(
                reinterpret_cast<uintptr_t>(stream)), 0,
                &payload, sizeof(payload));
    return;
  }

  // For H2D (kind=1): capture the source data
  // For D2H (kind=2): we'd need to capture after the copy completes
  // For now, capture H2D source data as a blob
  Hash128 blob_hash = {};
  if (kind == 1 && src && size_bytes > 0) {  // hipMemcpyHostToDevice
    if (g_state.max_blob_mb == 0 ||
        size_bytes <= g_state.max_blob_mb * 1024 * 1024) {
      blob_hash = write_blob(src, size_bytes);
    }
  }

  // Payload: dst(8) + src(8) + size(8) + kind(4) + blob_hash(16) = 44 bytes
  struct {
    uint64_t dst_addr;
    uint64_t src_addr;
    uint64_t size;
    uint32_t kind;
    uint64_t hash_lo;
    uint64_t hash_hi;
  } __attribute__((packed)) payload = {
    reinterpret_cast<uint64_t>(dst),
    reinterpret_cast<uint64_t>(src),
    size_bytes,
    kind,
    blob_hash.lo,
    blob_hash.hi
  };

  write_event(EVENT_MEMCPY,
              static_cast<uint32_t>(reinterpret_cast<uintptr_t>(stream)),
              0, &payload, sizeof(payload));
}

void record_memset(void* dst, int value, size_t size_bytes,
                   const void* stream) {
  if (!g_state.active) return;

  struct {
    uint64_t dst_addr;
    uint32_t value;
    uint64_t size;
  } __attribute__((packed)) payload = {
    reinterpret_cast<uint64_t>(dst),
    static_cast<uint32_t>(value),
    size_bytes
  };

  write_event(EVENT_MEMSET,
              static_cast<uint32_t>(reinterpret_cast<uintptr_t>(stream)),
              0, &payload, sizeof(payload));
}

// --- Module recording ---

void record_module_load(hipModule_t module, const void* image,
                        size_t image_size) {
  if (!g_state.active) return;

  uint64_t mod_handle;
  {
    std::lock_guard<std::mutex> lock(g_state.mu);
    mod_handle = g_state.next_module_handle++;
    g_state.modules[reinterpret_cast<uintptr_t>(module)] = mod_handle;
  }

  // Save code object to code_objects/ directory
  Hash128 co_hash = {};
  if (image && image_size > 0) {
    co_hash = hash_buffer(image, image_size);

    std::string hex = hash_hex(co_hash);
    std::string path = g_state.output_dir + HRR_PATH_SEP "code_objects" +
                       HRR_PATH_SEP + hex + ".hsaco";

    // Only write if not already present
    FILE* check = fopen(path.c_str(), "rb");
    if (!check) {
      FILE* f = fopen(path.c_str(), "wb");
      if (f) {
        fwrite(image, 1, image_size, f);
        fclose(f);
      }
    } else {
      fclose(check);
    }
  }

  // Payload: code_obj_hash(16) + module_handle(8) = 24 bytes
  struct {
    uint64_t hash_lo;
    uint64_t hash_hi;
    uint64_t module_handle;
  } __attribute__((packed)) payload = {
    co_hash.lo, co_hash.hi, mod_handle
  };

  write_event(EVENT_MODULE_LOAD, 0, 0, &payload, sizeof(payload));
}

void record_module_unload(hipModule_t module) {
  if (!g_state.active) return;

  uint64_t mod_handle = 0;
  {
    std::lock_guard<std::mutex> lock(g_state.mu);
    auto it = g_state.modules.find(reinterpret_cast<uintptr_t>(module));
    if (it != g_state.modules.end()) {
      mod_handle = it->second;
      g_state.modules.erase(it);
    }
  }

  write_event(EVENT_MODULE_UNLOAD, 0, 0, &mod_handle, sizeof(mod_handle));
}

// --- Kernel launch recording ---

void record_kernel_launch(const char* kernel_name,
                          uint32_t grid_x, uint32_t grid_y, uint32_t grid_z,
                          uint32_t block_x, uint32_t block_y, uint32_t block_z,
                          uint32_t shared_mem,
                          const void* stream,
                          void** kernel_args, size_t num_args) {
  if (!g_state.active) return;
  if (!matches_kernel_filter(kernel_name)) return;

  // Build a variable-length payload:
  //   kernel_name_len(2) + kernel_name(N) + grid(12) + block(12) +
  //   shared_mem(4) + num_args(2)
  // Kernel arg capture (buffer snapshots) will be added in Milestone 2.

  std::string name_str = kernel_name ? kernel_name : "<unknown>";
  uint16_t name_len = static_cast<uint16_t>(
      name_str.size() > 65535 ? 65535 : name_str.size());

  size_t payload_size = 2 + name_len + 12 + 12 + 4 + 2;
  std::vector<uint8_t> payload(payload_size);
  uint8_t* p = payload.data();

  // kernel_name_len
  memcpy(p, &name_len, 2); p += 2;
  // kernel_name
  memcpy(p, name_str.c_str(), name_len); p += name_len;
  // grid_dim
  memcpy(p, &grid_x, 4); p += 4;
  memcpy(p, &grid_y, 4); p += 4;
  memcpy(p, &grid_z, 4); p += 4;
  // block_dim
  memcpy(p, &block_x, 4); p += 4;
  memcpy(p, &block_y, 4); p += 4;
  memcpy(p, &block_z, 4); p += 4;
  // shared_mem
  memcpy(p, &shared_mem, 4); p += 4;
  // num_args (placeholder - full arg capture in Milestone 2)
  uint16_t n = static_cast<uint16_t>(num_args);
  memcpy(p, &n, 2); p += 2;

  write_event(EVENT_KERNEL_LAUNCH,
              static_cast<uint32_t>(reinterpret_cast<uintptr_t>(stream)),
              0, payload.data(),
              static_cast<uint16_t>(payload_size > 65535 ? 65535 : payload_size));
}

// --- Synchronization recording ---

void record_device_sync() {
  if (!g_state.active) return;
  write_event(EVENT_DEVICE_SYNC, 0, 0, nullptr, 0);
}

void record_stream_sync(const void* stream) {
  if (!g_state.active) return;
  write_event(EVENT_STREAM_SYNC,
              static_cast<uint32_t>(reinterpret_cast<uintptr_t>(stream)),
              0, nullptr, 0);
}

}  // namespace hrr
