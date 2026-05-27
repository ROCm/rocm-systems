/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

/*
 * hip_capture_writer.cpp — Streaming event serialization for the HRR capture layer.
 *
 * Writes events.bin, blobs/, and manifest.json to the output directory.
 * Binary format is compatible with hrr_reader.h / hrr_replay.cpp.
 *
 * Thread-safety: write_event_raw() and write_blob() acquire the file mutex.
 * open()/close()/flush() are called from a single thread (init/shutdown).
 */

#include "hip_capture_writer.h"
#include "hip_capture.h"

#include "os/os.hpp"           // amd::Os::timeNanos()
#include "utils/debug.hpp"     // LogPrintfError, LogPrintfWarning, LogPrintfInfo

#include <atomic>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <mutex>
#include <string>
#include <unordered_set>

#ifdef _WIN32
#  include <windows.h>
static inline uint64_t current_thread_id() {
  static thread_local uint64_t cached = static_cast<uint64_t>(GetCurrentThreadId());
  return cached;
}
#else
#  include <unistd.h>
#  include <sys/syscall.h>
static inline uint64_t current_thread_id() {
  static thread_local uint64_t cached = static_cast<uint64_t>(syscall(SYS_gettid));
  return cached;
}
#endif

namespace fs = std::filesystem;

namespace hrr_cap {
namespace writer {

// ---------------------------------------------------------------------------
// FNV-1a 128-bit hash (same algorithm as out-of-tree writer)
// ---------------------------------------------------------------------------

static Hash128 hash_buffer(const void* data, size_t len) {
  uint64_t h1 = 0xcbf29ce484222325ULL;
  uint64_t h2 = 0x100000001b3ULL;
  const auto* p = static_cast<const uint8_t*>(data);
  for (size_t i = 0; i < len; i++) {
    h1 ^= p[i]; h1 *= 0x100000001b3ULL;
    h2 ^= p[i]; h2 *= 0xcbf29ce484222325ULL;
  }
  return {h1, h2};
}

static void hash_hex(Hash128 h, char buf[33]) {
  snprintf(buf, 33, "%016llx%016llx",
           static_cast<unsigned long long>(h.lo),
           static_cast<unsigned long long>(h.hi));
}

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------

static std::mutex   g_file_mu;
static FILE*        g_events_file = nullptr;
static std::string  g_output_dir;
static std::atomic<uint64_t> g_seq_id{0};
static std::atomic<uint64_t> g_event_count{0};
static std::atomic<uint64_t> g_blob_count{0};

// In-memory set of blob hex keys already written to disk.
// Eliminates the fs::exists() stat syscall on repeated blobs (common for weight tensors).
// Protected by g_blob_mu (separate from g_file_mu to avoid head-of-line blocking).
// "co:" prefix for code objects matches the playback-side load_code_object key convention.
static std::mutex                      g_blob_mu;
static std::unordered_set<std::string> g_written_blobs;

// ---------------------------------------------------------------------------
// Directory helpers
// ---------------------------------------------------------------------------

static void ensure_dir(const std::string& path) {
  if (path.empty()) return;
  fs::create_directories(path);
}

// ---------------------------------------------------------------------------
// open / close / flush
// ---------------------------------------------------------------------------

bool open(const char* output_dir) {
  if (g_events_file) return true;  // already open — guard against double-invocation
  g_output_dir = output_dir;
  ensure_dir(g_output_dir);
  ensure_dir(g_output_dir + "/blobs");
  ensure_dir(g_output_dir + "/code_objects");

  std::string events_path = g_output_dir + "/events.bin";
  g_events_file = fopen(events_path.c_str(), "wb");
  if (!g_events_file) {
    LogPrintfError("[HRR capture] Failed to open %s for writing", events_path.c_str());
    return false;
  }

  hrr_file_header fh{HRR_MAGIC, HRR_VERSION, 0};
  fwrite(&fh, sizeof(fh), 1, g_events_file);
  return true;
}

void flush(const char* output_dir) {
  {
    std::lock_guard<std::mutex> lk(g_file_mu);
    if (g_events_file) fflush(g_events_file);
  }

  // Write manifest.json
  std::string manifest_path = std::string(output_dir) + "/manifest.json";
  FILE* mf = fopen(manifest_path.c_str(), "w");
  if (mf) {
    fprintf(mf,
            "{\n"
            "  \"version\": 1,\n"
            "  \"capture_mode\": \"in-tree\",\n"
            "  \"event_count\": %llu,\n"
            "  \"blob_count\": %llu\n"
            "}\n",
            static_cast<unsigned long long>(g_event_count.load()),
            static_cast<unsigned long long>(g_blob_count.load()));
    fclose(mf);
  }
}

void close() {
  std::lock_guard<std::mutex> lk(g_file_mu);
  if (g_events_file) {
    fclose(g_events_file);
    g_events_file = nullptr;
  }
}

// ---------------------------------------------------------------------------
// write_event_raw — unified write path for all events
//
// hdr points to the hrr_event_header at the front of an hrr_args_* struct.
// payload_len is sizeof the full hrr_args_* struct (header + fields).
// Fills all header fields then does a single fwrite of the whole struct.
// ---------------------------------------------------------------------------

void write_event_raw(uint16_t api_id, hrr_event_header* hdr, uint16_t payload_len) {
  // Fill fields that don't require the lock (timestamp and thread_id are
  // cheap and per-thread; getting them outside the lock keeps contention low).
  hdr->event_type     = api_id;
  hdr->timestamp_ns   = amd::Os::timeNanos();
  hdr->thread_id      = current_thread_id();
  hdr->payload_length = payload_len;
  memset(hdr->reserved, 0, sizeof(hdr->reserved));

  // Acquire once: assign sequence_id and write atomically so IDs are only
  // consumed for events that are actually written.  Avoids the TOCTOU gap
  // between the old double-lock pattern where a concurrent close() could
  // cause the seq ID to be burned with no corresponding event on disk.
  std::lock_guard<std::mutex> lk(g_file_mu);
  if (!g_events_file) return;
  hdr->sequence_id = g_seq_id.fetch_add(1, std::memory_order_relaxed);
  fwrite(hdr, 1, payload_len, g_events_file);
  g_event_count.fetch_add(1, std::memory_order_relaxed);
}

// ---------------------------------------------------------------------------
// Atomic file write: write to a temp file then rename into place.
//
// g_written_blobs ensures only one thread ever reaches here for a given path,
// so there is no concurrent write to the same temp file. The rename makes the
// blob visible to readers only when fully written — a process crash mid-fwrite
// leaves only the temp file, not a partial final blob.
//
// On Windows, rename() fails when the destination already exists (unlike POSIX
// where it is atomic). Use MoveFileExA(MOVEFILE_REPLACE_EXISTING) instead.
// ---------------------------------------------------------------------------

static bool atomic_write_file(const std::string& path,
                              const void* data, size_t len) {
  std::string tmp = path + ".tmp";
  FILE* f = fopen(tmp.c_str(), "wb");
  if (!f) return false;
  bool ok = (fwrite(data, 1, len, f) == len);
  fclose(f);
  if (!ok) { remove(tmp.c_str()); return false; }
#ifdef _WIN32
  ok = MoveFileExA(tmp.c_str(), path.c_str(), MOVEFILE_REPLACE_EXISTING) != 0;
#else
  ok = (rename(tmp.c_str(), path.c_str()) == 0);
#endif
  if (!ok) remove(tmp.c_str());
  return ok;
}

// ---------------------------------------------------------------------------
// write_blob
// ---------------------------------------------------------------------------

Hash128 write_blob(const void* data, size_t len) {
  {
    std::lock_guard<std::mutex> lk(g_file_mu);
    if (!g_events_file) return {};  // writer not open — drop silently
  }

  Hash128 h = hash_buffer(data, len);

  char hex[33];
  hash_hex(h, hex);
  std::string key(hex);  // no prefix — plain blobs

  {
    std::lock_guard<std::mutex> lk(g_blob_mu);
    if (!g_written_blobs.insert(key).second) return h;  // already written
  }

  // blobs/<2-char-prefix>/<fullhash>.blob
  std::string subdir = g_output_dir + "/blobs/" + std::string(hex, 2);
  ensure_dir(subdir);
  std::string path = subdir + "/" + key + ".blob";

  if (atomic_write_file(path, data, len)) {
    g_blob_count.fetch_add(1, std::memory_order_relaxed);
  } else {
    // Write failed — remove from set so a later call can retry.
    LogPrintfWarning("[HRR capture] Failed to write blob %s", hex);
    std::lock_guard<std::mutex> lk(g_blob_mu);
    g_written_blobs.erase(key);
  }
  return h;
}

// ---------------------------------------------------------------------------
// write_code_object
// ---------------------------------------------------------------------------

Hash128 write_code_object(const void* image, size_t image_size) {
  {
    std::lock_guard<std::mutex> lk(g_file_mu);
    if (!g_events_file) return {};  // writer not open — drop silently
  }

  Hash128 h = hash_buffer(image, image_size);
  char hex[33];
  hash_hex(h, hex);
  std::string key = std::string("co:") + hex;  // namespace to match playback load_code_object key

  {
    std::lock_guard<std::mutex> lk(g_blob_mu);
    if (!g_written_blobs.insert(key).second) return h;  // already written
  }

  std::string path = g_output_dir + "/code_objects/" + hex + ".hsaco";
  if (atomic_write_file(path, image, image_size)) {
    g_blob_count.fetch_add(1, std::memory_order_relaxed);
  } else {
    LogPrintfWarning("[HRR capture] Failed to write code object %s", hex);
    std::lock_guard<std::mutex> lk(g_blob_mu);
    g_written_blobs.erase(key);
  }
  return h;
}

// ---------------------------------------------------------------------------
// Counters / state queries
// ---------------------------------------------------------------------------

bool     is_open()      { std::lock_guard<std::mutex> lk(g_file_mu); return g_events_file != nullptr; }
uint64_t event_count()  { return g_event_count.load(); }
uint64_t blob_count()   { return g_blob_count.load(); }

}  // namespace writer
}  // namespace hrr_cap
