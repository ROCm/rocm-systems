/* Copyright (c) 2026 Advanced Micro Devices, Inc. - MIT License */

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

#include <atomic>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <mutex>
#include <string>

#ifdef _WIN32
#  include <windows.h>
static uint64_t now_ns() {
  LARGE_INTEGER freq, count;
  QueryPerformanceFrequency(&freq);
  QueryPerformanceCounter(&count);
  return static_cast<uint64_t>(count.QuadPart * 1000000000LL / freq.QuadPart);
}
#else
#  include <time.h>
static uint64_t now_ns() {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return static_cast<uint64_t>(ts.tv_sec) * 1000000000ULL +
         static_cast<uint64_t>(ts.tv_nsec);
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

// ---------------------------------------------------------------------------
// Directory helpers
// ---------------------------------------------------------------------------

static void ensure_dir(const std::string& path) {
  fs::create_directories(path);
}

// ---------------------------------------------------------------------------
// open / close / flush
// ---------------------------------------------------------------------------

bool open(const char* output_dir) {
  g_output_dir = output_dir;
  ensure_dir(g_output_dir);
  ensure_dir(g_output_dir + "/blobs");
  ensure_dir(g_output_dir + "/code_objects");

  std::string events_path = g_output_dir + "/events.bin";
  g_events_file = fopen(events_path.c_str(), "wb");
  if (!g_events_file) {
    fprintf(stderr, "[HRR capture] Failed to open %s for writing\n",
            events_path.c_str());
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
// Thread ID helper — cached per-thread (OS call runs once per thread)
// ---------------------------------------------------------------------------

#ifdef _WIN32
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

// ---------------------------------------------------------------------------
// write_event_raw — unified write path for all events
//
// hdr points to the hrr_event_header at the front of an hrr_args_* struct.
// payload_len is sizeof the full hrr_args_* struct (header + fields).
// Fills all header fields then does a single fwrite of the whole struct.
// ---------------------------------------------------------------------------

void write_event_raw(uint16_t api_id, hrr_event_header* hdr, uint16_t payload_len) {
  hdr->event_type     = api_id;
  hdr->sequence_id    = g_seq_id.fetch_add(1, std::memory_order_relaxed);
  hdr->timestamp_ns   = now_ns();
  hdr->thread_id      = current_thread_id();
  hdr->payload_length = payload_len;
  memset(hdr->reserved, 0, sizeof(hdr->reserved));

  std::lock_guard<std::mutex> lk(g_file_mu);
  if (!g_events_file) return;
  fwrite(hdr, 1, payload_len, g_events_file);
  g_event_count.fetch_add(1, std::memory_order_relaxed);
}

// ---------------------------------------------------------------------------
// write_blob
// ---------------------------------------------------------------------------

Hash128 write_blob(const void* data, size_t len) {
  Hash128 h = hash_buffer(data, len);
  char hex[33];
  hash_hex(h, hex);

  // blobs/<2-char-prefix>/<fullhash>.blob
  std::string subdir = g_output_dir + "/blobs/" + std::string(hex, 2);
  ensure_dir(subdir);
  std::string path = subdir + "/" + hex + ".blob";

  // Skip if already written
  if (fs::exists(path)) return h;

  FILE* f = fopen(path.c_str(), "wb");
  if (f) {
    fwrite(data, 1, len, f);
    fclose(f);
    g_blob_count.fetch_add(1, std::memory_order_relaxed);
  }
  return h;
}

// ---------------------------------------------------------------------------
// write_code_object
// ---------------------------------------------------------------------------

Hash128 write_code_object(const void* image, size_t image_size) {
  Hash128 h = hash_buffer(image, image_size);
  char hex[33];
  hash_hex(h, hex);

  std::string path = g_output_dir + "/code_objects/" + hex + ".hsaco";
  if (fs::exists(path)) return h;

  FILE* f = fopen(path.c_str(), "wb");
  if (f) {
    fwrite(image, 1, image_size, f);
    fclose(f);
    g_blob_count.fetch_add(1, std::memory_order_relaxed);
  }
  return h;
}

// ---------------------------------------------------------------------------
// Counters
// ---------------------------------------------------------------------------

uint64_t event_count() { return g_event_count.load(); }
uint64_t blob_count()  { return g_blob_count.load(); }

}  // namespace writer
}  // namespace hrr_cap
