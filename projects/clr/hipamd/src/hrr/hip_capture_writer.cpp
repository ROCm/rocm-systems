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
 * Crash resilience: events.bin is written through a raw file descriptor with a
 * small app-managed buffer (not buffered stdio). The writer checkpoints
 * (flush+fsync) every kCheckpointEvents events to bound how much a crash can
 * lose, and emergency_finalize can be called from CLR's crash callback as a
 * final best-effort flush. A clean shutdown appends an hrr_eof_record trailer;
 * its absence marks the archive as crash-truncated for the reader, which
 * recovers all complete records.
 *
 * Thread-safety: write_event_raw() and write_blob() acquire the file mutex.
 * open()/close()/flush() are called from a single thread (init/shutdown).
 */

#include "hip_capture_writer.h"
#include "hip_capture.h"

#include "os/os.hpp"           // amd::Os::timeNanos()
#include "utils/debug.hpp"     // LogPrintfError, LogPrintfWarning, LogPrintfInfo

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cerrno>
#include <filesystem>
#include <mutex>
#include <string>
#include <unordered_set>
#include <algorithm>
#include <vector>

#ifdef _WIN32
#  include <windows.h>
#  include <io.h>
#  include <fcntl.h>
#  include <process.h>
#  include <sys/stat.h>
// Truncate: new archive. Append: resume into existing events.bin (must NOT use _O_TRUNC).
#  define HRR_OPEN(p)          _open((p), _O_WRONLY | _O_CREAT | _O_TRUNC | _O_BINARY, _S_IREAD | _S_IWRITE)
#  define HRR_OPEN_APPEND(p)   _open((p), _O_RDWR | _O_CREAT | _O_BINARY, _S_IREAD | _S_IWRITE)
#  define HRR_WRITE(fd,b,n)    _write((fd), (b), (unsigned)(n))
#  define HRR_CLOSE(fd)        _close((fd))
#  define HRR_FSYNC(fd)        _commit((fd))

using hrr_stat_t = struct _stat64;
static int hrr_stat_file(const char* path, hrr_stat_t* st) { return _stat64(path, st); }
static std::int64_t hrr_stat_size(const hrr_stat_t& st) { return st.st_size; }

static int hrr_ftruncate_fd(int fd, std::int64_t len) {
  return _chsize_s(fd, len) == 0 ? 0 : -1;
}

static std::int64_t hrr_seek_end(int fd) {
  return static_cast<std::int64_t>(_lseeki64(fd, 0, SEEK_END));
}

static inline uint64_t current_thread_id() {
  static thread_local uint64_t cached = static_cast<uint64_t>(GetCurrentThreadId());
  return cached;
}

static inline uint64_t current_process_id() {
  return static_cast<uint64_t>(_getpid());
}

static inline uint64_t current_parent_process_id() {
  return 0;
}
#else
#  include <unistd.h>
#  include <fcntl.h>
#  include <sys/stat.h>
#  include <sys/syscall.h>
#  include <pthread.h>
#  define HRR_OPEN(p)        ::open((p), O_WRONLY | O_CREAT | O_TRUNC, 0644)
#  define HRR_OPEN_APPEND(p) ::open((p), O_RDWR | O_CREAT, 0644)
#  define HRR_WRITE(fd,b,n)  ::write((fd), (b), (n))
#  define HRR_CLOSE(fd)      ::close((fd))
#  define HRR_FSYNC(fd)      ::fsync((fd))

using hrr_stat_t = struct stat;
static int hrr_stat_file(const char* path, hrr_stat_t* st) { return stat(path, st); }
static std::int64_t hrr_stat_size(const hrr_stat_t& st) {
  return static_cast<std::int64_t>(st.st_size);
}

static int hrr_ftruncate_fd(int fd, std::int64_t len) {
  return ftruncate(fd, static_cast<off_t>(len));
}

static std::int64_t hrr_seek_end(int fd) {
  return static_cast<std::int64_t>(lseek(fd, 0, SEEK_END));
}

static inline uint64_t current_thread_id() {
  static thread_local uint64_t cached = static_cast<uint64_t>(syscall(SYS_gettid));
  return cached;
}

static inline uint64_t current_process_id() {
  return static_cast<uint64_t>(getpid());
}

static inline uint64_t current_parent_process_id() {
  return static_cast<uint64_t>(getppid());
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

// Buffer must comfortably hold the largest single record. payload_length is a
// uint32_t (v4); in practice the largest record is a kernel-launch event whose
// serialized args/name/by-value structs are well under this. 256 KiB amortizes
// write() syscalls. buffer_append_locked() handles a record larger than kBufCap
// by flushing and writing it directly, so this is a performance bound, not a cap.
static constexpr size_t   kBufCap           = 256u * 1024u;
// Flush+fsync every N events to bound crash data loss.
static constexpr uint64_t kCheckpointEvents = 4096;
// Path buffers are filled at open() so the signal path never touches std::string.
static constexpr size_t   kPathMax          = 4096;

static std::mutex   g_file_mu;
static int          g_events_fd = -1;
// g_base_dir is the archive path requested via HIP_HRR_CAPTURE_OUTPUT.
// g_output_dir is the per-process archive directory this process writes to:
// g_base_dir/pid-<pid>/.
static std::string  g_base_dir;
static std::string  g_output_dir;
static char         g_manifest_path[kPathMax] = {0};
static uint64_t     g_pid = 0;
static uint64_t     g_parent_pid = 0;

// App-managed write buffer for events.bin (protected by g_file_mu).
static uint8_t  g_buf[kBufCap];
static size_t   g_buf_len            = 0;
static uint64_t g_events_since_ckpt  = 0;
static bool     g_trailer_written    = false;

// Set when an event could not be serialized losslessly and had to be dropped
// (e.g. an oversized kernel launch). A capture with this flag set is finalized
// WITHOUT the clean-shutdown trailer and with manifest "complete": false, so the
// reader treats it like a truncated archive rather than a faithful capture.
static std::atomic<bool> g_capture_incomplete{false};

// Crash-callback guard over g_buf / g_buf_len, raised by writer threads while
// they mutate the buffer. If the crash interrupts a writer mid-lock,
// emergency_finalize must not poke g_file_mu's futex from the handler. Writers
// still hold g_file_mu for thread<->thread exclusion; the flag is purely the
// crash-callback <-> writer coordination point.
static std::atomic_flag g_buf_busy = ATOMIC_FLAG_INIT;

// RAII for writer threads: take the thread<->thread mutex AND raise g_buf_busy so
// the crash callback can tell a g_buf mutation is in flight. Member order
// matters: the mutex locks first and unlocks last, with the busy window nested
// strictly inside it.
struct BufWriteGuard {
  std::lock_guard<std::mutex> lk_;
  BufWriteGuard() : lk_(g_file_mu) {
    g_buf_busy.test_and_set(std::memory_order_acquire);
  }
  ~BufWriteGuard() { g_buf_busy.clear(std::memory_order_release); }
};

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
// Low-level fd helpers
// ---------------------------------------------------------------------------

// Write the entire buffer, retrying short writes. Async-signal-safe: uses only
// write(). Returns true if all bytes were written.
static bool write_all_fd(int fd, const void* data, size_t len) {
  const uint8_t* p = static_cast<const uint8_t*>(data);
  size_t off = 0;
  while (off < len) {
    auto n = HRR_WRITE(fd, p + off, len - off);
    if (n <= 0) {
#ifndef _WIN32
      if (n < 0 && errno == EINTR) continue;
#endif
      return false;
    }
    off += static_cast<size_t>(n);
  }
  return true;
}

// Drain the app buffer to the events fd. Caller must hold g_file_mu (or be the
// crash callback that has claimed g_buf_busy via test_and_set). Does not fsync.
static void flush_buffer_locked() {
  if (g_events_fd < 0 || g_buf_len == 0) { g_buf_len = 0; return; }
  write_all_fd(g_events_fd, g_buf, g_buf_len);
  g_buf_len = 0;
}

// Append `len` bytes of one complete record to the buffer, flushing first if it
// would not fit. A single record larger than the buffer (possible now that
// payload_length is a uint32_t — e.g. a kernel launch with very large by-value
// args) is flushed-then-written directly so it never overruns g_buf. Caller must
// hold g_file_mu.
static void buffer_append_locked(const void* data, size_t len) {
  if (g_buf_len + len > kBufCap) flush_buffer_locked();
  if (len > kBufCap) {
    // Oversized record: buffer is now empty (flushed above); write it straight
    // through rather than memcpy'ing past the end of g_buf.
    if (g_events_fd >= 0) write_all_fd(g_events_fd, data, len);
    return;
  }
  memcpy(g_buf + g_buf_len, data, len);
  g_buf_len += len;
}

// ---------------------------------------------------------------------------
// Directory helpers
// ---------------------------------------------------------------------------

static void ensure_dir(const std::string& path) {
  if (path.empty()) return;
  fs::create_directories(path);
}

static bool atomic_write_file(const std::string& path, const void* data, size_t len);

// ---------------------------------------------------------------------------
// manifest writers
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Subprocess / resume helpers
//
// vLLM server mode runs GPU work in a spawned EngineCore child. Each HIP-owning
// process must append to the same events.bin instead of truncating it. On resume
// we scan the existing log (or trust writer_state.json from the last checkpoint),
// strip a clean-shutdown trailer if present, and continue sequence IDs.
// ---------------------------------------------------------------------------

struct ScanResult {
  uint64_t max_seq   = 0;
  uint64_t count     = 0;
  std::int64_t append_at = 0;
  bool     had_trailer = false;
  bool     torn_tail   = false;
};

static ScanResult scan_events_for_resume(FILE* f, std::int64_t file_size) {
  ScanResult r;
  const uint16_t hdr_size = static_cast<uint16_t>(sizeof(hrr_event_header));
  if (fseek(f, static_cast<long>(sizeof(hrr_file_header)), SEEK_SET) != 0)
    return r;

  r.append_at = sizeof(hrr_file_header);
  while (true) {
    long pos = ftell(f);
    if (pos < 0 || static_cast<std::int64_t>(pos) >= file_size) break;

    hrr_event_header h{};
    if (fread(&h, hdr_size, 1, f) != 1) {
      r.torn_tail = true;
      r.append_at = pos;
      break;
    }

    // Valid trailer only if full hrr_eof_record + magic (same rule as hrr_reader).
    if (h.event_type == HRR_EOF_MARKER &&
        h.payload_length == static_cast<uint32_t>(sizeof(hrr_eof_record))) {
      uint64_t total_events = 0;
      uint32_t eof_magic    = 0;
      if (fread(&total_events, sizeof(total_events), 1, f) != 1 ||
          fread(&eof_magic, sizeof(eof_magic), 1, f) != 1) {
        r.torn_tail = true;
        r.append_at = pos;
        break;
      }
      if (eof_magic == HRR_EOF_MAGIC) {
        r.had_trailer = true;
        r.append_at = pos;
        break;
      }
      // Bogus EOF-shaped record: count it and skip past the bytes we read.
      r.max_seq = (h.sequence_id > r.max_seq) ? h.sequence_id : r.max_seq;
      r.count++;
      r.append_at = static_cast<std::int64_t>(ftell(f));
      continue;
    }

    if (h.payload_length < hdr_size) {
      r.torn_tail = true;
      r.append_at = pos;
      break;
    }

    r.max_seq = (h.sequence_id > r.max_seq) ? h.sequence_id : r.max_seq;
    r.count++;

    long body = static_cast<long>(h.payload_length) - static_cast<long>(hdr_size);
    // fseek past EOF "succeeds" on most platforms (it only fails to read on the
    // next fread), so a record claiming e.g. 65535 bytes of body in a 100-byte
    // file would otherwise be accepted and leave append_at pointing past EOF —
    // corrupting the resuming capture's offset and sequence IDs. Validate that
    // the full record body actually fits in the file before trusting it.
    if (fseek(f, body, SEEK_CUR) != 0 ||
        ftell(f) < 0 ||
        static_cast<std::int64_t>(ftell(f)) > file_size) {
      r.torn_tail = true;
      r.append_at = pos;
      break;
    }
    r.append_at = static_cast<std::int64_t>(ftell(f));
  }

  if (!r.had_trailer && !r.torn_tail)
    r.append_at = file_size;
  return r;
}

static bool try_load_writer_state(const std::string& path, std::int64_t file_size,
                                  uint64_t* next_seq, uint64_t* ev_count,
                                  uint64_t* bl_count) {
  FILE* f = fopen(path.c_str(), "r");
  if (!f) return false;

  uint64_t ns = 0, ec = 0, bc = 0;
  long long stored_size = -1;
  char line[256];
  while (fgets(line, sizeof(line), f)) {
    unsigned long long u = 0;
    long long s = 0;
    if (sscanf(line, " \"next_seq\": %llu", &u) == 1) { ns = u; continue; }
    if (sscanf(line, " \"event_count\": %llu", &u) == 1) { ec = u; continue; }
    if (sscanf(line, " \"blob_count\": %llu", &u) == 1) { bc = u; continue; }
    if (sscanf(line, " \"events_file_size\": %lld", &s) == 1) { stored_size = s; continue; }
  }
  fclose(f);

  if (stored_size != file_size)
    return false;
  *next_seq = ns;
  *ev_count = ec;
  *bl_count = bc;
  return true;
}

static void save_writer_state_locked() {
  if (g_output_dir.empty() || g_events_fd < 0) return;
  std::int64_t sz = hrr_seek_end(g_events_fd);
  if (sz < 0) return;

  std::string path = g_output_dir + "/writer_state.json";
  FILE* f = fopen(path.c_str(), "w");
  if (!f) return;
  fprintf(f,
          "{\n"
          "  \"next_seq\": %llu,\n"
          "  \"event_count\": %llu,\n"
          "  \"blob_count\": %llu,\n"
          "  \"events_file_size\": %lld\n"
          "}\n",
          static_cast<unsigned long long>(g_seq_id.load()),
          static_cast<unsigned long long>(g_event_count.load()),
          static_cast<unsigned long long>(g_blob_count.load()),
          static_cast<long long>(sz));
  fclose(f);
}

static void index_existing_blobs_locked() {
  std::lock_guard<std::mutex> lk(g_blob_mu);
  g_written_blobs.clear();

  fs::path blobs_root = g_output_dir + "/blobs";
  if (fs::exists(blobs_root)) {
    for (const auto& ent : fs::recursive_directory_iterator(blobs_root)) {
      if (ent.is_regular_file() && ent.path().extension() == ".blob")
        g_written_blobs.insert(ent.path().stem().string());
    }
  }

  fs::path co_root = g_output_dir + "/code_objects";
  if (fs::exists(co_root)) {
    for (const auto& ent : fs::directory_iterator(co_root)) {
      if (ent.is_regular_file() && ent.path().extension() == ".hsaco")
        g_written_blobs.insert(std::string("co:") + ent.path().stem().string());
    }
  }
}

#ifndef _WIN32
static void atfork_prepare() {
  BufWriteGuard lk;
  if (g_events_fd >= 0)
    flush_buffer_locked();
}

static void atfork_child() {
  std::string dir;
  {
    std::lock_guard<std::mutex> lk(g_file_mu);
    if (g_events_fd >= 0) {
      HRR_CLOSE(g_events_fd);
      g_events_fd = -1;
    }
    g_buf_len = 0;
    g_events_since_ckpt = 0;
    g_trailer_written = false;
    // Re-open from the *base* dir so the forked child selects its own
    // pid-<pid> sub-archive.
    dir = g_base_dir;
  }
  // NOTE: this is hrr_cap::writer::open(const char*) — the writer's archive-open
  // routine — NOT POSIX ::open(). It runs fs::create_directories / fopen,
  // which are not async-signal-safe in general, but pthread_atfork's child
  // handler runs in the (single-threaded) child immediately after fork() with no
  // mutex held, so these calls are safe here. We deliberately do NOT call this
  // from any async-signal context.
  if (!dir.empty())
    (void)writer::open(dir.c_str());
}

static void install_atfork_handlers_once() {
  static std::once_flag once;
  std::call_once(once, [] {
    pthread_atfork(atfork_prepare, nullptr, atfork_child);
  });
}
#endif

static void write_manifest_stdio(const char* output_dir, bool complete) {
  std::string manifest_path = std::string(output_dir) + "/manifest.json";
  FILE* mf = fopen(manifest_path.c_str(), "w");
  if (!mf) return;
  fprintf(mf,
          "{\n"
          "  \"pid\": %llu,\n"
          "  \"parent_pid\": %llu,\n"
          "  \"complete\": %s,\n"
          "  \"event_count\": %llu,\n"
          "  \"blob_count\": %llu\n"
          "}\n",
          static_cast<unsigned long long>(g_pid),
          static_cast<unsigned long long>(g_parent_pid),
          complete ? "true" : "false",
          static_cast<unsigned long long>(g_event_count.load()),
          static_cast<unsigned long long>(g_blob_count.load()));
  fclose(mf);
}

struct ProcessManifestEntry {
  uint64_t pid = 0;
  uint64_t parent_pid = 0;
  uint64_t event_count = 0;
  uint64_t blob_count = 0;
  bool complete = false;
};

static bool read_process_manifest(const std::string& path, ProcessManifestEntry* out) {
  FILE* f = fopen(path.c_str(), "r");
  if (!f) return false;

  ProcessManifestEntry e{};
  bool saw_pid = false;
  char line[256];
  while (fgets(line, sizeof(line), f)) {
    unsigned long long u = 0;
    char b[8] = {};
    if (sscanf(line, " \"pid\": %llu", &u) == 1) {
      e.pid = static_cast<uint64_t>(u);
      saw_pid = true;
      continue;
    }
    if (sscanf(line, " \"parent_pid\": %llu", &u) == 1) {
      e.parent_pid = static_cast<uint64_t>(u);
      continue;
    }
    if (sscanf(line, " \"event_count\": %llu", &u) == 1) {
      e.event_count = static_cast<uint64_t>(u);
      continue;
    }
    if (sscanf(line, " \"blob_count\": %llu", &u) == 1) {
      e.blob_count = static_cast<uint64_t>(u);
      continue;
    }
    if (sscanf(line, " \"complete\": %7[^,\n ]", b) == 1) {
      e.complete = (strcmp(b, "true") == 0);
      continue;
    }
  }
  fclose(f);
  if (!saw_pid) return false;
  *out = e;
  return true;
}

static uint64_t derive_owner_pid(const std::vector<ProcessManifestEntry>& entries) {
  if (entries.empty()) return 0;
  for (const auto& candidate : entries) {
    for (const auto& child : entries) {
      if (child.parent_pid == candidate.pid)
        return candidate.pid;
    }
  }
  return entries.front().pid;
}

static void update_root_manifest() {
  if (g_base_dir.empty()) return;

  std::vector<ProcessManifestEntry> entries;
  for (const auto& ent : fs::directory_iterator(g_base_dir)) {
    if (!ent.is_directory()) continue;
    const std::string name = ent.path().filename().string();
    if (name.rfind("pid-", 0) != 0) continue;
    ProcessManifestEntry entry{};
    if (read_process_manifest((ent.path() / "manifest.json").string(), &entry))
      entries.push_back(entry);
  }

  std::sort(entries.begin(), entries.end(),
            [](const auto& a, const auto& b) { return a.pid < b.pid; });
  const uint64_t owner_pid = derive_owner_pid(entries);

  std::string json;
  json += "{\n";
  json += "  \"version\": 1,\n";
  json += "  \"capture_mode\": \"in-tree\",\n";
  json += "  \"owner_pid\": " + std::to_string(owner_pid) + ",\n";
  json += "  \"processes\": [\n";
  for (size_t i = 0; i < entries.size(); ++i) {
    const auto& e = entries[i];
    json += "    { \"pid\": " + std::to_string(e.pid) +
            ", \"parent_pid\": " + std::to_string(e.parent_pid) +
            ", \"complete\": " + (e.complete ? "true" : "false") +
            ", \"event_count\": " + std::to_string(e.event_count) +
            ", \"blob_count\": " + std::to_string(e.blob_count) + " }";
    json += (i + 1 == entries.size()) ? "\n" : ",\n";
  }
  json += "  ]\n";
  json += "}\n";

  (void)atomic_write_file(g_base_dir + "/manifest.json", json.data(), json.size());
}

// ---------------------------------------------------------------------------
// open / close / flush / checkpoint
// ---------------------------------------------------------------------------

bool open(const char* output_dir) {
  if (g_events_fd >= 0) return true;  // already open — guard against double-invocation
#ifndef _WIN32
  install_atfork_handlers_once();
#endif
  g_base_dir = output_dir;
  ensure_dir(g_base_dir);
  g_pid = current_process_id();
  g_parent_pid = current_parent_process_id();
  char sub[64];
  snprintf(sub, sizeof(sub), "/pid-%llu",
           static_cast<unsigned long long>(g_pid));
  g_output_dir = g_base_dir + sub;

  // Buffer/checkpoint state is reset here so it is consistent for this
  // process's pid-<pid> sub-archive.
  g_buf_len           = 0;
  g_events_since_ckpt = 0;
  g_trailer_written   = false;

  ensure_dir(g_output_dir);
  ensure_dir(g_output_dir + "/blobs");
  ensure_dir(g_output_dir + "/code_objects");

  std::string events_path = g_output_dir + "/events.bin";
  std::string manifest_path = g_output_dir + "/manifest.json";
  snprintf(g_manifest_path, sizeof(g_manifest_path), "%s", manifest_path.c_str());

  hrr_stat_t st{};
  bool exists = false;
  exists = (hrr_stat_file(events_path.c_str(), &st) == 0 && hrr_stat_size(st) > 0);

  if (exists) {
    if (g_events_fd < 0) {
      g_events_fd = HRR_OPEN_APPEND(events_path.c_str());
    }
    if (g_events_fd < 0) {
      LogPrintfError("[HRR capture] Failed to open %s for append", events_path.c_str());
      return false;
    }

    uint64_t next_seq = 0, ev_count = 0, bl_count = 0;
    const std::string state_path = g_output_dir + "/writer_state.json";
    const bool fast = try_load_writer_state(state_path, hrr_stat_size(st),
                                            &next_seq, &ev_count, &bl_count);

    ScanResult scan{};
    if (!fast) {
      FILE* rf = fopen(events_path.c_str(), "rb");
      if (rf) {
        scan = scan_events_for_resume(rf, hrr_stat_size(st));
        fclose(rf);
      }
      next_seq = (scan.count > 0) ? (scan.max_seq + 1) : 0;
      ev_count = scan.count;
    } else if (hrr_stat_file(events_path.c_str(), &st) == 0) {
      FILE* rf = fopen(events_path.c_str(), "rb");
      if (rf) {
        scan = scan_events_for_resume(rf, hrr_stat_size(st));
        fclose(rf);
      }
    }

    if (scan.append_at > 0 && (scan.had_trailer || scan.torn_tail)) {
      if (hrr_ftruncate_fd(g_events_fd, scan.append_at) != 0) {
        LogPrintfWarning("[HRR capture] ftruncate resume at %lld failed", (long long)scan.append_at);
      }
    }
    if (hrr_seek_end(g_events_fd) < 0) {
      LogPrintfError("[HRR capture] seek end of %s failed", events_path.c_str());
      HRR_CLOSE(g_events_fd);
      g_events_fd = -1;
      return false;
    }

    g_seq_id.store(next_seq, std::memory_order_relaxed);
    g_event_count.store(ev_count, std::memory_order_relaxed);
    if (fast)
      g_blob_count.store(bl_count, std::memory_order_relaxed);
    index_existing_blobs_locked();

    LogPrintfInfo("[HRR capture] Resumed archive at %s (events=%llu next_seq=%llu%s%s)",
                  events_path.c_str(),
                  static_cast<unsigned long long>(ev_count),
                  static_cast<unsigned long long>(next_seq),
                  scan.had_trailer ? ", stripped trailer" : "",
                  scan.torn_tail ? ", trimmed torn tail" : "");
    return true;
  }

  // Fresh per-process archive.
  g_seq_id.store(0, std::memory_order_relaxed);
  g_event_count.store(0, std::memory_order_relaxed);
  g_blob_count.store(0, std::memory_order_relaxed);
  {
    std::lock_guard<std::mutex> lk(g_blob_mu);
    g_written_blobs.clear();
  }

  if (g_events_fd < 0) {
#ifndef _WIN32
    g_events_fd = ::open(events_path.c_str(), O_RDWR | O_CREAT | O_TRUNC, 0644);
#else
    g_events_fd = HRR_OPEN(events_path.c_str());
#endif
  }
  if (g_events_fd < 0) {
    LogPrintfError("[HRR capture] Failed to open %s for writing", events_path.c_str());
    return false;
  }

  hrr_file_header fh{HRR_MAGIC, HRR_VERSION, 0};
  buffer_append_locked(&fh, sizeof(fh));
  return true;
}

void checkpoint() {
  BufWriteGuard lk;
  if (g_events_fd < 0) return;
  flush_buffer_locked();
  HRR_FSYNC(g_events_fd);
  save_writer_state_locked();
  g_events_since_ckpt = 0;
}

void mark_incomplete(const char* reason) {
  // Record once; the loud, AMD_LOG_LEVEL-routed message is emitted by the caller
  // (e.g. serialize_kernel_launch) which has the relevant context. Here we only
  // need the durable flag and a single stderr breadcrumb so a bare run still
  // surfaces it.
  if (!g_capture_incomplete.exchange(true, std::memory_order_relaxed)) {
    fprintf(stderr,
            "[HRR capture] Archive marked INCOMPLETE: %s. The clean-shutdown "
            "trailer will be omitted and manifest.complete=false so replay "
            "cannot treat this capture as faithful.\n",
            reason ? reason : "(unspecified)");
  }
}

bool is_incomplete() { return g_capture_incomplete.load(std::memory_order_relaxed); }

void flush(const char* /*output_dir*/) {
  // Always finalize the *effective* directory this process actually wrote to
  // (g_output_dir), which is always a pid-<pid> sub-archive. The caller passes
  // the base HIP_HRR_CAPTURE_OUTPUT path.
  const bool incomplete = g_capture_incomplete.load(std::memory_order_relaxed);
  std::string out_dir;
  {
    BufWriteGuard lk;
    out_dir = g_output_dir;
    // Skip the clean-shutdown trailer when the capture is known incomplete: its
    // absence is exactly how the reader detects a non-faithful archive.
    if (g_events_fd >= 0 && !g_trailer_written && !incomplete) {
      hrr_eof_record rec = hrr_make_eof_record(
          g_seq_id.fetch_add(1, std::memory_order_relaxed), g_event_count.load());
      rec.hdr.timestamp_ns = amd::Os::timeNanos();
      rec.hdr.thread_id    = current_thread_id();
      buffer_append_locked(&rec, sizeof(rec));
      flush_buffer_locked();
      HRR_FSYNC(g_events_fd);
      g_trailer_written = true;
    } else if (g_events_fd >= 0 && incomplete) {
      // Still flush buffered events so nothing is lost, just no trailer.
      flush_buffer_locked();
      HRR_FSYNC(g_events_fd);
    }
  }

  if (out_dir.empty()) return;
  write_manifest_stdio(out_dir.c_str(), /*complete=*/!incomplete);
  update_root_manifest();
  remove((out_dir + "/writer_state.json").c_str());
}

void close() {
  BufWriteGuard lk;
  if (g_events_fd >= 0) {
    flush_buffer_locked();
    HRR_FSYNC(g_events_fd);
    HRR_CLOSE(g_events_fd);
    g_events_fd = -1;
  }
}

// ---------------------------------------------------------------------------
// emergency_finalize — best-effort crash callback path
// ---------------------------------------------------------------------------

// Async-signal-safe unsigned-to-decimal. Writes into out (no NUL), returns len.
static size_t u64_to_dec(uint64_t v, char* out) {
  char tmp[20];
  size_t n = 0;
  if (v == 0) { out[0] = '0'; return 1; }
  while (v) { tmp[n++] = static_cast<char>('0' + (v % 10)); v /= 10; }
  for (size_t i = 0; i < n; i++) out[i] = tmp[n - 1 - i];
  return n;
}

static size_t append_lit(char* out, size_t off, const char* s) {
  size_t i = 0;
  while (s[i]) { out[off + i] = s[i]; i++; }
  return off + i;
}

void emergency_finalize(bool clean_shutdown) {
  if (g_events_fd < 0) return;

  // Flush the in-memory buffer only if no writer thread is mid-mutation.
  // We must NOT touch g_file_mu here: if the crash interrupts a writer mid-lock,
  // probing the futex from the crash callback can deadlock or corrupt state.
  // Instead probe g_buf_busy with test_and_set: if it was already set a writer
  // holds g_buf (possibly a torn record in flight) so we only fsync
  // already-written bytes; if it was clear we now own it and can safely flush.
  // clear() releases it on the way out.
  bool locked = !g_buf_busy.test_and_set(std::memory_order_acquire);
  if (locked) {
    flush_buffer_locked();
    // Clean shutdowns append the fixed-size trailer so the reader does not treat
    // the archive as crash-truncated. The CLR crash callback passes
    // clean_shutdown=false; normal shutdown uses flush().
    if (clean_shutdown && !g_trailer_written) {
      hrr_eof_record rec = hrr_make_eof_record(
          g_seq_id.fetch_add(1, std::memory_order_relaxed), g_event_count.load());
      write_all_fd(g_events_fd, &rec, sizeof(rec));
      g_trailer_written = true;
    }
    g_buf_busy.clear(std::memory_order_release);
  }
  HRR_FSYNC(g_events_fd);

  // Best-effort manifest via raw open/write only. A clean shutdown writes
  // complete:true (the trailer is present); a crash writes complete:false — its
  // absence-of-trailer is how the reader detects truncation.
  if (g_manifest_path[0] == '\0') return;
  bool complete = clean_shutdown && locked;
  int mfd = HRR_OPEN(g_manifest_path);
  if (mfd < 0) return;
  char buf[256];
  size_t p = 0;
  p = append_lit(buf, p,
                 "{\n"
                 "  \"pid\": ");
  p += u64_to_dec(g_pid, buf + p);
  p = append_lit(buf, p, ",\n  \"parent_pid\": ");
  p += u64_to_dec(g_parent_pid, buf + p);
  p = append_lit(buf, p, ",\n  \"complete\": ");
  p = append_lit(buf, p, complete ? "true" : "false");
  p = append_lit(buf, p, ",\n  \"event_count\": ");
  p += u64_to_dec(g_event_count.load(), buf + p);
  p = append_lit(buf, p, ",\n  \"blob_count\": ");
  p += u64_to_dec(g_blob_count.load(), buf + p);
  p = append_lit(buf, p, "\n}\n");
  write_all_fd(mfd, buf, p);
  HRR_FSYNC(mfd);
  HRR_CLOSE(mfd);
}

// ---------------------------------------------------------------------------
// write_event_raw — unified write path for all events
//
// hdr points to the hrr_event_header at the front of an hrr_args_* struct.
// payload_len is sizeof the full hrr_args_* struct (header + fields).
// Fills all header fields then copies the whole struct into the app buffer.
// ---------------------------------------------------------------------------

void write_event_raw(uint16_t api_id, hrr_event_header* hdr, uint32_t payload_len) {
  // Fill fields that don't require the lock (timestamp and thread_id are
  // cheap and per-thread; getting them outside the lock keeps contention low).
  hdr->event_type     = api_id;
  hdr->timestamp_ns   = amd::Os::timeNanos();
  hdr->thread_id      = current_thread_id();
  hdr->payload_length = payload_len;
  memset(hdr->reserved, 0, sizeof(hdr->reserved));

  // Acquire once: assign sequence_id and buffer the record atomically so IDs are
  // only consumed for events that are actually written. A full record is always
  // appended under the lock, so the buffer never holds a torn record — which is
  // what makes the crash-callback flush in emergency_finalize() safe.
  //
  // The checkpoint flush+fsync happens inside this single lock scope. An earlier
  // version released the lock and re-acquired it for the fsync, which let two
  // threads that both crossed the kCheckpointEvents boundary race into back-to-
  // back fsyncs (a thundering herd at every 4096-event boundary). Doing the
  // fsync under the lock blocks other writers for the duration of the syscall,
  // but guarantees exactly one fsync per checkpoint and removes the race.
  {
    BufWriteGuard lk;
    if (g_events_fd < 0) return;
    hdr->sequence_id = g_seq_id.fetch_add(1, std::memory_order_relaxed);
    buffer_append_locked(hdr, payload_len);
    g_event_count.fetch_add(1, std::memory_order_relaxed);
    if (++g_events_since_ckpt >= kCheckpointEvents) {
      flush_buffer_locked();
      HRR_FSYNC(g_events_fd);
      g_events_since_ckpt = 0;
    }
  }
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
  std::string tmp = path + "." + std::to_string(current_process_id()) + ".tmp";
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
    if (g_events_fd < 0) return {};  // writer not open — drop silently
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
    if (g_events_fd < 0) return {};  // writer not open — drop silently
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

bool     is_open()      { std::lock_guard<std::mutex> lk(g_file_mu); return g_events_fd >= 0; }
uint64_t event_count()  { return g_event_count.load(); }
uint64_t blob_count()   { return g_blob_count.load(); }

}  // namespace writer
}  // namespace hrr_cap
