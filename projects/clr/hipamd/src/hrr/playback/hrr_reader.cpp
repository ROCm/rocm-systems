/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include "hrr_reader.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <vector>

namespace fs = std::filesystem;

namespace hrr {

// ---------------------------------------------------------------------------
// Event move semantics
// ---------------------------------------------------------------------------

Event::Event(Event&& o) noexcept
    : raw_payload(std::move(o.raw_payload)),
      stream_handle(o.stream_handle),
      handle64(o.handle64),
      malloc_ev(o.malloc_ev),
      memcpy_ev(o.memcpy_ev),
      module_load_ev(o.module_load_ev),
      malloc_async_ev(o.malloc_async_ev),
      free_async_ev(o.free_async_ev),
      stream_create_ev(o.stream_create_ev),
      event_record_ev(o.event_record_ev),
      stream_wait_ev(o.stream_wait_ev),
      kernel_launch(o.kernel_launch) {
  o.kernel_launch = nullptr;
}

Event& Event::operator=(Event&& o) noexcept {
  if (this != &o) {
    delete kernel_launch;
    raw_payload      = std::move(o.raw_payload);
    stream_handle    = o.stream_handle;
    handle64         = o.handle64;
    malloc_ev        = o.malloc_ev;
    memcpy_ev        = o.memcpy_ev;
    module_load_ev   = o.module_load_ev;
    malloc_async_ev  = o.malloc_async_ev;
    free_async_ev    = o.free_async_ev;
    stream_create_ev = o.stream_create_ev;
    event_record_ev  = o.event_record_ev;
    stream_wait_ev   = o.stream_wait_ev;
    kernel_launch    = o.kernel_launch;
    o.kernel_launch  = nullptr;
  }
  return *this;
}

// ---------------------------------------------------------------------------
// Utilities
// ---------------------------------------------------------------------------

std::string hash_hex(uint64_t lo, uint64_t hi) {
  char buf[33];
  snprintf(buf, sizeof(buf), "%016llx%016llx",
           static_cast<unsigned long long>(lo),
           static_cast<unsigned long long>(hi));
  return std::string(buf);
}

const char* event_type_name(uint16_t type) {
  if (type < HRR_API_COUNT) return hrr_api_names[type];
  return "UNKNOWN";
}

static std::vector<fs::path> find_pid_subarchives(const fs::path& root) {
  std::vector<fs::path> dirs;
  if (!fs::exists(root) || !fs::is_directory(root)) return dirs;
  for (const auto& ent : fs::directory_iterator(root)) {
    if (!ent.is_directory()) continue;
    const std::string name = ent.path().filename().string();
    if (name.rfind("pid-", 0) == 0 && fs::exists(ent.path() / "events.bin"))
      dirs.push_back(ent.path());
  }
  std::sort(dirs.begin(), dirs.end());
  return dirs;
}

static bool resolve_archive_path(const std::string& input, std::string& resolved) {
  fs::path path(input);
  if (fs::exists(path / "events.bin")) {
    resolved = path.string();
    return true;
  }

  std::vector<fs::path> pid_dirs = find_pid_subarchives(path);
  if (pid_dirs.empty()) {
    fprintf(stderr, "[HRR] Cannot open %s\n", (path / "events.bin").string().c_str());
    return false;
  }
  if (pid_dirs.size() == 1) {
    resolved = pid_dirs.front().string();
    return true;
  }

  fprintf(stderr,
          "[HRR] Archive root %s contains multiple process captures; "
          "point hrr-playback at a specific pid directory:\n",
          input.c_str());
  for (const auto& dir : pid_dirs)
    fprintf(stderr, "[HRR]   %s\n", dir.string().c_str());
  return false;
}

// ---------------------------------------------------------------------------
// Payload read helpers (bytes after the 32-byte EventHeader already consumed)
// ---------------------------------------------------------------------------
// Kernel launch payload parser
//
// Binary layout (bytes in raw_payload, after 32-byte EventHeader):
//   [0..7]   stream_handle (uint64_t, raw hipStream_t pointer)
//   [8..9]   name_len (uint16_t)
//   [10..]   kernel_name (name_len bytes, no NUL)
//   [+0..7]  co_hash_lo (uint64_t)
//   [+8..15] co_hash_hi (uint64_t)
//   [+0..11] grid[3] (uint32_t[3])
//   [+12..23] block[3] (uint32_t[3])
//   [+24..27] shared_mem (uint32_t)
//   [+28..29] num_args (uint16_t)
//   [+30..31] num_snapshots (uint16_t, always 0)
//   per arg: u8 value_kind, u16 size, <size> bytes data
// ---------------------------------------------------------------------------

static bool parse_kernel_launch(const uint8_t* data, size_t len,
                                uint64_t& out_stream_handle,
                                KernelLaunchEvent& kl) {
  const uint8_t* p   = data;
  const uint8_t* end = data + len;

  // stream_handle
  if (p + 8 > end) return false;
  memcpy(&out_stream_handle, p, 8); p += 8;

  // kernel name
  if (p + 2 > end) return false;
  uint16_t name_len;
  memcpy(&name_len, p, 2); p += 2;
  if (p + name_len > end) return false;
  kl.kernel_name.assign(reinterpret_cast<const char*>(p), name_len);
  p += name_len;

  // co_hash
  kl.co_hash_lo = 0; kl.co_hash_hi = 0;
  if (p + 16 <= end) {
    memcpy(&kl.co_hash_lo, p, 8); p += 8;
    memcpy(&kl.co_hash_hi, p, 8); p += 8;
  }

  // grid / block / shared / counts
  if (p + 12 + 12 + 4 + 2 + 2 > end) return false;
  memcpy(kl.grid,        p, 12); p += 12;
  memcpy(kl.block,       p, 12); p += 12;
  memcpy(&kl.shared_mem, p,  4); p +=  4;

  uint16_t num_args, num_snapshots;
  memcpy(&num_args,      p, 2); p += 2;
  memcpy(&num_snapshots, p, 2); p += 2;

  // args
  for (uint16_t i = 0; i < num_args; i++) {
    if (p + 3 > end) return false;
    KernelArg arg;
    arg.value_kind = *p++;
    memcpy(&arg.size, p, 2); p += 2;
    if (p + arg.size > end) return false;
    arg.data.assign(p, p + arg.size);
    p += arg.size;
    if (arg.value_kind == 3) {  // trailing embedded-pointer offset list
      if (p + 2 > end) return false;
      uint16_t n_ptrs; memcpy(&n_ptrs, p, 2); p += 2;
      for (uint16_t k = 0; k < n_ptrs; k++) {
        if (p + 2 > end) return false;
        uint16_t off; memcpy(&off, p, 2); p += 2;
        arg.ptr_offsets.push_back(off);
      }
    }
    kl.args.push_back(std::move(arg));
  }

  // buffer snapshots (always 0 in in-tree captures)
  for (uint16_t i = 0; i < num_snapshots; i++) {
    if (p + 41 > end) return false;
    BufferSnapshot snap;
    memcpy(&snap.ptr_handle, p, 8); p += 8;
    memcpy(&snap.offset,     p, 8); p += 8;
    memcpy(&snap.length,     p, 8); p += 8;
    memcpy(&snap.hash_lo,    p, 8); p += 8;
    memcpy(&snap.hash_hi,    p, 8); p += 8;
    snap.direction = *p++;
    kl.snapshots.push_back(snap);
  }

  return true;
}

// ---------------------------------------------------------------------------
// Archive loader
// ---------------------------------------------------------------------------

bool load_archive(const std::string& path, Archive& archive) {
  std::string archive_dir;
  if (!resolve_archive_path(path, archive_dir))
    return false;
  archive.path = archive_dir;

  std::string events_path = archive_dir + "/events.bin";
  FILE* f = fopen(events_path.c_str(), "rb");
  if (!f) {
    fprintf(stderr, "[HRR] Cannot open %s\n", events_path.c_str());
    return false;
  }

  // Read and validate file header
  hrr_file_header fh{};
  if (fread(&fh, sizeof(fh), 1, f) != 1) {
    fprintf(stderr, "[HRR] events.bin too short (missing file header)\n");
    fclose(f); return false;
  }
  if (fh.magic != HRR_MAGIC) {
    fprintf(stderr, "[HRR] Bad magic 0x%08x (expected 0x%08x)\n", fh.magic, HRR_MAGIC);
    fclose(f); return false;
  }
  if (fh.version != HRR_VERSION) {
    fprintf(stderr, "[HRR] Version mismatch: file=%u reader=%u\n", fh.version, HRR_VERSION);
    fclose(f); return false;
  }
  archive.version = fh.version;

  // Read events sequentially.
  //
  // Crash resilience: events.bin is an append-only log of self-delimiting
  // records, so a partial write can only ever be the LAST record. A torn header
  // or payload at the tail is therefore treated as a recovery point — we keep
  // every complete record already parsed and mark the archive truncated, rather
  // than discarding the whole capture. The clean-shutdown trailer
  // (hrr_eof_record) explicitly marks a whole archive.
  bool truncated = false;
  bool complete  = false;
  const uint16_t hdr_size = static_cast<uint16_t>(sizeof(hrr_event_header));
  while (true) {
    Event ev;
    // Read the header first to get payload_length, then read the rest into
    // raw_payload as one contiguous block: header(32) + fields.
    ev.raw_payload.resize(hdr_size);
    size_t got = fread(ev.raw_payload.data(), 1, hdr_size, f);
    if (got == 0) break;  // clean EOF at a record boundary
    if (got < hdr_size) {
      fprintf(stderr,
              "[HRR] Truncated event header (%zu/%u bytes) at tail — recovered %zu events\n",
              got, hdr_size, archive.events.size());
      truncated = true; break;
    }

    const uint16_t etype = ev.header().event_type;
    const uint32_t total = ev.header().payload_length;

    if (total < hdr_size) {
      fprintf(stderr,
              "[HRR] Torn trailing record (payload_length %u < %u) — recovered %zu events\n",
              total, hdr_size, archive.events.size());
      truncated = true; break;
    }
    if (total > hdr_size) {
      ev.raw_payload.resize(total);
      uint32_t pl_size = total - hdr_size;
      if (fread(ev.raw_payload.data() + hdr_size, 1, pl_size, f) != pl_size) {
        fprintf(stderr,
                "[HRR] Truncated event payload (expected %u bytes) at tail — recovered %zu events\n",
                pl_size, archive.events.size());
        truncated = true; break;
      }
    }

    // Clean-shutdown trailer: full hrr_eof_record with valid magic only.
    // event_type 0xFFFF alone is not enough — Unit_HRR_Format_UnknownEventType
    // uses 0xFFFF with header-sized payload as an opaque unknown record.
    if (etype == HRR_EOF_MARKER &&
        total == static_cast<uint32_t>(sizeof(hrr_eof_record))) {
      const auto* er = reinterpret_cast<const hrr_eof_record*>(ev.raw_payload.data());
      if (er->eof_magic == HRR_EOF_MAGIC) {
        complete = true;
        continue;  // do not append trailer as a replay event
      }
    }

    // Convenience cast macro — raw_payload.data() is the full hrr_args_* struct.
    // SIZE_OK(T): verify payload is large enough for the typed struct before casting.
    // A malformed archive could supply a valid event_type with payload_length == 32
    // (header only), which would cause an OOB read in the typed cast.
    #define AS(T) reinterpret_cast<const T*>(ev.raw_payload.data())
    #define SIZE_OK(T) (total >= static_cast<uint32_t>(sizeof(T)))

    switch (ev.header().event_type) {

      // --- Memory allocation ---

      case HRR_API_HIPMALLOC: {
        if (!SIZE_OK(hrr_args_hipMalloc)) break;
        const auto* a = AS(hrr_args_hipMalloc);
        ev.malloc_ev.ptr_handle = a->ptr;
        ev.malloc_ev.size       = a->size;
        break;
      }
      case HRR_API_HIPFREE:
        if (SIZE_OK(hrr_args_hipFree))
          ev.malloc_ev.ptr_handle = AS(hrr_args_hipFree)->ptr;
        break;

      case HRR_API_HIPMALLOCASYNC: {
        if (!SIZE_OK(hrr_args_hipMallocAsync)) break;
        const auto* a = AS(hrr_args_hipMallocAsync);
        ev.malloc_async_ev.ptr_handle    = a->dev_ptr;
        ev.malloc_async_ev.size          = a->size;
        ev.malloc_async_ev.stream_handle = a->stream;
        break;
      }
      case HRR_API_HIPFREEASYNC: {
        if (!SIZE_OK(hrr_args_hipFreeAsync)) break;
        const auto* a = AS(hrr_args_hipFreeAsync);
        ev.free_async_ev.ptr_handle    = a->dev_ptr;
        ev.free_async_ev.stream_handle = a->stream;
        break;
      }

      // --- Data transfer ---

      case HRR_API_HIPMEMCPY: {
        if (!SIZE_OK(hrr_args_hipMemcpy)) break;
        const auto* a = AS(hrr_args_hipMemcpy);
        ev.memcpy_ev.dst_addr = a->dst;
        ev.memcpy_ev.src_addr = a->src;
        ev.memcpy_ev.size     = a->sizeBytes;
        ev.memcpy_ev.kind     = a->kind;
        ev.memcpy_ev.hash_lo  = a->blob_hash_lo;
        ev.memcpy_ev.hash_hi  = a->blob_hash_hi;
        break;
      }
      case HRR_API_HIPMEMCPYASYNC: {
        if (!SIZE_OK(hrr_args_hipMemcpyAsync)) break;
        const auto* a = AS(hrr_args_hipMemcpyAsync);
        ev.memcpy_ev.dst_addr = a->dst;
        ev.memcpy_ev.src_addr = a->src;
        ev.memcpy_ev.size     = a->sizeBytes;
        ev.memcpy_ev.kind     = a->kind;
        ev.stream_handle      = a->stream;
        ev.memcpy_ev.hash_lo  = a->blob_hash_lo;
        ev.memcpy_ev.hash_hi  = a->blob_hash_hi;
        break;
      }
      case HRR_API_HIPMEMCPYHTOD: {
        if (!SIZE_OK(hrr_args_hipMemcpyHtoD)) break;
        const auto* a = AS(hrr_args_hipMemcpyHtoD);
        ev.memcpy_ev.dst_addr = a->dst;
        ev.memcpy_ev.src_addr = a->src;
        ev.memcpy_ev.size     = a->sizeBytes;
        ev.memcpy_ev.kind     = 1;  // hipMemcpyHostToDevice
        ev.memcpy_ev.hash_lo  = a->blob_hash_lo;
        ev.memcpy_ev.hash_hi  = a->blob_hash_hi;
        break;
      }
      case HRR_API_HIPMEMCPYHTODASYNC: {
        if (!SIZE_OK(hrr_args_hipMemcpyHtoDAsync)) break;
        const auto* a = AS(hrr_args_hipMemcpyHtoDAsync);
        ev.memcpy_ev.dst_addr = a->dst;
        ev.memcpy_ev.src_addr = a->src;
        ev.memcpy_ev.size     = a->sizeBytes;
        ev.memcpy_ev.kind     = 1;  // hipMemcpyHostToDevice
        ev.stream_handle      = a->stream;
        ev.memcpy_ev.hash_lo  = a->blob_hash_lo;
        ev.memcpy_ev.hash_hi  = a->blob_hash_hi;
        break;
      }

      case HRR_API_HIPMEMCPY2D: {
        if (!SIZE_OK(hrr_args_hipMemcpy2D)) break;
        const auto* a = AS(hrr_args_hipMemcpy2D);
        ev.memcpy_ev.dst_addr = a->dst;
        ev.memcpy_ev.src_addr = a->src;
        ev.memcpy_ev.size     = a->width * a->height;  // logical bytes (best-effort)
        ev.memcpy_ev.kind     = a->kind;
        // H2D blob if present, else D2H expected-output blob.
        ev.memcpy_ev.hash_lo  = a->blob_hash_lo ? a->blob_hash_lo : a->d2h_hash_lo;
        ev.memcpy_ev.hash_hi  = a->blob_hash_lo ? a->blob_hash_hi : a->d2h_hash_hi;
        break;
      }
      case HRR_API_HIPMEMCPY2DASYNC: {
        if (!SIZE_OK(hrr_args_hipMemcpy2DAsync)) break;
        const auto* a = AS(hrr_args_hipMemcpy2DAsync);
        ev.memcpy_ev.dst_addr = a->dst;
        ev.memcpy_ev.src_addr = a->src;
        ev.memcpy_ev.size     = a->width * a->height;  // logical bytes (best-effort)
        ev.memcpy_ev.kind     = a->kind;
        ev.stream_handle      = a->stream;
        ev.memcpy_ev.hash_lo  = a->blob_hash_lo ? a->blob_hash_lo : a->d2h_hash_lo;
        ev.memcpy_ev.hash_hi  = a->blob_hash_lo ? a->blob_hash_hi : a->d2h_hash_hi;
        break;
      }

      case HRR_API_HIPMEMSET:
      case HRR_API_HIPMEMSETASYNC:
        break;  // raw_payload retained for replay

      // --- Modules ---

      case HRR_API_HIPMODULELOADDATA:
      case HRR_API_HIPMODULELOADDATAEX: {
        if (!SIZE_OK(hrr_args_hipModuleLoadData)) break;
        const auto* a = AS(hrr_args_hipModuleLoadData);
        ev.module_load_ev.module_handle = a->module;
        ev.module_load_ev.hash_lo       = a->co_hash_lo;
        ev.module_load_ev.hash_hi       = a->co_hash_hi;
        break;
      }
      case HRR_API_HIPMODULELOAD: {
        if (!SIZE_OK(hrr_args_hipModuleLoad)) break;
        const auto* a = AS(hrr_args_hipModuleLoad);
        ev.module_load_ev.module_handle = a->module;
        ev.module_load_ev.hash_lo       = a->co_hash_lo;
        ev.module_load_ev.hash_hi       = a->co_hash_hi;
        break;
      }
      case HRR_API_HIPMODULEUNLOAD:
        if (SIZE_OK(hrr_args_hipModuleUnload))
          ev.handle64 = AS(hrr_args_hipModuleUnload)->module;
        break;

      // --- Kernel launch ---

      case HRR_API_HIPMODULELAUNCHKERNEL:
      case HRR_API_HIPEXTMODULELAUNCHKERNEL:
      case HRR_API_HIPLAUNCHKERNEL:
      case HRR_API_HIPLAUNCHBYPTR: {
        auto* kl = new KernelLaunchEvent();
        uint64_t sh = 0;
        const uint8_t* kl_data = ev.raw_payload.data() + hdr_size;
        size_t         kl_len  = static_cast<size_t>(total) - hdr_size;
        if (parse_kernel_launch(kl_data, kl_len, sh, *kl)) {
          ev.stream_handle  = sh;
          ev.kernel_launch  = kl;
          archive.kernel_count++;
        } else {
          delete kl;
        }
        break;
      }

      // --- Streams ---

      case HRR_API_HIPSTREAMCREATE: {
        if (!SIZE_OK(hrr_args_hipStreamCreate)) break;
        const auto* a = AS(hrr_args_hipStreamCreate);
        ev.stream_create_ev.stream_handle = a->stream;
        ev.stream_create_ev.flags         = 0;
        ev.stream_create_ev.priority      = 0;
        break;
      }
      case HRR_API_HIPSTREAMCREATEWITHFLAGS: {
        if (!SIZE_OK(hrr_args_hipStreamCreateWithFlags)) break;
        const auto* a = AS(hrr_args_hipStreamCreateWithFlags);
        ev.stream_create_ev.stream_handle = a->stream;
        ev.stream_create_ev.flags         = a->flags;
        ev.stream_create_ev.priority      = 0;
        break;
      }
      case HRR_API_HIPSTREAMCREATEWITHPRIORITY: {
        if (!SIZE_OK(hrr_args_hipStreamCreateWithPriority)) break;
        const auto* a = AS(hrr_args_hipStreamCreateWithPriority);
        ev.stream_create_ev.stream_handle = a->stream;
        ev.stream_create_ev.flags         = a->flags;
        ev.stream_create_ev.priority      = a->priority;
        break;
      }
      case HRR_API_HIPSTREAMDESTROY:
        if (SIZE_OK(hrr_args_hipStreamDestroy))
          ev.handle64 = AS(hrr_args_hipStreamDestroy)->stream;
        break;
      case HRR_API_HIPSTREAMSYNCHRONIZE:
        if (SIZE_OK(hrr_args_hipStreamSynchronize))
          ev.handle64 = AS(hrr_args_hipStreamSynchronize)->stream;
        break;

      case HRR_API_HIPSTREAMWAITEVENT: {
        if (!SIZE_OK(hrr_args_hipStreamWaitEvent)) break;
        const auto* a = AS(hrr_args_hipStreamWaitEvent);
        ev.stream_wait_ev.stream_handle = a->stream;
        ev.stream_wait_ev.event_handle  = a->event;
        ev.stream_wait_ev.flags         = a->flags;
        break;
      }

      // --- Events ---

      case HRR_API_HIPEVENTCREATE:
        if (SIZE_OK(hrr_args_hipEventCreate))
          ev.handle64 = AS(hrr_args_hipEventCreate)->event;
        break;
      case HRR_API_HIPEVENTCREATEWITHFLAGS:
        if (SIZE_OK(hrr_args_hipEventCreateWithFlags))
          ev.handle64 = AS(hrr_args_hipEventCreateWithFlags)->event;
        break;
      case HRR_API_HIPEVENTRECORD: {
        if (!SIZE_OK(hrr_args_hipEventRecord)) break;
        const auto* a = AS(hrr_args_hipEventRecord);
        ev.event_record_ev.event_handle  = a->event;
        ev.event_record_ev.stream_handle = a->stream;
        break;
      }
      case HRR_API_HIPEVENTSYNCHRONIZE:
        if (SIZE_OK(hrr_args_hipEventSynchronize))
          ev.handle64 = AS(hrr_args_hipEventSynchronize)->event;
        break;
      case HRR_API_HIPEVENTDESTROY:
        if (SIZE_OK(hrr_args_hipEventDestroy))
          ev.handle64 = AS(hrr_args_hipEventDestroy)->event;
        break;

      // --- Device sync ---

      case HRR_API_HIPDEVICESYNCHRONIZE:
      default:
        break;
    }

    #undef AS
    #undef SIZE_OK

    archive.events.push_back(std::move(ev));
  }

  fclose(f);

  archive.complete  = complete;
  archive.truncated = truncated;
  if (!complete) {
    fprintf(stderr,
            "[HRR] Archive has no clean-shutdown trailer (capture likely crashed); "
            "recovered %zu events%s\n",
            archive.events.size(), truncated ? ", trailing torn record discarded" : "");
  }

  // Sort events by sequence_id to restore causal ordering across threads.
  // Multi-threaded captures write events from different threads in file-arrival
  // order, which may differ from the logical call order. sequence_id is assigned
  // atomically by the writer, so sorting by it restores the true call sequence.
  std::stable_sort(archive.events.begin(), archive.events.end(),
    [](const Event& a, const Event& b) {
      return a.header().sequence_id < b.header().sequence_id;
    });

  archive.event_count = archive.events.size();

  // Collect distinct thread IDs in first-seen order (after sort = sequence order).
  // One linear pass; avoids a separate scan by the replayer.
  {
    std::unordered_map<uint64_t, bool> seen;
    for (const auto& ev : archive.events) {
      if (seen.emplace(ev.header().thread_id, true).second)
        archive.threads.push_back(ev.header().thread_id);
    }
  }

  // Enumerate blobs
  std::string blobs_dir = archive_dir + "/blobs";
  if (fs::exists(blobs_dir)) {
    for (auto& entry : fs::recursive_directory_iterator(blobs_dir)) {
      if (entry.is_regular_file() && entry.path().extension() == ".blob") {
        archive.blobs[entry.path().stem().string()] = entry.path().string();
        archive.blob_count++;
      }
    }
  }

  // Enumerate code objects
  std::string co_dir = archive_dir + "/code_objects";
  if (fs::exists(co_dir)) {
    for (auto& entry : fs::directory_iterator(co_dir)) {
      if (entry.is_regular_file() && entry.path().extension() == ".hsaco") {
        archive.code_objects[entry.path().stem().string()] = entry.path().string();
        archive.code_object_count++;
      }
    }
  }

  return true;
}

// ---------------------------------------------------------------------------
// Blob / code object readers
// ---------------------------------------------------------------------------

static bool read_file(const std::string& file_path, std::vector<uint8_t>& data) {
  FILE* f = fopen(file_path.c_str(), "rb");
  if (!f) return false;
  if (fseek(f, 0, SEEK_END) != 0) { fclose(f); return false; }
  long size = ftell(f);
  if (size < 0) { fclose(f); return false; }
  if (fseek(f, 0, SEEK_SET) != 0) { fclose(f); return false; }
  data.resize(static_cast<size_t>(size));
  bool ok = fread(data.data(), 1, data.size(), f) == data.size();
  fclose(f);
  return ok;
}

bool read_blob(const Archive& archive, uint64_t hash_lo, uint64_t hash_hi,
               std::vector<uint8_t>& data) {
  auto it = archive.blobs.find(hash_hex(hash_lo, hash_hi));
  return it != archive.blobs.end() && read_file(it->second, data);
}

bool read_code_object(const Archive& archive, uint64_t hash_lo, uint64_t hash_hi,
                      std::vector<uint8_t>& data) {
  auto it = archive.code_objects.find(hash_hex(hash_lo, hash_hi));
  return it != archive.code_objects.end() && read_file(it->second, data);
}

}  // namespace hrr
