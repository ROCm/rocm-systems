/* Copyright (c) 2026 Advanced Micro Devices, Inc. - MIT License */

#include "hrr_reader.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <filesystem>

namespace fs = std::filesystem;

namespace hrr {

// ---------------------------------------------------------------------------
// Event move semantics
// ---------------------------------------------------------------------------

Event::Event(Event&& o) noexcept
    : header(o.header),
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
      kernel_launch(o.kernel_launch),
      raw_payload(std::move(o.raw_payload)) {
  o.kernel_launch = nullptr;
}

Event& Event::operator=(Event&& o) noexcept {
  if (this != &o) {
    delete kernel_launch;
    header           = o.header;
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
    raw_payload      = std::move(o.raw_payload);
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

// ---------------------------------------------------------------------------
// Payload read helpers (bytes after the 32-byte EventHeader already consumed)
//
// All hrr_args_* structs are pack(1). The first field after hdr is always
// ret (int32_t, 4 bytes), then the API parameters as uint64_t/uint32_t/etc.
// raw_payload[0] = ret, raw_payload[4] = first parameter, etc.
// ---------------------------------------------------------------------------

static inline uint32_t rd32(const uint8_t* p, size_t off) {
  uint32_t v; memcpy(&v, p + off, 4); return v;
}
static inline uint64_t rd64(const uint8_t* p, size_t off) {
  uint64_t v; memcpy(&v, p + off, 8); return v;
}
static inline int32_t rdi32(const uint8_t* p, size_t off) {
  int32_t v; memcpy(&v, p + off, 4); return v;
}

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
  archive.path = path;

  std::string events_path = path + "/events.bin";
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

  // Read events sequentially
  while (true) {
    Event ev;
    if (fread(&ev.header, sizeof(hrr_event_header), 1, f) != 1) break;

    // Payload = everything after the 32-byte header
    uint16_t total = ev.header.payload_length;
    uint16_t hdr_size = static_cast<uint16_t>(sizeof(hrr_event_header));
    if (total > hdr_size) {
      uint16_t pl_size = total - hdr_size;
      ev.raw_payload.resize(pl_size);
      if (fread(ev.raw_payload.data(), 1, pl_size, f) != pl_size) break;
    }

    const uint8_t* pl     = ev.raw_payload.data();
    size_t         pl_len = ev.raw_payload.size();

    // Parse typed payload fields from raw bytes.
    // Payload layout per event: ret(int32_t,4) then API params as uint64_t/uint32_t.
    switch (ev.header.event_type) {

      // --- Memory allocation ---

      case HRR_API_HIPMALLOC:
        // ret(4) ptr(8) size(8)
        if (pl_len >= 20) {
          ev.malloc_ev.ptr_handle = rd64(pl,  4);
          ev.malloc_ev.size       = rd64(pl, 12);
        }
        break;

      case HRR_API_HIPFREE:
        // ret(4) ptr(8)
        if (pl_len >= 12)
          ev.malloc_ev.ptr_handle = rd64(pl, 4);
        break;

      case HRR_API_HIPMALLOCASYNC:
        // ret(4) dev_ptr(8) size(8) stream(8)
        if (pl_len >= 28) {
          ev.malloc_async_ev.ptr_handle    = rd64(pl,  4);
          ev.malloc_async_ev.size          = rd64(pl, 12);
          ev.malloc_async_ev.stream_handle = rd64(pl, 20);
        }
        break;

      case HRR_API_HIPFREEASYNC:
        // ret(4) dev_ptr(8) stream(8)
        if (pl_len >= 20) {
          ev.free_async_ev.ptr_handle    = rd64(pl,  4);
          ev.free_async_ev.stream_handle = rd64(pl, 12);
        }
        break;

      // --- Data transfer ---

      case HRR_API_HIPMEMCPY:
        // ret(4) dst(8) src(8) sizeBytes(8) kind(4) blob_hash_lo(8) blob_hash_hi(8)
        if (pl_len >= 48) {
          ev.memcpy_ev.dst_addr = rd64(pl,  4);
          ev.memcpy_ev.src_addr = rd64(pl, 12);
          ev.memcpy_ev.size     = rd64(pl, 20);
          ev.memcpy_ev.kind     = rdi32(pl, 28);
          ev.memcpy_ev.hash_lo  = rd64(pl, 32);
          ev.memcpy_ev.hash_hi  = rd64(pl, 40);
        }
        break;

      case HRR_API_HIPMEMCPYASYNC:
        // ret(4) dst(8) src(8) sizeBytes(8) kind(4) stream(8) blob_hash_lo(8) blob_hash_hi(8)
        if (pl_len >= 56) {
          ev.memcpy_ev.dst_addr     = rd64(pl,  4);
          ev.memcpy_ev.src_addr     = rd64(pl, 12);
          ev.memcpy_ev.size         = rd64(pl, 20);
          ev.memcpy_ev.kind         = rdi32(pl, 28);
          ev.stream_handle          = rd64(pl, 32);
          ev.memcpy_ev.hash_lo      = rd64(pl, 40);
          ev.memcpy_ev.hash_hi      = rd64(pl, 48);
        }
        break;

      case HRR_API_HIPMEMCPYHTOD:
        // ret(4) dst(8) src(8) sizeBytes(8) blob_hash_lo(8) blob_hash_hi(8)
        if (pl_len >= 44) {
          ev.memcpy_ev.dst_addr = rd64(pl,  4);
          ev.memcpy_ev.src_addr = rd64(pl, 12);
          ev.memcpy_ev.size     = rd64(pl, 20);
          ev.memcpy_ev.kind     = 1;  // hipMemcpyHostToDevice
          ev.memcpy_ev.hash_lo  = rd64(pl, 28);
          ev.memcpy_ev.hash_hi  = rd64(pl, 36);
        }
        break;

      case HRR_API_HIPMEMCPYHTODASYNC:
        // ret(4) dst(8) src(8) sizeBytes(8) stream(8) blob_hash_lo(8) blob_hash_hi(8)
        if (pl_len >= 52) {
          ev.memcpy_ev.dst_addr = rd64(pl,  4);
          ev.memcpy_ev.src_addr = rd64(pl, 12);
          ev.memcpy_ev.size     = rd64(pl, 20);
          ev.memcpy_ev.kind     = 1;  // hipMemcpyHostToDevice
          ev.stream_handle      = rd64(pl, 28);
          ev.memcpy_ev.hash_lo  = rd64(pl, 36);
          ev.memcpy_ev.hash_hi  = rd64(pl, 44);
        }
        break;

      case HRR_API_HIPMEMSET:
        // ret(4) dst(8) value(4) sizeBytes(8)  — raw_payload retained for replay
        break;

      case HRR_API_HIPMEMSETASYNC:
        // ret(4) dst(8) value(4) sizeBytes(8) stream(8) — retained
        break;

      // --- Modules ---

      case HRR_API_HIPMODULELOADDATA:
      case HRR_API_HIPMODULELOADDATAEX:
        // ret(4) module(8) image(8) co_hash_lo(8) co_hash_hi(8) module_id(4)
        if (pl_len >= 40) {
          ev.module_load_ev.module_handle = rd64(pl,  4);
          ev.module_load_ev.hash_lo       = rd64(pl, 20);
          ev.module_load_ev.hash_hi       = rd64(pl, 28);
        }
        break;

      case HRR_API_HIPMODULELOAD:
        // ret(4) module(8) fname(8) co_hash_lo(8) co_hash_hi(8) module_id(4)
        if (pl_len >= 40) {
          ev.module_load_ev.module_handle = rd64(pl,  4);
          ev.module_load_ev.hash_lo       = rd64(pl, 20);
          ev.module_load_ev.hash_hi       = rd64(pl, 28);
        }
        break;

      case HRR_API_HIPMODULEUNLOAD:
        // ret(4) module(8)
        if (pl_len >= 12)
          ev.handle64 = rd64(pl, 4);
        break;

      // --- Kernel launch ---

      case HRR_API_HIPMODULELAUNCHKERNEL:
      case HRR_API_HIPEXTMODULELAUNCHKERNEL:
      case HRR_API_HIPLAUNCHKERNEL:
      case HRR_API_HIPLAUNCHBYPTR: {
        auto* kl = new KernelLaunchEvent();
        uint64_t sh = 0;
        if (parse_kernel_launch(pl, pl_len, sh, *kl)) {
          ev.stream_handle  = sh;
          ev.kernel_launch  = kl;
          archive.kernel_count++;
        } else {
          delete kl;
        }
        break;
      }

      // --- Streams ---

      case HRR_API_HIPSTREAMCREATE:
        // ret(4) stream(8)
        if (pl_len >= 12) {
          ev.stream_create_ev.stream_handle = rd64(pl, 4);
          ev.stream_create_ev.flags         = 0;
          ev.stream_create_ev.priority      = 0;
        }
        break;

      case HRR_API_HIPSTREAMCREATEWITHFLAGS:
        // ret(4) stream(8) flags(4)
        if (pl_len >= 16) {
          ev.stream_create_ev.stream_handle = rd64(pl, 4);
          ev.stream_create_ev.flags         = rd32(pl, 12);
          ev.stream_create_ev.priority      = 0;
        }
        break;

      case HRR_API_HIPSTREAMCREATEWITHPRIORITY:
        // ret(4) stream(8) flags(4) priority(4)
        if (pl_len >= 20) {
          ev.stream_create_ev.stream_handle = rd64(pl, 4);
          ev.stream_create_ev.flags         = rd32(pl, 12);
          ev.stream_create_ev.priority      = rdi32(pl, 16);
        }
        break;

      case HRR_API_HIPSTREAMDESTROY:
      case HRR_API_HIPSTREAMSYNCHRONIZE:
        // ret(4) stream(8)
        if (pl_len >= 12)
          ev.handle64 = rd64(pl, 4);
        break;

      case HRR_API_HIPSTREAMWAITEVENT:
        // ret(4) stream(8) event(8) flags(4)
        if (pl_len >= 24) {
          ev.stream_wait_ev.stream_handle = rd64(pl,  4);
          ev.stream_wait_ev.event_handle  = rd64(pl, 12);
          ev.stream_wait_ev.flags         = rd32(pl, 20);
        }
        break;

      // --- Events ---

      case HRR_API_HIPEVENTCREATE:
        // ret(4) event(8)
        if (pl_len >= 12)
          ev.handle64 = rd64(pl, 4);
        break;

      case HRR_API_HIPEVENTCREATEWITHFLAGS:
        // ret(4) event(8) flags(4)
        if (pl_len >= 16)
          ev.handle64 = rd64(pl, 4);
        break;

      case HRR_API_HIPEVENTRECORD:
        // ret(4) event(8) stream(8)
        if (pl_len >= 20) {
          ev.event_record_ev.event_handle  = rd64(pl,  4);
          ev.event_record_ev.stream_handle = rd64(pl, 12);
        }
        break;

      case HRR_API_HIPEVENTSYNCHRONIZE:
      case HRR_API_HIPEVENTDESTROY:
        // ret(4) event(8)
        if (pl_len >= 12)
          ev.handle64 = rd64(pl, 4);
        break;

      // --- Device sync ---

      case HRR_API_HIPDEVICESYNCHRONIZE:
        break;

      default:
        break;
    }

    archive.events.push_back(std::move(ev));
  }

  fclose(f);

  // Sort events by sequence_id to restore causal ordering across threads.
  // Multi-threaded captures write events from different threads in file-arrival
  // order, which may differ from the logical call order. sequence_id is assigned
  // atomically by the writer, so sorting by it restores the true call sequence.
  std::stable_sort(archive.events.begin(), archive.events.end(),
    [](const Event& a, const Event& b) {
      return a.header.sequence_id < b.header.sequence_id;
    });

  archive.event_count = archive.events.size();

  // Collect distinct thread IDs in first-seen order (after sort = sequence order).
  // One linear pass; avoids a separate scan by the replayer.
  {
    std::unordered_map<uint64_t, bool> seen;
    for (const auto& ev : archive.events) {
      if (seen.emplace(ev.header.thread_id, true).second)
        archive.threads.push_back(ev.header.thread_id);
    }
  }

  // Enumerate blobs
  std::string blobs_dir = path + "/blobs";
  if (fs::exists(blobs_dir)) {
    for (auto& entry : fs::recursive_directory_iterator(blobs_dir)) {
      if (entry.is_regular_file() && entry.path().extension() == ".blob") {
        archive.blobs[entry.path().stem().string()] = entry.path().string();
        archive.blob_count++;
      }
    }
  }

  // Enumerate code objects
  std::string co_dir = path + "/code_objects";
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
  fseek(f, 0, SEEK_END);
  long size = ftell(f);
  fseek(f, 0, SEEK_SET);
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
