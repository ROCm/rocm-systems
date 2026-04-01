/* Copyright (c) 2026 Advanced Micro Devices, Inc. - MIT License */

#include "hrr_reader.h"

#include <cstdio>
#include <cstring>
#include <filesystem>

namespace fs = std::filesystem;

namespace hrr {

// Union copy helper: sizeof(memcpy_ev) is the largest union member (44 bytes
// packed) so this covers all fields of all three union variants.
static constexpr size_t kEventUnionBytes = sizeof(MemcpyEvent);

Event::Event(Event&& o) noexcept
    : header(o.header), kernel_launch(o.kernel_launch),
      raw_payload(std::move(o.raw_payload)) {
  memcpy(&malloc_ev, &o.malloc_ev, kEventUnionBytes);
  o.kernel_launch = nullptr;
}

Event& Event::operator=(Event&& o) noexcept {
  if (this != &o) {
    delete kernel_launch;
    header = o.header;
    memcpy(&malloc_ev, &o.malloc_ev, kEventUnionBytes);
    kernel_launch = o.kernel_launch;
    raw_payload = std::move(o.raw_payload);
    o.kernel_launch = nullptr;
  }
  return *this;
}

std::string hash_hex(uint64_t lo, uint64_t hi) {
  char buf[33];
  snprintf(buf, sizeof(buf), "%016llx%016llx",
           (unsigned long long)lo, (unsigned long long)hi);
  return std::string(buf);
}

const char* event_type_name(uint16_t type) {
  switch (type) {
    case EVENT_MALLOC:        return "MALLOC";
    case EVENT_FREE:          return "FREE";
    case EVENT_MEMCPY:        return "MEMCPY";
    case EVENT_MEMSET:        return "MEMSET";
    case EVENT_MODULE_LOAD:   return "MODULE_LOAD";
    case EVENT_MODULE_UNLOAD: return "MODULE_UNLOAD";
    case EVENT_KERNEL_LAUNCH: return "KERNEL_LAUNCH";
    case EVENT_STREAM_CREATE: return "STREAM_CREATE";
    case EVENT_STREAM_DESTROY:return "STREAM_DESTROY";
    case EVENT_STREAM_SYNC:   return "STREAM_SYNC";
    case EVENT_DEVICE_SYNC:   return "DEVICE_SYNC";
    default:                  return "UNKNOWN";
  }
}

static bool parse_kernel_launch(const uint8_t* data, size_t len,
                                KernelLaunchEvent& kl) {
  const uint8_t* p = data;
  const uint8_t* end = data + len;

  if (p + 2 > end) return false;
  uint16_t name_len;
  memcpy(&name_len, p, 2); p += 2;

  if (p + name_len > end) return false;
  kl.kernel_name.assign(reinterpret_cast<const char*>(p), name_len);
  p += name_len;

  // co_hash (16 bytes): identifies the specific code object this kernel belongs to.
  // Present in traces recorded after the co_hash fix; older traces have payload too
  // short here, so we default to 0,0 (search all modules) and skip gracefully.
  kl.co_hash_lo = 0; kl.co_hash_hi = 0;
  if (p + 16 <= end) {
    memcpy(&kl.co_hash_lo, p, 8); p += 8;
    memcpy(&kl.co_hash_hi, p, 8); p += 8;
  }

  if (p + 12 + 12 + 4 + 2 + 2 > end) return false;
  memcpy(kl.grid, p, 12); p += 12;
  memcpy(kl.block, p, 12); p += 12;
  memcpy(&kl.shared_mem, p, 4); p += 4;

  uint16_t num_args, num_snapshots;
  memcpy(&num_args, p, 2); p += 2;
  memcpy(&num_snapshots, p, 2); p += 2;

  // Parse args
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

  // Parse buffer snapshots
  for (uint16_t i = 0; i < num_snapshots; i++) {
    if (p + 41 > end) return false;  // 8+8+8+16+1
    BufferSnapshot snap;
    memcpy(&snap.ptr_handle, p, 8); p += 8;
    memcpy(&snap.offset, p, 8); p += 8;
    memcpy(&snap.length, p, 8); p += 8;
    memcpy(&snap.hash_lo, p, 8); p += 8;
    memcpy(&snap.hash_hi, p, 8); p += 8;
    snap.direction = *p++;
    kl.snapshots.push_back(snap);
  }

  return true;
}

bool load_archive(const std::string& path, Archive& archive) {
  archive.path = path;

  // Read events.bin
  std::string events_path = path + "/events.bin";
  FILE* f = fopen(events_path.c_str(), "rb");
  if (!f) {
    fprintf(stderr, "[HRR] Cannot open %s\n", events_path.c_str());
    return false;
  }

  while (true) {
    Event ev;
    if (fread(&ev.header, sizeof(EventHeader), 1, f) != 1) break;

    if (ev.header.magic != MAGIC) {
      fprintf(stderr, "[HRR] Bad magic at event %zu\n", archive.events.size());
      break;
    }

    // Read payload
    if (ev.header.payload_length > 0) {
      ev.raw_payload.resize(ev.header.payload_length);
      if (fread(ev.raw_payload.data(), 1, ev.header.payload_length, f) !=
          ev.header.payload_length) {
        break;
      }
    }

    // Parse typed payloads
    const uint8_t* pl = ev.raw_payload.data();
    size_t pl_len = ev.raw_payload.size();

    switch (ev.header.event_type) {
      case EVENT_MALLOC:
        if (pl_len >= 20) {
          memcpy(&ev.malloc_ev.ptr_handle, pl, 8);
          memcpy(&ev.malloc_ev.size, pl + 8, 8);
          memcpy(&ev.malloc_ev.flags, pl + 16, 4);
        }
        break;

      case EVENT_FREE:
        if (pl_len >= 8) {
          memcpy(&ev.malloc_ev.ptr_handle, pl, 8);
        }
        break;

      case EVENT_MEMCPY:
        if (pl_len >= 44) {
          memcpy(&ev.memcpy_ev, pl, 44);
        }
        break;

      case EVENT_MODULE_LOAD:
        if (pl_len >= 24) {
          memcpy(&ev.module_load_ev, pl, 24);
        }
        break;

      case EVENT_KERNEL_LAUNCH: {
        auto* kl = new KernelLaunchEvent();
        if (parse_kernel_launch(pl, pl_len, *kl)) {
          ev.kernel_launch = kl;
          archive.kernel_count++;
        } else {
          delete kl;
        }
        break;
      }

      default:
        break;
    }

    archive.events.push_back(std::move(ev));
  }

  fclose(f);
  archive.event_count = archive.events.size();

  // Enumerate blobs
  std::string blobs_dir = path + "/blobs";
  if (fs::exists(blobs_dir)) {
    for (auto& entry : fs::recursive_directory_iterator(blobs_dir)) {
      if (entry.is_regular_file() && entry.path().extension() == ".blob") {
        std::string stem = entry.path().stem().string();
        archive.blobs[stem] = entry.path().string();
        archive.blob_count++;
      }
    }
  }

  // Enumerate code objects
  std::string co_dir = path + "/code_objects";
  if (fs::exists(co_dir)) {
    for (auto& entry : fs::directory_iterator(co_dir)) {
      if (entry.is_regular_file() && entry.path().extension() == ".hsaco") {
        std::string stem = entry.path().stem().string();
        archive.code_objects[stem] = entry.path().string();
        archive.code_object_count++;
      }
    }
  }

  return true;
}

bool read_blob(const Archive& archive, uint64_t hash_lo, uint64_t hash_hi,
               std::vector<uint8_t>& data) {
  std::string hex = hash_hex(hash_lo, hash_hi);
  auto it = archive.blobs.find(hex);
  if (it == archive.blobs.end()) return false;

  FILE* f = fopen(it->second.c_str(), "rb");
  if (!f) return false;

  fseek(f, 0, SEEK_END);
  long size = ftell(f);
  fseek(f, 0, SEEK_SET);

  data.resize(size);
  size_t read = fread(data.data(), 1, size, f);
  fclose(f);
  return read == static_cast<size_t>(size);
}

bool read_code_object(const Archive& archive, uint64_t hash_lo, uint64_t hash_hi,
                      std::vector<uint8_t>& data) {
  std::string hex = hash_hex(hash_lo, hash_hi);
  auto it = archive.code_objects.find(hex);
  if (it == archive.code_objects.end()) return false;

  FILE* f = fopen(it->second.c_str(), "rb");
  if (!f) return false;

  fseek(f, 0, SEEK_END);
  long size = ftell(f);
  fseek(f, 0, SEEK_SET);

  data.resize(size);
  size_t read = fread(data.data(), 1, size, f);
  fclose(f);
  return read == static_cast<size_t>(size);
}

}  // namespace hrr
