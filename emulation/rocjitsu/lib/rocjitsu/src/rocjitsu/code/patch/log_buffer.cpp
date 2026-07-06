// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/code/patch/log_buffer.h"

#include "rocjitsu/code/patch/error_report.h"

#include "util/bit.h"

#include <cassert>
#include <cinttypes>
#include <cstdio>
#include <cstring>
#include <new>

namespace rocjitsu {

namespace {

constexpr std::size_t kBufferAlignment = alignof(RjLogBufferHeader); // 64

} // namespace

void LogBuffer::AlignedStorageDeleter::operator()(void *p) const noexcept {
  ::operator delete(p, std::align_val_t{kBufferAlignment});
}

LogBuffer::LogBuffer(void *storage, uint32_t slot_count, size_t total_bytes)
    : storage_(storage), slot_count_(slot_count), total_bytes_(total_bytes) {}

std::unique_ptr<LogBuffer> LogBuffer::create_for_host_tests(uint32_t slot_count,
                                                            std::string *error_out) {
  if (!util::is_power_of_2(slot_count)) {
    report(error_out, "slot_count must be a nonzero power of two");
    return nullptr;
  }

  // Header and records are each 64-byte sized/aligned, so the contiguous
  // [header][slot_count * record] image is exactly the sum of the two.
  const size_t total_bytes =
      sizeof(RjLogBufferHeader) + static_cast<size_t>(slot_count) * sizeof(RjLogRecord);

  void *storage = ::operator new(total_bytes, std::align_val_t{kBufferAlignment}, std::nothrow);
  if (storage == nullptr) {
    report(error_out, "failed to allocate logging buffer");
    return nullptr;
  }

  std::memset(storage, 0, total_bytes);

  // RjLogBufferHeader and RjLogRecord are implicit-lifetime types. The
  // operator new above implicitly creates such objects in the returned storage,
  // so the header()/records() reinterpret_casts below yield pointers to live
  // objects without an explicit placement-new. Do not add non-trivial members
  // to these structs or this guarantee no longer holds.

  // std::unique_ptr private-ctor dance: construct, then initialize the header.
  auto buffer = std::unique_ptr<LogBuffer>(new LogBuffer(storage, slot_count, total_bytes));

  RjLogBufferHeader *hdr = buffer->header();
  hdr->magic = kRjLogMagic;
  hdr->abi_version = kRjLogAbiVersion;
  hdr->header_size = static_cast<uint16_t>(sizeof(RjLogBufferHeader));
  hdr->record_size = kRjLogRecordSize;
  hdr->slot_count = slot_count;
  // flags and all counters are already zeroed by the memset above.

  return buffer;
}

LogBuffer::~LogBuffer() = default;

RjLogBufferHeader *LogBuffer::header() { return static_cast<RjLogBufferHeader *>(storage_.get()); }

const RjLogBufferHeader *LogBuffer::header() const {
  return static_cast<const RjLogBufferHeader *>(storage_.get());
}

RjLogRecord *LogBuffer::records() { return reinterpret_cast<RjLogRecord *>(header() + 1); }

const RjLogRecord *LogBuffer::records() const {
  return reinterpret_cast<const RjLogRecord *>(header() + 1);
}

uint32_t LogBuffer::slot_count() const { return slot_count_; }

uint64_t LogBuffer::overflow_count() const { return header()->overflow_count; }

size_t LogBuffer::total_bytes() const { return total_bytes_; }

DrainStats LogBuffer::drain(const LogRecordCallback &callback) {
  DrainStats stats;
  RjLogBufferHeader *hdr = header();
  RjLogRecord *recs = records();
  const uint32_t mask = slot_count_ - 1;

  // drain runs single-threaded after dispatch completion, so these are the
  // final settled counter values (no concurrent producer to race).
  const uint64_t write_ptr = hdr->write_ptr;
  const uint64_t read_ptr = hdr->read_ptr;
  const uint64_t pending = write_ptr - read_ptr;

  // Producer contract: a full ring bumps overflow_count without advancing
  // write_ptr, so pending never exceeds slot_count and the `pending` indices
  // map to distinct live slots. Check in debug build.
  assert(pending <= slot_count_ &&
         "log ring producer overran the buffer: write_ptr - read_ptr > slot_count");
  // Walk `pending` monotonic indices, mapping each to its slot. The
  // `read_ptr + n` and `& mask` arithmetic is unsigned and wrap-safe so the
  // walk stays correct even if the counters roll over.
  for (uint64_t n = 0; n < pending; ++n) {
    RjLogRecord &record = recs[(read_ptr + n) & mask];
    if (record.valid == 1) {
      if (callback)
        callback(record);
      record.valid = 0;
      ++stats.records_drained;
    } else {
      // Advertised by write_ptr but not published: a protocol error after
      // completion, not a race to spin on.
      ++stats.invalid_slots;
    }
  }

  hdr->read_ptr = write_ptr;
  stats.overflow_count = hdr->overflow_count;
  return stats;
}

void LogBuffer::reinitialize() {
  RjLogBufferHeader *hdr = header();
  hdr->write_ptr = 0;
  hdr->read_ptr = 0;
  hdr->overflow_count = 0;

  RjLogRecord *recs = records();
  for (uint32_t i = 0; i < slot_count_; ++i)
    recs[i].valid = 0;
}

std::string format_log_line(const RjLogRecord &record) {
  char buf[128];
  std::snprintf(buf, sizeof(buf),
                "rj-log record_type=%" PRIu32 " site=0x%" PRIx32 " wave=0x%" PRIx32 " lane=%" PRIu32
                " payload=0x%" PRIx64,
                record.record_type, record.site, record.wave_id, record.writer_lane,
                record.payload);
  return std::string(buf);
}

} // namespace rocjitsu
