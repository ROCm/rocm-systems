// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/code/patch/log_buffer.h"

#include "rocjitsu/code/patch/error_report.h"

#include "util/bit.h"

#include <cstring>
#include <format>
#include <new>

namespace rocjitsu {

namespace {

constexpr std::size_t kBufferAlignment = alignof(RjLogBufferHeader); // 64

// Stamp the self-describing control fields of a logging header. Factored out of
// the allocator so a future device-shared factory can reuse it over an
// externally-provided [header][slots] image. Assumes the reserved padding is
// already zeroed; this sets every meaningful field explicitly.
void init_header(RjLogBufferHeader *hdr, uint32_t slot_count) {
  hdr->magic = kRjLogMagic;
  hdr->abi_version = kRjLogAbiVersion;
  hdr->header_size = static_cast<uint16_t>(sizeof(RjLogBufferHeader));
  hdr->record_size = kRjLogRecordSize;
  hdr->slot_count = slot_count;
  hdr->flags = 0;
  hdr->write_ptr = 0;
  hdr->read_ptr = 0;
  hdr->overflow_count = 0;
}

} // namespace

void LogBuffer::AlignedStorageDeleter::operator()(void *p) const noexcept {
  ::operator delete(p, std::align_val_t{kBufferAlignment});
}

LogBuffer::LogBuffer(void *storage, uint32_t slot_count, size_t total_bytes)
    : storage_(storage), slot_count_(slot_count), total_bytes_(total_bytes) {}

std::unique_ptr<LogBuffer> LogBuffer::create(uint32_t slot_count, std::string *error_out) {
  if (!util::is_power_of_2(slot_count)) {
    report(error_out, "slot_count must be a nonzero power of two");
    return nullptr;
  }

  // Header and records are each 64-byte sized/aligned, so the contiguous
  // [header][slot_count * record] image is exactly the sum of the two.
  const size_t total_bytes =
      sizeof(RjLogBufferHeader) + static_cast<size_t>(slot_count) * sizeof(RjLogRecord);

  // Own the backing allocation via RAII immediately so it cannot leak before
  // ownership transfers into the LogBuffer. Both allocations below use
  // non-throwing new -- failure is a nullptr return, not an exception -- so the
  // guard's job is the null-return window: if the wrapper allocation fails, the
  // early return must still free this backing buffer (see release() below).
  std::unique_ptr<void, AlignedStorageDeleter> storage(
      ::operator new(total_bytes, std::align_val_t{kBufferAlignment}, std::nothrow));
  if (storage == nullptr) {
    report(error_out, "failed to allocate logging buffer");
    return nullptr;
  }

  std::memset(storage.get(), 0, total_bytes);

  // RjLogBufferHeader and RjLogRecord are implicit-lifetime types. The
  // operator new above implicitly creates such objects in the returned storage,
  // so the header()/records() reinterpret_casts below yield pointers to live
  // objects without an explicit placement-new. Do not add non-trivial members
  // to these structs or this guarantee no longer holds.

  // std::unique_ptr private-ctor dance: construct, then initialize the header.
  // Non-throwing new so a wrapper allocation failure returns nullptr (freeing
  // storage) rather than throwing; ownership of storage transfers only after the
  // LogBuffer is live.
  auto buffer = std::unique_ptr<LogBuffer>(new (std::nothrow)
                                               LogBuffer(storage.get(), slot_count, total_bytes));
  if (buffer == nullptr) {
    report(error_out, "failed to allocate logging buffer");
    return nullptr;
  }
  storage.release();
  init_header(buffer->header(), slot_count);
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

bool LogBuffer::validate(std::string *error_out) const {
  const RjLogBufferHeader *hdr = header();
  if (hdr->magic != kRjLogMagic) {
    report(error_out, "log buffer magic mismatch");
    return false;
  }
  if (hdr->abi_version != kRjLogAbiVersion) {
    report(error_out, "log buffer abi_version mismatch");
    return false;
  }
  if (hdr->header_size != sizeof(RjLogBufferHeader)) {
    report(error_out, "log buffer header_size mismatch");
    return false;
  }
  if (hdr->record_size != kRjLogRecordSize) {
    report(error_out, "log buffer record_size mismatch");
    return false;
  }
  if (hdr->slot_count != slot_count_) {
    report(error_out, "log buffer slot_count mismatch");
    return false;
  }
  return true;
}

DrainStats LogBuffer::drain(const LogRecordCallback &callback) {
  DrainStats stats;
  RjLogBufferHeader *hdr = header();
  RjLogRecord *recs = records();
  const uint32_t mask = slot_count_ - 1;

  // drain runs single-threaded after dispatch completion, so these are the
  // final settled counter values (no concurrent producer to race).
  const uint64_t write_ptr = hdr->write_ptr;
  const uint64_t read_ptr = hdr->read_ptr;
  uint64_t pending = write_ptr - read_ptr;

  // Producer contract is drop-newest: a full ring drops the incoming record and
  // bumps overflow_count WITHOUT advancing write_ptr, so a well-behaved producer
  // never lets pending exceed slot_count. The clamp below is therefore not a
  // recovery path -- it is a safety guard for the untrusted cross-agent
  // boundary. If a buggy device advances write_ptr past read_ptr + slot_count
  // anyway, an unclamped walk would let (read_ptr + n) & mask alias earlier
  // slots and hand the same record to the callback more than once (silent
  // duplicate instrumentation data); clamping the walk to the physical slots
  // prevents that. dropped_overrun records the clamped excess purely as a debug
  // signal that a producer broke contract -- we do NOT try to reconstruct which
  // records survived or in what order, because once the contract is violated
  // that ordering is undefined.
  if (pending > slot_count_) {
    stats.dropped_overrun = pending - slot_count_;
    pending = slot_count_;
  }
  // Walk `pending` monotonic indices, mapping each to its slot. The
  // `read_ptr + n` and `& mask` arithmetic is unsigned and wrap-safe so the
  // walk stays correct even if the counters roll over.
  for (uint64_t n = 0; n < pending; ++n) {
    RjLogRecord &record = recs[(read_ptr + n) & mask];
    if (record.valid != 1) {
      // Advertised by write_ptr but not published: a protocol error after
      // completion, not a race to spin on.
      ++stats.invalid_slots;
      continue;
    }
    // Defense-in-depth against a single corrupted record, not the primary layout
    // guard: drain() strides by a fixed sizeof(RjLogRecord), so a wholesale
    // record_size mismatch desyncs the stride itself and must be caught by
    // validate() on the header (the required precondition). Past that, a record
    // tripping this was built against a different layout -- decoding it with our
    // offsets would be silently wrong, so drop it.
    if (record.abi_version != kRjLogAbiVersion || record.record_size != kRjLogRecordSize) {
      record.valid = 0;
      ++stats.incompatible_records;
      continue;
    }
    if (callback)
      callback(record);
    record.valid = 0;
    ++stats.records_drained;
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
  return std::format("rj-log record_type={} site=0x{:x} wave=0x{:x} lane={} payload=0x{:x}",
                     record.record_type, record.site, record.wave_id, record.writer_lane,
                     record.payload);
}

} // namespace rocjitsu
