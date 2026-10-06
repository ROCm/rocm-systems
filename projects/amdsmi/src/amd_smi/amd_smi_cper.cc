// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "amd_smi/impl/amd_smi_cper.h"

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cerrno>
#include <cstring>
#include <limits>
#include <memory>
#include <sstream>
#include <vector>

#include "amd_smi/impl/amd_smi_cper_testing.h"
#include "rocm_smi/rocm_smi_logger.h"

namespace {
// Upper bound on the CPER ring file we will allocate for. debugfs reports the
// ring capacity (a few MiB) in st_size; this cap guards against a nonsensical
// size exhausting memory or throwing across the C API boundary.
constexpr off_t kMaxCperBufferSize = 64 * 1024 * 1024;  // 64 MiB

// read() indirection so unit tests can reproduce the debugfs short-read shape
// (st_size advertises ring capacity, read() returns fewer/zero bytes) that a
// regular file cannot. Defaults to POSIX read(); the production path is unchanged.
// Not thread-safe: only the single-threaded tests mutate it (via
// cper_set_read_fn_for_testing); production never writes it.
ssize_t (*g_cper_read_fn)(int, void*, size_t) = ::read;

static std::vector<const amdsmi_cper_hdr_t*> amdsmi_get_gpu_cper_headers(const char* buffer,
                                                                         size_t buffer_sz) {
  std::ostringstream ss;
  ss << __PRETTY_FUNCTION__ << "\n:" << __LINE__ << "[CPER] buffer_sz: " << buffer_sz;
  LOG_DEBUG(ss);

  std::vector<const amdsmi_cper_hdr_t*> headers;
  if (!buffer) {
    ss << __PRETTY_FUNCTION__ << "\n:" << __LINE__ << "[CPER] buffer is null";
    LOG_ERROR(ss);
    return headers;
  }
  // A valid record begins with a full header. Only scan offsets where an entire
  // amdsmi_cper_hdr_t fits, so reading the header fields below never reads past
  // the end of the buffer.
  if (buffer_sz < sizeof(amdsmi_cper_hdr_t)) {
    return headers;
  }
  for (size_t data_idx = 0; data_idx <= buffer_sz - sizeof(amdsmi_cper_hdr_t); ++data_idx) {
    const amdsmi_cper_hdr_t* hdr = reinterpret_cast<const amdsmi_cper_hdr_t*>(&buffer[data_idx]);
    if (hdr->signature[0] != 'C' || hdr->signature[1] != 'P' || hdr->signature[2] != 'E' ||
        hdr->signature[3] != 'R') {
      continue;
    }
    if (hdr->signature_end != 0xFFFFFFFF) {
      continue;
    }
    // The record must hold a full header and fit from its start offset, not merely
    // within buffer_sz overall. A short record leaves inject_product_serial_number
    // reading record_length and sec_cnt from bytes the memcpy never wrote, and an
    // overlong one would memcpy past the end.
    if ((hdr->record_length < sizeof(amdsmi_cper_hdr_t)) ||
        (hdr->record_length > buffer_sz - data_idx)) {
      continue;
    }
    ss << __PRETTY_FUNCTION__ << "\n:" << __LINE__ << "[CPER] add header at data_idx: " << data_idx
       << ", sig: " << hdr->signature[0] << hdr->signature[1] << hdr->signature[2]
       << hdr->signature[3];
    LOG_DEBUG(ss);
    headers.emplace_back(hdr);
  }
  return headers;
}

struct CperFileCtx {
  amdsmi_status_t status = AMDSMI_STATUS_FILE_ERROR;
  std::unique_ptr<char[]> buffer;
  size_t file_size = 0;
};

static auto amdsmi_read_cper_file(const std::string& filepath) -> CperFileCtx {
  std::ostringstream ss;

  CperFileCtx ctx;
  ctx.status = AMDSMI_STATUS_FILE_ERROR;
  ctx.file_size = 0;

  struct stat file_stats;
  if (stat(filepath.c_str(), &file_stats) != 0) {
    int stat_errno = errno;
    ss << __PRETTY_FUNCTION__ << "\n:" << __LINE__ << "[CPER] file does not exist: " << filepath
       << ", errno: " << stat_errno << "): " << strerror(stat_errno);
    ctx.status = AMDSMI_STATUS_NOT_SUPPORTED;
    return ctx;
  }
  if (!S_ISREG(file_stats.st_mode)) {
    ss << __PRETTY_FUNCTION__ << "\n:" << __LINE__
       << "[CPER] file is not a regular file: " << filepath;
    return ctx;
  }

  // st_size carries the ring capacity from debugfs; reject a negative or absurd
  // value before allocating rather than risk std::bad_alloc or OOM.
  if (file_stats.st_size < 0 || file_stats.st_size > kMaxCperBufferSize) {
    ss << __PRETTY_FUNCTION__ << "\n:" << __LINE__
       << "[CPER] implausible file size: " << file_stats.st_size << " for " << filepath;
    LOG_ERROR(ss);
    return ctx;
  }

  // st_size can be 0 here (e.g. an empty regular file). We do not special-case
  // it: the uniform open/read/close path below reads 0 bytes and reports an
  // empty ring, keeping the empty / short / full read handling in one place.
  ctx.file_size = static_cast<size_t>(file_stats.st_size);
  ctx.buffer = std::make_unique<char[]>(ctx.file_size);

  // Use POSIX open/read/close, not std::ifstream: its basic_filebuf is freed by
  // this library's libstdc++, an invalid free when libamd_smi.so is LD_PRELOAD-ed
  // under a different host libstdc++.
  int fd = open(filepath.c_str(), O_RDONLY);
  if (fd == -1) {
    ss << __PRETTY_FUNCTION__ << "\n:" << __LINE__ << "[CPER] failed to open file: " << filepath
       << ", errno:(" << errno << "): " << strerror(errno);
    LOG_ERROR(ss);
    return ctx;
  }
  ssize_t bytes_read = g_cper_read_fn(fd, ctx.buffer.get(), ctx.file_size);
  if (bytes_read < 0) {
    ss << __PRETTY_FUNCTION__ << "\n:" << __LINE__ << "[CPER] failed to read file: " << filepath
       << ", errno:(" << errno << "): " << strerror(errno);
    LOG_ERROR(ss);
    close(fd);
    return ctx;
  }
  close(fd);

  // Empty ring: st_size advertises capacity but read() returns 0 bytes. Not an
  // error, so report success ("no CPER records") rather than FILE_ERROR.
  if (bytes_read == 0) {
    ss << __PRETTY_FUNCTION__ << "\n:" << __LINE__ << "[CPER] ring is empty (no records)";
    LOG_DEBUG(ss);
  } else if (bytes_read < file_stats.st_size) {
    // Short read is normal: the ring advertises its full capacity in st_size but
    // read() returns only the bytes currently populated. Parse whatever was
    // returned (the parser bounds-checks each record); log at debug only.
    ss << __PRETTY_FUNCTION__ << "\n:" << __LINE__ << "[CPER] short read: " << bytes_read << " of "
       << file_stats.st_size << " bytes; parsing available records";
    LOG_DEBUG(ss);
  }

  ctx.status = AMDSMI_STATUS_SUCCESS;
  ctx.file_size = static_cast<size_t>(bytes_read);
  return ctx;
}

static int cper_num_sec(const amdsmi_cper_hdr_t* hdr) { return hdr->sec_cnt; }

static size_t cper_sec_desc_offset(int idx) {
  return (sizeof(amdsmi_cper_hdr_t) + (sizeof(struct cper_sec_desc) * static_cast<size_t>(idx)));
}

static void inject_product_serial_number(amdsmi_cper_hdr_t* cper, uint64_t product_serial) {
  // record_length was validated to fit the copied record; stop once sec_cnt walks
  // a descriptor past it rather than writing out of bounds.
  const size_t bound = cper->record_length;
  const std::string serial = std::to_string(product_serial);
  for (int i = 0; i < cper_num_sec(cper); i++) {
    const size_t offset = cper_sec_desc_offset(i);
    if ((offset > bound) || (sizeof(struct cper_sec_desc) > (bound - offset))) {
      break;
    }
    auto* sec_desc =
        reinterpret_cast<struct cper_sec_desc*>(reinterpret_cast<char*>(cper) + offset);
    strncpy(sec_desc->fru_id, serial.c_str(), (sizeof(sec_desc->fru_id) - 1));
    sec_desc->fru_id[(sizeof(sec_desc->fru_id) - 1)] = '\0';
  }
}

}  // namespace

// See amd_smi/impl/amd_smi_cper_testing.h for the contract.
void cper_set_read_fn_for_testing(ssize_t (*read_fn)(int, void*, size_t)) {
  g_cper_read_fn = read_fn ? read_fn : ::read;
}

amdsmi_status_t amdsmi_get_gpu_cper_entries_by_path(const char* amdgpu_ring_cper_file,
                                                    uint32_t severity_mask, char* cper_data,
                                                    uint64_t* buf_size,
                                                    amdsmi_cper_hdr_t** cper_hdrs,
                                                    uint64_t* entry_count, uint64_t* cursor,
                                                    uint64_t product_serial) {
  std::ostringstream ss;
  if (!amdgpu_ring_cper_file) {
    if (entry_count) {
      *entry_count = 0;
    }
    if (buf_size) {
      *buf_size = 0;
    }
    return AMDSMI_STATUS_OUT_OF_RESOURCES;
  }
  ss << __PRETTY_FUNCTION__ << "\n:" << __LINE__ << "[CPER] begin\n"
     << ", amdgpu_ring_cper_file: " << amdgpu_ring_cper_file
     << ", severity_mask: " << severity_mask;
  LOG_DEBUG(ss);

  if (!cper_data) {
    ss << __PRETTY_FUNCTION__ << "\n:" << __LINE__
       << "[CPER] cper_data should be a valid memory address\n";
    LOG_ERROR(ss);
    if (entry_count) {
      *entry_count = 0;
    }
    if (buf_size) {
      *buf_size = 0;
    }
    return AMDSMI_STATUS_OUT_OF_RESOURCES;
  } else if (!buf_size) {
    ss << __PRETTY_FUNCTION__ << "\n:" << __LINE__
       << "[CPER] buf_size should be a valid memory address";
    LOG_ERROR(ss);
    if (entry_count) {
      *entry_count = 0;
    }
    return AMDSMI_STATUS_OUT_OF_RESOURCES;
  } else if (!entry_count) {
    ss << __PRETTY_FUNCTION__ << "\n:" << __LINE__
       << "[CPER] entry_count should be a valid memory address";
    LOG_ERROR(ss);
    *buf_size = 0;
    return AMDSMI_STATUS_OUT_OF_RESOURCES;
  } else if (!*buf_size) {
    ss << __PRETTY_FUNCTION__ << "\n:" << __LINE__ << "[CPER] buf_size should be greater than zero";
    LOG_ERROR(ss);
    *entry_count = 0;
    return AMDSMI_STATUS_OUT_OF_RESOURCES;
  } else if (!*entry_count) {
    ss << __PRETTY_FUNCTION__ << "\n:" << __LINE__ << "[CPER] entry_count should be greater than 0";
    LOG_ERROR(ss);
    *buf_size = 0;
    return AMDSMI_STATUS_OUT_OF_RESOURCES;
  } else if (!cper_hdrs) {
    ss << __PRETTY_FUNCTION__ << "\n:" << __LINE__
       << "[CPER] cper_hdrs should be a valid memory address";
    LOG_ERROR(ss);
    *entry_count = 0;
    *buf_size = 0;
    return AMDSMI_STATUS_OUT_OF_RESOURCES;
  } else if (!cursor) {
    ss << __PRETTY_FUNCTION__ << "\n:" << __LINE__
       << "[CPER] cursor should be a valid memory address";
    LOG_ERROR(ss);
    *entry_count = 0;
    *buf_size = 0;
    return AMDSMI_STATUS_OUT_OF_RESOURCES;
  }

  auto ctx = amdsmi_read_cper_file(amdgpu_ring_cper_file);
  if (ctx.status != AMDSMI_STATUS_SUCCESS) {
    *entry_count = 0;
    *buf_size = 0;
    return ctx.status;
  }

  auto headers = amdsmi_get_gpu_cper_headers(ctx.buffer.get(), ctx.file_size);
  ss << __PRETTY_FUNCTION__ << "\n:" << __LINE__ << "[CPER] num headers: " << headers.size();
  LOG_DEBUG(ss);

  uint64_t data_idx = 0;
  uint64_t header_idx = 0;
  size_t num_headers_copied = 0;
  // error_severity is typed as an enum but arrives from the record, so it can hold
  // a value the enum never names. Read the bytes instead of the enum, and treat a
  // severity wider than the mask as unselectable rather than shifting by it.
  constexpr auto kSeverityMaskBits =
      static_cast<uint32_t>(std::numeric_limits<decltype(severity_mask)>::digits);
  static_assert(sizeof(amdsmi_cper_sev_t) == sizeof(uint32_t),
                "error_severity is copied out as a uint32_t to keep the enum's own bytes");
  for (const amdsmi_cper_hdr_t* header : headers) {
    uint32_t severity = 0;
    std::memcpy(&severity, &header->error_severity, sizeof(severity));
    const bool is_severity_selected =
        ((severity < kSeverityMaskBits) && ((severity_mask & (1U << severity)) != 0U));
    if (!is_severity_selected) {
      ss << __PRETTY_FUNCTION__ << "\n:" << __LINE__
         << "[CPER] cper header rejected with severity: 0x" << std::hex << severity
         << ", given severity_mask: 0x" << std::hex << severity_mask
         << ", record_length:" << std::dec << header->record_length;
      LOG_DEBUG(ss);
      continue;
    } else {
      ss << __PRETTY_FUNCTION__ << "\n:" << __LINE__
         << "[CPER] cper header accepted with severity: 0x" << std::hex << severity
         << ", given severity_mask: 0x" << std::hex << severity_mask
         << ", record_length:" << std::dec << header->record_length;
      LOG_DEBUG(ss);
    }
    if ((*buf_size - data_idx) < header->record_length) {
      ss << __PRETTY_FUNCTION__ << "\n:" << __LINE__
         << "[CPER] buffer filled up without copying all cper entries, buf_size: " << std::dec
         << *buf_size;
      LOG_ERROR(ss);
      *entry_count = num_headers_copied;
      *buf_size = data_idx;
      return (data_idx == 0) ? AMDSMI_STATUS_OUT_OF_RESOURCES : AMDSMI_STATUS_MORE_DATA;
    }
    if (num_headers_copied == *entry_count) {
      ss << __PRETTY_FUNCTION__ << "\n:" << __LINE__
         << "[CPER] cper_hdrs filled up before finished with copying all header pointers, "
            "entry_count: "
         << std::dec << *entry_count;
      LOG_ERROR(ss);
      *entry_count = num_headers_copied;
      *buf_size = data_idx;
      return (data_idx == 0) ? AMDSMI_STATUS_OUT_OF_RESOURCES : AMDSMI_STATUS_MORE_DATA;
    }
    if (*cursor != header_idx) {
      ++header_idx;
      continue;
    }
    cper_hdrs[num_headers_copied] = reinterpret_cast<amdsmi_cper_hdr_t*>(&cper_data[data_idx]);
    ++num_headers_copied;
    *cursor = ++header_idx;
    std::memcpy(&cper_data[data_idx], reinterpret_cast<const char*>(header), header->record_length);
    inject_product_serial_number(reinterpret_cast<amdsmi_cper_hdr_t*>(&cper_data[data_idx]),
                                 product_serial);
    data_idx += header->record_length;
  }
  *entry_count = num_headers_copied;
  *buf_size = data_idx;

  ss << __PRETTY_FUNCTION__ << "\n:" << __LINE__ << "[CPER] *entry_count: " << *entry_count
     << ", *cursor: " << *cursor << ", *buf_size: " << *buf_size;

  LOG_DEBUG(ss);
  return AMDSMI_STATUS_SUCCESS;
}
