// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "smi_nic_cper.h"

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cerrno>
#include <cstring>
#include <limits>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "smi_nic_log.h"

namespace {
constexpr off_t kMaxCperBufferSize = 64 * 1024 * 1024;  // 64 MiB

struct CperFileCtx {
  smi_nic_status_t status = SMI_NIC_STATUS_ERROR;
  std::unique_ptr<char[]> buffer;
  size_t file_size = 0;
};

static CperFileCtx nic_read_cper_file(const std::string& filepath) {
  CperFileCtx ctx;
  std::ostringstream ss;

  struct stat file_stats;
  if (stat(filepath.c_str(), &file_stats) != 0) {
    ss << "[NIC CPER] file does not exist: " << filepath << " errno:" << errno;
    NIC_LOG_DEBUG(ss.str());
    ctx.status = SMI_NIC_STATUS_FILE_NOT_FOUND;
    return ctx;
  }
  if (!S_ISREG(file_stats.st_mode)) {
    ss << "[NIC CPER] not a regular file: " << filepath;
    NIC_LOG_DEBUG(ss.str());
    return ctx;
  }
  if (file_stats.st_size < 0 || file_stats.st_size > kMaxCperBufferSize) {
    ss << "[NIC CPER] implausible size " << file_stats.st_size << " for " << filepath;
    NIC_LOG_DEBUG(ss.str());
    return ctx;
  }

  ctx.file_size = static_cast<size_t>(file_stats.st_size);
  ctx.buffer = std::make_unique<char[]>(ctx.file_size);

  int fd = open(filepath.c_str(), O_RDONLY);
  if (fd == -1) {
    ctx.status = (errno == EACCES || errno == EPERM) ? SMI_NIC_STATUS_NO_PERM
                                                     : SMI_NIC_STATUS_FILE_NOT_FOUND;
    ss << "[NIC CPER] open failed: " << filepath << " errno:" << errno;
    NIC_LOG_DEBUG(ss.str());
    return ctx;
  }
  ssize_t bytes_read = ::read(fd, ctx.buffer.get(), ctx.file_size);
  close(fd);

  if (bytes_read < 0) {
    ss << "[NIC CPER] read failed: " << filepath << " errno:" << errno;
    NIC_LOG_DEBUG(ss.str());
    return ctx;
  }

  ctx.status = SMI_NIC_STATUS_SUCCESS;
  ctx.file_size = static_cast<size_t>(bytes_read);
  return ctx;
}

static std::vector<const amdsmi_cper_hdr_t*> nic_get_cper_headers(const char* buffer,
                                                                  size_t buffer_sz) {
  std::vector<const amdsmi_cper_hdr_t*> headers;
  if (!buffer || buffer_sz < sizeof(amdsmi_cper_hdr_t)) {
    return headers;
  }
  for (size_t i = 0; i <= buffer_sz - sizeof(amdsmi_cper_hdr_t); ++i) {
    const auto* hdr = reinterpret_cast<const amdsmi_cper_hdr_t*>(&buffer[i]);
    if (hdr->signature[0] != 'C' || hdr->signature[1] != 'P' || hdr->signature[2] != 'E' ||
        hdr->signature[3] != 'R') {
      continue;
    }
    if (hdr->signature_end != 0xFFFFFFFF) continue;
    if (hdr->record_length < sizeof(amdsmi_cper_hdr_t) || hdr->record_length > buffer_sz - i) {
      continue;
    }
    headers.emplace_back(hdr);
  }
  return headers;
}
}  // namespace

smi_nic_status_t smi_nic_read_cper_entries(const SmiNic& nic, uint32_t severity_mask,
                                           char* cper_data, uint64_t* buf_size,
                                           amdsmi_cper_hdr_t** cper_hdrs, uint64_t* entry_count,
                                           uint64_t* cursor) {
  const std::string& mgmt_bdf = nic.mgmt_bdf();
  if (mgmt_bdf.empty()) {
    NIC_LOG_DEBUG("[NIC CPER] no mgmt BDF for this NIC");
    return SMI_NIC_STATUS_NOT_SUPPORTED;
  }

  std::string path = "/sys/kernel/debug/pds_core/" + mgmt_bdf + "/cper_events";
  auto ctx = nic_read_cper_file(path);
  if (ctx.status != SMI_NIC_STATUS_SUCCESS) {
    *entry_count = 0;
    *buf_size = 0;
    return ctx.status;
  }

  auto headers = nic_get_cper_headers(ctx.buffer.get(), ctx.file_size);

  constexpr auto kBits = static_cast<uint32_t>(std::numeric_limits<uint32_t>::digits);
  uint64_t data_idx = 0;
  uint64_t header_idx = 0;
  size_t num_copied = 0;

  for (const amdsmi_cper_hdr_t* hdr : headers) {
    uint32_t severity = 0;
    std::memcpy(&severity, &hdr->error_severity, sizeof(severity));
    if (!((severity < kBits) && ((severity_mask & (1U << severity)) != 0U))) {
      continue;
    }
    if (*cursor != header_idx) {
      ++header_idx;
      continue;
    }
    if ((*buf_size - data_idx) < hdr->record_length || num_copied == *entry_count) {
      *entry_count = num_copied;
      *buf_size = data_idx;
      return (data_idx == 0) ? SMI_NIC_STATUS_NO_RESOURCE : SMI_NIC_STATUS_MORE_DATA;
    }
    cper_hdrs[num_copied] = reinterpret_cast<amdsmi_cper_hdr_t*>(&cper_data[data_idx]);
    ++num_copied;
    *cursor = ++header_idx;
    std::memcpy(&cper_data[data_idx], reinterpret_cast<const char*>(hdr), hdr->record_length);
    data_idx += hdr->record_length;
  }

  *entry_count = num_copied;
  *buf_size = data_idx;
  return SMI_NIC_STATUS_SUCCESS;
}
