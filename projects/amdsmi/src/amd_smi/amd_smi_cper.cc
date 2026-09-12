// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <cstring>
#include <iterator>
#include <limits>
#include <memory>
#include <sstream>
#include <string_view>

extern "C" {
#include "ras-decode/aca_decode.h"
#include "ras-decode/ras_decode_constants.h"
}
#include "amd_smi/impl/amd_smi_cper.h"
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

// Bounds-checked view over one raw CPER record: every structural offset in a
// record (sec_cnt, sec_offset, ...) is untrusted, so each typed access is gated.
class CperReader {
 public:
  CperReader(const void* base, size_t size) : base_(static_cast<const char*>(base)), size_(size) {}

  // Overflow-safe: (offset + n) is never formed.
  bool is_buffer_fit(size_t offset, size_t needed_size) const {
    return ((offset <= size_) && (needed_size <= (size_ - offset)));
  }

  template <typename Tp>
  const Tp* at(size_t offset) const {
    return is_buffer_fit(offset, sizeof(Tp)) ? reinterpret_cast<const Tp*>(base_ + offset)
                                             : nullptr;
  }

 private:
  const char* base_;
  size_t size_;
};

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

#define GUID_INIT(a, b, c, d0, d1, d2, d3, d4, d5, d6, d7) \
  {(a) & 0xff,                                             \
   ((a) >> 8) & 0xff,                                      \
   ((a) >> 16) & 0xff,                                     \
   ((a) >> 24) & 0xff,                                     \
   (b) & 0xff,                                             \
   ((b) >> 8) & 0xff,                                      \
   (c) & 0xff,                                             \
   ((c) >> 8) & 0xff,                                      \
   (d0),                                                   \
   (d1),                                                   \
   (d2),                                                   \
   (d3),                                                   \
   (d4),                                                   \
   (d5),                                                   \
   (d6),                                                   \
   (d7)};

/* Machine Check Exception */
#define CPER_NOTIFY_MCE \
  GUID_INIT(0xE8F56FFE, 0x919C, 0x4cc5, 0xBA, 0x88, 0x65, 0xAB, 0xE1, 0x49, 0x13, 0xBB)
#define CPER_NOTIFY_CMC \
  GUID_INIT(0x2DCE8BB1, 0xBDD7, 0x450e, 0xB9, 0xAD, 0x9C, 0xF4, 0xEB, 0xD4, 0xF8, 0x90)
#define BOOT_TYPE \
  GUID_INIT(0x3D61A466, 0xAB40, 0x409a, 0xA6, 0x98, 0xF3, 0x62, 0xD4, 0x64, 0xB3, 0x8F)
#define AMD_OOB_CRASHDUMP \
  GUID_INIT(0x32AC0C78, 0x2623, 0x48F6, 0xB0, 0xD0, 0x73, 0x65, 0x72, 0x5F, 0xD6, 0xAE)
#define AMD_GPU_NONSTANDARD_ERROR \
  GUID_INIT(0x32AC0C78, 0x2623, 0x48F6, 0x81, 0xA2, 0xAC, 0x69, 0x17, 0x80, 0x55, 0x1D)
#define PROC_ERR_SECTION_TYPE \
  GUID_INIT(0xDC3EA0B0, 0xA144, 0x4797, 0xB9, 0x5B, 0x53, 0xFA, 0x24, 0x2B, 0x6E, 0x1D)

static amdsmi_cper_guid_t bt = BOOT_TYPE;
static amdsmi_cper_guid_t cr = AMD_OOB_CRASHDUMP;
static amdsmi_cper_guid_t nonstd = AMD_GPU_NONSTANDARD_ERROR;
static amdsmi_cper_guid_t proc_err = PROC_ERR_SECTION_TYPE;

static int cper_is_cr(const amdsmi_cper_guid_t* guid) {
  return !memcmp(&cr, guid, sizeof(amdsmi_cper_guid_t));
}

static int cper_is_nonstd(const amdsmi_cper_guid_t* guid) {
  return !memcmp(&nonstd, guid, sizeof(amdsmi_cper_guid_t));
}

static int cper_is_proc_err(const amdsmi_cper_guid_t* guid) {
  return !memcmp(&proc_err, guid, sizeof(amdsmi_cper_guid_t));
}

static int cper_is_bt(const amdsmi_cper_guid_t* guid) {
  return !memcmp(&bt, guid, sizeof(amdsmi_cper_guid_t));
}

static int cper_num_sec(const amdsmi_cper_hdr_t* hdr) { return hdr->sec_cnt; }

static const amdsmi_cper_guid_t* get_sec_desc_type(const struct cper_sec_desc* desc) {
  return &desc->sec_type;
}

static const amdsmi_cper_guid_t* get_cper_type(const amdsmi_cper_hdr_t* hdr) {
  return &hdr->notify_type;
}

static size_t cper_sec_desc_offset(int idx) {
  return (sizeof(amdsmi_cper_hdr_t) + (sizeof(struct cper_sec_desc) * static_cast<size_t>(idx)));
}

// The record's fixed-width text fields carry no guaranteed NUL, so streaming one
// as a C string walks off the end of the field it was read from.
static auto bounded_field(const char* field, size_t capacity) -> std::string_view {
  return {field, ::strnlen(field, capacity)};
}

static int cper_dump_sec_desc(const struct cper_sec_desc* desc) {
  std::ostringstream ss;

  ss << __PRETTY_FUNCTION__ << "\n:" << __LINE__ << "[AFIDS]\n~~~~SECTION DESCRIPTION~~~\n";

  ss << "[SEC DESC] REV Major = 0x" << std::hex << static_cast<int>(desc->revision_major) << "\n";
  ss << "[SEC DESC] REV Minor = 0x" << std::hex << static_cast<int>(desc->revision_minor) << "\n";
  ss << "[SEC DESC] Length    = 0x" << std::hex << desc->sec_length << "\n";
  ss << "[SEC DESC] Offset    = 0x" << std::hex << desc->sec_offset << "\n";

  ss << "[SEC DESC] fru_id    = " << bounded_field(desc->fru_id, sizeof(desc->fru_id)) << "\n";
  ss << "[SEC DESC] fru_text  = " << bounded_field(desc->fru_text, sizeof(desc->fru_text)) << "\n";

  ss << std::dec << "\n";

  if (cper_is_cr(&desc->sec_type))
    ss << "[SEC DESC] AMD CrashDump Section\n";
  else if (cper_is_nonstd(&desc->sec_type))
    ss << "[SEC DESC] AMD NonStandard Section\n";
  else if (cper_is_proc_err(&desc->sec_type))
    ss << "[SEC DESC] AMD Proc Error Section\n";
  else
    ss << "UNKNOWN ERROR TYPE!!\n";

  ss << "~~~~SECTION DESCRIPTION~~~\n\n";

  LOG_DEBUG(ss);
  return 0;
}

// decode_afid reads its argument as uint64_t, but both register sources below sit
// in packed structs reached at a record-supplied section offset, so neither
// address is aligned for that read. Each copies into an aligned local first.

static int aca_decode_fatal(const cper_sec_crashdump_data& data, uint32_t flag,
                            uint16_t hw_revision, uint16_t register_context_type) {
  constexpr size_t kFatalErrRegs = (sizeof(data.dump.fatal_err) / sizeof(uint64_t));
  uint64_t register_array[kFatalErrRegs] = {};
  std::memcpy(register_array, &data.dump.fatal_err, sizeof(data.dump.fatal_err));
  return decode_afid(register_array, kFatalErrRegs, flag, hw_revision, register_context_type);
}

// num_regs counts uint64_t registers, matching decode_afid's array_len. It derives
// from an untrusted field and sizes a memcpy at both ends, so the staging array
// enforces its own capacity here. The caller rejects the same counts earlier to
// name the offending record in the log; this bound is what keeps the copy in range.
static int aca_decode_corrected_error(const uint32_t* reg_dump, size_t num_regs, uint32_t flag,
                                      uint16_t hw_revision, uint16_t register_context_type) {
  uint64_t register_array[RAS_DECODE_REGISTER_ARRAY_SIZE_128_BYTES] = {};
  if (num_regs > std::size(register_array)) {
    return -1;
  }
  std::memcpy(register_array, reg_dump, (num_regs * sizeof(register_array[0])));
  return decode_afid(register_array, num_regs, flag, hw_revision, register_context_type);
}

static int cper_dump_nonstd_err(const struct cper_sec_nonstd_err* nonstd_err,
                                const cper_sec_desc* section) {
  std::ostringstream ss;

  struct cper_sec_nonstd_err_body* body = nullptr;

  ss << __PRETTY_FUNCTION__ << "\n:" << __LINE__ << "[AFIDS]\n~~~~NON STANDARD SECTION~~~\n";

  ss << "[NonSTD SEC] Err Info Count    = 0x" << std::hex << nonstd_err->hdr.valid_bits.err_info_cnt
     << "\n";
  ss << "[NonSTD SEC] Err Context Count = 0x" << std::hex
     << nonstd_err->hdr.valid_bits.err_context_cnt << "\n";

  if (nonstd_err->hdr.valid_bits.err_info_cnt != nonstd_err->hdr.valid_bits.err_context_cnt) {
    ss << "~~~~Malformed Non Standard Section!~~~~\n\n";
    goto exit;
  }

  body = reinterpret_cast<struct cper_sec_nonstd_err_body*>((char*)nonstd_err +
                                                            sizeof(struct cper_sec_nonstd_err_hdr));

  ss << "[NonSTD SEC] Reg Ctx Type   = 0x" << std::hex << body->err_ctx.reg_ctx_type << "\n";
  ss << "[NonSTD SEC] Reg Array Size = 0x" << std::hex << body->err_ctx.reg_arr_size << "\n";

  for (int i = 0; i < CPER_ACA_REG_COUNT; i++) {
    ss << "[NonSTD SEC] reg_dump[" << std::dec << i << "] = 0x" << std::hex
       << body->err_ctx.reg_dump[i] << "\n";
  }

exit:
  ss << std::dec << "~~~~NON STANDARD SECTION~~~\n\n";

  LOG_DEBUG(ss);

  if (!body) return -1;

  // reg_arr_size is untrusted and reg_dump is fixed-size (declared uint32_t[] but
  // consumed as uint64_t), so a larger claim describes registers the record does
  // not carry. Reject rather than decode a truncated view of it: clamping would
  // turn a malformed record into an AFID it never earned. The bound doubles as
  // the capacity of the staging array aca_decode_corrected_error copies into, and
  // boot context decodes whatever length it is handed, so nothing below caps it.
  constexpr size_t reg_dump_bytes = sizeof(body->err_ctx.reg_dump);
  constexpr size_t max_regs = (reg_dump_bytes / sizeof(uint64_t));
  static_assert(max_regs == static_cast<size_t>(RAS_DECODE_REGISTER_ARRAY_SIZE_128_BYTES),
                "Bound must stay at the largest array length decode_error_info accepts, "
                "otherwise a full register dump is rejected instead of decoded");
  const size_t num_regs = (body->err_ctx.reg_arr_size / sizeof(uint64_t));
  if (num_regs > max_regs) {
    ss << __PRETTY_FUNCTION__ << "\n:" << __LINE__ << "[AFIDS] reg_arr_size " << std::dec
       << body->err_ctx.reg_arr_size << " exceeds the " << reg_dump_bytes
       << "-byte reg_dump capacity; rejecting section\n";
    LOG_ERROR(ss);
    return -1;
  }

  return aca_decode_corrected_error(body->err_ctx.reg_dump, num_regs, section->flags_mask,
                                    section->revision_major, body->err_ctx.reg_ctx_type);
}

static int cper_dump_cr_fatal(const struct cper_sec_crashdump* crashdump,
                              const cper_sec_desc* section) {
  std::ostringstream ss;
  ss << __PRETTY_FUNCTION__ << "\n:" << __LINE__ << "[AFIDS]\n~~~~CRASH DUMP - FATAL~~~\n";

  ss << "[Crash Dump - Fatal] status_lo = 0x" << std::hex
     << crashdump->data.dump.fatal_err.status_lo << "\n";
  ss << "[Crash Dump - Fatal] status_hi = 0x" << std::hex
     << crashdump->data.dump.fatal_err.status_hi << "\n";
  ss << "[Crash Dump - Fatal] addr_lo   = 0x" << std::hex << crashdump->data.dump.fatal_err.addr_lo
     << "\n";
  ss << "[Crash Dump - Fatal] addr_hi   = 0x" << std::hex << crashdump->data.dump.fatal_err.addr_hi
     << "\n";
  ss << "[Crash Dump - Fatal] ipid_lo   = 0x" << std::hex << crashdump->data.dump.fatal_err.ipid_lo
     << "\n";
  ss << "[Crash Dump - Fatal] ipid_hi   = 0x" << std::hex << crashdump->data.dump.fatal_err.ipid_hi
     << "\n";
  ss << "[Crash Dump - Fatal] synd_lo   = 0x" << std::hex << crashdump->data.dump.fatal_err.synd_lo
     << "\n";
  ss << "[Crash Dump - Fatal] synd_hi   = 0x" << std::hex << crashdump->data.dump.fatal_err.synd_hi
     << "\n";

  ss << std::dec << "~~~~CRASH DUMP - FATAL~~~\n\n";

  LOG_DEBUG(ss);

  return aca_decode_fatal(crashdump->data, section->flags_mask, section->revision_major,
                          crashdump->data.reg_ctx_type);
}

static int cper_dump_cr_boot(const struct cper_sec_crashdump* crashdump,
                             const cper_sec_desc* section) {
  std::ostringstream ss;
  ss << __PRETTY_FUNCTION__ << "\n:" << __LINE__ << "[AFIDS]\n~~~~CRASH DUMP - BOOT TIME~~~\n";

  for (int i = 0; i < CPER_MAX_OAM_COUNT; i++) {
    ss << "[Crash Dump - Boot] bootmsg[" << std::dec << i << "] = 0x" << std::hex
       << crashdump->data.dump.boot_err.msg[i] << "\n";
  }

  ss << "~~~~CRASH DUMP - BOOT TIME~~~\n\n";
  LOG_DEBUG(ss);

  return aca_decode_fatal(crashdump->data, section->flags_mask, section->revision_major,
                          crashdump->data.reg_ctx_type);
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

std::vector<int> cper_decode(const amdsmi_cper_hdr_t* cper, size_t buf_size) {
  std::vector<int> afids;
  std::ostringstream ss;

  // Every check here is repeated by amdsmi_get_afids_from_cper. It is repeated
  // rather than assumed so that walking the record is safe for any (pointer, size)
  // the caller hands over; the public entry point re-checks only to distinguish
  // UNEXPECTED_SIZE from UNEXPECTED_DATA, which a returned vector cannot express.
  if (buf_size < sizeof(amdsmi_cper_hdr_t)) {
    ss << __PRETTY_FUNCTION__ << "\n:" << __LINE__ << "[AFIDS] cper buffer size: " << std::dec
       << buf_size << " is smaller than the cper header: (" << sizeof(amdsmi_cper_hdr_t) << ")\n";
    LOG_ERROR(ss);
    return afids;
  }

  // Without this, non-CPER bytes walk as a record and the sec_cnt and sec_offset
  // that happen to land there yield AFIDs no device reported.
  if (strncmp(cper->signature, "CPER", 4) != 0) {
    ss << __PRETTY_FUNCTION__ << "\n:" << __LINE__
       << "[AFIDS] cper buffer does not have the correct signature\n";
    LOG_ERROR(ss);
    return afids;
  }

  // record_length is untrusted; bound the view by the smaller of it and the
  // caller's buffer.
  const CperReader rec(cper, std::min<size_t>(cper->record_length, buf_size));
  const amdsmi_cper_guid_t* cper_guid = get_cper_type(cper);

  for (int i = 0; i < cper_num_sec(cper); i++) {
    const struct cper_sec_desc* section = rec.at<struct cper_sec_desc>(cper_sec_desc_offset(i));
    if (!section) {
      ss << __PRETTY_FUNCTION__ << "\n:" << __LINE__ << "[AFIDS] section descriptor: " << i
         << " runs past the record; stopping\n";
      LOG_ERROR(ss);
      break;
    }
    const amdsmi_cper_guid_t* sec_guid = get_sec_desc_type(section);
    cper_dump_sec_desc(section);

    int afid = -1;
    if (cper_is_cr(sec_guid)) {
      const struct cper_sec_crashdump* crashdump =
          rec.at<struct cper_sec_crashdump>(section->sec_offset);
      if (!crashdump) {
        ss << __PRETTY_FUNCTION__ << "\n:" << __LINE__ << "[AFIDS] crash dump section: " << i
           << " out of bounds; skipping\n";
        LOG_ERROR(ss);
      } else if (cper_is_bt(cper_guid)) {
        ss << __PRETTY_FUNCTION__ << "\n:" << __LINE__ << "[AFIDS] decoding boot crash dump\n";
        LOG_DEBUG(ss);
        afid = cper_dump_cr_boot(crashdump, section);
      } else {
        ss << __PRETTY_FUNCTION__ << "\n:" << __LINE__ << "[AFIDS] decoding crash dump\n";
        LOG_DEBUG(ss);
        afid = cper_dump_cr_fatal(crashdump, section);
      }
    } else if (cper_is_nonstd(sec_guid) || cper_is_proc_err(sec_guid)) {
      // cper_dump_nonstd_err reads the hdr plus one body immediately after it. The
      // body lives in a flexible array member, so it is outside sizeof() and only
      // this explicit extent covers it; at<> below cannot re-check it.
      const size_t need =
          (sizeof(struct cper_sec_nonstd_err_hdr) + sizeof(struct cper_sec_nonstd_err_body));
      if (!rec.is_buffer_fit(section->sec_offset, need)) {
        ss << __PRETTY_FUNCTION__ << "\n:" << __LINE__ << "[AFIDS] non-standard section: " << i
           << " out of bounds; skipping\n";
        LOG_ERROR(ss);
      } else {
        ss << __PRETTY_FUNCTION__ << "\n:" << __LINE__
           << "[AFIDS] decoding non-standard/proc-error section\n";
        LOG_DEBUG(ss);
        afid =
            cper_dump_nonstd_err(rec.at<struct cper_sec_nonstd_err>(section->sec_offset), section);
      }
    } else {
      ss << __PRETTY_FUNCTION__ << "\n:" << __LINE__ << "[AFIDS] Unknown error type!!\n";
      for (size_t j = 0; j < sizeof(sec_guid->b); ++j) {
        ss << std::hex << static_cast<int>(sec_guid->b[j]) << ":";
      }
      // LOG_ERROR clears the stream contents but not its format flags, so the hex
      // above would follow this record into every later line of the decode loop.
      ss << std::dec << "\n";
      LOG_ERROR(ss);
    }
    if (afid != -1) {
      afids.emplace_back(afid);
    }
  }

  return afids;
}
