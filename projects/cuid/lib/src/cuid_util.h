// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef CUID_UTIL_H
#define CUID_UTIL_H

#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "hmac.h"
#include "include/amd_cuid.h"
#include "src/cuid_internal.h"

enum LogLevel { DEBUG, INFO, WARN, ERROR };

class Logger {
 public:
  static Logger& instance() {
    static Logger logger_;
    return logger_;
  }

  void set_level(LogLevel level) { level_ = level; }
  LogLevel level() const { return level_; }

  const char* LogLevelName(LogLevel level) const;

  void log(LogLevel level, const std::string& msg) const;

 private:
  Logger() : level_(INFO) {}
  LogLevel level_;
};

// NOLINTBEGIN(bugprone-macro-parentheses)
// `msg` is deliberately left unparenthesised. Callers pass a stream-
// continuation fragment such as `"failed: " << path`, which is only valid
// glued onto the left of `_log_stream_ <<`. Wrapping it would evaluate
// `const char[] << std::string` as an expression of its own, which does not
// compile. clang-tidy cannot see that, so the check is suppressed here rather
// than obeyed.
#define LOG(level, msg)                                  \
  do {                                                   \
    std::ostringstream _log_stream_;                     \
    _log_stream_ << msg;                                 \
    Logger::instance().log((level), _log_stream_.str()); \
  } while (0)
// NOLINTEND(bugprone-macro-parentheses)

namespace CuidUtilities {
// Thread-safe replacement for strerror(). strerror() returns a pointer into a
// static buffer, so two threads reporting errors at once can read a torn or
// wrong message. libamdcuid is linked into multithreaded hosts -- amd_smi and
// libhsa-runtime64.so among them -- so it must not use it.
std::string errno_string(int err);

// A zero hardware fingerprint is the absence of an identity, not an identity.
// Unprogrammed DSN capabilities and unconfigured MAC addresses both read back
// as all-zero, and reporting that as a successful fingerprint gives every such
// device on every machine the same primary CUID. Callers use this to convert
// "read succeeded, value is meaningless" into HW_FINGERPRINT_NOT_FOUND, which
// routes the device onto the temporary-CUID path it should have been on.
inline amdcuid_status_t validate_fingerprint(uint64_t fingerprint) {
  return (fingerprint == 0) ? AMDCUID_STATUS_HW_FINGERPRINT_NOT_FOUND : AMDCUID_STATUS_SUCCESS;
}

std::string read_sysfs_file(const std::string& path);
std::string readlink_bdf(const std::string& device_path);
std::string bdf_to_device_path(const std::string& bdf, amdcuid_device_type_t device_type);
std::string real_dev_path_from_fd(int fd);
std::string get_real_path(const std::string& path);
amdcuid_status_t generate_derived_cuid(const amdcuid_primary_id* primary_id,
                                       amdcuid_derived_id* derived_id, cuid_hmac* hmac);
amdcuid_status_t generate_primary_cuid(uint64_t serial_number, uint16_t unit_id,
                                       uint8_t revision_id, uint16_t device_id, uint16_t vendor_id,
                                       uint8_t device_type, amdcuid_primary_id* primary_id,
                                       bool temp = false);
void remove_UUIDv8_bits(amdcuid_id_t* id, uint8_t out_raw_bits[16]);
void add_UUIDv8_bits(const uint8_t raw_bits[16], amdcuid_id_t* id);
std::string get_cuid_as_string(const amdcuid_id_t* id);
amdcuid_status_t uuid_string_to_uint8(const std::string& uuid_str, uint8_t* uuid);
std::string device_type_to_string(amdcuid_device_type_t type);

bool is_valid_bdf(const std::string& bdf);
amdcuid_status_t make_fallback_fingerprint(const std::string& id, uint64_t& fingerprint);

// GPU VF (SR-IOV Virtual Function) utilities
int extract_render_minor(const std::string& path);
uint16_t get_gpu_vf_id(const std::string& device_path);

inline const std::string& cuid_file() {
  static const std::string path = "/tmp/cuid";
  return path;
}
inline const std::string& priv_cuid_file() {
  static const std::string path = "/tmp/priv_cuid";
  return path;
}
}  // namespace CuidUtilities

#endif
