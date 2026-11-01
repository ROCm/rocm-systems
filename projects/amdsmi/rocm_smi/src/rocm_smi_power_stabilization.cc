// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Power Stabilization (HAWK dummy-compute insertion plus Ramp Management):
// reads/writes the driver's power_stabilization/ sysfs tree at
// /sys/class/drm/<card>/device/power_stabilization/. See
// include/amd_smi/amdsmi.h (tagPowerStabilization) for the public API
// contract this file implements (through rsmi_dev_power_stabilization_*
// below).

#include <fcntl.h>
#include <unistd.h>

#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <sstream>
#include <string>
#include <type_traits>

#include "rocm_smi/rocm_smi.h"
#include "rocm_smi/rocm_smi_common.h"  // Should go before rocm_smi.h
#include "rocm_smi/rocm_smi_device.h"
#include "rocm_smi/rocm_smi_exception.h"
#include "rocm_smi/rocm_smi_logger.h"
#include "rocm_smi/rocm_smi_main.h"
#include "rocm_smi/rocm_smi_utils.h"

#define TRY try {
#define CATCH                           \
  }                                     \
  catch (...) {                         \
    return amd::smi::handleException(); \
  }

namespace {

namespace fs = std::filesystem;

// Local mirror of amd_smi::ScopedFD (include/amd_smi/impl/scoped_fd.h) --
// rocm_smi must not include amd_smi's internal headers.
class ScopedFD {
 public:
  ScopedFD(const std::string& path, int flags) : fd_(open(path.c_str(), flags)), path_(path) {
    if (fd_ < 0) {
      // Save errno before LOG_ERROR (mutex + file write) gets a chance to
      // clobber it; callers must read open_errno() instead of the global
      // errno after this constructor runs.
      saved_errno_ = errno;
      std::ostringstream ss;
      ss << __PRETTY_FUNCTION__ << " | Failed to open file: " << path_
         << " | Error: " << strerror(saved_errno_);
      LOG_ERROR(ss);
    }
  }
  ~ScopedFD() {
    if (fd_ >= 0) {
      close(fd_);
    }
  }

  ScopedFD(const ScopedFD&) = delete;
  ScopedFD& operator=(const ScopedFD&) = delete;

  ScopedFD(ScopedFD&& other) noexcept : fd_(other.fd_), path_(std::move(other.path_)) {
    other.fd_ = -1;
  }
  ScopedFD& operator=(ScopedFD&& other) noexcept {
    if (this != &other) {
      if (fd_ >= 0) close(fd_);
      fd_ = other.fd_;
      path_ = std::move(other.path_);
      saved_errno_ = other.saved_errno_;
      other.fd_ = -1;
    }
    return *this;
  }

  int get() const { return fd_; }
  bool valid() const { return fd_ >= 0; }
  operator int() const { return fd_; }
  // errno captured at construction time if open() failed; see the
  // constructor's comment for why this must be used instead of the global
  // errno after construction returns.
  int open_errno() const { return saved_errno_; }

 private:
  int fd_;
  std::string path_;
  int saved_errno_ = 0;
};

// Maps a Power Stabilization param id to its sysfs filename, relative to
// the device's power_stabilization/ directory.
const char* power_stabilization_param_relpath(rsmi_power_stabilization_param_id_t param_id) {
  switch (param_id) {
    case RSMI_POWER_STABILIZATION_PARAM_ENABLE:
      return "enable";
    case RSMI_POWER_STABILIZATION_PARAM_POWER_FLOOR:
      return "power_floor";
    case RSMI_POWER_STABILIZATION_PARAM_DELAY:
      return "delay";
    case RSMI_POWER_STABILIZATION_PARAM_INSERTION_PRIORITY:
      return "insertion_priority";
    case RSMI_POWER_STABILIZATION_PARAM_RAMP_ENABLE:
      return "ramp/enable";
    case RSMI_POWER_STABILIZATION_PARAM_RAMP_MAX_RAMP_UP_RATE:
      return "ramp/max_ramp_up_rate";
    case RSMI_POWER_STABILIZATION_PARAM_RAMP_MAX_RAMP_DOWN_RATE:
      return "ramp/max_ramp_down_rate";
    case RSMI_POWER_STABILIZATION_PARAM_RAMP_MIN_TIME_AT_POWER_FLOOR:
      return "ramp/min_time_at_power_floor";
    case RSMI_POWER_STABILIZATION_PARAM_RAMP_STOP_DELAY:
      return "ramp/stop_delay";
  }
  return nullptr;
}

// Shared by both the read (ifstream open failure) and write (open()/write()
// failure) paths, so a permission-restricted or driver-rejected node is
// classified identically regardless of which helper hit it. The read path
// and write()'s post-write failure read the global errno immediately, with
// nothing in between to clobber it. ScopedFD's open() failure is the
// exception: its constructor's own LOG_ERROR call can clobber errno before
// the caller gets to read it, so that path must use ScopedFD::open_errno()
// (saved at construction time) instead.
rsmi_status_t classify_power_stabilization_errno(int err) {
  if (err == EACCES || err == EPERM) {
    return RSMI_STATUS_PERMISSION;
  }
  if (err == ENOENT) {
    return RSMI_STATUS_NOT_SUPPORTED;
  }
  if (err == EINVAL) {
    return RSMI_STATUS_INVALID_ARGS;
  }
  if (err == ENOTSUP || err == EOPNOTSUPP || err == EBUSY || err == ENODEV) {
    return RSMI_STATUS_NOT_SUPPORTED;
  }
  return RSMI_STATUS_FILE_ERROR;
}

// int64_t and double sysfs reads differ only in the type `f >> *out`
// extracts into, so both share this template. After a successful
// extraction, trailing non-whitespace content (e.g. "0x10", which `>>`
// would otherwise silently parse as 0) is rejected rather than ignored.
template <typename T>
rsmi_status_t read_sysfs_value(const std::string& path, T* out) {
  std::ifstream f(path);
  if (!f.good()) {
    return classify_power_stabilization_errno(errno);
  }
  f >> *out;
  if (f.fail()) {
    return RSMI_STATUS_FILE_ERROR;
  }
  f >> std::ws;
  if (!f.eof()) {
    return RSMI_STATUS_UNEXPECTED_DATA;
  }
  return RSMI_STATUS_SUCCESS;
}

// Directory presence is the sole support signal: absent both when the
// platform doesn't implement Power Stabilization and for multi-VF devices
// (the driver never creates the node in either case).
bool power_stabilization_supported(const std::string& root) {
  std::error_code ec;
  return fs::is_directory(root, ec);
}

rsmi_status_t get_power_stabilization_info_internal(const std::string& base,
                                                    rsmi_power_stabilization_info_t* info) {
  std::memset(info, 0, sizeof(*info));

  static const struct {
    const char* relpath;
    bool rsmi_power_stabilization_info_t::* bool_field;
    uint32_t rsmi_power_stabilization_info_t::* uint_field;
  } kIntFields[] = {
      {"host_control", &rsmi_power_stabilization_info_t::host_control_enabled, nullptr},
      {"enable", &rsmi_power_stabilization_info_t::enabled, nullptr},
      {"power_floor", nullptr, &rsmi_power_stabilization_info_t::power_floor},
      {"delay", nullptr, &rsmi_power_stabilization_info_t::delay},
      {"ramp/enable", &rsmi_power_stabilization_info_t::ramp_enabled, nullptr},
      {"ramp/min_time_at_power_floor", nullptr,
       &rsmi_power_stabilization_info_t::ramp_min_time_at_power_floor},
      {"ramp/stop_delay", nullptr, &rsmi_power_stabilization_info_t::ramp_stop_delay},
  };

  for (const auto& field : kIntFields) {
    int64_t v = 0;
    rsmi_status_t ret = read_sysfs_value(base + field.relpath, &v);
    if (ret != RSMI_STATUS_SUCCESS) {
      return ret;
    }
    if (field.bool_field != nullptr) {
      info->*field.bool_field = (v != 0);
    } else {
      // Reject rather than silently truncate a negative/oversized sysfs
      // value into the display field (e.g. -1 would otherwise wrap to
      // 4294967295).
      if (v < 0 || v > static_cast<int64_t>(std::numeric_limits<uint32_t>::max())) {
        return RSMI_STATUS_UNEXPECTED_DATA;
      }
      info->*field.uint_field = static_cast<uint32_t>(v);
    }
  }

  int64_t priority = 0;
  rsmi_status_t ret = read_sysfs_value(base + "insertion_priority", &priority);
  if (ret != RSMI_STATUS_SUCCESS) {
    return ret;
  }
  if (priority < RSMI_HAWK_INSERTION_PRIORITY_OPPORTUNISTIC ||
      priority > RSMI_HAWK_INSERTION_PRIORITY_DYNAMIC) {
    // Out-of-range sysfs content -- either driver/firmware skew or a
    // malformed node. Reject here rather than propagating an unvalidated
    // value into an enum, which would later raise an uncaught ValueError
    // in the Python binding.
    return RSMI_STATUS_UNEXPECTED_DATA;
  }
  info->insertion_priority = static_cast<rsmi_hawk_insertion_priority_t>(priority);

  ret = read_sysfs_value(base + "ramp/max_ramp_up_rate", &info->ramp_max_ramp_up_rate);
  if (ret != RSMI_STATUS_SUCCESS) {
    return ret;
  }
  ret = read_sysfs_value(base + "ramp/max_ramp_down_rate", &info->ramp_max_ramp_down_rate);
  if (ret != RSMI_STATUS_SUCCESS) {
    return ret;
  }

  return RSMI_STATUS_SUCCESS;
}

// int64_t is formatted exactly via std::to_string; double uses enough
// precision to round-trip exactly (std::to_string(double) truncates to 6
// decimal places, e.g. writing 5e-07 as "0.000000", which would defeat the
// driver's range validation on the truncated value instead of the one the
// caller asked for).
template <typename T>
std::string format_power_stabilization_value(T value) {
  if constexpr (std::is_floating_point_v<T>) {
    char buf[32];
    snprintf(buf, sizeof(buf), "%.17g", value);
    return std::string(buf);
  } else {
    return std::to_string(value);
  }
}

// int64_t and double sysfs writes differ only in value formatting, so both
// share this template. Writes as decimal ASCII via a truncating open()+
// write() (rather than ofstream) so a failed write's errno can be inspected
// directly, distinguishing permission failures from driver-side
// value/dependency rejection. O_NOFOLLOW refuses to write through a symlink
// planted at the sysfs leaf path. write()'s short-write check is intentional
// (treated as an error below) -- only the retry here is for EINTR.
template <typename T>
rsmi_status_t write_sysfs_value(const std::string& path, T value) {
  ScopedFD fd(path, O_WRONLY | O_TRUNC | O_CLOEXEC | O_NOFOLLOW);
  if (!fd.valid()) {
    return classify_power_stabilization_errno(fd.open_errno());
  }

  std::string text = format_power_stabilization_value(value);
  ssize_t written;
  do {
    written = write(fd.get(), text.c_str(), text.size());
  } while (written < 0 && errno == EINTR);
  if (written < 0) {
    return classify_power_stabilization_errno(errno);
  }
  if (static_cast<size_t>(written) != text.size()) {
    return RSMI_STATUS_FILE_ERROR;
  }

  return RSMI_STATUS_SUCCESS;
}

}  // namespace

rsmi_status_t rsmi_dev_power_stabilization_info_get(uint32_t dv_ind,
                                                    rsmi_power_stabilization_info_t* info) {
  TRY std::ostringstream ss;
  ss << __PRETTY_FUNCTION__ << "| ======= start =======, dv_ind=" << dv_ind;
  LOG_TRACE(ss);

  if (info == nullptr) {
    return RSMI_STATUS_INVALID_ARGS;
  }

  CHK_SUPPORT_NAME_ONLY(info)

  DEVICE_MUTEX

  std::string base = dev->get_power_stabilization_root_path();
  if (!power_stabilization_supported(base)) {
    return RSMI_STATUS_NOT_SUPPORTED;
  }

  return get_power_stabilization_info_internal(base, info);
  CATCH
}

rsmi_status_t rsmi_dev_power_stabilization_param_set(uint32_t dv_ind,
                                                     rsmi_power_stabilization_param_id_t param_id,
                                                     const rsmi_value_t* value) {
  TRY std::ostringstream ss;
  ss << __PRETTY_FUNCTION__ << "| ======= start =======, dv_ind=" << dv_ind;
  LOG_TRACE(ss);

  if (value == nullptr) {
    return RSMI_STATUS_INVALID_ARGS;
  }

  const char* relpath = power_stabilization_param_relpath(param_id);
  if (relpath == nullptr) {
    return RSMI_STATUS_INVALID_ARGS;
  }

  CHK_SUPPORT_NAME_ONLY(value)

  DEVICE_MUTEX

  std::string base = dev->get_power_stabilization_root_path();
  if (!power_stabilization_supported(base)) {
    return RSMI_STATUS_NOT_SUPPORTED;
  }

  // All params apply directly to dv_ind's own sysfs node. No NPM-aware
  // routing here: when NPM coordination makes a param node-wide, selecting
  // the node's primary GPU handle (OAM ID 0) is the caller's responsibility
  // (enforced by the CLI), not this function's.
  bool is_ramp_rate = param_id == RSMI_POWER_STABILIZATION_PARAM_RAMP_MAX_RAMP_UP_RATE ||
                      param_id == RSMI_POWER_STABILIZATION_PARAM_RAMP_MAX_RAMP_DOWN_RATE;

  if (is_ramp_rate) {
    return write_sysfs_value(base + relpath, value->data.double_value);
  }
  return write_sysfs_value(base + relpath, value->data.int64_value);
  CATCH
}
