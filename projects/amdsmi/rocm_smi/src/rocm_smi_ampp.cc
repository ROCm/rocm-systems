// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// AMPP (amdsmi power profile): reads/writes the driver's app_modes/ sysfs
// tree at /sys/class/drm/<card>/device/app_modes/. All profile names, field
// names, field counts, and units are dynamically enumerated at runtime --
// nothing about the recipe table is hardcoded here, since the driver does
// not guarantee a fixed field set, profile count, or naming across SoC
// generations. See include/amd_smi/amdsmi.h (tagAMPP) for the public API
// contract this file implements (through rsmi_dev_ampp_* below).

#include "rocm_smi/rocm_smi_ampp.h"

#include <fcntl.h>
#include <unistd.h>

#include <algorithm>
#include <cctype>
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

#include "rocm_smi/rocm_smi.h"
#include "rocm_smi/rocm_smi_common.h"  // Should go before rocm_smi.h
#include "rocm_smi/rocm_smi_device.h"
#include "rocm_smi/rocm_smi_exception.h"
#include "rocm_smi/rocm_smi_logger.h"
#include "rocm_smi/rocm_smi_main.h"
#include "rocm_smi/rocm_smi_utils.h"

using amd::smi::getRSMIStatusString;

#define TRY try {
#define CATCH                           \
  }                                     \
  catch (...) {                         \
    return amd::smi::handleException(); \
  }

namespace {

namespace fs = std::filesystem;

std::string ltrim(const std::string& s) {
  size_t start = s.find_first_not_of(" \t\r\n");
  return (start == std::string::npos) ? std::string() : s.substr(start);
}

std::string rtrim(const std::string& s) {
  size_t end = s.find_last_not_of(" \t\r\n");
  return (end == std::string::npos) ? std::string() : s.substr(0, end + 1);
}

std::string trim(const std::string& s) { return rtrim(ltrim(s)); }

// Maps an errno value observed while reading/writing an app_modes/ sysfs
// file to the closest rsmi_status_t. ENOENT means the file/dir does not
// exist -- for AMPP that is the "unconfigured slot" or "unpublished
// profile" case, not a hard I/O error, so callers decide INVALID_ARGS vs
// NO_DATA based on context; this helper covers the remaining, unambiguous
// cases.
rsmi_status_t convert_ampp_errno(int errno_val) {
  switch (errno_val) {
    case EINVAL:
      return RSMI_STATUS_INVALID_ARGS;
    case EPERM:
    case EACCES:
      return RSMI_STATUS_PERMISSION;
    case ENOTSUP:
      return RSMI_STATUS_NOT_SUPPORTED;
    default:
      return RSMI_STATUS_FILE_ERROR;
  }
}

bool app_modes_supported(const std::string& root) {
  std::error_code ec;
  return fs::exists(root, ec) && !ec;
}

// Reads a single line from a sysfs file. ENOENT is preserved so callers can
// distinguish an optional file from a permission or I/O failure.
rsmi_status_t read_sysfs_line(const std::string& path, std::string* out) {
  int fd = open(path.c_str(), O_RDONLY | O_CLOEXEC);
  if (fd < 0) return errno == ENOENT ? RSMI_STATUS_NO_DATA : convert_ampp_errno(errno);

  char buffer[4096];
  ssize_t bytes_read;
  do {
    bytes_read = read(fd, buffer, sizeof(buffer) - 1);
  } while (bytes_read < 0 && errno == EINTR);
  int read_errno = errno;
  int close_result = close(fd);
  if (bytes_read < 0) return convert_ampp_errno(read_errno);
  if (close_result != 0) return convert_ampp_errno(errno);
  buffer[bytes_read] = '\0';
  *out = std::string(buffer);
  size_t newline = out->find('\n');
  if (newline != std::string::npos) out->resize(newline);
  return RSMI_STATUS_SUCCESS;
}

rsmi_status_t write_sysfs_value(const std::string& path, const std::string& value) {
  int fd = open(path.c_str(), O_WRONLY | O_CLOEXEC);
  if (fd < 0) return convert_ampp_errno(errno);

  size_t written = 0;
  while (written < value.size()) {
    ssize_t result = write(fd, value.data() + written, value.size() - written);
    while (result < 0 && errno == EINTR) {
      result = write(fd, value.data() + written, value.size() - written);
    }
    if (result < 0) {
      int write_errno = errno;
      close(fd);
      return write_errno == EINVAL ? RSMI_STATUS_INVALID_ARGS : convert_ampp_errno(write_errno);
    }
    if (result == 0) {
      close(fd);
      return RSMI_STATUS_FILE_ERROR;
    }
    written += static_cast<size_t>(result);
  }

  if (close(fd) != 0) return convert_ampp_errno(errno);
  return RSMI_STATUS_SUCCESS;
}

// Reads app_modes/profile_abi (e.g. "1.0\n") verbatim, trimmed of the
// trailing newline. This is the SMI-visible ABI version string; the
// driver's internal recipe-blob version (currently v3) is not surfaced
// here and has no SMI-visible impact. Returns false if profile_abi does
// not exist, leaving *version untouched.
rsmi_status_t read_profile_abi(const std::string& root, std::string* version) {
  std::string line;
  rsmi_status_t status = read_sysfs_line(root + "profile_abi", &line);
  if (status != RSMI_STATUS_SUCCESS) {
    return status;
  }
  *version = trim(line);
  return RSMI_STATUS_SUCCESS;
}

// Parses "<value> <unit...>" sysfs field content generically: first
// whitespace-delimited token is the numeric value, the remainder (verbatim,
// trimmed) is an opaque unit string. Never enumerates known unit types.
bool parse_field_content(const std::string& content, int64_t* value, std::string* unit) {
  std::istringstream iss(content);
  std::string first_token;
  if (!(iss >> first_token)) {
    return false;
  }
  char* end = nullptr;
  errno = 0;
  int64_t parsed_value = std::strtoll(first_token.c_str(), &end, 10);
  if (errno == ERANGE || end == first_token.c_str() || *end != '\0') {
    return false;
  }
  *value = parsed_value;
  std::string rest;
  std::getline(iss, rest);
  *unit = trim(rest);
  return true;
}

// Reads and parses config/writable_slot_mask into the set of writable
// custom slot indices. The pure parsing logic (no filesystem I/O) lives in
// amd::smi::parse_ampp_writable_slot_mask() (rocm_smi_ampp.h) so it can
// be unit-tested directly with malformed/adversarial input, without
// needing sysfs, root, or a mock-sysfs overlay.
rsmi_status_t read_writable_slots(const std::string& root, std::vector<uint32_t>* out_slots) {
  std::string line;
  rsmi_status_t status = read_sysfs_line(root + "config/writable_slot_mask", &line);
  if (status == RSMI_STATUS_NO_DATA) {
    // No config/writable_slot_mask file at all -- nothing is writable, not
    // a parse error.
    out_slots->clear();
    return RSMI_STATUS_SUCCESS;
  }
  if (status != RSMI_STATUS_SUCCESS) return status;
  return amd::smi::parse_ampp_writable_slot_mask(trim(line), out_slots);
}

bool is_writable_slot(const std::vector<uint32_t>& writable_slots, uint32_t index) {
  for (uint32_t s : writable_slots) {
    if (s == index) return true;
  }
  return false;
}

// A profile_N directory is "configured" if it contains at least one
// regular field file. Empty custom slots (before first configure)
// enumerate to zero entries.
bool directory_has_entries(const std::string& dir_path, bool* has_entries) {
  *has_entries = false;
  std::error_code ec;
  if (!fs::exists(dir_path, ec) || !fs::is_directory(dir_path, ec)) {
    return !ec;
  }
  for (const auto& entry : fs::directory_iterator(dir_path, ec)) {
    if (ec) return false;
    std::error_code entry_ec;
    if (entry.is_regular_file(entry_ec)) {
      *has_entries = true;
      break;
    }
    if (entry_ec) return false;
  }
  return !ec;
}

// Resolves a profile_name (e.g. "profile_2") to its numeric slot index.
// Returns false if the name doesn't parse as "profile_<N>".
bool parse_profile_index(const std::string& profile_name, uint32_t* index) {
  static const std::string kPrefix = "profile_";
  if (profile_name.rfind(kPrefix, 0) != 0) {
    return false;
  }
  std::string suffix = profile_name.substr(kPrefix.size());
  if (suffix.empty()) {
    return false;
  }
  for (char c : suffix) {
    if (!std::isdigit(static_cast<unsigned char>(c))) return false;
  }
  char* end = nullptr;
  errno = 0;
  unsigned long parsed_index = std::strtoul(suffix.c_str(), &end, 10);
  if (errno == ERANGE || end == suffix.c_str() || *end != '\0' ||
      parsed_index > std::numeric_limits<uint32_t>::max()) {
    return false;
  }
  *index = static_cast<uint32_t>(parsed_index);
  return true;
}

bool parse_active_profile(const std::string& content, uint32_t* index) {
  char* end = nullptr;
  errno = 0;
  unsigned long parsed_index = std::strtoul(content.c_str(), &end, 10);
  if (errno == ERANGE || end == content.c_str() || *end != '\0' ||
      parsed_index > std::numeric_limits<uint32_t>::max()) {
    return false;
  }
  *index = static_cast<uint32_t>(parsed_index);
  return true;
}

// Enumerates the currently published profile_N directories under
// app_modes/. Discovered by iterating the directory rather than assuming
// any fixed slot count/range -- the driver does not guarantee how many
// profile_N slots exist across SoC generations. Sorted by numeric slot
// index (not directory-listing order) so callers get stable, predictable
// results.
struct AmppProfileInternal {
  uint32_t index;
  std::string name;
  bool configured;
};

std::vector<AmppProfileInternal> enumerate_profiles(const std::string& root,
                                                    rsmi_status_t* status) {
  std::vector<AmppProfileInternal> profiles;
  *status = RSMI_STATUS_SUCCESS;
  std::error_code ec;
  if (!fs::exists(root, ec) || !fs::is_directory(root, ec)) {
    if (ec) *status = convert_ampp_errno(ec.value());
    return profiles;
  }
  for (const auto& entry : fs::directory_iterator(root, ec)) {
    if (ec) {
      *status = convert_ampp_errno(ec.value());
      return {};
    }
    if (!entry.is_directory()) {
      continue;
    }
    std::string name = entry.path().filename().string();
    uint32_t idx = 0;
    if (!parse_profile_index(name, &idx)) {
      continue;
    }
    bool configured = false;
    if (!directory_has_entries(entry.path().string(), &configured)) {
      *status = RSMI_STATUS_FILE_ERROR;
      return {};
    }
    profiles.push_back({idx, name, configured});
  }
  if (ec) {
    *status = convert_ampp_errno(ec.value());
    return {};
  }
  std::sort(
      profiles.begin(), profiles.end(),
      [](const AmppProfileInternal& a, const AmppProfileInternal& b) { return a.index < b.index; });
  return profiles;
}

// Enumerates the field files that currently exist directly under
// app_modes/<profile_name>/. Field names are opaque strings -- whatever
// the driver happens to expose is what gets returned.
std::vector<std::string> enumerate_field_names(const std::string& profile_dir,
                                               rsmi_status_t* status) {
  std::vector<std::string> names;
  *status = RSMI_STATUS_SUCCESS;
  std::error_code ec;
  if (!fs::exists(profile_dir, ec) || !fs::is_directory(profile_dir, ec)) {
    if (ec) *status = convert_ampp_errno(ec.value());
    return names;
  }
  for (const auto& entry : fs::directory_iterator(profile_dir, ec)) {
    if (ec) {
      *status = convert_ampp_errno(ec.value());
      return {};
    }
    if (entry.is_regular_file()) {
      names.push_back(entry.path().filename().string());
    }
  }
  if (ec) {
    *status = convert_ampp_errno(ec.value());
    return {};
  }
  return names;
}

}  // namespace

namespace amd::smi {

rsmi_status_t parse_ampp_writable_slot_mask(const std::string& trimmed_line,
                                            std::vector<uint32_t>* out_slots) {
  out_slots->clear();
  if (trimmed_line.empty()) {
    return RSMI_STATUS_SUCCESS;
  }
  // Require an explicit "0x"/"0X" prefix rather than letting strtoul accept
  // bare hex digits -- matches the driver's published format exactly.
  if (trimmed_line.size() < 3 || trimmed_line[0] != '0' ||
      (trimmed_line[1] != 'x' && trimmed_line[1] != 'X')) {
    return RSMI_STATUS_UNEXPECTED_DATA;
  }

  errno = 0;
  char* end_ptr = nullptr;
  uint64_t mask = std::strtoull(trimmed_line.c_str(), &end_ptr, 16);
  if (end_ptr != trimmed_line.c_str() + trimmed_line.size() || errno == ERANGE) {
    return RSMI_STATUS_UNEXPECTED_DATA;
  }

  // No slot-count ceiling: the number of published slots is a runtime
  // property of the driver, so every set bit in the mask is honored.
  for (uint32_t bit = 0; bit < 64; ++bit) {
    if (mask & (UINT64_C(1) << bit)) {
      out_slots->push_back(bit);
    }
  }
  return RSMI_STATUS_SUCCESS;
}

}  // namespace amd::smi

rsmi_status_t rsmi_dev_ampp_profiles_get(uint32_t dv_ind, char version[RSMI_AMPP_MAX_STRING_LENGTH],
                                         rsmi_ampp_profile_t* profiles, uint32_t* num_profiles) {
  TRY std::ostringstream ss;
  ss << __PRETTY_FUNCTION__ << "| ======= start =======, dv_ind=" << dv_ind;
  LOG_TRACE(ss);

  if (num_profiles == nullptr) {
    return RSMI_STATUS_INVALID_ARGS;
  }

  CHK_SUPPORT_NAME_ONLY(num_profiles)

  DEVICE_MUTEX

  std::string root = dev->get_ampp_root_path();
  if (!app_modes_supported(root)) {
    return RSMI_STATUS_NOT_SUPPORTED;
  }

  if (version != nullptr) {
    std::string abi_version_str;
    version[0] = '\0';
    rsmi_status_t abi_status = read_profile_abi(root, &abi_version_str);
    if (abi_status != RSMI_STATUS_SUCCESS && abi_status != RSMI_STATUS_NO_DATA) {
      return abi_status;
    }
    if (abi_status == RSMI_STATUS_SUCCESS) {
      snprintf(version, RSMI_AMPP_MAX_STRING_LENGTH, "%s", abi_version_str.c_str());
    }
  }

  uint32_t active_index = 0;
  std::string active_line;
  rsmi_status_t active_status = read_sysfs_line(root + "active_profile", &active_line);
  if (active_status != RSMI_STATUS_SUCCESS) return active_status;
  if (!parse_active_profile(active_line, &active_index)) {
    return RSMI_STATUS_UNEXPECTED_DATA;
  }

  std::vector<uint32_t> writable_slots;
  rsmi_status_t writable_slots_ret = read_writable_slots(root, &writable_slots);
  if (writable_slots_ret != RSMI_STATUS_SUCCESS) {
    return writable_slots_ret;
  }
  rsmi_status_t enumerate_status = RSMI_STATUS_SUCCESS;
  std::vector<AmppProfileInternal> found = enumerate_profiles(root, &enumerate_status);
  if (enumerate_status != RSMI_STATUS_SUCCESS) return enumerate_status;

  const uint32_t required = static_cast<uint32_t>(found.size());

  if (profiles == nullptr) {
    *num_profiles = required;
    return RSMI_STATUS_SUCCESS;
  }

  const uint32_t capacity = *num_profiles;
  *num_profiles = required;
  if (capacity < required) {
    return RSMI_STATUS_OUT_OF_RESOURCES;
  }

  for (uint32_t i = 0; i < required; ++i) {
    rsmi_ampp_profile_t& p = profiles[i];
    memset(&p, 0, sizeof(p));
    snprintf(p.name, sizeof(p.name), "%s", found[i].name.c_str());
    p.index = found[i].index;
    p.is_active = (found[i].index == active_index);
    p.is_writable = is_writable_slot(writable_slots, found[i].index);
    p.is_configured = found[i].configured;
  }

  return RSMI_STATUS_SUCCESS;
  CATCH
}

rsmi_status_t rsmi_dev_ampp_fields_get(uint32_t dv_ind, const char* profile_name,
                                       uint32_t* num_fields, rsmi_ampp_field_t* fields) {
  TRY std::ostringstream ss;
  ss << __PRETTY_FUNCTION__ << "| ======= start =======, dv_ind=" << dv_ind;
  LOG_TRACE(ss);

  if (profile_name == nullptr || num_fields == nullptr) {
    return RSMI_STATUS_INVALID_ARGS;
  }

  CHK_SUPPORT_NAME_ONLY(num_fields)

  DEVICE_MUTEX

  std::string root = dev->get_ampp_root_path();
  if (!app_modes_supported(root)) {
    return RSMI_STATUS_NOT_SUPPORTED;
  }

  // Reject anything that doesn't parse as "profile_<N>" before doing any
  // path construction / filesystem lookups -- otherwise a caller-supplied
  // profile_name like "../../etc" would be concatenated onto root and
  // probed on disk, and control directories (e.g. "config", "limits")
  // would be treated as pseudo-profiles.
  uint32_t index = 0;
  if (!parse_profile_index(profile_name, &index)) {
    return RSMI_STATUS_INVALID_ARGS;
  }

  std::string profile_dir = root + profile_name;
  std::error_code ec;
  if (!fs::exists(profile_dir, ec) || !fs::is_directory(profile_dir, ec)) {
    if (ec) return convert_ampp_errno(ec.value());
    return RSMI_STATUS_INVALID_ARGS;
  }

  rsmi_status_t enumerate_status = RSMI_STATUS_SUCCESS;
  std::vector<std::string> field_names = enumerate_field_names(profile_dir, &enumerate_status);
  if (enumerate_status != RSMI_STATUS_SUCCESS) return enumerate_status;

  if (field_names.empty()) {
    // Distinguish "writable but unconfigured" (NO_DATA) from a profile
    // that is simply not writable and yet somehow has no fields (treat
    // as INVALID_ARGS, since a driver-authored / PMFW-default profile is
    // expected to always have its full field set).
    std::vector<uint32_t> writable_slots;
    rsmi_status_t writable_slots_ret = read_writable_slots(root, &writable_slots);
    if (writable_slots_ret != RSMI_STATUS_SUCCESS) {
      return writable_slots_ret;
    }
    if (is_writable_slot(writable_slots, index)) {
      *num_fields = 0;
      return RSMI_STATUS_NO_DATA;
    }
    return RSMI_STATUS_INVALID_ARGS;
  }

  const uint32_t required = static_cast<uint32_t>(field_names.size());

  if (fields == nullptr) {
    *num_fields = required;
    return RSMI_STATUS_SUCCESS;
  }

  const uint32_t capacity = *num_fields;
  *num_fields = required;
  if (capacity < required) {
    return RSMI_STATUS_OUT_OF_RESOURCES;
  }

  for (uint32_t i = 0; i < required; ++i) {
    rsmi_ampp_field_t& f = fields[i];
    memset(&f, 0, sizeof(f));
    snprintf(f.name, sizeof(f.name), "%s", field_names[i].c_str());

    std::string content;
    rsmi_status_t field_status = read_sysfs_line(profile_dir + "/" + field_names[i], &content);
    if (field_status != RSMI_STATUS_SUCCESS) return field_status;
    std::string unit;
    if (!parse_field_content(content, &f.value, &unit)) {
      return RSMI_STATUS_UNEXPECTED_DATA;
    }
    snprintf(f.unit, sizeof(f.unit), "%s", unit.c_str());

    std::string min_line, max_line;
    rsmi_status_t min_status = read_sysfs_line(root + "limits/min/" + field_names[i], &min_line);
    rsmi_status_t max_status = read_sysfs_line(root + "limits/max/" + field_names[i], &max_line);
    if (min_status != RSMI_STATUS_SUCCESS && min_status != RSMI_STATUS_NO_DATA) {
      return min_status;
    }
    if (max_status != RSMI_STATUS_SUCCESS && max_status != RSMI_STATUS_NO_DATA) {
      return max_status;
    }
    bool has_min = min_status == RSMI_STATUS_SUCCESS;
    bool has_max = max_status == RSMI_STATUS_SUCCESS;
    if (has_min) {
      std::string unused_unit;
      if (!parse_field_content(min_line, &f.limit_min, &unused_unit)) {
        return RSMI_STATUS_UNEXPECTED_DATA;
      }
    }
    if (has_max) {
      std::string unused_unit;
      if (!parse_field_content(max_line, &f.limit_max, &unused_unit)) {
        return RSMI_STATUS_UNEXPECTED_DATA;
      }
    }
    f.has_limits = has_min && has_max;
  }

  return RSMI_STATUS_SUCCESS;
  CATCH
}

namespace {

// Shared prefix of both rsmi_dev_ampp_profile_activate/_configure: validates
// profile_name, resolves the app_modes root, and confirms the slot exists on
// disk. Returns the parsed slot index via *out_index; any non-success return
// means the caller should propagate the status without touching sysfs.
rsmi_status_t resolve_ampp_profile_dir(const std::shared_ptr<amd::smi::Device>& dev,
                                       const char* profile_name, std::string* out_root,
                                       uint32_t* out_index) {
  if (profile_name == nullptr) {
    return RSMI_STATUS_INVALID_ARGS;
  }

  *out_root = dev->get_ampp_root_path();
  if (!app_modes_supported(*out_root)) {
    return RSMI_STATUS_NOT_SUPPORTED;
  }

  // Reject anything that doesn't parse as "profile_<N>" before doing any
  // path construction / filesystem lookups -- otherwise a caller-supplied
  // profile_name like "../../etc" would be concatenated onto root and
  // probed on disk, and control directories (e.g. "config", "limits")
  // would be treated as pseudo-profiles.
  if (!parse_profile_index(profile_name, out_index)) {
    return RSMI_STATUS_INVALID_ARGS;
  }

  std::string profile_dir = *out_root + profile_name;
  std::error_code ec;
  if (!fs::exists(profile_dir, ec) || !fs::is_directory(profile_dir, ec)) {
    if (ec) return convert_ampp_errno(ec.value());
    return RSMI_STATUS_INVALID_ARGS;
  }

  return RSMI_STATUS_SUCCESS;
}

}  // namespace

rsmi_status_t rsmi_dev_ampp_profile_activate(uint32_t dv_ind, const char* profile_name) {
  TRY std::ostringstream ss;
  ss << __PRETTY_FUNCTION__ << "| ======= start =======, dv_ind=" << dv_ind;
  LOG_TRACE(ss);

  if (profile_name == nullptr) {
    return RSMI_STATUS_INVALID_ARGS;
  }

  CHK_SUPPORT_NAME_ONLY(profile_name)

  DEVICE_MUTEX

  std::string root;
  uint32_t index = 0;
  rsmi_status_t resolve_ret = resolve_ampp_profile_dir(dev, profile_name, &root, &index);
  if (resolve_ret != RSMI_STATUS_SUCCESS) {
    return resolve_ret;
  }

  return write_sysfs_value(root + "active_profile", std::to_string(index));
  CATCH
}

rsmi_status_t rsmi_dev_ampp_profile_configure(uint32_t dv_ind, const char* profile_name,
                                              const rsmi_ampp_field_t* fields,
                                              uint32_t num_fields) {
  TRY std::ostringstream ss;
  ss << __PRETTY_FUNCTION__ << "| ======= start =======, dv_ind=" << dv_ind;
  LOG_TRACE(ss);

  if (profile_name == nullptr) {
    return RSMI_STATUS_INVALID_ARGS;
  }

  CHK_SUPPORT_NAME_ONLY(profile_name)

  DEVICE_MUTEX

  std::string root;
  uint32_t index = 0;
  rsmi_status_t resolve_ret = resolve_ampp_profile_dir(dev, profile_name, &root, &index);
  if (resolve_ret != RSMI_STATUS_SUCCESS) {
    return resolve_ret;
  }

  // The driver's config/commit rejects with -EINVAL if config/profile was
  // never set for this attempt, or if no field was ever staged for this
  // attempt -- it does not fall back to "pull everything from the active
  // profile" the way an empty commit might suggest. Defensively require at
  // least one field before doing any sysfs I/O at all, so callers get a
  // clear RSMI_STATUS_INVALID_ARGS instead of an uninterpreted kernel
  // -EINVAL bubbling up from the commit write below. (config/profile is
  // always written by this function immediately before staging, so that
  // half of the driver's precondition is unconditionally satisfied here.)
  if (fields == nullptr || num_fields == 0) {
    return RSMI_STATUS_INVALID_ARGS;
  }

  std::vector<uint32_t> writable_slots;
  rsmi_status_t writable_slots_ret = read_writable_slots(root, &writable_slots);
  if (writable_slots_ret != RSMI_STATUS_SUCCESS) {
    return writable_slots_ret;
  }
  if (!is_writable_slot(writable_slots, index)) {
    return RSMI_STATUS_NOT_SUPPORTED;
  }

  // Validate requested field names against whatever fields the driver
  // currently recognizes for staging (config/<field> mirrors the field
  // set of a fully-configured profile of this generation).
  rsmi_status_t enumerate_status = RSMI_STATUS_SUCCESS;
  std::vector<std::string> stageable_fields =
      enumerate_field_names(root + "config", &enumerate_status);
  if (enumerate_status != RSMI_STATUS_SUCCESS) return enumerate_status;
  // config/ also contains non-field control files; filter those out.
  auto is_control_file = [](const std::string& name) {
    return name == "profile" || name == "writable_slot_mask" || name == "commit";
  };

  auto field_is_stageable = [&](const std::string& name) {
    if (is_control_file(name)) return false;
    for (const auto& f : stageable_fields) {
      if (!is_control_file(f) && f == name) return true;
    }
    return false;
  };

  for (uint32_t i = 0; i < num_fields; ++i) {
    std::string name(fields[i].name, strnlen(fields[i].name, sizeof(fields[i].name)));
    if (!field_is_stageable(name)) {
      return RSMI_STATUS_INVALID_ARGS;
    }
  }

  // Select the target slot.
  rsmi_status_t write_status = write_sysfs_value(root + "config/profile", std::to_string(index));
  if (write_status != RSMI_STATUS_SUCCESS) return write_status;

  // Stage only the fields the caller asked for -- partial staging is
  // intentional; do not force-write the full field set (the driver
  // resolves any unset fields from the active profile or this slot's
  // last-committed values).
  for (uint32_t i = 0; i < num_fields; ++i) {
    std::string field_path =
        root + "config/" +
        std::string(fields[i].name, strnlen(fields[i].name, sizeof(fields[i].name)));
    write_status = write_sysfs_value(field_path, std::to_string(fields[i].value));
    if (write_status != RSMI_STATUS_SUCCESS) return write_status;
  }

  // Commit.
  return write_sysfs_value(root + "config/commit", "1");
  CATCH
}
