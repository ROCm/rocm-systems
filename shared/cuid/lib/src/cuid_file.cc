// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "src/cuid_file.h"

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <cctype>
#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>

#include "smbios_util.h"
#include "src/cuid_cpu.h"
#include "src/cuid_device.h"
#include "src/cuid_gpu.h"
#include "src/cuid_internal.h"
#include "src/cuid_nic.h"
#include "src/cuid_npu.h"
#include "src/cuid_platform.h"
#include "src/cuid_util.h"
#include "src/hmac.h"

namespace {

// The lock file is 0666, so any local process can hold it indefinitely; no
// acquisition here may block on that without a bound.
constexpr int kLockTimeoutSeconds = 5;

// Create the record directory 0755 when missing, as the packaging script would.
// False for a non-root caller, which becomes PERMISSION_DENIED rather than a
// fallback to anywhere more writable.
bool ensure_parent_dir(const std::string& path) {
  const size_t slash = path.find_last_of('/');
  if (slash == std::string::npos || slash == 0) return true;
  const std::string dir = path.substr(0, slash);

  struct stat st;
  if (stat(dir.c_str(), &st) == 0) return S_ISDIR(st.st_mode);
  if (mkdir(dir.c_str(), S_IRWXU | S_IRGRP | S_IXGRP | S_IROTH | S_IXOTH) == 0) return true;
  return errno == EEXIST;
}

// Write `content` to `path` atomically, with `mode`. Same shape as
// cuid_hmac::store_key(). A predictable temp name (the old `path + ".tmp"`) in
// a directory an unprivileged user can write lets that user pre-create it as a
// symlink and have the next root-privileged refresh truncate the target, so the
// name comes from mkstemp; and every write is checked, so an ENOSPC cannot
// rename a truncated record over a complete one and return success.
amdcuid_status_t write_record_file(const std::string& path, const std::string& content,
                                   mode_t mode) {
  if (!ensure_parent_dir(path)) {
    LOG(ERROR, "CuidFile: cannot create the directory for " << path << ": "
                                                            << CuidUtilities::errno_string(errno));
    return AMDCUID_STATUS_PERMISSION_DENIED;
  }

  std::string temp_path = path + ".XXXXXX";
  const int fd = mkstemp(&temp_path[0]);
  if (fd < 0) {
    const int err = errno;
    LOG(ERROR, "CuidFile: cannot create a temporary file beside "
                   << path << ": " << CuidUtilities::errno_string(err));
    return (err == EACCES || err == EPERM || err == EROFS) ? AMDCUID_STATUS_PERMISSION_DENIED
                                                           : AMDCUID_STATUS_FILE_ERROR;
  }

  auto fail = [&](const char* what) {
    LOG(ERROR,
        "CuidFile: " << what << " for " << path << ": " << CuidUtilities::errno_string(errno));
    close(fd);
    unlink(temp_path.c_str());
    return AMDCUID_STATUS_FILE_ERROR;
  };

  // mkstemp() creates 0600; widen before there is anything in it to read.
  if (fchmod(fd, mode) != 0) return fail("fchmod failed");

  size_t written = 0;
  while (written < content.size()) {
    const ssize_t n = write(fd, content.data() + written, content.size() - written);
    if (n < 0) {
      if (errno == EINTR) continue;
      return fail("write failed");
    }
    written += static_cast<size_t>(n);
  }

  // Durable before the rename: otherwise a crash leaves the record present and
  // empty, which reads back as a node with no components.
  if (fsync(fd) != 0) return fail("fsync failed");
  if (close(fd) != 0) {
    LOG(ERROR, "CuidFile: close failed for " << path << ": " << CuidUtilities::errno_string(errno));
    unlink(temp_path.c_str());
    return AMDCUID_STATUS_FILE_ERROR;
  }

  if (rename(temp_path.c_str(), path.c_str()) != 0) {
    const int err = errno;
    LOG(ERROR, "CuidFile: rename failed for " << path << ": " << CuidUtilities::errno_string(err));
    unlink(temp_path.c_str());
    return (err == EACCES || err == EPERM) ? AMDCUID_STATUS_PERMISSION_DENIED
                                           : AMDCUID_STATUS_FILE_ERROR;
  }

  // Persist the directory entry so the rename survives power loss.
  const size_t slash = path.find_last_of('/');
  const std::string dir = (slash == std::string::npos) ? "." : path.substr(0, slash);
  const int dir_fd = open(dir.empty() ? "/" : dir.c_str(), O_RDONLY | O_DIRECTORY | O_CLOEXEC);
  if (dir_fd >= 0) {
    (void)fsync(dir_fd);
    close(dir_fd);
  }

  return AMDCUID_STATUS_SUCCESS;
}

// Open (creating if need be) the advisory lock file, for all three acquire
// paths. Do not reintroduce the umask(0) dance around the O_CREAT: umask is
// process-global, libamdcuid is linked into multithreaded hosts (libamd_smi.so
// among them), and the fchmod below sets the mode without touching global
// state. O_NOFOLLOW because the lock path is derived from the record path and
// so predictable: a symlink there is somebody redirecting us.
int open_lock_file(const std::string& path, CuidLockType lock_type) {
  const int base_flags = O_CLOEXEC | O_NOFOLLOW;
  int fd =
      open(path.c_str(), ((lock_type == CuidLockType::EXCLUSIVE) ? O_RDWR : O_RDONLY) | base_flags);
  if (fd >= 0 || errno != ENOENT) return fd;

  fd = open(path.c_str(), O_RDWR | O_CREAT | base_flags, 0666);
  if (fd >= 0) {
    // An unprivileged reader has to be able to take a shared lock on a file
    // root created, whatever the creator's umask was.
    (void)fchmod(fd, 0666);
    return fd;
  }

  // Ordinary, not an error: the store is root-owned, so where no refresh has
  // run there is no directory to create it in, and load() reads on unlocked.
  const int err = errno;
  const LogLevel level = (err == ENOENT || err == EACCES || err == EPERM) ? DEBUG : ERROR;
  LOG(level, "CuidFileLock: Failed to open lock file " << path << ": "
                                                       << CuidUtilities::errno_string(err));
  errno = err;
  return fd;
}

// Parse an unsigned integer field of a record. The record is untrusted input,
// and std::stoul/stoull/stol throw across the C entry points that reach here.
// `max` is the width of the destination field: a value that does not fit is
// malformed input, not something to truncate silently.
bool parse_uint(const std::string& text, int base, uint64_t max, uint64_t& out) {
  if (text.empty()) return false;
  // strtoull accepts leading whitespace and a sign, so "-1" would arrive as
  // ULLONG_MAX.
  if (!std::isalnum(static_cast<unsigned char>(text[0]))) return false;

  errno = 0;
  char* end = nullptr;
  const unsigned long long value = std::strtoull(text.c_str(), &end, base);
  if (errno == ERANGE || end == text.c_str() || *end != '\0') return false;
  if (value > max) return false;

  out = static_cast<uint64_t>(value);
  return true;
}

// Same, for the signed last_update timestamp.
bool parse_time(const std::string& text, time_t& out) {
  if (text.empty()) return false;

  errno = 0;
  char* end = nullptr;
  const long long value = std::strtoll(text.c_str(), &end, 10);
  if (errno == ERANGE || end == text.c_str() || *end != '\0') return false;

  out = static_cast<time_t>(value);
  return true;
}

// A record's CUID field, validated: uuid_string_to_uint8() rejects a non-hex
// digit and a wrong length.
bool parse_cuid(const std::string& text, amdcuid_id_t& id) {
  memset(id.bytes, 0, sizeof(id.bytes));
  if (CuidUtilities::uuid_string_to_uint8(text, id.bytes) != AMDCUID_STATUS_SUCCESS) {
    memset(id.bytes, 0, sizeof(id.bytes));
    return false;
  }
  return true;
}

}  // namespace

// ============================================================================
// CuidFileLock Implementation
// ============================================================================

CuidFileLock::CuidFileLock(const std::string& file_path, CuidLockType lock_type)
    : lock_file_path_(file_path + ".lock"),
      lock_type_(lock_type),
      lock_fd_(-1),
      is_locked_(false) {}

CuidFileLock::~CuidFileLock() { release(); }

bool CuidFileLock::acquire() {
  if (is_locked_) {
    return true;  // Already locked
  }

  lock_fd_ = open_lock_file(lock_file_path_, lock_type_);

  if (lock_fd_ < 0) {
    return false;  // open_lock_file() has already said why
  }

  // Set up the lock structure
  struct flock fl;
  memset(&fl, 0, sizeof(fl));
  fl.l_whence = SEEK_SET;
  fl.l_start = 0;
  fl.l_len = 0;  // Lock entire file
  fl.l_type = (lock_type_ == CuidLockType::EXCLUSIVE) ? F_WRLCK : F_RDLCK;

  // F_SETLKW: blocking call - wait until lock is available
  if (fcntl(lock_fd_, F_SETLKW, &fl) < 0) {
    LOG(ERROR, "CuidFileLock: Failed to acquire lock on " << lock_file_path_ << ": "
                                                          << CuidUtilities::errno_string(errno));
    close(lock_fd_);
    lock_fd_ = -1;
    return false;
  }

  is_locked_ = true;
  LOG(DEBUG, "CuidFileLock: Acquired "
                 << (lock_type_ == CuidLockType::EXCLUSIVE ? "exclusive" : "shared") << " lock on "
                 << lock_file_path_);
  return true;
}

bool CuidFileLock::acquire_with_timeout(int timeout_seconds) {
  // Special cases
  if (timeout_seconds == 0) {
    return try_acquire();  // Non-blocking
  }
  if (timeout_seconds < 0) {
    return acquire();  // Infinite wait
  }

  if (is_locked_) {
    return true;  // Already locked
  }

  lock_fd_ = open_lock_file(lock_file_path_, lock_type_);

  if (lock_fd_ < 0) {
    return false;  // open_lock_file() has already said why
  }

  // Set up the lock structure
  struct flock fl;
  memset(&fl, 0, sizeof(fl));
  fl.l_whence = SEEK_SET;
  fl.l_start = 0;
  fl.l_len = 0;  // Lock entire file
  fl.l_type = (lock_type_ == CuidLockType::EXCLUSIVE) ? F_WRLCK : F_RDLCK;

  // Retry loop with timeout
  const int retry_interval_ms = 50;  // 50ms between retries
  const int max_retries = (timeout_seconds * 1000) / retry_interval_ms;

  for (int retry = 0; retry <= max_retries; ++retry) {
    // Try non-blocking acquire
    if (fcntl(lock_fd_, F_SETLK, &fl) == 0) {
      is_locked_ = true;
      LOG(DEBUG, "CuidFileLock: Acquired "
                     << (lock_type_ == CuidLockType::EXCLUSIVE ? "exclusive" : "shared")
                     << " lock on " << lock_file_path_ << " after " << retry << " retries");
      return true;
    }

    // Check if it's a "lock held" error vs real error
    if (errno != EACCES && errno != EAGAIN) {
      LOG(ERROR, "CuidFileLock: Failed to acquire lock on " << lock_file_path_ << ": "
                                                            << CuidUtilities::errno_string(errno));
      close(lock_fd_);
      lock_fd_ = -1;
      return false;
    }

    // Wait before retry (unless this is the last iteration)
    if (retry < max_retries) {
      usleep(retry_interval_ms * 1000);  // Convert ms to microseconds
    }
  }

  // Timeout reached
  LOG(WARN, "CuidFileLock: Timeout after " << timeout_seconds << " seconds waiting for lock on "
                                           << lock_file_path_);
  close(lock_fd_);
  lock_fd_ = -1;
  return false;
}

bool CuidFileLock::try_acquire() {
  if (is_locked_) {
    return true;  // Already locked
  }

  lock_fd_ = open_lock_file(lock_file_path_, lock_type_);

  if (lock_fd_ < 0) {
    return false;  // open_lock_file() has already said why
  }

  // Set up the lock structure
  struct flock fl;
  memset(&fl, 0, sizeof(fl));
  fl.l_whence = SEEK_SET;
  fl.l_start = 0;
  fl.l_len = 0;  // Lock entire file
  fl.l_type = (lock_type_ == CuidLockType::EXCLUSIVE) ? F_WRLCK : F_RDLCK;

  // F_SETLK: non-blocking - return immediately if can't acquire
  if (fcntl(lock_fd_, F_SETLK, &fl) < 0) {
    if (errno == EACCES || errno == EAGAIN) {
      // Lock is held by another process
      LOG(DEBUG, "CuidFileLock: Lock on " << lock_file_path_ << " held by another process");
    } else {
      LOG(ERROR, "CuidFileLock: Failed to try_acquire lock on "
                     << lock_file_path_ << ": " << CuidUtilities::errno_string(errno));
    }
    close(lock_fd_);
    lock_fd_ = -1;
    return false;
  }

  is_locked_ = true;
  LOG(DEBUG, "CuidFileLock: Acquired "
                 << (lock_type_ == CuidLockType::EXCLUSIVE ? "exclusive" : "shared") << " lock on "
                 << lock_file_path_);
  return true;
}

void CuidFileLock::release() {
  if (!is_locked_ || lock_fd_ < 0) {
    return;
  }

  // Unlock the file
  struct flock fl;
  memset(&fl, 0, sizeof(fl));
  fl.l_whence = SEEK_SET;
  fl.l_start = 0;
  fl.l_len = 0;  // Unlock entire file
  fl.l_type = F_UNLCK;

  if (fcntl(lock_fd_, F_SETLK, &fl) < 0) {
    LOG(ERROR, "CuidFileLock: Failed to release lock on " << lock_file_path_ << ": "
                                                          << CuidUtilities::errno_string(errno));
  }

  close(lock_fd_);
  lock_fd_ = -1;
  is_locked_ = false;
  LOG(DEBUG, "CuidFileLock: Released lock on " << lock_file_path_);
}

// ============================================================================
// CuidFile Implementation
// ============================================================================

CuidFile::CuidFile(const std::string& file_path, bool is_privileged)
    : file_path_(file_path), is_privileged_(is_privileged) {}

bool CuidFile::exists() const {
  struct stat buffer;
  return (stat(file_path_.c_str(), &buffer) == 0);
}

std::string CuidFile::trim(const std::string& str) const {
  size_t first = str.find_first_not_of(" \t\r\n");
  if (first == std::string::npos) return "";
  size_t last = str.find_last_not_of(" \t\r\n");
  return str.substr(first, last - first + 1);
}

amdcuid_device_type_t CuidFile::string_to_device_type(const std::string& str) const {
  if (str == "PLATFORM") return AMDCUID_DEVICE_TYPE_PLATFORM;
  if (str == "CPU") return AMDCUID_DEVICE_TYPE_CPU;
  if (str == "GPU") return AMDCUID_DEVICE_TYPE_GPU;
  if (str == "NIC") return AMDCUID_DEVICE_TYPE_NIC;
  if (str == "NPU") return AMDCUID_DEVICE_TYPE_NPU;
  if (str == "STORAGE") return AMDCUID_DEVICE_TYPE_STORAGE;
  if (str == "MEMORY") return AMDCUID_DEVICE_TYPE_MEMORY;
  if (str == "GENPCIE") return AMDCUID_DEVICE_TYPE_GENPCIE;
  if (str == "GENC") return AMDCUID_DEVICE_TYPE_GENC;
  if (str == "RACKTRAY") return AMDCUID_DEVICE_TYPE_RACKTRAY;
  if (str == "RACK") return AMDCUID_DEVICE_TYPE_RACK;
  if (str == "OTHER") return AMDCUID_DEVICE_TYPE_OTHER;
  // Exact inverse of CuidUtilities::device_type_to_string() over every valid
  // Component Type; anything else, its "UNKNOWN" included, is not one.
  return AMDCUID_DEVICE_TYPE_NONE;
}

amdcuid_id_t CuidFile::string_to_cuid(const std::string& str) const {
  amdcuid_id_t id;
  (void)parse_cuid(str, id);  // zero-filled on parse error, as before
  return id;
}

bool CuidFile::parse_section_header(const std::string& line, amdcuid_device_type_t& type,
                                    uint32_t& index) const {
  // Parse lines like [GPU:0] or [PLATFORM]
  if (line.empty() || line[0] != '[' || line.back() != ']') {
    return false;
  }

  std::string content = line.substr(1, line.length() - 2);
  size_t colon_pos = content.find(':');

  if (colon_pos != std::string::npos) {
    // Format: TYPE:INDEX (index is in hex)
    std::string type_str = content.substr(0, colon_pos);
    std::string index_str = content.substr(colon_pos + 1);
    type = string_to_device_type(type_str);
    uint64_t parsed = 0;
    if (!parse_uint(index_str, 16, std::numeric_limits<uint32_t>::max(), parsed)) {
      return false;
    }
    index = static_cast<uint32_t>(parsed);
  } else {
    // Format: PLATFORM (no index)
    type = string_to_device_type(content);
    index = 0;
  }

  return type != AMDCUID_DEVICE_TYPE_NONE;
}

amdcuid_status_t CuidFile::load() {
  entries_.clear();

  // Bounded rather than F_SETLKW: CuidDeviceManager reaches load() holding
  // manager_mutex_ on some paths, and the 0666 lock file can be held by any
  // local process. Failing to lock is not fatal, because the writer swaps the
  // record in with rename(), so an unlocked reader still sees one whole version
  // of it.
  CuidFileLock lock(file_path_, CuidLockType::SHARED);
  if (!lock.acquire_with_timeout(kLockTimeoutSeconds)) {
    LOG(DEBUG, "CuidFile::load: reading " << file_path_ << " without the shared lock");
  }

  std::ifstream file(file_path_);
  if (!file.is_open()) {
    return AMDCUID_STATUS_FILE_NOT_FOUND;
  }

  // No exception may leave this function: it is reached from
  // amdcuid_query_device_property() and amdcuid_get_all_handles(), which are C
  // entry points. The field parsers below do not throw; this is the backstop
  // against std::bad_alloc and against a later edit.
  try {
    std::string line;
    CuidFileEntry current_entry;
    bool in_section = false;

    while (std::getline(file, line)) {
      line = trim(line);

      // Skip empty lines and comments
      if (line.empty() || line[0] == '#' || line[0] == ';') {
        continue;
      }

      // Check for section header
      if (line[0] == '[') {
        // Save previous entry if valid
        if (in_section) {
          entries_.push_back(current_entry);
        }

        // Start new section
        amdcuid_device_type_t type;
        uint32_t index;
        if (parse_section_header(line, type, index)) {
          current_entry = CuidFileEntry();
          current_entry.device_type = type;
          current_entry.device_index = index;
          in_section = true;
        } else {
          in_section = false;
        }
        continue;
      }

      // Parse key=value pairs
      if (!in_section) continue;
      const size_t eq_pos = line.find('=');
      if (eq_pos == std::string::npos) continue;

      const std::string key = trim(line.substr(0, eq_pos));
      const std::string value = trim(line.substr(eq_pos + 1));

      // A malformed record is rejected whole: defaulting the field would hand
      // the caller a CUID association assembled partly out of zeroes.
      bool ok = true;
      uint64_t number = 0;
      if (key == "primary_cuid") {
        ok = parse_cuid(value, current_entry.primary_cuid);
      } else if (key == "is_temporary") {
        current_entry.is_temporary = (value == "true");
      } else if (key == "derived_cuid") {
        ok = parse_cuid(value, current_entry.derived_cuid);
      } else if (key == "device_node") {
        current_entry.device_node = value;
      } else if (key == "bdf") {
        current_entry.bdf = value;
      } else if (key == "mac_address") {
        current_entry.mac_address = value;
      } else if (key == "hardware_fingerprint") {
        ok = parse_uint(value, 16, std::numeric_limits<uint64_t>::max(), number);
        current_entry.hardware_fingerprint = number;
      } else if (key == "revision_id") {
        ok = parse_uint(value, 16, std::numeric_limits<uint8_t>::max(), number);
        current_entry.revision_id = static_cast<uint8_t>(number);
      } else if (key == "last_update") {
        ok = parse_time(value, current_entry.last_update);
      } else {
        // The uint16_t fields, which are all parsed identically.
        uint16_t* field = nullptr;
        if (key == "package_id") {
          field = &current_entry.package_id;
        } else if (key == "core_id") {
          field = &current_entry.core_id;
        } else if (key == "vendor_id") {
          field = &current_entry.vendor_id;
        } else if (key == "device_id") {
          field = &current_entry.device_id;
        } else if (key == "family") {
          field = &current_entry.family;
        } else if (key == "model") {
          field = &current_entry.model;
        } else if (key == "pci_class") {
          field = &current_entry.pci_class;
        } else if (key == "unit_id") {
          field = &current_entry.unit_id;
        }
        if (field) {
          ok = parse_uint(value, 16, std::numeric_limits<uint16_t>::max(), number);
          *field = static_cast<uint16_t>(number);
        }
      }

      if (!ok) {
        LOG(ERROR, "CuidFile::load: " << file_path_ << ": malformed value for '" << key << "'");
        entries_.clear();
        return AMDCUID_STATUS_FILE_ERROR;
      }
    }

    // Save last entry
    if (in_section) {
      entries_.push_back(current_entry);
    }
  } catch (const std::exception& e) {
    LOG(ERROR, "CuidFile::load: " << file_path_ << ": " << e.what());
    entries_.clear();
    return AMDCUID_STATUS_FILE_ERROR;
  } catch (...) {
    LOG(ERROR, "CuidFile::load: " << file_path_ << ": unknown error");
    entries_.clear();
    return AMDCUID_STATUS_FILE_ERROR;
  }

  file.close();
  return AMDCUID_STATUS_SUCCESS;
}

amdcuid_status_t CuidFile::save() {
  // Before the lock, which lives in the same directory: PERMISSION_DENIED here
  // is a better answer than a lock failure reported as a file error.
  if (!ensure_parent_dir(file_path_)) {
    LOG(ERROR, "CuidFile::save: cannot create the directory for "
                   << file_path_ << ": " << CuidUtilities::errno_string(errno));
    return AMDCUID_STATUS_PERMISSION_DENIED;
  }

  // Acquire exclusive lock for writing
  CuidFileLock lock(file_path_, CuidLockType::EXCLUSIVE);
  if (!lock.acquire_with_timeout(kLockTimeoutSeconds)) {
    LOG(ERROR, "CuidFile::save: Failed to acquire exclusive lock for " << file_path_);
    return AMDCUID_STATUS_FILE_ERROR;
  }

  // Rendered in memory; write_record_file() below owns the temp file, its mode
  // and the rename.
  std::ostringstream file;

  // Write header comment
  file << "# AMD CUID Device Information File\n";
  file << "# Auto-generated by AMD CUID library\n";
  file << "# DO NOT EDIT MANUALLY\n";
  file << "# File: " << file_path_ << "\n";
  if (is_privileged_) {
    file << "# Type: Privileged (contains primary CUIDs)\n";
    file << "# Permissions: Root access only\n";
  } else {
    file << "# Type: Unprivileged (derived CUIDs only)\n";
    file << "# Permissions: Readable by all users\n";
  }
  file << "\n";

  // Group entries by type for better organization
  std::map<amdcuid_device_type_t, std::vector<CuidFileEntry>> grouped;
  get_grouped_entries(grouped);

  // A filter as much as an ordering: a type absent from this list is skipped
  // and never written, so every named Component Type has to be listed. The
  // first five keep their historical order so an existing record does not
  // churn; the rest follow in on-wire order.
  std::vector<amdcuid_device_type_t> order = {
      AMDCUID_DEVICE_TYPE_GPU,      AMDCUID_DEVICE_TYPE_CPU,      AMDCUID_DEVICE_TYPE_NIC,
      AMDCUID_DEVICE_TYPE_NPU,      AMDCUID_DEVICE_TYPE_PLATFORM, AMDCUID_DEVICE_TYPE_STORAGE,
      AMDCUID_DEVICE_TYPE_MEMORY,   AMDCUID_DEVICE_TYPE_GENPCIE,  AMDCUID_DEVICE_TYPE_GENC,
      AMDCUID_DEVICE_TYPE_RACKTRAY, AMDCUID_DEVICE_TYPE_RACK,     AMDCUID_DEVICE_TYPE_OTHER};

  for (auto type : order) {
    if (grouped.find(type) == grouped.end()) continue;

    for (const auto& entry : grouped[type]) {
      // Write section header
      if (entry.device_type == AMDCUID_DEVICE_TYPE_PLATFORM) {
        file << "[" << CuidUtilities::device_type_to_string(entry.device_type) << "]\n";
      } else {
        file << "[" << CuidUtilities::device_type_to_string(entry.device_type) << ":"
             << entry.device_index << "]\n";
      }

      // Write primary CUID (privileged file only)
      if (is_privileged_) {
        file << "primary_cuid=" << CuidUtilities::get_cuid_as_string(&entry.primary_cuid) << "\n";
      }

      // Write if CUID is temporary
      file << "is_temporary=" << (entry.is_temporary ? "true" : "false") << "\n";

      // Write derived CUID
      file << "derived_cuid=" << CuidUtilities::get_cuid_as_string(&entry.derived_cuid) << "\n";

      // Write hardware fingerprint (privileged file only)
      if (is_privileged_) {
        file << "hardware_fingerprint=0x" << std::hex << std::setw(16) << std::setfill('0')
             << entry.hardware_fingerprint << "\n";
      }

      // Write device-specific fields
      if (entry.vendor_id != 0) {
        file << "vendor_id=0x" << std::hex << std::setw(4) << std::setfill('0') << entry.vendor_id
             << "\n";
      }
      if (entry.device_id != 0) {
        file << "device_id=0x" << std::hex << std::setw(4) << std::setfill('0') << entry.device_id
             << "\n";
      }
      if (entry.revision_id != 0) {
        file << "revision_id=0x" << std::hex << std::setw(2) << std::setfill('0')
             << static_cast<uint16_t>(entry.revision_id) << "\n";
      }
      if (entry.family != 0) {
        file << "family=0x" << std::hex << std::setw(4) << std::setfill('0') << entry.family
             << "\n";
      }
      if (entry.model != 0) {
        file << "model=0x" << std::hex << std::setw(4) << std::setfill('0') << entry.model << "\n";
      }
      if (entry.pci_class != 0) {
        file << "pci_class=0x" << std::hex << std::setw(4) << std::setfill('0') << entry.pci_class
             << "\n";
      }
      if (entry.unit_id != std::numeric_limits<uint16_t>::max()) {
        file << "unit_id=0x" << std::hex << std::setw(4) << std::setfill('0') << entry.unit_id
             << "\n";
      }
      if (!entry.device_node.empty()) {
        file << "device_node=" << entry.device_node << "\n";
      }
      if (entry.package_id != std::numeric_limits<uint16_t>::max()) {
        file << "package_id=0x" << std::hex << std::setw(4) << std::setfill('0') << entry.package_id
             << "\n";
      }
      if (entry.core_id != std::numeric_limits<uint16_t>::max()) {
        file << "core_id=0x" << std::hex << std::setw(4) << std::setfill('0') << entry.core_id
             << "\n";
      }
      if (!entry.bdf.empty()) {
        file << "bdf=" << entry.bdf << "\n";
      }
      if (!entry.mac_address.empty()) {
        file << "mac_address=" << entry.mac_address << "\n";
      }

      // Write timestamp
      file << std::dec << "last_update=" << entry.last_update << "\n";
      file << "\n";
    }
  }

  // The privileged record carries primary CUIDs and raw hardware fingerprints,
  // so it is 0600 from the moment it exists rather than chmod'ed down
  // afterwards. The unprivileged record is 0644; every consumer reads it.
  const mode_t mode =
      is_privileged_ ? (S_IRUSR | S_IWUSR) : (S_IRUSR | S_IWUSR | S_IRGRP | S_IROTH);
  return write_record_file(file_path_, file.str(), mode);
}

amdcuid_status_t CuidFile::add_entry(const CuidFileEntry& entry) {
  // Check if entry with same derived CUID exists
  for (auto& existing : entries_) {
    if (memcmp(existing.derived_cuid.bytes, entry.derived_cuid.bytes,
               sizeof(amdcuid_id_t::bytes)) == 0) {
      // Update existing entry
      existing = entry;
      return AMDCUID_STATUS_SUCCESS;
    }
  }

  // Add new entry
  entries_.push_back(entry);
  return AMDCUID_STATUS_SUCCESS;
}

amdcuid_status_t CuidFile::remove_entry(const amdcuid_id_t& handle) {
  // search for entry by derived CUID and move that entry to the back, then
  // erase it
  auto it = std::remove_if(entries_.begin(), entries_.end(), [&handle](const CuidFileEntry& e) {
    return (memcmp(e.derived_cuid.bytes, handle.bytes, sizeof(amdcuid_id_t::bytes)) == 0);
  });
  if (it != entries_.end()) {
    entries_.erase(it, entries_.end());
    return AMDCUID_STATUS_SUCCESS;
  }
  return AMDCUID_STATUS_DEVICE_NOT_FOUND;
}

amdcuid_status_t CuidFile::find_by_device_node(const std::string& device_node,
                                               CuidFileEntry& entry) const {
  for (const auto& e : entries_) {
    if (e.device_node == device_node) {
      entry = e;
      return AMDCUID_STATUS_SUCCESS;
    }
  }
  return AMDCUID_STATUS_DEVICE_NOT_FOUND;
}

amdcuid_status_t CuidFile::find_by_bdf(const std::string& bdf, CuidFileEntry& entry) const {
  for (const auto& e : entries_) {
    if (e.bdf == bdf) {
      entry = e;
      return AMDCUID_STATUS_SUCCESS;
    }
  }
  return AMDCUID_STATUS_DEVICE_NOT_FOUND;
}

amdcuid_status_t CuidFile::find_by_package_id(uint16_t package_id, CuidFileEntry& entry) const {
  for (const auto& e : entries_) {
    if (e.package_id == package_id) {
      entry = e;
      return AMDCUID_STATUS_SUCCESS;
    }
  }
  return AMDCUID_STATUS_DEVICE_NOT_FOUND;
}

amdcuid_status_t CuidFile::find_by_device_type(amdcuid_device_type_t device_type,
                                               CuidFileEntry& entry) const {
  for (const auto& e : entries_) {
    if (e.device_type == device_type) {
      entry = e;
      return AMDCUID_STATUS_SUCCESS;
    }
  }
  return AMDCUID_STATUS_DEVICE_NOT_FOUND;
}

amdcuid_status_t CuidFile::find_by_derived_cuid(const amdcuid_id_t& derived_cuid,
                                                CuidFileEntry& entry) const {
  for (const auto& e : entries_) {
    if (memcmp(e.derived_cuid.bytes, derived_cuid.bytes, sizeof(amdcuid_id_t::bytes)) == 0) {
      entry = e;
      return AMDCUID_STATUS_SUCCESS;
    }
  }
  return AMDCUID_STATUS_DEVICE_NOT_FOUND;
}

void CuidFile::get_grouped_entries(
    std::map<amdcuid_device_type_t, std::vector<CuidFileEntry>>& grouped) const {
  grouped.clear();
  for (const auto& entry : entries_) {
    grouped[entry.device_type].push_back(entry);
  }
}

// ============================================================================
// CuidFileGenerator Implementation
// ============================================================================

namespace {

// helper function to generate CUID files from a list of devices
amdcuid_status_t generate_from_devices(const std::vector<std::shared_ptr<CuidDevice>>& devices,
                                       const std::string& file, bool is_privileged) {
  // Create file handlers
  CuidFile cuid_file(file, is_privileged);

  // Clear existing entries
  cuid_file.clear();

  time_t now = time(nullptr);
  cuid_hmac hmac = cuid_hmac();

  // Track device indices per type
  std::map<amdcuid_device_type_t, uint32_t> device_counters;

  // Process each device
  for (const auto& device : devices) {
    if (!device) continue;

    CuidFileEntry entry;
    entry.device_type = device->type();
    entry.device_index = device_counters[entry.device_type]++;
    entry.last_update = now;

    amdcuid_status_t status;
    // Check if the CUID is temporary
    bool is_temporary = false;
    status = device->is_temporary_cuid(&is_temporary);
    if (status != AMDCUID_STATUS_SUCCESS) {
      std::cerr << "Warning: Failed to get temporary CUID status for device type "
                << entry.device_type << " status: " << status << std::endl;
    }
    entry.is_temporary = is_temporary;

    if (is_privileged) {
      // Get primary CUID
      amdcuid_primary_id primary_id = {};
      status = device->get_primary_cuid(primary_id);
      if (status != AMDCUID_STATUS_SUCCESS) {
        std::cerr << "Warning: Failed to get primary CUID for device type " << entry.device_type
                  << " status: " << status << std::endl;
      }
      entry.primary_cuid = primary_id.UUIDv8_representation;

      // get hardware fingerprint
      uint64_t fingerprint = 0;
      status = device->get_hardware_fingerprint(fingerprint);
      if (status != AMDCUID_STATUS_SUCCESS && !is_temporary) {
        std::cerr << "Warning: Failed to get hardware fingerprint for device type "
                  << entry.device_type << " status: " << status << std::endl;
      } else {
        entry.hardware_fingerprint = fingerprint;
      }
    }

    amdcuid_derived_id derived_id = {};
    status = device->get_derived_cuid(derived_id, &hmac);
    if (status != AMDCUID_STATUS_SUCCESS) {
      // No entry at all, rather than an entry holding the zero-initialised
      // derived_id: every device with no obtainable primary would otherwise be
      // recorded under the same all-zero identifier. A device the record does
      // not name is answered by the lookup path; one it names wrongly is not.
      std::cerr << "Warning: Failed to generate derived CUID for device type " << entry.device_type
                << " status: " << status << "; omitting it from " << file << std::endl;
      --device_counters[entry.device_type];
      continue;
    }
    entry.derived_cuid = derived_id.UUIDv8_representation;

    // Fill in device-specific information
    switch (entry.device_type) {
      case AMDCUID_DEVICE_TYPE_GPU: {
        auto gpu = std::dynamic_pointer_cast<CuidGpu>(device);
        if (gpu) {
          const auto& info = gpu->get_info();
          entry.vendor_id = info.header.fields.gpu.vendor_id;
          entry.device_id = info.header.fields.gpu.device_id;
          entry.revision_id = info.header.fields.gpu.revision_id;
          entry.pci_class = info.header.fields.gpu.pci_class;
          entry.unit_id = info.header.fields.gpu.unit_id;
          entry.device_node = info.render_node;
          entry.bdf = info.bdf;

          // if temp CUID, rebuild the auxiliary fingerprint the device used
          if (is_temporary) {
            CuidUtilities::AuxiliaryInput aux;
            aux.format = CuidUtilities::kAuxFormatPcie;
            aux.routing_id = CuidUtilities::routing_id_from_bdf(info.bdf);
            aux.revision_id = info.header.fields.gpu.revision_id;
            aux.device_id = info.header.fields.gpu.device_id;
            aux.vendor_id = info.header.fields.gpu.vendor_id;
            aux.component_type = static_cast<uint8_t>(AMDCUID_DEVICE_TYPE_GPU);
            CuidUtilities::make_fallback_fingerprint(aux, entry.hardware_fingerprint);
          }
        }
        break;
      }
      case AMDCUID_DEVICE_TYPE_CPU: {
        auto cpu = std::dynamic_pointer_cast<CuidCpu>(device);
        if (cpu) {
          const auto& info = cpu->get_info();
          entry.vendor_id = info.header.fields.cpu.vendor_id;
          entry.device_id = info.header.fields.cpu.device_id;
          entry.revision_id = info.header.fields.cpu.revision_id;
          entry.family = info.header.fields.cpu.family;
          entry.model = info.header.fields.cpu.model;
          entry.unit_id = info.header.fields.cpu.unit_id;
          entry.package_id = info.header.fields.cpu.physical_id;
          entry.core_id = info.header.fields.cpu.core;
          // Store device path (unique per logical CPU, needed for SMT)
          std::string cpu_device_path;
          if (cpu->get_device_path(cpu_device_path) == AMDCUID_STATUS_SUCCESS) {
            entry.device_node = cpu_device_path;
          }
          // if temp CUID, rebuild the auxiliary fingerprint the device used
          if (is_temporary) {
            CuidUtilities::AuxiliaryInput aux;
            aux.format = CuidUtilities::kAuxFormatCpu;
            aux.routing_id = 0;
            aux.revision_id = info.header.fields.cpu.revision_id;
            aux.device_id = info.header.fields.cpu.device_id;
            aux.vendor_id = info.header.fields.cpu.vendor_id;
            aux.component_type = static_cast<uint8_t>(AMDCUID_DEVICE_TYPE_CPU);
            CuidUtilities::make_fallback_fingerprint(aux, entry.hardware_fingerprint);
          }
        }
        break;
      }
      case AMDCUID_DEVICE_TYPE_NIC: {
        auto nic = std::dynamic_pointer_cast<CuidNic>(device);
        if (nic) {
          const auto& info = nic->get_info();
          entry.vendor_id = info.header.fields.nic.vendor_id;
          entry.device_id = info.header.fields.nic.device_id;
          entry.revision_id = info.header.fields.nic.revision_id;
          entry.pci_class = info.header.fields.nic.pci_class;
          entry.device_node = info.network_interface;
          std::string mac_address;
          if (nic->get_mac_address(mac_address) == AMDCUID_STATUS_SUCCESS) {
            entry.mac_address = mac_address;
          }
          entry.bdf = info.bdf;

          // if temp CUID, rebuild the auxiliary fingerprint the device used
          if (is_temporary) {
            CuidUtilities::AuxiliaryInput aux;
            aux.format = CuidUtilities::kAuxFormatPcie;
            aux.routing_id = CuidUtilities::routing_id_from_bdf(info.bdf);
            aux.revision_id = info.header.fields.nic.revision_id;
            aux.device_id = info.header.fields.nic.device_id;
            aux.vendor_id = info.header.fields.nic.vendor_id;
            aux.component_type = static_cast<uint8_t>(AMDCUID_DEVICE_TYPE_NIC);
            CuidUtilities::make_fallback_fingerprint(aux, entry.hardware_fingerprint);
          }
        }
        break;
      }
      case AMDCUID_DEVICE_TYPE_NPU: {
        auto npu = std::dynamic_pointer_cast<CuidNpu>(device);
        if (npu) {
          const auto& info = npu->get_info();
          entry.vendor_id = info.header.fields.npu.vendor_id;
          entry.device_id = info.header.fields.npu.device_id;
          entry.revision_id = info.header.fields.npu.revision_id;
          entry.pci_class = info.header.fields.npu.pci_class;
          entry.device_node = info.accel_node;
          entry.bdf = info.bdf;
          // if temp CUID, rebuild the auxiliary fingerprint the device used
          if (is_temporary) {
            CuidUtilities::AuxiliaryInput aux;
            aux.format = CuidUtilities::kAuxFormatPcie;
            aux.routing_id = CuidUtilities::routing_id_from_bdf(info.bdf);
            aux.revision_id = info.header.fields.npu.revision_id;
            aux.device_id = info.header.fields.npu.device_id;
            aux.vendor_id = info.header.fields.npu.vendor_id;
            aux.component_type = static_cast<uint8_t>(AMDCUID_DEVICE_TYPE_NPU);
            CuidUtilities::make_fallback_fingerprint(aux, entry.hardware_fingerprint);
          }
        }
        break;
      }
      case AMDCUID_DEVICE_TYPE_PLATFORM: {
        auto platform = std::dynamic_pointer_cast<CuidPlatform>(device);
        // Platform only has vendor_id
        if (platform) {
          const auto& info = platform->get_info();
          entry.vendor_id = info.header.fields.platform.vendor_id;

          // The Platform CUID has no auxiliary form: it is the SMBIOS system
          // UUID verbatim, or built from the system serial through the normal
          // layout, or absent. Nothing to synthesise.
        }
        break;
      }
      default:
        break;
    }

    // Add to file
    cuid_file.add_entry(entry);
  }

  // Save file
  amdcuid_status_t status = cuid_file.save();
  if (status != AMDCUID_STATUS_SUCCESS) {
    std::cerr << "Error: Failed to save CUID file: " << file << std::endl;
    return status;
  }

  return AMDCUID_STATUS_SUCCESS;
}

}  // namespace

amdcuid_status_t CuidFileGenerator::generate_unpriv_from_devices(
    const std::vector<std::shared_ptr<CuidDevice>>& devices, const std::string& unpriv_file_path) {
  return generate_from_devices(devices, unpriv_file_path, false);
}

amdcuid_status_t CuidFileGenerator::generate_priv_from_devices(
    const std::vector<std::shared_ptr<CuidDevice>>& devices, const std::string& priv_file_path) {
  return generate_from_devices(devices, priv_file_path, true);
}
