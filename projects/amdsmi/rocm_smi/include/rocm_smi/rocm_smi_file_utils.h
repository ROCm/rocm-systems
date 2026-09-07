// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef ROCM_SMI_INCLUDE_ROCM_SMI_ROCM_SMI_FILE_UTILS_H_
#define ROCM_SMI_INCLUDE_ROCM_SMI_ROCM_SMI_FILE_UTILS_H_

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cerrno>

namespace amd::smi {

// Set the permission bits of the existing regular file at `path`, refusing to
// follow a symlink at the final path component. This avoids the TOCTOU race in
// chmod(path): between file creation and a path-based chmod, `path` can be
// replaced with a symlink, redirecting the mode change to an attacker-chosen
// file (CWE-367 / CWE-732). Returns 0 on success, -1 on error (errno set).
inline int SetFileModeNoFollow(const char* path, mode_t mode) {
  // O_NOFOLLOW makes the open fail if the final path component is a symlink, and
  // fchmod() acts on the opened descriptor rather than re-resolving the path --
  // together these close the chmod(path) TOCTOU window.
  const int fd = ::open(path, O_WRONLY | O_NOFOLLOW | O_CLOEXEC);
  if (fd < 0) {
    return -1;
  }
  const int rc = ::fchmod(fd, mode);
  const int saved_errno = errno;
  ::close(fd);
  errno = saved_errno;
  return rc;
}

}  // namespace amd::smi

#endif  // ROCM_SMI_INCLUDE_ROCM_SMI_ROCM_SMI_FILE_UTILS_H_
