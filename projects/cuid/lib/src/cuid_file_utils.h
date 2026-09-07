// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef CUID_LIB_SRC_CUID_FILE_UTILS_H_
#define CUID_LIB_SRC_CUID_FILE_UTILS_H_

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

// Create a new regular file at `path` with the given mode, failing if `path`
// already exists or is a symlink. fchmod() then sets the exact mode on the
// returned descriptor regardless of umask. Returns an open write descriptor on
// success (the caller closes it), or -1 on error.
//
// Creating the file this way lets callers set permissions on a descriptor they
// own before renaming it into place, instead of a TOCTOU-prone chmod() on the
// destination path after the rename (CWE-367).
inline int CuidCreateExclusiveFile(const char* path, mode_t mode) {
  // O_EXCL fails if `path` already exists (including as a symlink), and
  // O_NOFOLLOW additionally refuses a symlink at the final component, so the
  // file is always freshly created and owned by us.
  int fd = ::open(path, O_WRONLY | O_CREAT | O_EXCL | O_NOFOLLOW | O_CLOEXEC, mode);
  if (fd < 0) {
    return -1;
  }
  ::fchmod(fd, mode);  // exact mode regardless of umask
  return fd;
}

#endif  // CUID_LIB_SRC_CUID_FILE_UTILS_H_
