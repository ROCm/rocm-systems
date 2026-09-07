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
  // NOTE: without O_EXCL|O_NOFOLLOW this follows a symlink planted at `path` and
  // reuses/overwrites an existing file -- see the accompanying test, which
  // fails here. The fix adds O_EXCL|O_NOFOLLOW.
  int fd = ::open(path, O_WRONLY | O_CREAT | O_TRUNC, mode);
  if (fd < 0) {
    return -1;
  }
  ::fchmod(fd, mode);
  return fd;
}

#endif  // CUID_LIB_SRC_CUID_FILE_UTILS_H_
