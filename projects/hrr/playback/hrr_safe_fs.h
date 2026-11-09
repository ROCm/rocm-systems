/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */
#pragma once

// File writes that never go through a symbolic link.
//
// `hrr-playback --repair` rewrites files named by the archive it was pointed
// at. An archive is input from another principal: a link planted in it where
// repair expects a file or a directory would send the write to whatever the
// link names, with the repairing user's rights. These helpers do the writes the
// way the capture writer does: open the directory once with
// O_NOFOLLOW, check it, and then create, test and rename names inside it through
// that descriptor, so a name cannot be swapped for a link between the check and
// the use.
//
// POSIX only. The Windows build keeps its own path in hrr_playback.cpp.

#ifndef _WIN32

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <string>
#include <system_error>
#include <utility>

namespace hrr {
namespace safefs {

// Owning file descriptor.
class Fd {
 public:
  Fd() = default;
  explicit Fd(int fd) : fd_(fd) {}
  Fd(Fd&& o) noexcept : fd_(o.release()) {}
  Fd& operator=(Fd&& o) noexcept {
    if (this != &o) { reset(); fd_ = o.release(); }
    return *this;
  }
  Fd(const Fd&) = delete;
  Fd& operator=(const Fd&) = delete;
  ~Fd() { reset(); }

  int get() const { return fd_; }
  bool valid() const { return fd_ >= 0; }
  int release() { int f = fd_; fd_ = -1; return f; }
  void reset() { if (fd_ >= 0) ::close(fd_); fd_ = -1; }

 private:
  int fd_ = -1;
};

inline std::string errno_text(const std::string& what) {
  return what + ": " + std::strerror(errno);
}

// Open the directory called `name` inside `parent_fd`. `name` must be a single
// path component. A symbolic link, even one to a directory, is refused, and so
// is anything that is not a directory. The directory must belong to the
// calling user (or the caller must be root): repair does not write into a
// directory someone else owns.
inline Fd open_dir_at(int parent_fd, const std::string& name, std::string* err) {
  if (name.empty() || name.find('/') != std::string::npos || name == "." || name == "..") {
    if (err) *err = "not a single path component: '" + name + "'";
    return Fd();
  }
  int fd = ::openat(parent_fd, name.c_str(),
                    O_RDONLY | O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC);
  if (fd < 0) {
    if (err) {
      *err = (errno == ELOOP || errno == ENOTDIR)
                 ? "refusing " + name + ": not a real directory (symbolic link or other file)"
                 : errno_text("cannot open directory " + name);
    }
    return Fd();
  }
  Fd d(fd);
  struct stat st;
  if (::fstat(d.get(), &st) != 0) {
    if (err) *err = errno_text("cannot stat directory " + name);
    return Fd();
  }
  if (st.st_uid != ::geteuid() && ::geteuid() != 0) {
    if (err) *err = "refusing " + name + ": the directory belongs to another user";
    return Fd();
  }
  return d;
}

// Open the directory at `path`. The directories above it may be links (the
// caller named the path and resolves them), but the last component may not.
inline Fd open_dir_nofollow_leaf(const std::string& path, std::string* err) {
  namespace fs = std::filesystem;
  std::error_code ec;
  fs::path p = fs::absolute(path, ec);
  if (ec) {
    if (err) *err = "cannot resolve " + path + ": " + ec.message();
    return Fd();
  }
  // No lexical normalisation: "link/../x" must mean what the kernel makes of
  // it (the parent of what link points to), the same as when the archive was
  // read, and canonical() below resolves the parent that way.
  if (p.filename().empty()) p = p.parent_path();  // trailing slash
  if (p.filename().empty()) {                      // "/"
    if (err) *err = "refusing to write into /";
    return Fd();
  }
  const fs::path parent = fs::canonical(p.parent_path(), ec);
  if (ec) {
    if (err) *err = "cannot resolve " + p.parent_path().string() + ": " + ec.message();
    return Fd();
  }
  Fd pfd(::open(parent.c_str(), O_RDONLY | O_DIRECTORY | O_CLOEXEC));
  if (!pfd.valid()) {
    if (err) *err = errno_text("cannot open " + parent.string());
    return Fd();
  }
  return open_dir_at(pfd.get(), p.filename().string(), err);
}

// Result of looking at a name inside a directory without following a link.
enum class Entry { Absent, Regular, Other, Error };

// Is `name` in `dir_fd` absent, a regular file, or something else (a symbolic
// link, a directory, a device...)? The link itself is examined, never its
// target.
inline Entry stat_entry(int dir_fd, const std::string& name, std::string* err = nullptr) {
  struct stat st;
  if (::fstatat(dir_fd, name.c_str(), &st, AT_SYMLINK_NOFOLLOW) != 0) {
    if (errno == ENOENT) return Entry::Absent;
    if (err) *err = errno_text("cannot stat " + name);
    return Entry::Error;
  }
  return S_ISREG(st.st_mode) ? Entry::Regular : Entry::Other;
}

// 16 hex digits from the kernel's random source, for names an attacker cannot
// guess ahead of time.
inline bool random_hex(std::string* out) {
  uint64_t v = 0;
  int fd = ::open("/dev/urandom", O_RDONLY | O_CLOEXEC);
  if (fd < 0) return false;
  ssize_t n = ::read(fd, &v, sizeof(v));
  ::close(fd);
  if (n != static_cast<ssize_t>(sizeof(v))) return false;
  char buf[17];
  std::snprintf(buf, sizeof(buf), "%016llx", static_cast<unsigned long long>(v));
  *out = buf;
  return true;
}

// Create a new file in `dir_fd` named "<stem>.<random>" that did not exist
// (O_EXCL) and is not a link (O_NOFOLLOW). The file is created with `mode`
// (default private, 0600; the umask applies). *name receives the name.
// Retries a few times if the random name is taken.
inline Fd create_exclusive_temp(int dir_fd, const std::string& stem, std::string* name,
                                std::string* err, mode_t mode = 0600) {
  for (int attempt = 0; attempt < 8; ++attempt) {
    std::string hex;
    if (!random_hex(&hex)) {
      if (err) *err = "cannot read /dev/urandom for a temporary file name";
      return Fd();
    }
    const std::string candidate = stem + "." + hex;
    int fd = ::openat(dir_fd, candidate.c_str(),
                      O_WRONLY | O_CREAT | O_EXCL | O_NOFOLLOW | O_CLOEXEC, mode);
    if (fd >= 0) {
      *name = candidate;
      return Fd(fd);
    }
    if (errno != EEXIST) {
      if (err) *err = errno_text("cannot create " + candidate);
      return Fd();
    }
  }
  if (err) *err = "cannot find an unused temporary file name for " + stem;
  return Fd();
}

// Move the finished, already-fsynced temporary file over `final_name` in
// `dir_fd`, then persist the rename. renameat replaces a name that is a link
// with the new file; it does not write through the link. Callers refuse a link
// at `final_name` first so repair does not silently replace one either.
inline bool commit_replace(int dir_fd, const std::string& tmp_name,
                           const std::string& final_name, std::string* err) {
  if (::renameat(dir_fd, tmp_name.c_str(), dir_fd, final_name.c_str()) != 0) {
    if (err) *err = errno_text("cannot replace " + final_name);
    return false;
  }
  (void)::fsync(dir_fd);  // best effort: persist the rename
  return true;
}

inline void unlink_at(int dir_fd, const std::string& name) {
  (void)::unlinkat(dir_fd, name.c_str(), 0);
}

}  // namespace safefs
}  // namespace hrr

#endif  // !_WIN32
