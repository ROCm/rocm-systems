#include "amd_smi/impl/scoped_fd.h"

#include <fcntl.h>

#include <cstring>
#include <string>

#include "rocm_smi/rocm_smi_logger.h"

ScopedFD::ScopedFD(const std::string& path, int flags)
    : fd_(open(path.c_str(), flags)), path_(path) {
  std::ostringstream ss;
  if (fd_ < 0) {
    ss << __PRETTY_FUNCTION__ << " | Failed to open file: " << path_
       << " | Error: " << strerror(errno);
    LOG_ERROR(ss);
  }
}

ScopedFD::~ScopedFD() {
  if (fd_ >= 0) {
    close(fd_);
  }
}

ScopedFD::ScopedFD(ScopedFD&& other) noexcept : fd_(other.fd_), path_(std::move(other.path_)) {
  other.fd_ = -1;
}

ScopedFD& ScopedFD::operator=(ScopedFD&& other) noexcept {
  if (this != &other) {
    if (fd_ >= 0) close(fd_);
    fd_ = other.fd_;
    path_ = std::move(other.path_);
    other.fd_ = -1;
  }
  return *this;
}

int ScopedFD::get() const { return fd_; }

bool ScopedFD::valid() const { return fd_ >= 0; }

ScopedFD::operator int() const { return fd_; }
