// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>

#include <unistd.h>

namespace rocjitsu::test {

class ScopedTempFile {
public:
  explicit ScopedTempFile(std::string_view prefix) : path_(make_pattern(prefix)) {
    const int fd = ::mkstemp(path_.data());
    if (fd == -1)
      throw std::runtime_error("Failed to create temporary file");
    (void)::close(fd);
  }

  ~ScopedTempFile() { cleanup(); }

  ScopedTempFile(const ScopedTempFile &) = delete;
  ScopedTempFile &operator=(const ScopedTempFile &) = delete;

  ScopedTempFile(ScopedTempFile &&other) noexcept : path_(std::move(other.path_)) {
    other.path_.clear();
  }

  ScopedTempFile &operator=(ScopedTempFile &&other) noexcept {
    if (this != &other) {
      cleanup();
      path_ = std::move(other.path_);
      other.path_.clear();
    }
    return *this;
  }

  [[nodiscard]] const std::string &path() const { return path_; }

  void write(std::string_view contents) const {
    std::ofstream output(path_, std::ios::binary | std::ios::trunc);
    output.write(contents.data(), static_cast<std::streamsize>(contents.size()));
    if (!output)
      throw std::runtime_error("Failed to write temporary file");
  }

private:
  static std::string make_pattern(std::string_view prefix) {
    return (std::filesystem::temp_directory_path() / (std::string(prefix) + "XXXXXX")).string();
  }

  void cleanup() noexcept {
    if (path_.empty())
      return;
    std::error_code error;
    std::filesystem::remove(path_, error);
  }

  std::string path_;
};

class ScopedTempDirectory {
public:
  explicit ScopedTempDirectory(std::string_view prefix) : path_(make_pattern(prefix)) {
    if (::mkdtemp(path_.data()) == nullptr)
      throw std::runtime_error("Failed to create temporary directory");
  }

  ~ScopedTempDirectory() { cleanup(); }

  ScopedTempDirectory(const ScopedTempDirectory &) = delete;
  ScopedTempDirectory &operator=(const ScopedTempDirectory &) = delete;

  ScopedTempDirectory(ScopedTempDirectory &&other) noexcept : path_(std::move(other.path_)) {
    other.path_.clear();
  }

  ScopedTempDirectory &operator=(ScopedTempDirectory &&other) noexcept {
    if (this != &other) {
      cleanup();
      path_ = std::move(other.path_);
      other.path_.clear();
    }
    return *this;
  }

  [[nodiscard]] const std::string &path() const { return path_; }

private:
  static std::string make_pattern(std::string_view prefix) {
    return (std::filesystem::temp_directory_path() / (std::string(prefix) + "XXXXXX")).string();
  }

  void cleanup() noexcept {
    if (path_.empty())
      return;
    std::error_code error;
    std::filesystem::remove_all(path_, error);
  }

  std::string path_;
};

} // namespace rocjitsu::test
