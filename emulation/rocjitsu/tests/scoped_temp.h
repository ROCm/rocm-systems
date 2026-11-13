// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include <gtest/gtest.h>

#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <optional>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>

#include <unistd.h>

namespace rocjitsu::test {

class ScopedTempFile {
public:
  ScopedTempFile() = default;

  /// @brief Create the resource; check success before using path().
  /// @details An initial creation failure leaves the path empty. Reinitializing
  /// an owned resource fails and preserves its existing path.
  [[nodiscard]] testing::AssertionResult initialize(std::string_view prefix) {
    if (!path_.empty())
      return testing::AssertionFailure() << "Temporary resource is already initialized";
    std::error_code error;
    const std::filesystem::path directory = std::filesystem::temp_directory_path(error);
    if (error)
      return testing::AssertionFailure()
             << "Failed to locate temporary directory: " << error.message();
    std::string pattern = (directory / (std::string(prefix) + "XXXXXX")).string();
    const int fd = ::mkstemp(pattern.data());
    if (fd == -1)
      return testing::AssertionFailure()
             << "Failed to create temporary file: " << std::strerror(errno);
    (void)::close(fd);
    path_ = std::move(pattern);
    return testing::AssertionSuccess();
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

  /// @brief Write an initialized file; check success before consuming its contents.
  [[nodiscard]] testing::AssertionResult write(std::string_view contents) const {
    if (path_.empty())
      return testing::AssertionFailure() << "Temporary file is not initialized";
    std::ofstream output(path_, std::ios::binary | std::ios::trunc);
    output.write(contents.data(), static_cast<std::streamsize>(contents.size()));
    output.close();
    if (!output)
      return testing::AssertionFailure() << "Failed to write temporary file: " << path_;
    return testing::AssertionSuccess();
  }

private:
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
  ScopedTempDirectory() = default;

  /// @brief Create the resource; check success before using path().
  /// @details An initial creation failure leaves the path empty. Reinitializing
  /// an owned resource fails and preserves its existing path.
  [[nodiscard]] testing::AssertionResult initialize(std::string_view prefix) {
    if (!path_.empty())
      return testing::AssertionFailure() << "Temporary resource is already initialized";
    std::error_code error;
    const std::filesystem::path directory = std::filesystem::temp_directory_path(error);
    if (error)
      return testing::AssertionFailure()
             << "Failed to locate temporary directory: " << error.message();
    std::string pattern = (directory / (std::string(prefix) + "XXXXXX")).string();
    if (::mkdtemp(pattern.data()) == nullptr)
      return testing::AssertionFailure()
             << "Failed to create temporary directory: " << std::strerror(errno);
    path_ = std::move(pattern);
    return testing::AssertionSuccess();
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
  void cleanup() noexcept {
    if (path_.empty())
      return;
    std::error_code error;
    std::filesystem::remove_all(path_, error);
  }

  std::string path_;
};

/// @brief Sets an environment variable for the current scope and restores it on exit.
/// @details Tests that steer config discovery through the environment share one process
/// with every other test in the binary, so an unrestored variable silently changes what
/// later tests resolve. Restoring means putting back the original value, or unsetting the
/// variable again when it was absent to begin with.
class ScopedEnvironmentVariable {
public:
  ScopedEnvironmentVariable(const char *name, const std::string &value) : name_(name) {
    if (const char *original = std::getenv(name))
      original_ = original;
    setenv(name_.c_str(), value.c_str(), 1);
  }

  ~ScopedEnvironmentVariable() {
    if (original_)
      setenv(name_.c_str(), original_->c_str(), 1);
    else
      unsetenv(name_.c_str());
  }

  ScopedEnvironmentVariable(const ScopedEnvironmentVariable &) = delete;
  ScopedEnvironmentVariable &operator=(const ScopedEnvironmentVariable &) = delete;

private:
  std::string name_;
  std::optional<std::string> original_;
};

} // namespace rocjitsu::test
