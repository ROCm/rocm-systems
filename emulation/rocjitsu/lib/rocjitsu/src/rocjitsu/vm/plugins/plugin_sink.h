// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file plugin_sink.h
/// @brief Output sink abstraction for the plugin system.
///
/// Plugins produce diagnostic output (race reports, profiling data, kernel
/// logs). Rather than writing directly to stderr, plugins write to a
/// PluginSink. This decouples output destination from output generation:
///
///   - Production: StderrSink (default) — same behavior as writing to stderr.
///   - Tests: StringSink — captures output for assertions, no stderr scraping.
///   - Integration tests: FileSink — writes to <dir>/<plugin_name>.log for
///     structured parsing by test harnesses.
///   - Mixed: CompositeSink — dispatches to multiple sinks simultaneously
///     (e.g., stderr for interactive viewing + file for test parsing).
///
/// ## Wiring
///
/// The ExecutionPluginGroup owns the sink configuration. When a plugin is
/// added to a group via add(), the group constructs a sink for that plugin
/// (combining shared sinks like stderr with per-plugin file sinks) and
/// assigns it. Plugins access their sink via sink().write("message").
///
/// If a plugin is used without a group (unusual), it falls back to a
/// static StderrSink so output is never silently lost.
///
/// ## Environment variables (production path)
///
///   RJ_SINKS=stderr,file   Comma-separated sink types: stderr, stdout, file
///                          (default: stderr).
///   RJ_SINK_DIR=/tmp/out   Directory for file sinks. Each plugin writes
///                          to <dir>/<plugin_name>.log.

#pragma once

#include <cstdio>
#include <string>
#include <string_view>
#include <vector>

namespace rocjitsu {

/// @brief Abstract base for plugin output destinations.
class PluginSink {
public:
  virtual ~PluginSink() = default;
  virtual void write(std::string_view msg) = 0;
};

/// @brief Writes to stderr. Thread-safe via stdio locking.
class StderrSink : public PluginSink {
public:
  void write(std::string_view msg) override { std::fwrite(msg.data(), 1, msg.size(), stderr); }

  static StderrSink &instance() {
    static StderrSink s;
    return s;
  }
};

/// @brief Writes to stdout.
class StdoutSink : public PluginSink {
public:
  void write(std::string_view msg) override { std::fwrite(msg.data(), 1, msg.size(), stdout); }

  static StdoutSink &instance() {
    static StdoutSink s;
    return s;
  }
};

/// @brief Writes to a file. Owns the FILE* handle; flushes after each write.
class FileSink : public PluginSink {
public:
  explicit FileSink(const std::string &path) : file_(std::fopen(path.c_str(), "w")) {}

  ~FileSink() override {
    if (file_)
      std::fclose(file_);
  }

  FileSink(const FileSink &) = delete;
  FileSink &operator=(const FileSink &) = delete;

  void write(std::string_view msg) override {
    if (file_) {
      std::fwrite(msg.data(), 1, msg.size(), file_);
      std::fflush(file_);
    }
  }

private:
  FILE *file_ = nullptr;
};

/// @brief Captures output into a string. For use in unit tests.
class StringSink : public PluginSink {
public:
  void write(std::string_view msg) override { buf_ += msg; }

  const std::string &str() const { return buf_; }
  void clear() { buf_.clear(); }

private:
  std::string buf_;
};

/// @brief Dispatches writes to multiple child sinks.
class CompositeSink : public PluginSink {
public:
  void add(PluginSink *s) {
    if (s)
      children_.push_back(s);
  }

  void write(std::string_view msg) override {
    for (auto *s : children_)
      s->write(msg);
  }

  bool empty() const { return children_.empty(); }

private:
  std::vector<PluginSink *> children_;
};

} // namespace rocjitsu
