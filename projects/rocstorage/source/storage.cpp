// MIT License
//
// Copyright (c) 2025 Advanced Micro Devices, Inc. All Rights Reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include <rocstorage/storage.hpp>

#include "data_storage/database.hpp"

#include <filesystem>
#include <fstream>
#include <stdexcept>

namespace rocm {

// ==================== storage_config implementation ====================

namespace {

bool is_directory_writable(const std::string &path) {
  if (!std::filesystem::exists(path)) {
    return false;
  }
  if (!std::filesystem::is_directory(path)) {
    return false;
  }
  // Try to create a temp file to verify write access
  auto test_path = std::filesystem::path(path) / ".rocstorage_write_test";
  std::ofstream test_file(test_path);
  if (!test_file.good()) {
    return false;
  }
  test_file.close();
  std::filesystem::remove(test_path);
  return true;
}

} // namespace

storage_config storage_config::write_only() {
  storage_config config;
  config.storage_mode = mode::write;
  return config;
}

storage_config storage_config::read_write() {
  storage_config config;
  config.storage_mode = mode::read_write;
  config.wal_directory = default_wal_directory();
  return config;
}

storage_config storage_config::read_only() {
  storage_config config;
  config.storage_mode = mode::read;
  return config;
}

storage_config storage_config::detect_defaults() {
  // Default to write-only mode (current behavior, max performance)
  return write_only();
}

std::string storage_config::default_wal_directory() {
  // Use RAM disk if available, otherwise temp directory
  if (is_ram_disk_available()) {
    return "/dev/shm/rocstorage";
  }
  return (std::filesystem::temp_directory_path() / "rocstorage").string();
}

bool storage_config::is_ram_disk_available() {
#ifdef __linux__
  return is_directory_writable("/dev/shm");
#else
  // RAM disk not available on other platforms by default
  return false;
#endif
}

std::string storage_config::effective_wal_directory() const {
  // If explicitly set, use it
  if (!wal_directory.empty()) {
    return wal_directory;
  }

  // Otherwise, compute based on use_ram_disk preference
  if (use_ram_disk && is_ram_disk_available()) {
    return "/dev/shm/rocstorage";
  }

  return (std::filesystem::temp_directory_path() / "rocstorage").string();
}

// ==================== storage implementation ====================

struct storage::Impl {
  std::string path;
  std::string uuid;
  storage_config config;
  std::shared_ptr<rocstorage::data_storage::database> database;
  std::shared_ptr<rocstorage::writer> writer;
  mutable std::shared_ptr<rocstorage::reader> reader;
  mutable std::vector<track_view> tracks;
  mutable bool loaded = false;
};

storage::storage() : impl_(std::make_unique<Impl>()) {}

storage::storage(std::string database_path, std::string uuid)
    : impl_(std::make_unique<Impl>()) {
  impl_->path = std::move(database_path);
  impl_->uuid = std::move(uuid);
  impl_->database = std::make_shared<rocstorage::data_storage::database>(
      impl_->path, impl_->uuid);
  if (!impl_->database) {
    throw std::invalid_argument("Unable to create database!");
  }
  impl_->writer =
      std::shared_ptr<rocstorage::writer>(new rocstorage::writer(impl_->database, impl_->uuid));
}

std::unique_ptr<storage> storage::open(const std::string &path) {
  auto s = std::unique_ptr<storage>(new storage());
  s->impl_->path = path;
  s->impl_->config = storage_config::read_only();
  s->impl_->reader = rocstorage::reader::open(path);
  if (!s->impl_->reader) {
    return nullptr;
  }
  return s;
}

std::unique_ptr<storage> storage::create(const std::string &path,
                                         const std::string &uuid,
                                         const storage_config &config) {
  auto s = std::unique_ptr<storage>(new storage());
  s->impl_->path = path;
  s->impl_->uuid = uuid;
  s->impl_->config = config;

  switch (config.storage_mode) {
  case storage_config::mode::read:
    // Read-only mode: just open for reading
    s->impl_->reader = rocstorage::reader::open(path);
    if (!s->impl_->reader) {
      return nullptr;
    }
    break;

  case storage_config::mode::write:
    // Write-only mode: in-memory database, flush at end
    s->impl_->database = std::make_shared<rocstorage::data_storage::database>(
        path, uuid, rocstorage::data_storage::database_mode::in_memory);
    if (!s->impl_->database) {
      return nullptr;
    }
    s->impl_->writer = std::shared_ptr<rocstorage::writer>(
        new rocstorage::writer(s->impl_->database, uuid));
    break;

  case storage_config::mode::read_write:
    // Read-write mode: file-based WAL for concurrent access
    s->impl_->database = std::make_shared<rocstorage::data_storage::database>(
        path, uuid, rocstorage::data_storage::database_mode::wal);
    if (!s->impl_->database) {
      return nullptr;
    }
    s->impl_->writer = std::shared_ptr<rocstorage::writer>(
        new rocstorage::writer(s->impl_->database, uuid));
    break;

  case storage_config::mode::unknown:
  default:
    throw std::invalid_argument(
        "Invalid storage mode. Use write_only(), read_write(), or read_only().");
  }

  return s;
}

storage::~storage() = default;

bool storage::load() {
  if (impl_->loaded) {
    return true;
  }

  if (!impl_->reader) {
    impl_->reader = rocstorage::reader::open(impl_->path);
  }

  if (!impl_->reader) {
    return false;
  }

  if (!impl_->reader->read_metadata()) {
    return false;
  }

  uint64_t n = impl_->reader->num_tracks();
  impl_->tracks.clear();
  impl_->tracks.reserve(n);

  for (uint64_t i = 0; i < n; ++i) {
    auto t = impl_->reader->get_track(i);
    if (t) {
      track_view tv;
      tv.id_ = t->id();
      tv.category_ = t->category();
      tv.category_string_ = t->category_string();
      tv.node_id_ = t->node_id();
      tv.process_name_ = t->process_name();
      tv.subprocess_name_ = t->subprocess_name();
      tv.num_records_ = t->num_records();
      tv.min_timestamp_ = t->min_timestamp();
      tv.max_timestamp_ = t->max_timestamp();
      impl_->tracks.push_back(std::move(tv));
    }
  }

  impl_->loaded = true;
  return true;
}

bool storage::is_loaded() const { return impl_->loaded; }

uint64_t storage::start_time() const {
  if (!impl_->reader) {
    return 0;
  }
  return impl_->reader->start_time();
}

uint64_t storage::end_time() const {
  if (!impl_->reader) {
    return 0;
  }
  return impl_->reader->end_time();
}

size_t storage::num_tracks() const { return impl_->tracks.size(); }

const std::vector<track_view> &storage::tracks() const { return impl_->tracks; }

const track_view &storage::track(size_t index) const {
  if (index >= impl_->tracks.size()) {
    throw std::out_of_range("Track index out of range");
  }
  return impl_->tracks[index];
}

std::string_view storage::path() const { return impl_->path; }

std::shared_ptr<rocstorage::writer> storage::get_writer() const {
  return impl_->writer;
}

std::shared_ptr<rocstorage::reader> storage::get_reader() const {
  if (!impl_->reader && !impl_->path.empty()) {
    // WARNING: This is a workaround, not a proper solution.
    //
    // The writer uses an in-memory SQLite database (data_storage::database)
    // that gets flushed to disk once via sqlite3_backup. The reader uses a
    // completely separate database layer (RocProfVis::DataModel::Database)
    // that opens the on-disk file with connection pooling.
    //
    // These two layers do not share a connection, so we must:
    // 1. Flush the writer's in-memory DB to disk
    // 2. Open a new reader connection to that file
    //
    // Limitations:
    // - Caller cannot read data until after flush (writes not visible)
    // - flush() can only be called once, so this only works once
    // - Reader and writer are never truly connected
    //
    // A proper solution would allow the reader to use the writer's sqlite3*
    // handle directly, enabling read-write mode without flushing.
    if (impl_->writer) {
      impl_->writer->flush();
    }
    impl_->reader = rocstorage::reader::open(impl_->path);
  }
  return impl_->reader;
}

} // namespace rocm
