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
#include <rocstorage/writer.hpp>

#include "data_storage/database.hpp"

namespace rocm {

struct storage::impl {
  explicit impl(std::string database_path, const std::string &uuid)
      : m_database(std::make_shared<rocstorage::data_storage::database>(
            database_path, uuid)),
        m_uuid(uuid), m_writer(new rocstorage::writer(m_database, m_uuid)) {
    if (!m_database) {
      throw std::invalid_argument("Unable to create database!");
    }
  }

  std::shared_ptr<rocstorage::data_storage::database> m_database;
  std::string m_uuid;
  std::shared_ptr<rocstorage::writer> m_writer;
};

storage::storage(std::string database_path, std::string uuid)
    : m_impl(
          std::make_unique<impl>(std::move(database_path), std::move(uuid))) {}

storage::~storage() { m_impl.reset(); }

std::shared_ptr<rocstorage::writer> storage::get_writer() const {
  return m_impl->m_writer;
}

std::shared_ptr<rocstorage::reader> storage::get_reader() const {
  // Reader facade not yet implemented.
  // Use rocstorage::reader::open() directly for now.
  return nullptr;
}

} // namespace rocm
