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

#include <gtest/gtest.h>

#include <cstdio>
#include <memory>
#include <string>

namespace {

class storage_test : public ::testing::Test {
protected:
  void SetUp() override {
    m_database_path =
        "test_storage_" +
        std::to_string(
            ::testing::UnitTest::GetInstance()->current_test_info()->line()) +
        ".db";
    m_uuid = "12345678";
  }

  void TearDown() override { std::remove(m_database_path.c_str()); }

  std::string m_database_path;
  std::string m_uuid;
};

TEST_F(storage_test, create_storage_instance) {
  auto storage = std::make_unique<rocm::storage>(m_database_path, m_uuid);
  ASSERT_NE(storage, nullptr);
}

TEST_F(storage_test, get_writer_returns_valid_ptr) {
  auto storage = std::make_unique<rocm::storage>(m_database_path, m_uuid);
  auto writer = storage->get_writer();
  ASSERT_NE(writer, nullptr);
}

TEST_F(storage_test, get_reader_returns_valid_ptr) {
  auto storage = std::make_unique<rocm::storage>(m_database_path, m_uuid);
  auto reader = storage->get_reader();
  ASSERT_NE(reader, nullptr);
}

TEST_F(storage_test, multiple_get_writer_returns_same_instance) {
  auto storage = std::make_unique<rocm::storage>(m_database_path, m_uuid);
  auto writer1 = storage->get_writer();
  auto writer2 = storage->get_writer();
  EXPECT_EQ(writer1.get(), writer2.get());
}

TEST_F(storage_test, multiple_get_reader_returns_same_instance) {
  auto storage = std::make_unique<rocm::storage>(m_database_path, m_uuid);
  auto reader1 = storage->get_reader();
  auto reader2 = storage->get_reader();
  EXPECT_EQ(reader1.get(), reader2.get());
}

} // namespace
