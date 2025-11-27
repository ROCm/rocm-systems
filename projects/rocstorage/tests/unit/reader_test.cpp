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

#include <rocstorage/reader.hpp>
#include <rocstorage/storage.hpp>

#include <gtest/gtest.h>

#include <cstdio>
#include <memory>
#include <string>

namespace {

class reader_test : public ::testing::Test {
protected:
  void SetUp() override {
    m_database_path =
        "test_reader_" +
        std::to_string(
            ::testing::UnitTest::GetInstance()->current_test_info()->line()) +
        ".db";
    m_uuid = "12345";
    m_storage = std::make_unique<rocm::storage>(m_database_path, m_uuid);
    m_reader = m_storage->get_reader();
  }

  void TearDown() override {
    m_reader.reset();
    m_storage.reset();
    std::remove(m_database_path.c_str());
  }

  std::string m_database_path;
  std::string m_uuid;
  std::unique_ptr<rocm::storage> m_storage;
  std::shared_ptr<rocstorage::reader> m_reader;
};

TEST_F(reader_test, reader_instance_is_valid) { ASSERT_NE(m_reader, nullptr); }

TEST_F(reader_test, reader_can_be_retrieved_multiple_times) {
  auto reader1 = m_storage->get_reader();
  auto reader2 = m_storage->get_reader();

  ASSERT_NE(reader1, nullptr);
  ASSERT_NE(reader2, nullptr);
  EXPECT_EQ(reader1.get(), reader2.get());
}

} // namespace
