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

#pragma once

#include "database.hpp"

#include <cstdint>
#include <functional>
#include <memory>
#include <string>

namespace rocstorage {
namespace data_storage {

struct insert_statements {
  explicit insert_statements(std::shared_ptr<database> database,
                             std::string uuid);
  insert_statements() = delete;
  insert_statements(const insert_statements &) = delete;
  insert_statements(insert_statements &&) = delete;
  insert_statements &operator=(const insert_statements &) = delete;
  insert_statements &operator=(insert_statements &&) = delete;
  ~insert_statements() = default;

  using insert_event_statement =
      std::function<void(const char *, size_t, size_t, size_t, size_t,
                         const char *, const char *, const char *)>;
  using insert_pmc_event_statement =
      std::function<void(const char *, size_t, size_t, double, const char *)>;
  using insert_sample_statement =
      std::function<void(const char *, size_t, uint64_t, size_t, const char *)>;
  using insert_region_statement =
      std::function<void(const char *, size_t, size_t, size_t, uint64_t,
                         uint64_t, size_t, size_t, const char *)>;
  using insert_kernel_dispatch_statement = std::function<void(
      const char *, size_t, size_t, size_t, size_t, size_t, size_t, size_t,
      size_t, uint64_t, uint64_t, size_t, size_t, size_t, size_t, size_t,
      size_t, size_t, size_t, size_t, size_t, const char *)>;
  using insert_memory_copy_statement =
      std::function<void(const char *, size_t, size_t, size_t, uint64_t,
                         uint64_t, size_t, size_t, size_t, size_t, size_t,
                         size_t, size_t, size_t, size_t, size_t, const char *)>;
  using insert_memory_alloc_statement =
      std::function<void(const char *, size_t, size_t, size_t, size_t,
                         const char *, const char *, uint64_t, uint64_t, size_t,
                         size_t, size_t, size_t, size_t, const char *)>;
  using insert_memory_alloc_no_agent_statement =
      std::function<void(const char *, size_t, size_t, size_t, const char *,
                         const char *, uint64_t, uint64_t, size_t, size_t,
                         size_t, size_t, size_t, const char *)>;
  using insert_kernel_symbol_statement = std::function<void(
      size_t, const char *, size_t, size_t, uint64_t, const char *,
      const char *, uint64_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t,
      uint32_t, uint32_t, const char *)>;
  using insert_code_object_statement = std::function<void(
      size_t, const char *, size_t, size_t, size_t, const char *, uint64_t,
      uint64_t, uint64_t, const char *, const char *)>;
  using insert_args_statement =
      std::function<void(const char *, size_t, size_t, const char *,
                         const char *, const char *, const char *)>;
  using insert_string_statement =
      std::function<void(const char *, const char *)>;

private:
  void initialize_pmc_event_stmt();
  void initialize_event_stmt();
  void initialize_sample_stmt();
  void initialize_region_stmt();
  void initialize_kernel_dispatch_stmt();
  void initialize_memory_copy_stmt();
  void initialize_kernel_symbol_stmt();
  void initialize_code_object_stmt();
  void initialize_args_stmt();
  void initialize_memory_alloc_stmt();
  void initialize_string_stmt();

private:
  std::shared_ptr<database> m_database;
  std::string m_uuid;

public:
  insert_event_statement m_insert_event_statement;
  insert_pmc_event_statement m_insert_pmc_event_statement;
  insert_sample_statement m_insert_sample_statement;
  insert_region_statement m_insert_region_statement;
  insert_kernel_dispatch_statement m_insert_kernel_dispatch_statement;
  insert_memory_copy_statement m_insert_memory_copy_statement;
  insert_kernel_symbol_statement m_insert_kernel_symbol_statement;
  insert_code_object_statement m_insert_code_object_statement;
  insert_args_statement m_insert_args_statement;
  insert_memory_alloc_statement m_insert_memory_alloc_statement;
  insert_memory_alloc_no_agent_statement
      m_insert_memory_alloc_no_agent_statement;
  insert_string_statement m_insert_string_statement;
};

} // namespace data_storage
} // namespace rocstorage
