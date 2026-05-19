/******************************************************************************
 * Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 *****************************************************************************/

#ifndef ROCSHMEM_ENVVAR_GTEST_HPP
#define ROCSHMEM_ENVVAR_GTEST_HPP

#include <cstdlib>
#include <sstream>
#include <string>
#include <type_traits>

#include "gtest/gtest.h"
#include "../src/envvar.hpp"

namespace rocshmem {

class EnvVarTestFixture : public ::testing::Test
{
  public:
    static constexpr envvar::category::tag category_ = envvar::category::tag::ROCSHMEM;
    static constexpr const char* var_name_ = "GTEST";
    static constexpr const char* var_prefix_ = envvar::category::prefix<category_>;
    static inline const std::string var_full_name_{std::string(var_prefix_).append("_").append(var_name_)};
    static constexpr const char* var_doc_ = "Test envvar documentation: documents the test envvar.";

    static int setenv(const char* value) {
      return ::setenv(var_full_name_.c_str(), value, true);
    }

    static int setenv(const std::string& value) {
      return setenv(value.c_str());
    }

    static int unsetenv() {
      return ::unsetenv(var_full_name_.c_str());
    }

  protected:
    static void SetUpTestSuite() {
      unsetenv();
    }

    static void TearDownTestSuite() {
      unsetenv();
    }

    void SetUp() override {
      unsetenv();
    }

    void TearDown() override {
      unsetenv();
    }
};

template <typename T>
class EnvVarUnsetTestFixture : public EnvVarTestFixture
{
  protected:
    const envvar::var<T> var_{var_name_, var_doc_};
};

template <typename T>
class EnvVarSetTestFixture : public EnvVarTestFixture
{
  protected:
    const envvar::var<T> var_{var_name_, var_doc_};

    static void SetUpTestSuite() {
      using namespace envvar::parser::_type_traits;
      std::ostringstream oss{};
      // operator<<(std::ostream&, {un}signed char) prints output as a character
      // so signed or unsigned char need to be widened to a larger type
      if constexpr (is_standard_integer_v<T> && is_narrow_character_v<T>) {
        using parsechar_t = std::conditional_t<std::is_signed_v<T>, signed int, unsigned int>;
        oss << static_cast<parsechar_t>(T{});
      } else {
        oss << T{};
      }
      setenv(oss.str());
    }
};
} // namespace rocshmem

#endif // ROCSHMEM_ENVVAR_GTEST_HPP
