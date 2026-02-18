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

#include <limits>

#include "envvar_gtest.hpp"

using namespace rocshmem;

using VarTypes = ::testing::Types<bool,
                                  uint32_t,
                                  size_t,
                                  int64_t,
                                  std::string,
                                  envvar::types::socket_family,
                                  envvar::types::debug_level>;

TYPED_TEST_SUITE(EnvVarUnsetTestFixture, VarTypes);
TYPED_TEST_SUITE(EnvVarSetTestFixture, VarTypes);

TYPED_TEST(EnvVarUnsetTestFixture, name) {
  EXPECT_EQ(this->var_.get_name(), this->var_full_name_);
}

TYPED_TEST(EnvVarUnsetTestFixture, doc) {
  EXPECT_EQ(this->var_.get_doc(), this->var_doc_);
}

TYPED_TEST(EnvVarUnsetTestFixture, is_default) {
  EXPECT_TRUE(this->var_.is_default());
  EXPECT_EQ(this->var_.get_value(), this->var_.get_default());
}

TYPED_TEST(EnvVarSetTestFixture, is_default) {
  EXPECT_FALSE(this->var_.is_default());
  EXPECT_EQ(this->var_.get_value(), this->var_.get_default());
}

TEST_F(EnvVarTestFixture, string_custom_default) {
  const std::string default_value_{"This is the default value."};
  const envvar::var<std::string> var_{this->var_name_, this->var_doc_, default_value_};
  EXPECT_TRUE(var_.is_default());
  EXPECT_EQ(var_.get_default(), default_value_);
}

TEST_F(EnvVarTestFixture, parse_integer) {
  this->setenv("1073741824");
  const envvar::var<int64_t> var_{this->var_name_, this->var_doc_};
  EXPECT_FALSE(var_.is_default());
  EXPECT_EQ(var_.get_default(), 0);
  EXPECT_EQ(var_.get_value(), 1L << 30);
}

TEST_F(EnvVarTestFixture, parse_integer_notaninteger) {
  this->setenv("Ceci n'est pas un entier.");
  const envvar::var<int64_t> var_{this->var_name_, this->var_doc_, 1L << 30};
  EXPECT_TRUE(var_.is_default());
  EXPECT_EQ(var_.get_value(), 1L << 30);
}

TEST_F(EnvVarTestFixture, parse_integer_large) {
  this->setenv("9223372036854775807");
  const envvar::var<int64_t> var_{this->var_name_, this->var_doc_};
  EXPECT_FALSE(var_.is_default());
  EXPECT_EQ(var_.get_default(), 0);
  EXPECT_EQ(var_.get_value(), std::numeric_limits<int64_t>::max());
}

TEST_F(EnvVarTestFixture, parse_integer_too_large) {
  this->setenv("9223372036854775808");
  const envvar::var<int64_t> var_{this->var_name_, this->var_doc_};
  EXPECT_TRUE(var_.is_default());
  EXPECT_EQ(var_.get_default(), 0);
  EXPECT_EQ(var_.get_value(), var_.get_default());
}

TEST_F(EnvVarTestFixture, parse_integer_negative) {
  this->setenv("-1073741824");
  const envvar::var<int64_t> var_{this->var_name_, this->var_doc_};
  EXPECT_FALSE(var_.is_default());
  EXPECT_EQ(var_.get_default(), 0);
  EXPECT_EQ(var_.get_value(), -1L << 30);
}

TEST_F(EnvVarTestFixture, parse_integer_negative_large) {
  this->setenv("-9223372036854775808");
  const envvar::var<int64_t> var_{this->var_name_, this->var_doc_};
  EXPECT_FALSE(var_.is_default());
  EXPECT_EQ(var_.get_default(), 0);
  EXPECT_EQ(var_.get_value(), std::numeric_limits<int64_t>::min());
}

TEST_F(EnvVarTestFixture, parse_integer_negative_too_large) {
  this->setenv("-9223372036854775809");
  const envvar::var<int64_t> var_{this->var_name_, this->var_doc_};
  EXPECT_TRUE(var_.is_default());
  EXPECT_EQ(var_.get_default(), 0);
  EXPECT_EQ(var_.get_value(), var_.get_default());
}

TEST_F(EnvVarTestFixture, parse_integer_hex) {
  this->setenv("0x40000000");
  const envvar::var<int64_t> var_{this->var_name_, this->var_doc_};
  EXPECT_FALSE(var_.is_default());
  EXPECT_EQ(var_.get_default(), 0);
  EXPECT_EQ(var_.get_value(), 1L << 30);
}

TEST_F(EnvVarTestFixture, parse_integer_hex_only) {
  this->setenv("0x40000000");
  const envvar::var<int64_t> var_{this->var_name_, this->var_doc_,
                                  envvar::parser::parse_hex<int64_t>{}};
  EXPECT_FALSE(var_.is_default());
  EXPECT_EQ(var_.get_default(), 0);
  EXPECT_EQ(var_.get_value(), 1L << 30);
}

// parse_hex<> interprets input digits as hexadecimal, even without 0x prefix
TEST_F(EnvVarTestFixture, parse_integer_hex_only_noprefix) {
  this->setenv("40000000");
  const envvar::var<int64_t> var_{this->var_name_, this->var_doc_,
                                  envvar::parser::parse_hex<int64_t>{}};
  EXPECT_FALSE(var_.is_default());
  EXPECT_EQ(var_.get_default(), 0);
  EXPECT_EQ(var_.get_value(), 1L << 30);
}

TEST_F(EnvVarTestFixture, parse_unsigned_integer) {
  this->setenv("1073741824");
  const envvar::var<uint32_t> var_{this->var_name_, this->var_doc_};
  EXPECT_FALSE(var_.is_default());
  EXPECT_EQ(var_.get_default(), 0);
  EXPECT_EQ(var_.get_value(), 1L << 30);
}

TEST_F(EnvVarTestFixture, parse_unsigned_integer_negative) {
  this->setenv("-1");
  const envvar::var<uint32_t> var_{this->var_name_, this->var_doc_};
  EXPECT_TRUE(var_.is_default());
  EXPECT_EQ(var_.get_default(), 0);
  EXPECT_EQ(var_.get_value(), var_.get_default());
}

TEST_F(EnvVarTestFixture, parse_unsigned_large) {
  this->setenv("4294967296");
  const envvar::var<uint32_t> var_{this->var_name_, this->var_doc_};
  EXPECT_TRUE(var_.is_default());
  EXPECT_EQ(var_.get_default(), 0);
  EXPECT_EQ(var_.get_value(), var_.get_default());
}

TEST_F(EnvVarTestFixture, parse_bool_zero) {
  this->setenv("0");
  const envvar::var<bool> var_{this->var_name_, this->var_doc_};
  EXPECT_FALSE(var_.is_default());
  EXPECT_EQ(var_.get_default(), false);
  EXPECT_EQ(var_.get_value(), false);
}

TEST_F(EnvVarTestFixture, parse_bool_one) {
  this->setenv("1");
  const envvar::var<bool> var_{this->var_name_, this->var_doc_};
  EXPECT_FALSE(var_.is_default());
  EXPECT_EQ(var_.get_default(), false);
  EXPECT_EQ(var_.get_value(), true);
}

TEST_F(EnvVarTestFixture, parse_bool_negative) {
  this->setenv("-1");
  const envvar::var<bool> var_{this->var_name_, this->var_doc_};
  EXPECT_TRUE(var_.is_default());
  EXPECT_EQ(var_.get_default(), false);
  EXPECT_EQ(var_.get_value(), false);
}

TEST_F(EnvVarTestFixture, parse_bool_two) {
  this->setenv("2");
  const envvar::var<bool> var_{this->var_name_, this->var_doc_};
  EXPECT_TRUE(var_.is_default());
  EXPECT_EQ(var_.get_default(), false);
  EXPECT_EQ(var_.get_value(), false);
}

TEST_F(EnvVarTestFixture, parse_bool_false) {
  this->setenv("false");
  const envvar::var<bool> var_{this->var_name_, this->var_doc_};
  EXPECT_FALSE(var_.is_default());
  EXPECT_EQ(var_.get_default(), false);
  EXPECT_EQ(var_.get_value(), false);
}

TEST_F(EnvVarTestFixture, parse_bool_true) {
  this->setenv("true");
  const envvar::var<bool> var_{this->var_name_, this->var_doc_};
  EXPECT_FALSE(var_.is_default());
  EXPECT_EQ(var_.get_default(), false);
  EXPECT_EQ(var_.get_value(), true);
}

TEST_F(EnvVarTestFixture, parse_bool_other) {
  this->setenv("other");
  const envvar::var<bool> var_{this->var_name_, this->var_doc_};
  EXPECT_TRUE(var_.is_default());
  EXPECT_EQ(var_.get_default(), false);
  EXPECT_EQ(var_.get_value(), false);
}

TEST_F(EnvVarTestFixture, parse_socket_family_unspec) {
  this->setenv("UNSPEC");
  const envvar::var<envvar::types::socket_family> var_{this->var_name_, this->var_doc_};
  EXPECT_FALSE(var_.is_default());
  EXPECT_EQ(var_.get_default(), envvar::types::socket_family::UNSPEC);
  EXPECT_EQ(var_.get_value(), envvar::types::socket_family::UNSPEC);
}

TEST_F(EnvVarTestFixture, parse_socket_family_af_unspec) {
  this->setenv("AF_UNSPEC");
  const envvar::var<envvar::types::socket_family> var_{this->var_name_, this->var_doc_};
  EXPECT_FALSE(var_.is_default());
  EXPECT_EQ(var_.get_default(), envvar::types::socket_family::UNSPEC);
  EXPECT_EQ(var_.get_value(), envvar::types::socket_family::UNSPEC);
}

TEST_F(EnvVarTestFixture, parse_socket_family_inet) {
  this->setenv("INET");
  const envvar::var<envvar::types::socket_family> var_{this->var_name_, this->var_doc_};
  EXPECT_FALSE(var_.is_default());
  EXPECT_EQ(var_.get_default(), envvar::types::socket_family::UNSPEC);
  EXPECT_EQ(var_.get_value(), envvar::types::socket_family::INET);
}

TEST_F(EnvVarTestFixture, parse_socket_family_af_inet) {
  this->setenv("AF_INET");
  const envvar::var<envvar::types::socket_family> var_{this->var_name_, this->var_doc_};
  EXPECT_FALSE(var_.is_default());
  EXPECT_EQ(var_.get_default(), envvar::types::socket_family::UNSPEC);
  EXPECT_EQ(var_.get_value(), envvar::types::socket_family::INET);
}

TEST_F(EnvVarTestFixture, parse_socket_family_inet6) {
  this->setenv("INET6");
  const envvar::var<envvar::types::socket_family> var_{this->var_name_, this->var_doc_};
  EXPECT_FALSE(var_.is_default());
  EXPECT_EQ(var_.get_default(), envvar::types::socket_family::UNSPEC);
  EXPECT_EQ(var_.get_value(), envvar::types::socket_family::INET6);
}

TEST_F(EnvVarTestFixture, parse_socket_family_af_inet6) {
  this->setenv("AF_INET6");
  const envvar::var<envvar::types::socket_family> var_{this->var_name_, this->var_doc_};
  EXPECT_FALSE(var_.is_default());
  EXPECT_EQ(var_.get_default(), envvar::types::socket_family::UNSPEC);
  EXPECT_EQ(var_.get_value(), envvar::types::socket_family::INET6);
}

TEST_F(EnvVarTestFixture, parse_socket_family_nonesense) {
  this->setenv("'Twas brillig, and the slithy toves, did gyre and gimble in the wabe.");
  const envvar::var<envvar::types::socket_family> var_{this->var_name_, this->var_doc_};
  EXPECT_TRUE(var_.is_default());
  EXPECT_EQ(var_.get_default(), envvar::types::socket_family::UNSPEC);
  EXPECT_EQ(var_.get_value(), var_.get_default());
}
