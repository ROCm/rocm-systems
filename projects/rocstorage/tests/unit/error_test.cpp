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

#include "common/error.hpp"
#include <rocstorage/result.hpp>

#include <gtest/gtest.h>

using namespace rocstorage;

// ============================================================================
// error class tests
// ============================================================================

TEST(ErrorTest, ConstructWithCodeOnly) {
  error e(error_code::invalid_parameter);
  EXPECT_EQ(e.code(), error_code::invalid_parameter);
  EXPECT_TRUE(e.message().empty());
  EXPECT_TRUE(e.query().empty());
  EXPECT_EQ(e.sqlite_code(), 0);
}

TEST(ErrorTest, ConstructWithCodeAndMessage) {
  error e(error_code::db_access_failed, "connection refused");
  EXPECT_EQ(e.code(), error_code::db_access_failed);
  EXPECT_EQ(e.message(), "connection refused");
  EXPECT_TRUE(e.query().empty());
}

TEST(ErrorTest, ConstructWithFullContext) {
  error e(error_code::query_error, "syntax error", "SELECT * FROM", 1);
  EXPECT_EQ(e.code(), error_code::query_error);
  EXPECT_EQ(e.message(), "syntax error");
  EXPECT_EQ(e.query(), "SELECT * FROM");
  EXPECT_EQ(e.sqlite_code(), 1);
}

TEST(ErrorTest, ToCResultMapsCorrectly) {
  // Test direct mappings (codes 0-10)
  EXPECT_EQ(error(error_code::success).to_c_result(), 0);
  EXPECT_EQ(error(error_code::unknown_error).to_c_result(), 1);
  EXPECT_EQ(error(error_code::timeout).to_c_result(), 2);
  EXPECT_EQ(error(error_code::not_loaded).to_c_result(), 3);
  EXPECT_EQ(error(error_code::alloc_failure).to_c_result(), 4);
  EXPECT_EQ(error(error_code::invalid_parameter).to_c_result(), 5);
  EXPECT_EQ(error(error_code::db_access_failed).to_c_result(), 6);
  EXPECT_EQ(error(error_code::invalid_property).to_c_result(), 7);
  EXPECT_EQ(error(error_code::not_supported).to_c_result(), 8);
  EXPECT_EQ(error(error_code::resource_busy).to_c_result(), 9);
  EXPECT_EQ(error(error_code::db_abort).to_c_result(), 10);
}

TEST(ErrorTest, ExtendedCodesMappedToBaseTypes) {
  // Extended codes should map to appropriate base types
  EXPECT_EQ(error(error_code::file_not_found).to_c_result(),
            6); // db_access_failed
  EXPECT_EQ(error(error_code::invalid_database).to_c_result(),
            6); // db_access_failed
  EXPECT_EQ(error(error_code::constraint_violation).to_c_result(),
            6); // db_access_failed
  EXPECT_EQ(error(error_code::index_out_of_bounds).to_c_result(),
            5); // invalid_parameter
}

TEST(ErrorTest, ToStringFormatsCorrectly) {
  error e(error_code::db_access_failed, "connection failed", "OPEN db.sqlite",
          14);
  std::string s = e.to_string();

  EXPECT_NE(s.find("db_access_failed"), std::string::npos);
  EXPECT_NE(s.find("connection failed"), std::string::npos);
  EXPECT_NE(s.find("OPEN db.sqlite"), std::string::npos);
  EXPECT_NE(s.find("14"), std::string::npos);
}

TEST(ErrorTest, FromCResult) {
  auto e = from_c_result(5, "test message");
  EXPECT_EQ(e.code(), error_code::invalid_parameter);
  EXPECT_EQ(e.message(), "test message");
}

// ============================================================================
// result<T> tests
// ============================================================================

TEST(ResultTest, ConstructWithValue) {
  result<int> r(42);
  EXPECT_TRUE(r);
  EXPECT_TRUE(r.has_value());
  EXPECT_FALSE(r.has_error());
  EXPECT_EQ(r.value(), 42);
  EXPECT_EQ(*r, 42);
}

TEST(ResultTest, ConstructWithError) {
  result<int> r(error(error_code::invalid_parameter, "bad input"));
  EXPECT_FALSE(r);
  EXPECT_FALSE(r.has_value());
  EXPECT_TRUE(r.has_error());
  EXPECT_EQ(r.get_error().code(), error_code::invalid_parameter);
}

TEST(ResultTest, ConstructWithErrorCodeDirectly) {
  result<int> r(error_code::timeout, "operation timed out");
  EXPECT_FALSE(r);
  EXPECT_EQ(r.get_error().code(), error_code::timeout);
  EXPECT_EQ(r.get_error().message(), "operation timed out");
}

TEST(ResultTest, ValueOrReturnsValueWhenPresent) {
  result<int> r(42);
  EXPECT_EQ(r.value_or(0), 42);
}

TEST(ResultTest, ValueOrReturnsDefaultWhenError) {
  result<int> r(error_code::invalid_parameter);
  EXPECT_EQ(r.value_or(99), 99);
}

TEST(ResultTest, ValueThrowsWhenError) {
  result<int> r(error_code::invalid_parameter);
  EXPECT_THROW(r.value(), std::runtime_error);
}

TEST(ResultTest, ErrorThrowsWhenValue) {
  result<int> r(42);
  EXPECT_THROW(r.get_error(), std::logic_error);
}

TEST(ResultTest, PointerOperators) {
  struct Data {
    int x;
  };
  result<Data> r(Data{42});
  EXPECT_EQ(r->x, 42);
}

TEST(ResultTest, MoveSemantics) {
  result<std::string> r("hello");
  std::string s = std::move(r).value();
  EXPECT_EQ(s, "hello");
}

TEST(ResultTest, MapTransformsValue) {
  result<int> r(5);
  auto r2 = r.map([](int x) { return x * 2; });
  EXPECT_TRUE(r2);
  EXPECT_EQ(r2.value(), 10);
}

TEST(ResultTest, MapPreservesError) {
  result<int> r(error_code::invalid_parameter);
  auto r2 = r.map([](int x) { return x * 2; });
  EXPECT_FALSE(r2);
  EXPECT_EQ(r2.get_error().code(), error_code::invalid_parameter);
}

TEST(ResultTest, AndThenChainsOperations) {
  auto divide = [](int x) -> result<int> {
    if (x == 0)
      return error(error_code::invalid_parameter, "division by zero");
    return 100 / x;
  };

  result<int> r(5);
  auto r2 = r.and_then(divide);
  EXPECT_TRUE(r2);
  EXPECT_EQ(r2.value(), 20);
}

TEST(ResultTest, AndThenPropagatesError) {
  auto divide = [](int x) -> result<int> {
    if (x == 0)
      return error(error_code::invalid_parameter, "division by zero");
    return 100 / x;
  };

  result<int> r(error_code::timeout);
  auto r2 = r.and_then(divide);
  EXPECT_FALSE(r2);
  EXPECT_EQ(r2.get_error().code(), error_code::timeout);
}

// ============================================================================
// result<void> (status) tests
// ============================================================================

TEST(StatusTest, DefaultConstructIsSuccess) {
  status s;
  EXPECT_TRUE(s);
  EXPECT_TRUE(s.has_value());
  EXPECT_FALSE(s.has_error());
}

TEST(StatusTest, ConstructWithError) {
  status s(error(error_code::db_access_failed));
  EXPECT_FALSE(s);
  EXPECT_FALSE(s.has_value());
  EXPECT_TRUE(s.has_error());
  EXPECT_EQ(s.get_error().code(), error_code::db_access_failed);
}

TEST(StatusTest, ConstructWithErrorCode) {
  status s(error_code::timeout, "operation timed out");
  EXPECT_FALSE(s);
  EXPECT_EQ(s.get_error().code(), error_code::timeout);
}

TEST(StatusTest, ErrorThrowsWhenSuccess) {
  status s;
  EXPECT_THROW(s.get_error(), std::logic_error);
}

// ============================================================================
// Helper function tests
// ============================================================================

TEST(HelpersTest, OkCreatesSuccessValue) {
  auto r = ok(42);
  EXPECT_TRUE(r);
  EXPECT_EQ(r.value(), 42);
}

TEST(HelpersTest, OkVoidCreatesSuccessStatus) {
  auto s = ok();
  EXPECT_TRUE(s);
}

TEST(HelpersTest, ErrCreatesErrorResult) {
  auto r = err<int>(error_code::invalid_parameter, "bad");
  EXPECT_FALSE(r);
  EXPECT_EQ(r.get_error().code(), error_code::invalid_parameter);
}

TEST(HelpersTest, ErrVoidCreatesErrorStatus) {
  auto s = err(error_code::timeout);
  EXPECT_FALSE(s);
  EXPECT_EQ(s.get_error().code(), error_code::timeout);
}

// ============================================================================
// Practical usage pattern tests
// ============================================================================

namespace {
result<int> safe_divide(int a, int b) {
  if (b == 0) {
    return error(error_code::invalid_parameter, "division by zero");
  }
  return a / b;
}

status validate_positive(int x) {
  if (x <= 0) {
    return error(error_code::invalid_parameter, "value must be positive");
  }
  return ok();
}
} // namespace

TEST(UsagePatternTest, FunctionReturningResultSuccess) {
  auto r = safe_divide(10, 2);
  EXPECT_TRUE(r);
  EXPECT_EQ(r.value(), 5);
}

TEST(UsagePatternTest, FunctionReturningResultError) {
  auto r = safe_divide(10, 0);
  EXPECT_FALSE(r);
  EXPECT_EQ(r.get_error().code(), error_code::invalid_parameter);
  EXPECT_EQ(r.get_error().message(), "division by zero");
}

TEST(UsagePatternTest, FunctionReturningStatusSuccess) {
  auto s = validate_positive(5);
  EXPECT_TRUE(s);
}

TEST(UsagePatternTest, FunctionReturningStatusError) {
  auto s = validate_positive(-1);
  EXPECT_FALSE(s);
  EXPECT_EQ(s.get_error().code(), error_code::invalid_parameter);
}

TEST(UsagePatternTest, ChainedOperations) {
  auto r = safe_divide(100, 5)
               .and_then([](int x) { return safe_divide(x, 2); })
               .and_then([](int x) { return safe_divide(x, 2); });

  EXPECT_TRUE(r);
  EXPECT_EQ(r.value(), 5); // 100 / 5 / 2 / 2 = 5
}

TEST(UsagePatternTest, ChainedOperationsWithError) {
  auto r = safe_divide(100, 5)
               .and_then([](int x) { return safe_divide(x, 0); }) // Error here
               .and_then([](int x) { return safe_divide(x, 2); });

  EXPECT_FALSE(r);
  EXPECT_EQ(r.get_error().code(), error_code::invalid_parameter);
}
