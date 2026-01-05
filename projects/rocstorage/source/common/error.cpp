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

#include "error.hpp"

#include <sstream>

namespace rocstorage {

error::error(error_code code) noexcept : code_(code) {}

error::error(error_code code, std::string message)
    : code_(code), message_(std::move(message)) {}

error::error(error_code code, std::string message, std::string query)
    : code_(code), message_(std::move(message)), query_(std::move(query)) {}

error::error(error_code code, std::string message, std::string query,
             int sqlite_code)
    : code_(code), message_(std::move(message)), query_(std::move(query)),
      sqlite_code_(sqlite_code) {}

int error::to_c_result() const noexcept {
  // Map extended error codes to base C API codes
  switch (code_) {
  case error_code::success:
    return 0; // kRocProfVisDmResultSuccess
  case error_code::unknown_error:
    return 1; // kRocProfVisDmResultUnknownError
  case error_code::timeout:
    return 2; // kRocProfVisDmResultTimeout
  case error_code::not_loaded:
    return 3; // kRocProfVisDmResultNotLoaded
  case error_code::alloc_failure:
    return 4; // kRocProfVisDmResultAllocFailure
  case error_code::invalid_parameter:
    return 5; // kRocProfVisDmResultInvalidParameter
  case error_code::db_access_failed:
    return 6; // kRocProfVisDmResultDbAccessFailed
  case error_code::invalid_property:
    return 7; // kRocProfVisDmResultInvalidProperty
  case error_code::not_supported:
    return 8; // kRocProfVisDmResultNotSupported
  case error_code::resource_busy:
    return 9; // kRocProfVisDmResultResourceBusy
  case error_code::db_abort:
    return 10; // kRocProfVisDmResultDbAbort

  // Map extended codes to closest C API equivalents
  case error_code::file_not_found:
  case error_code::invalid_database:
  case error_code::connection_failed:
    return 6; // kRocProfVisDmResultDbAccessFailed

  case error_code::constraint_violation:
  case error_code::schema_error:
  case error_code::query_error:
    return 6; // kRocProfVisDmResultDbAccessFailed

  case error_code::index_out_of_bounds:
    return 5; // kRocProfVisDmResultInvalidParameter

  default:
    return 1; // kRocProfVisDmResultUnknownError
  }
}

namespace {
const char *error_code_name(error_code code) {
  switch (code) {
  case error_code::success:
    return "success";
  case error_code::unknown_error:
    return "unknown_error";
  case error_code::timeout:
    return "timeout";
  case error_code::not_loaded:
    return "not_loaded";
  case error_code::alloc_failure:
    return "alloc_failure";
  case error_code::invalid_parameter:
    return "invalid_parameter";
  case error_code::db_access_failed:
    return "db_access_failed";
  case error_code::invalid_property:
    return "invalid_property";
  case error_code::not_supported:
    return "not_supported";
  case error_code::resource_busy:
    return "resource_busy";
  case error_code::db_abort:
    return "db_abort";
  case error_code::file_not_found:
    return "file_not_found";
  case error_code::invalid_database:
    return "invalid_database";
  case error_code::constraint_violation:
    return "constraint_violation";
  case error_code::schema_error:
    return "schema_error";
  case error_code::query_error:
    return "query_error";
  case error_code::connection_failed:
    return "connection_failed";
  case error_code::index_out_of_bounds:
    return "index_out_of_bounds";
  default:
    return "unknown";
  }
}
} // namespace

std::string error::to_string() const {
  std::ostringstream ss;
  ss << "rocstorage::error[" << error_code_name(code_) << "]";

  if (!message_.empty()) {
    ss << ": " << message_;
  }

  if (!query_.empty()) {
    ss << " (query: " << query_ << ")";
  }

  if (sqlite_code_ != 0) {
    ss << " [sqlite3 error: " << sqlite_code_ << "]";
  }

  return ss.str();
}

error from_c_result(int c_result, std::string message) {
  error_code code;
  switch (c_result) {
  case 0:
    code = error_code::success;
    break;
  case 1:
    code = error_code::unknown_error;
    break;
  case 2:
    code = error_code::timeout;
    break;
  case 3:
    code = error_code::not_loaded;
    break;
  case 4:
    code = error_code::alloc_failure;
    break;
  case 5:
    code = error_code::invalid_parameter;
    break;
  case 6:
    code = error_code::db_access_failed;
    break;
  case 7:
    code = error_code::invalid_property;
    break;
  case 8:
    code = error_code::not_supported;
    break;
  case 9:
    code = error_code::resource_busy;
    break;
  case 10:
    code = error_code::db_abort;
    break;
  default:
    code = error_code::unknown_error;
    break;
  }

  return error(code, std::move(message));
}

} // namespace rocstorage
