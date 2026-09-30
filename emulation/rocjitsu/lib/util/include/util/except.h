// Copyright (c) 2025 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef UTIL_EXCEPT_H_
#define UTIL_EXCEPT_H_

#include <exception>
#include <string>
#include <string_view>

namespace util {

class Exception : public std::exception {
public:
  Exception(const std::string &msg) : msg_(msg) {}
  Exception(std::string_view msg) : msg_(msg) {}

  const char *what() const noexcept override { return msg_.c_str(); }

private:
  const std::string msg_;
};

class InvalidInst : public Exception {
public:
  InvalidInst(const std::string &msg, const std::string prefix = "Invalid instruction opcode: ")
      : Exception(prefix + msg) {}
};

class UnimplementedInst : public Exception {
public:
  explicit UnimplementedInst(std::string_view mnemonic)
      : Exception(std::string("Unimplemented instruction: ").append(mnemonic)) {}
};

class ConfigError : public Exception {
public:
  explicit ConfigError(std::string_view msg)
      : Exception(std::string("Config error: ").append(msg)) {}
};

} // namespace util

#endif // UTIL_EXCEPT_H_
