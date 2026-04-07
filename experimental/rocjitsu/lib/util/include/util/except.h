// Copyright (c) 2025 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef UTIL_EXCEPT_H_
#define UTIL_EXCEPT_H_

#include <exception>
#include <string>

namespace util {

class Exception : public std::exception {
public:
  Exception(const std::string &msg) : msg_(msg) {}

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
  explicit UnimplementedInst(const std::string &mnemonic)
      : Exception("Unimplemented instruction: " + mnemonic) {}
};

} // namespace util

#endif // UTIL_EXCEPT_H_
