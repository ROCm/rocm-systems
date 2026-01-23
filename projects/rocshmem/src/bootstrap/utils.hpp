/******************************************************************************
 * Copyright (c) 2016-2022, NVIDIA CORPORATION. All rights reserved.
 * Modifications Copyright (c) Microsoft Corporation.
 * Modifications Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
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

#ifndef ROCSHMEM_UTILS_HPP_
#define ROCSHMEM_UTILS_HPP_

#include <chrono>
#include <cstdint>
#include <cstdio>

#define ERROR(...) { fprintf(stderr, __VA_ARGS__); abort(); }

namespace rocshmem {

struct Timer {
  std::chrono::steady_clock::time_point start_;
  int timeout_;

  Timer(int timeout = -1);

  ~Timer();

  /// Returns the elapsed time in microseconds.
  int64_t elapsed() const;

  void set(int timeout);

  void reset();

  void print(const std::string& name);
};

struct ScopedTimer : public Timer {
  const std::string name_;

  ScopedTimer(const std::string& name);

  ~ScopedTimer();
};

std::string getHostName(int maxlen, const char delim);

// PCI Bus ID <-> int64 conversion functions
std::string int64ToBusId(int64_t id);
int64_t busIdToInt64(const std::string busId);

uint64_t getHash(const char* string, int n);
uint64_t getHostHash();
uint64_t getPidHash();
void getRandomData(void* buffer, size_t bytes);

struct netIf {
  char prefix[64];
  int port;
};

int parseStringList(const char* string, struct netIf* ifList, int maxList);
bool matchIfList(const char* string, int port, struct netIf* ifList, int listSize, bool matchExact);

template <class T>
inline void hashCombine(std::size_t& hash, const T& v) {
  std::hash<T> hasher;
  hash ^= hasher(v) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
}

struct PairHash {
 public:
  template <typename T, typename U>
  std::size_t operator()(const std::pair<T, U>& x) const {
    std::size_t hash = 0;
    hashCombine(hash, x.first);
    hashCombine(hash, x.second);
    return hash;
  }
};

}  // namespace rocshmem

#endif // ROCSHMEM_UTILS_HPP
