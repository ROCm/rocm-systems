// MIT License
//
// Copyright (c) 2017-2025 Advanced Micro Devices, Inc.
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
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.


#ifndef TEST_UTIL_HELPER_FUNCS_H_
#define TEST_UTIL_HELPER_FUNCS_H_

#include <time.h>

#include <cmath>
#include <iostream>
#include <sstream>
#include <string>

static inline void Error(std::string error_msg) {
  std::cerr << "Error: " << error_msg << std::endl;
}

template <typename T>
void PrintArray(const std::string header, const T* data, const int width, const int height) {
  std::clog << header << " :\n";
  for (int i = 0; i < height; i++) {
    std::clog << "> ";
    for (int j = 0; j < width; j++) {
      std::clog << data[i * width + j] << " ";
    }
    std::clog << "\n";
  }
}

template <typename T>
bool FillRandom(T* array_ptr, const int width, const int height, const T range_min,
                const T range_max, unsigned int seed = 123) {
  if (!array_ptr) {
    Error("Cannot fill array. NULL pointer.");
    return false;
  }

  if (!seed) seed = (unsigned int)time(NULL);

  srand(seed);
  double range = double(range_max - range_min) + 1.0;

  /* random initialisation of input */
  for (int i = 0; i < height; i++)
    for (int j = 0; j < width; j++) {
      int index = i * width + j;
      array_ptr[index] = range_min + T(range * rand() / (RAND_MAX + 1.0));
    }

  return true;
}

template <typename T>
T RoundToPowerOf2(T val) {
  int bytes = sizeof(T);

  val--;
  for (int i = 0; i < bytes; i++) val |= val >> (1 << i);
  val++;

  return val;
}

template <typename T>
bool IsPowerOf2(T val) {
  long long long_val = val;
  return (((long_val & (-long_val)) - long_val == 0) && (long_val != 0));
}

#endif  // TEST_UTIL_HELPER_FUNCS_H_
