/**********************************************************************
Copyright ©2013 Advanced Micro Devices, Inc. All rights reserved.

Redistribution and use in source and binary forms, with or without modification, are permitted
provided that the following conditions are met:

•	Redistributions of source code must retain the above copyright notice, this list of
conditions and the following disclaimer.
•	Redistributions in binary form must reproduce the above copyright notice, this list of
conditions and the following disclaimer in the documentation and/or
 other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR
IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT
SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY
 DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS
 OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
 NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.
********************************************************************/
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

template <typename T> T RoundToPowerOf2(T val) {
  int bytes = sizeof(T);

  val--;
  for (int i = 0; i < bytes; i++) val |= val >> (1 << i);
  val++;

  return val;
}

template <typename T> bool IsPowerOf2(T val) {
  long long long_val = val;
  return (((long_val & (-long_val)) - long_val == 0) && (long_val != 0));
}

#endif  // TEST_UTIL_HELPER_FUNCS_H_
