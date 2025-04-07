/*
 * =============================================================================
 *   ROC Runtime Conformance Release License
 * =============================================================================
 * The University of Illinois/NCSA
 * Open Source License (NCSA)
 *
 * Copyright (c) 2017, Advanced Micro Devices, Inc.
 * All rights reserved.
 *
 * Developed by:
 *
 *                 AMD Research and AMD ROC Software Development
 *
 *                 Advanced Micro Devices, Inc.
 *
 *                 www.amd.com
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal with the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 *  - Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimers.
 *  - Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimers in
 *    the documentation and/or other materials provided with the distribution.
 *  - Neither the names of <Name of Development Group, Name of Institution>,
 *    nor the names of its contributors may be used to endorse or promote
 *    products derived from this Software without specific prior written
 *    permission.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
 * THE CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR
 * OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
 * ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS WITH THE SOFTWARE.
 *
 */


#include "common/helper_funcs.h"
#ifndef _WIN32
#include <unistd.h>
#endif
#include <assert.h>
#include <cmath>
#include <iostream>
#include <string>
#include <vector>
#include <numeric>

namespace rocrtst {

template<typename T>
void PrintArray(const std::string header, const T* data, const int width,
                const int height) {
  std::cout << std::endl << header << std::endl;

  for (int i = 0; i < height; i++) {
    for (int j = 0; j < width; j++) {
      std::cout << data[i * width + j] << " ";
    }

    std::cout << std::endl;
  }

  std::cout << std::endl;
}

template<typename T>
int FillRandom(T* arrayPtr,
               const int width,
               const int height,
               const T rangeMin,
               const T rangeMax,
               unsigned int seed) {
  if (!arrayPtr) {
    return 1;
  }

  if (!seed) {
    seed = (unsigned int)time(NULL);
  }

  srand(seed);
  double range = static_cast<double>(rangeMax - rangeMin) + 1.0;

  /* random initialisation of input */
  for (int i = 0; i < height; i++) {
    for (int j = 0; j < width; j++) {
      int index = i * width + j;
      arrayPtr[index] = rangeMin + T(range * rand_r(&seed) / (RAND_MAX + 1.0));
    }
  }

  return 0;
}

uint64_t RoundToPowerOf2(uint64_t val) {
  val--;
  /*
   * Shift with amount larger than the bit width can result in
   * undefined behavior by compiler for release builds.
   * Shift till 32 bit only which is less than bit width of val.
   */
  for (int i = 1; i <= 32; i *= 2) val |= val >> i;

  val++;
  return val;
}

bool IsPowerOf2(uint64_t val) {
  uint64_t tmp = val;

  if ((tmp & (-tmp)) - tmp == 0 && tmp != 0) {
    return true;
  } else {
    return false;
  }
}

bool
Compare(const float* refData, const float* data,
        const int length, const float epsilon) {
  float error = 0.0f;
  float ref = 0.0f;

  for (int i = 1; i < length; ++i) {
    float diff = refData[i] - data[i];
    error += diff * diff;
    ref += refData[i] * refData[i];
  }

  float normRef =::sqrtf(static_cast<float>(ref));

  if (::fabs(static_cast<float>(ref)) < 1e-7f) {
    return false;
  }

  float normError = ::sqrtf(static_cast<float>(error));
  error = normError / normRef;

  return error < epsilon;
}

bool
Compare(const double* refData, const double* data,
        const int length, const double epsilon) {
  double error = 0.0;
  double ref = 0.0;

  for (int i = 1; i < length; ++i) {
    double diff = refData[i] - data[i];
    error += diff * diff;
    ref += refData[i] * refData[i];
  }

  double normRef =::sqrt(static_cast<double>(ref));

  if (::fabs(static_cast<double>(ref)) < 1e-7) {
    return false;
  }

  double normError = ::sqrt(static_cast<double>(error));
  error = normError / normRef;

  return error < epsilon;
}

intptr_t
AlignDown(intptr_t value, size_t alignment) {
    assert(alignment != 0 && "Zero alignment");
    return (intptr_t) (value & ~(alignment - 1));
}

void *
AlignDown(void* value, size_t alignment) {
    return reinterpret_cast<void*>(AlignDown(
                              reinterpret_cast<uintptr_t>(value), alignment));
}

void *
AlignUp(void* value, size_t alignment) {
    return reinterpret_cast<void*>(
     AlignDown((uintptr_t)(reinterpret_cast<uintptr_t>(value) + alignment - 1),
                                                                   alignment));
}

double CalcMedian(const std::vector<double> &scores) {
  double median;
  size_t size = scores.size();

  if (size % 2 == 0) {
    median = (scores[size / 2 - 1] + scores[size / 2]) / 2;
  } else {
    median = scores[size / 2];
  }

  return median;
}

double CalcMean(const std::vector<double> &scores) {
  double mean;

  mean = std::accumulate(scores.begin(), scores.end(), 0.0);
  return mean/scores.size();
}

double CalcMean(const std::vector<double>& v1, const std::vector<double>& v2) {
  double mean = 0;
  size_t size = v1.size();

  for (size_t i = 0; i < size; i++) {
    mean += v2[i] - v1[i];
  }

  return mean / size;
}

double CalcStdDeviation(std::vector<double> scores, int score_mean) {
  double ret = 0.0;

  for (size_t i = 0; i < scores.size(); ++i) {
    ret += (scores[i] - score_mean) * (scores[i] - score_mean);
  }

  ret /= scores.size();

  return sqrt(ret);
}

/////////////////////////////////////////////////////////////////
// Template Instantiations
/////////////////////////////////////////////////////////////////

template
void PrintArray<uint32_t>(const std::string, const unsigned int*, int, int);

template
void PrintArray<float>(const std::string, const float*, int, int);

template
int FillRandom<uint32_t>(uint32_t* arrayPtr,
                         const int width, const int height,
                         uint32_t rangeMin, uint32_t rangeMax,
                                                           unsigned int seed);

template
int FillRandom<float>(float* arrayPtr,
                      const int width, const int height,
                      float rangeMin, float rangeMax, unsigned int seed);

}  // namespace rocrtst
