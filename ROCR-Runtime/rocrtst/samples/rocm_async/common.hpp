#ifndef ROCM_ASYNC_BW_COMMON_HPP
#define ROCM_ASYNC_BW_COMMON_HPP

#include <cstdlib>
#include <iostream>
#include <vector>
#include <cmath>
#include "hsa/hsa.h"
#include "hsa/hsa_ext_amd.h"

using namespace std;

#if defined(_MSC_VER)
#define ALIGNED_(x) __declspec(align(x))
#else
#if defined(__GNUC__)
#define ALIGNED_(x) __attribute__((aligned(x)))
#endif  // __GNUC__
#endif  // _MSC_VER

#define MULTILINE(...) #__VA_ARGS__

#define HSA_ARGUMENT_ALIGN_BYTES 16

#define ErrorCheck(x) error_check(x, __LINE__, __FILE__)

// @Brief: Check HSA API return value
void error_check(hsa_status_t hsa_error_code, int line_num, const char* str);

// @Brief: Find the first avaliable GPU device
hsa_status_t FindGpuDevice(hsa_agent_t agent, void* data);

// @Brief: Find the first avaliable CPU device
hsa_status_t FindCpuDevice(hsa_agent_t agent, void* data);

// @Brief: Find the agent's global region / pool
hsa_status_t FindGlobalPool(hsa_amd_memory_pool_t region, void* data);

// @Brief: Calculate the mean number of the vector
double CalcMean(vector<double> scores);

// @Brief: Calculate the Median valud of the vector
double CalcMedian(vector<double> scores);

// @Brief: Calculate the standard deviation of the vector
double CalcStdDeviation(vector<double> scores, int score_mean);

#endif  // ROCM_ASYNC_BW_COMMON_HPP
