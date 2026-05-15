/*************************************************************************
 * Copyright (c) 2022 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/
#pragma once
#include <cstring>

namespace RcclUnitTesting
{
  typedef enum : int
  {
    TEST_SUCCESS = 0,
    TEST_FAIL    = 1,
    TEST_TIMEOUT = 2
  } ErrCode;

#define ERROR(...) printf("\033[0;31m" "[ ERROR    ] " "\033[0m" __VA_ARGS__)
#define INFO(...)  printf("[ INFO     ] " __VA_ARGS__)
#define WARN(...)  printf("[ WARNING  ] " __VA_ARGS__)
#define RETURN_RESULT(result) return (result)

#define CHECK_CALL_BASE(func, RESULT, RESULT_ARGS...) \
  do {                                                \
    ErrCode status = func;                            \
    if (status != TEST_SUCCESS)                       \
    {                                                 \
      ERROR("Error in call %s\n", #func);             \
      RESULT(status, ##RESULT_ARGS);                  \
    }                                                 \
  } while (false)
#define CHECK_CALL(func) CHECK_CALL_BASE(func, RETURN_RESULT)

#define CHECK_HIP_BASE(func, RESULT, RESULT_ARGS...)                    \
  do {                                                                  \
    hipError_t error = (func);                                          \
    if (error != hipSuccess)                                            \
    {                                                                   \
      fprintf(stderr, "\033[0;31m" "[ ERROR    ] HIP error: %s File:%s Line:%d\n" "\033[m", \
              hipGetErrorString(error), strrchr("/" __FILE__, '/') + 1, __LINE__); \
      RESULT(TEST_FAIL, ##RESULT_ARGS);                                 \
    }                                                                   \
  } while (false)
#define CHECK_HIP(func) CHECK_HIP_BASE(func, RETURN_RESULT)

#ifdef ENABLE_OPENMP
#define OMP_CANCEL_FOR(result, errCode) errCode = (result); _Pragma("omp cancel for")
#define RANK_RESULT(errCode, result) OMP_CANCEL_FOR(result, errCode)
#define CHECK_CALL_RANK(errCode, func) CHECK_CALL_BASE(func, OMP_CANCEL_FOR, errCode)
#define CHECK_HIP_RANK(errCode, func) CHECK_HIP_BASE(func, OMP_CANCEL_FOR, errCode)
#else
#define RANK_RESULT(errCode, result) RETURN_RESULT(result)
#define CHECK_CALL_RANK(errCode, func) CHECK_CALL(func)
#define CHECK_HIP_RANK(errCode, func) CHECK_HIP(func)
#endif
}

