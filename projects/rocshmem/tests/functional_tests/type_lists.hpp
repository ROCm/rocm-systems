/******************************************************************************
 * Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
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

#ifndef _TYPE_LISTS_HPP_
#define _TYPE_LISTS_HPP_

/**
 * Type lists used by Tester::create() to decide which typed tester
 * instances to dispatch for a given TypeCoverage mode.
 *
 * Each list is an X-macro taking X(T, "name"):
 *   T     - the C++ type to instantiate the tester with
 *   name  - the string a user passes to -tc to select that type
 *
 * Every list contains exactly the types the corresponding tester .cpp
 * actually compiles specializations for. Adding a type here without adding
 * the matching device-side specialization will fail to link.
 */

// ---------------------------------------------------------------------------
// Collective type lists
// ---------------------------------------------------------------------------

// team_broadcast, team_fcollect, fcollect_wave, team_alltoall, team_alltoallv
// Note: int64_t is an alias of long on LP64, so it is covered by X(long, ...).
#define ROCSHMEM_COLL_TYPES_FULL(X) \
  X(float,               "float") \
  X(double,              "double") \
  X(char,                "char") \
  X(signed char,         "signed char") \
  X(short,               "short") \
  X(int,                 "int") \
  X(long,                "long") \
  X(long long,           "long long") \
  X(unsigned char,       "unsigned char") \
  X(unsigned short,      "unsigned short") \
  X(unsigned int,        "unsigned int") \
  X(unsigned long,       "unsigned long") \
  X(unsigned long long,  "unsigned long long")

// broadcast_wave
#define ROCSHMEM_BCAST_WAVE_TYPES_FULL(X) \
  X(float,      "float") \
  X(double,     "double") \
  X(short,      "short") \
  X(int,        "int") \
  X(long,       "long") \
  X(long long,  "long long")

// alltoall_wave
#define ROCSHMEM_ALLTOALL_WAVE_TYPES_FULL(X) \
  X(float,      "float") \
  X(double,     "double") \
  X(char,       "char") \
  X(short,      "short") \
  X(int,        "int") \
  X(long,       "long") \
  X(long long,  "long long")

// ---------------------------------------------------------------------------
// AMO type lists
//
// The GDA backend only implements 8-byte atomics -- see the
// "not implemented for non-64bit types" aborts in
// src/gda/context_gda_tmpl_device.hpp. Types narrower than 8 bytes therefore
// live in a separate _NONGDA list that the call site dispatches only when the
// active backend is not GDA.
// ---------------------------------------------------------------------------

// AMO Standard: fetch_add, fetch_inc, fetch_cswap, add, inc
#define ROCSHMEM_AMO_STD_TYPES_ALWAYS(X) \
  X(long long,          "long long") \
  X(long,               "long") \
  X(unsigned long,      "unsigned long") \
  X(unsigned long long, "unsigned long long")

#define ROCSHMEM_AMO_STD_TYPES_NONGDA(X) \
  X(int,          "int") \
  X(unsigned int, "unsigned int")

// AMO Extended: fetch, set, swap -- adds float/double over the standard set
#define ROCSHMEM_AMO_EXT_TYPES_ALWAYS(X) \
  X(long long,          "long long") \
  X(long,               "long") \
  X(unsigned long,      "unsigned long") \
  X(unsigned long long, "unsigned long long") \
  X(double,             "double")

#define ROCSHMEM_AMO_EXT_TYPES_NONGDA(X) \
  X(int,          "int") \
  X(unsigned int, "unsigned int") \
  X(float,        "float")

// AMO Bitwise: fetch_and/or/xor, and/or/xor
#define ROCSHMEM_AMO_BIT_TYPES_ALWAYS(X) \
  X(unsigned long long, "unsigned long long") \
  X(unsigned long,      "unsigned long") \
  X(int64_t,            "int64_t")

#define ROCSHMEM_AMO_BIT_TYPES_NONGDA(X) \
  X(unsigned int, "unsigned int") \
  X(int32_t,      "int32_t")

// ---------------------------------------------------------------------------
// Reduction push helpers
//
// Reduction testers take two typed lambdas, so they cannot be dispatched with
// the plain "new Tester<T>(args)" shape the other lists use. Two variants:
//   ARITH - integer types; casts through long long to format the error string
//   FLOAT - floating-point types; formats directly
// Only ROCSHMEM_SUM is exercised; operation coverage is a separate concern.
// ---------------------------------------------------------------------------

#define ROCSHMEM_PUSH_REDUCTION_ARITH(TESTER, T, args, testers) \
  (testers).push_back(new TESTER<T, ROCSHMEM_SUM>( \
      args, \
      [](T& f1, T& f2) { f1 = static_cast<T>(1); f2 = static_cast<T>(1); }, \
      [](T v, T n_pes) { \
        return (v == n_pes) \
            ? std::make_pair(true, std::string("")) \
            : std::make_pair(false, \
                  "Got " + std::to_string(static_cast<long long>(v)) + \
                  ", Expect " + std::to_string(static_cast<long long>(n_pes))); \
      }));

#define ROCSHMEM_PUSH_REDUCTION_FLOAT(TESTER, T, args, testers) \
  (testers).push_back(new TESTER<T, ROCSHMEM_SUM>( \
      args, \
      [](T& f1, T& f2) { f1 = static_cast<T>(1); f2 = static_cast<T>(1); }, \
      [](T v, T n_pes) { \
        return (v == n_pes) \
            ? std::make_pair(true, std::string("")) \
            : std::make_pair(false, \
                  "Got " + std::to_string(v) + \
                  ", Expect " + std::to_string(n_pes)); \
      }));

// All types compiled by team_reduction / team_reduce_scatter / reduce_wave /
// team_reduce_scatter_wave, each gated the same way the X-macro lists are.
#define ROCSHMEM_PUSH_REDUCTION_ALL(TESTER, args, testers) \
  if ((args).type_coverage == TypeCoverage::Full || (args).type_enabled("float")) \
    { ROCSHMEM_PUSH_REDUCTION_FLOAT(TESTER, float,     args, testers) } \
  if ((args).type_coverage == TypeCoverage::Full || (args).type_enabled("double")) \
    { ROCSHMEM_PUSH_REDUCTION_FLOAT(TESTER, double,    args, testers) } \
  if ((args).type_coverage == TypeCoverage::Full || (args).type_enabled("int")) \
    { ROCSHMEM_PUSH_REDUCTION_ARITH(TESTER, int,       args, testers) } \
  if ((args).type_coverage == TypeCoverage::Full || (args).type_enabled("short")) \
    { ROCSHMEM_PUSH_REDUCTION_ARITH(TESTER, short,     args, testers) } \
  if ((args).type_coverage == TypeCoverage::Full || (args).type_enabled("long")) \
    { ROCSHMEM_PUSH_REDUCTION_ARITH(TESTER, long,      args, testers) } \
  if ((args).type_coverage == TypeCoverage::Full || (args).type_enabled("long long")) \
    { ROCSHMEM_PUSH_REDUCTION_ARITH(TESTER, long long, args, testers) }

#endif // _TYPE_LISTS_HPP_
