/*
 * =============================================================================
 *   ROC Runtime Conformance Release License
 * =============================================================================
 * The University of Illinois/NCSA
 * Open Source License (NCSA)
 *
 * Copyright (c) 2026, Advanced Micro Devices, Inc.
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

// This test guards the exported C++ ABI of the rocm_smi library.
//
// amd::smi::Device::writeDevInfoStr historically exported a three-argument
// overload whose final parameter is a `bool`. Dropping that parameter silently
// changes the mangled symbol name and breaks binary compatibility for consumers
// that were linked against an earlier release. The test below pins the exported
// symbol so that any future change to the signature is caught here rather than
// by downstream ABI tooling.

// RTLD_DEFAULT is a GNU extension exposed by <dlfcn.h> under _GNU_SOURCE.
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include <dlfcn.h>

#include "gtest/gtest.h"

namespace {

// Mangled name of
//   amd::smi::Device::writeDevInfoStr(amd::smi::DevInfoTypes, std::string, bool)
constexpr char kWriteDevInfoStrBoolSymbol[] =
    "_ZN3amd3smi6Device15writeDevInfoStrENS0_12DevInfoTypesENSt7__cxx1112basic_"
    "stringIcSt11char_traitsIcESaIcEEEb";

}  // namespace

TEST(RsmiAbiStability, WriteDevInfoStrRetainsBoolParameter) {
  // rsmitst links librocm_smi64, so the library's exported symbols are part of
  // the process' global symbol scope. Resolve the symbol directly from that
  // scope and use the standard dlerror() pattern to detect lookup failures.
  dlerror();  // Clear any stale error state.
  void* symbol = dlsym(RTLD_DEFAULT, kWriteDevInfoStrBoolSymbol);
  const char* error = dlerror();

  EXPECT_EQ(error, nullptr) << "dlsym failed to resolve the writeDevInfoStr symbol: " << error;
  EXPECT_NE(symbol, nullptr)
      << "amd::smi::Device::writeDevInfoStr(DevInfoTypes, std::string, bool) is "
         "missing from the exported ABI. Removing the trailing bool parameter "
         "breaks binary compatibility with consumers linked against earlier "
         "releases.";
}
