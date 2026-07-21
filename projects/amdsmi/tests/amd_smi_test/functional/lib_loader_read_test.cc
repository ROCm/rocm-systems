/*
 * Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */

// Loader fallback tests: a bogus primary soname followed by a valid candidate
// must still resolve. No GPU required.

#include <gtest/gtest.h>

#include <vector>

#include "amd_smi/impl/amd_smi_lib_loader.h"

namespace {

TEST(LibLoaderFallback, FirstCandidateWins) {
  amd::smi::AMDSmiLibraryLoader loader;
  EXPECT_EQ(loader.load(std::vector<const char*>{"libm.so.6", "does_not_exist.so"}),
            AMDSMI_STATUS_SUCCESS);
}

TEST(LibLoaderFallback, FallsBackWhenPrimaryMissing) {
  amd::smi::AMDSmiLibraryLoader loader;
  EXPECT_EQ(loader.load(std::vector<const char*>{"does_not_exist.so.9", "libm.so.6"}),
            AMDSMI_STATUS_SUCCESS);
}

TEST(LibLoaderFallback, AllCandidatesMissingFails) {
  amd::smi::AMDSmiLibraryLoader loader;
  EXPECT_EQ(loader.load(std::vector<const char*>{"nope_a.so.9", "nope_b.so.9"}),
            AMDSMI_STATUS_FAIL_LOAD_MODULE);
}

// When a candidate is already loaded in the process, the loader must still keep
// a usable handle so load_symbol() resolves.
TEST(LibLoaderFallback, AlreadyLoadedResolvesSymbol) {
  amd::smi::AMDSmiLibraryLoader keep_open;
  ASSERT_EQ(keep_open.load("libm.so.6"), AMDSMI_STATUS_SUCCESS);

  amd::smi::AMDSmiLibraryLoader loader;
  ASSERT_EQ(loader.load(std::vector<const char*>{"libm.so.6"}), AMDSMI_STATUS_SUCCESS);
  double (*cos_fn)(double) = nullptr;
  EXPECT_EQ(loader.load_symbol(&cos_fn, "cos"), AMDSMI_STATUS_SUCCESS);
  EXPECT_NE(cos_fn, nullptr);
}

}  // namespace
