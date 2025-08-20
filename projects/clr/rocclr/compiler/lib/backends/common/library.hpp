/* Copyright (c) 2008 - 2021 Advanced Micro Devices, Inc.

 Permission is hereby granted, free of charge, to any person obtaining a copy
 of this software and associated documentation files (the "Software"), to deal
 in the Software without restriction, including without limitation the rights
 to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 copies of the Software, and to permit persons to whom the Software is
 furnished to do so, subject to the following conditions:

 The above copyright notice and this permission notice shall be included in
 all copies or substantial portions of the Software.

 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 THE SOFTWARE. */

#ifndef LIBRARY_H_
#define LIBRARY_H_

#include <vector>
#include <string>
namespace amd {

typedef enum _library_selector {
  LibraryUndefined = 0,
  GPU_Library_7xx,
  GPU_Library_Evergreen,
  GPU_Library_SI,
  CPU_Library_Generic,
  CPU_Library_AVX,
  CPU_Library_FMA4,
  GPU_Library_Generic,
  CPU64_Library_Generic,
  CPU64_Library_AVX,
  CPU64_Library_FMA4,
  GPU64_Library_Evergreen,
  GPU64_Library_SI,
  GPU64_Library_Generic,
  GPU_Library_CI,
  GPU64_Library_CI,
  GPU_Library_HSAIL,
  LibraryTotal
} LibrarySelector;

/** Integrated Bitcode Libararies **/
class LibraryDescriptor {
 public:
  enum { MAX_NUM_LIBRARY_DESCS = 11 };

  const char* start;
  size_t size;
};

int getLibDescs(LibrarySelector LibType,     // input
                LibraryDescriptor* LibDesc,  // output
                int& LibDescSize             // output -- LibDesc[0:LibDescSize-1]
);

static constexpr const char* amdRTFuns[] = {"__amdrt_div_i64",        "__amdrt_div_u64",
                                            "__amdrt_mod_i64",        "__amdrt_mod_u64",
                                            "__amdrt_cvt_f64_to_u64", "__amdrt_cvt_f32_to_u64"};
}  // namespace amd

#endif  // LIBRARY_H_
