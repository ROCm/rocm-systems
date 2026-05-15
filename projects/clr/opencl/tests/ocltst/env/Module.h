/* Copyright (c) 2010 - 2021 Advanced Micro Devices, Inc.

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

#ifndef OCL_TEST_MODULE_H
#define OCL_TEST_MODULE_H

#include <string>

#include "OCLTest.h"
#include "OCLTestList.h"

struct Module {
  std::string name;
  ModuleHandle hmodule;
  TestCountFuncPtr get_count;
  TestNameFuncPtr get_name;
  CreateTestFuncPtr create_test;
  DestroyTestFuncPtr destroy_test;
  TestVersionFuncPtr get_version;
  TestLibNameFuncPtr get_libname;
  OCLTest** cached_test;

  Module()
      : name(""),
        hmodule(0),
        get_count(0),
        get_name(0),
        create_test(0),
        destroy_test(0),
        get_version(0),
        get_libname(0),
        cached_test(0) {
    // EMPTY!
  }
};

#endif  // OCL_TEST_MODULE_H
