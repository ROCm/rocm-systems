/*
Copyright (c) 2022 Advanced Micro Devices, Inc. All rights reserved.

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
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

#pragma once

#include <hip_test_common.hh>

#pragma clang diagnostic ignored "-Wunused-function"

struct BoxStruct {
  int count;
  BoxStruct() { INFO("Constructor called for Struct!\n"); }
  ~BoxStruct() { INFO("Destructor called for Struct!\n"); }
};

class BoxClass {
 public:
  int count;
  BoxClass() { INFO("Constructor called for Class!\n"); }
  ~BoxClass() { INFO("Destructor called for Class!\n"); }
};

namespace {

void destroyStructObj(void* ptr) {
  BoxStruct* ptr1 = reinterpret_cast<BoxStruct*>(ptr);
  delete ptr1;
}

void destroyClassObj(void* ptr) {
  BoxClass* ptr2 = reinterpret_cast<BoxClass*>(ptr);
  delete ptr2;
}

void destroyIntObj(void* ptr) {
  int* ptr2 = reinterpret_cast<int*>(ptr);
  delete ptr2;
}

void destroyFloatObj(void* ptr) {
  float* ptr2 = reinterpret_cast<float*>(ptr);
  delete ptr2;
}

}  // anonymous namespace
