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

#ifndef _RESULT_STRUCT_H_

struct IndicesRange {
  int startIndex;
  int endIndex;
};

#define INDEX_ALL_TESTS -1
#define EXTREMELY_SMALL_VALUE -10000.0f
#define EXTREMELY_LARGE_VALUE 10000.0f

class TestResult {
 public:
  float value;
  std::string resultString;
  bool passed;

  TestResult(float val) : resultString("\n"), passed(true) { value = val; }

  void reset(float val) {
    value = val;
    passed = true;
    resultString.assign("\n");
  }
};

class Report {
 public:
  TestResult* max;
  TestResult* min;
  bool success;
  int numFailedTests;

  Report() : success(true), numFailedTests(0) {
    max = new TestResult(EXTREMELY_SMALL_VALUE);
    min = new TestResult(EXTREMELY_LARGE_VALUE);
  }

  void reset() {
    max->reset(EXTREMELY_SMALL_VALUE);
    min->reset(EXTREMELY_LARGE_VALUE);
    success = true;
    numFailedTests = 0;
  }
  ~Report() {
    delete max;
    delete min;
  }
};

#endif  // _RESULT_STRUCT_H_
