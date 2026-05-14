/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

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
