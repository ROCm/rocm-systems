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

#ifndef TESTS_AMD_SMI_TEST_FUNCTIONAL_IFOE_INFO_READ_H_
#define TESTS_AMD_SMI_TEST_FUNCTIONAL_IFOE_INFO_READ_H_

#include "../test_base.h"

class TestIfoeInfoRead : public TestBase {
 public:
  TestIfoeInfoRead();

  // @Brief: Destructor for test case of TestIfoeInfoRead
  virtual ~TestIfoeInfoRead();

  // @Brief: Setup the environment for measurement
  void SetUp() override;

  // @Brief: Core measurement execution
  void Run() override;

  // @Brief: Clean up and retrieve the resource
  void Close() override;

  // @Brief: Display  results
  void DisplayResults() const override;

  // @Brief: Display information about what this test does
  void DisplayTestInfo() override;
};

#endif  // TESTS_AMD_SMI_TEST_FUNCTIONAL_IFOE_INFO_READ_H_
