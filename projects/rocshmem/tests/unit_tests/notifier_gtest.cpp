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

#include "notifier_gtest.hpp"

using namespace rocshmem;

/*****************************************************************************
 ******************************* Fixture Tests *******************************
 *****************************************************************************/

TEST_F(NotifierBlockTestFixture, run_all_threads_once_1_1) {
  run_all_threads_once(1, 1);
}

TEST_F(NotifierBlockTestFixture, run_all_threads_once_2_1) {
  run_all_threads_once(2, 1);
}

TEST_F(NotifierBlockTestFixture, run_all_threads_once_64_1) {
  run_all_threads_once(64, 1);
}

TEST_F(NotifierBlockTestFixture, run_all_threads_once_128_1) {
  run_all_threads_once(128, 1);
}

TEST_F(NotifierBlockTestFixture, run_all_threads_once_256_1) {
  run_all_threads_once(256, 1);
}

TEST_F(NotifierBlockTestFixture, run_all_threads_once_512_1) {
  run_all_threads_once(512, 1);
}

TEST_F(NotifierBlockTestFixture, run_all_threads_once_1024_1) {
  run_all_threads_once(1024, 1);
}

TEST_F(NotifierAgentTestFixture, run_all_threads_once_1_1) {
  run_all_threads_once(1, 1);
}

TEST_F(NotifierAgentTestFixture, run_all_threads_once_2_1) {
  run_all_threads_once(2, 1);
}

TEST_F(NotifierAgentTestFixture, run_all_threads_once_64_1) {
  run_all_threads_once(64, 1);
}

TEST_F(NotifierAgentTestFixture, run_all_threads_once_128_1) {
  run_all_threads_once(128, 1);
}

TEST_F(NotifierAgentTestFixture, run_all_threads_once_256_1) {
  run_all_threads_once(256, 1);
}

TEST_F(NotifierAgentTestFixture, run_all_threads_once_512_1) {
  run_all_threads_once(512, 1);
}

TEST_F(NotifierAgentTestFixture, run_all_threads_once_1024_1) {
  run_all_threads_once(1024, 1);
}

TEST_F(NotifierAgentTestFixture, run_all_threads_once_1_2) {
  run_all_threads_once(1, 2);
}

TEST_F(NotifierAgentTestFixture, run_all_threads_once_1024_2) {
  run_all_threads_once(1024, 2);
}

TEST_F(NotifierAgentTestFixture, run_all_threads_once_1_4) {
  run_all_threads_once(1, 4);
}

TEST_F(NotifierAgentTestFixture, run_all_threads_once_1024_4) {
  run_all_threads_once(1024, 4);
}

TEST_F(NotifierAgentTestFixture, run_all_threads_once_1_8) {
  run_all_threads_once(1, 8);
}

TEST_F(NotifierAgentTestFixture, run_all_threads_once_1024_8) {
  run_all_threads_once(1024, 8);
}

TEST_F(NotifierAgentTestFixture, run_all_threads_once_1_32) {
  run_all_threads_once(1, 32);
}

TEST_F(NotifierAgentTestFixture, run_all_threads_once_1024_32) {
  run_all_threads_once(1024, 32);
}

TEST_F(NotifierAgentTestFixture, run_all_threads_once_1_38) {
  run_all_threads_once(1, 38);  // MI300 CPX
}

TEST_F(NotifierAgentTestFixture, run_all_threads_once_1024_38) {
  run_all_threads_once(1024, 38);  // MI300 CPX
}
