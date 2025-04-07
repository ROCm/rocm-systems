/*
 * Copyright (C) 2014-2018 Advanced Micro Devices, Inc. All Rights Reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR
 * OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
 * ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
 * OTHER DEALINGS IN THE SOFTWARE.
 *
 */

#include <gtest/gtest.h>
#include "hsakmt/hsakmt.h"

#ifndef __KFD_OPEN_CLOSE_KFD_TEST__H__
#define __KFD_OPEN_CLOSE_KFD_TEST__H__

//  @class KFDOpenCloseKFDTest
class KFDOpenCloseKFDTest : public testing::Test {
 public:
    KFDOpenCloseKFDTest(void) {}
    ~KFDOpenCloseKFDTest(void) {}

 protected:
    // @brief Executed before every test that uses KFDOpenCloseKFDTest class, sets all common settings for the tests.
    virtual void SetUp();
    // @brief Executed after every test that uses KFDOpenCloseKFDTest class
    virtual void TearDown();
};

#endif  //  __KFD_OPEN_CLOSE_KFD_TEST__H__
