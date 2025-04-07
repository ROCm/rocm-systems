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

#ifndef __GOOGLETEST_EXTENSION__H__
#define __GOOGLETEST_EXTENSION__H__

#include <gtest/gtest.h>
#include "hsakmt/hsakmt.h"
#include "KFDTestFlags.hpp"

enum LOGTYPE {
    LOGTYPE_INFO,      // msg header in green
    LOGTYPE_WARNING    // msg header in yellow
};

class KFDLog{};
std::ostream& operator << (KFDLog log, LOGTYPE level);

// @brief  Log additional details, to be displayed in the same format as other google test outputs
// Currently not supported by gtest
// Should be used like cout: LOG() << "message" << value << std::endl;
#define LOG()      KFDLog() << LOGTYPE_INFO
#define WARN()     KFDLog() << LOGTYPE_WARNING

class KFDRecord: public testing::Test {
public:
    KFDRecord(const char *val): m_val(val) {}
    KFDRecord(std::string &val): m_val(val) {}
    KFDRecord(HSAint64 val): m_val(std::to_string(val)) {}
    KFDRecord(HSAuint64 val): m_val(std::to_string(val)) {}
    KFDRecord(double val): m_val(std::to_string(val)) {}
    ~KFDRecord() {
        RecordProperty(m_key.str().c_str(), m_val.c_str());
    }
    std::stringstream &get_key_stream() {
        return m_key;
    }
    virtual void TestBody() {};
private:
    std::string m_val;
    std::stringstream m_key;
};

#define RECORD(val)     (KFDRecord(val).get_key_stream())

// All tests MUST be in a try catch since the gtest flag to throw an exception on any fatal failure is enabled
#define TEST_START(testProfile)   if (Ok2Run(testProfile)) try {
#define TEST_END       } catch (...) {}

// Used to wrap setup and teardown functions, anything that is built-in gtest and is not a test
#define ROUTINE_START   try {
#define ROUTINE_END       }catch(...) {}

#define TEST_REQUIRE_ENV_CAPABILITIES(envCaps)          if (!TestReqEnvCaps(envCaps))  return;
#define TEST_REQUIRE_NO_ENV_CAPABILITIES(envCaps)  if (!TestReqNoEnvCaps(envCaps))  return;

#define ASSERT_SUCCESS(_val) ASSERT_EQ(HSAKMT_STATUS_SUCCESS, (_val))
#define EXPECT_SUCCESS(_val) EXPECT_EQ(HSAKMT_STATUS_SUCCESS, (_val))

#define EXPECT_EQ_GPU(expected, actual , gpuNode) EXPECT_EQ((expected), (actual)) << "gpuNodeID: " << std::to_string(gpuNode) << "\n"
#define ASSERT_SUCCESS_GPU(_val, gpuNode) ASSERT_EQ(HSAKMT_STATUS_SUCCESS, (_val)) << "gpuNodeID: " << std::to_string(gpuNode) << "\n"
#define EXPECT_SUCCESS_GPU(_val, gpuNode) EXPECT_EQ(HSAKMT_STATUS_SUCCESS, (_val)) << "gpuNodeID: " << std::to_string(gpuNode) << "\n"

#define ASSERT_NOTNULL(_val) ASSERT_NE((void *)NULL, _val)
#define EXPECT_NOTNULL(_val) EXPECT_NE((void *)NULL, _val)

#define ASSERT_NOTNULL_GPU(_val, gpuNode) ASSERT_NE((void *)NULL, _val) << "gpuNodeID: " << std::to_string(gpuNode) << "\n"
#define EXPECT_NOTNULL_GPU(_val, gpuNode) EXPECT_NE((void *)NULL, _val) << "gpuNodeID: " << std::to_string(gpuNode) << "\n"

#define EXPECT_NE_GPU(expected, actual, gpuNode) EXPECT_NE((expected), (actual)) << "gpuNodeID: " << std::to_string(gpuNode) << "\n"
#define EXPECT_GE_GPU(expected, actual, gpuNode) EXPECT_GE((expected), (actual)) << "gpuNodeID: " << std::to_string(gpuNode) << "\n"

#define ASSERT_GE_GPU(val1, val2, gpuNode) ASSERT_GE((val1), (val2)) << "gpuNodeID: " << std::to_string(gpuNode) << "\n"
#define ASSERT_NE_GPU(val1, val2, gpuNode) ASSERT_NE((val1), (val2)) << "gpuNodeID: " << std::to_string(gpuNode) << "\n"
#define ASSERT_EQ_GPU(val1, val2, gpuNode) ASSERT_EQ((val1), (val2)) << "gpuNodeID: " << std::to_string(gpuNode) << "\n"

#define EXPECT_TRUE_GPU(condition, gpuNode) EXPECT_TRUE(condition) << "gpuNodeID: " << std::to_string(gpuNode) << "\n"

// @brief  Determines if it is ok to run a test given input flags
bool Ok2Run(unsigned int testProfile);

// @brief  Checks if all HW capabilities needed for a test to run exist
bool TestReqEnvCaps(unsigned int hwCaps);

// @brief  Checks if all HW capabilities that prevents a test from running are absent
bool TestReqNoEnvCaps(unsigned int hwCaps);

#endif
