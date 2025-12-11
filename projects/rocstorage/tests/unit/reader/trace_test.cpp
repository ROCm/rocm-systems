// MIT License
//
// Copyright (c) 2025 Advanced Micro Devices, Inc. All Rights Reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include "rocprofvis_dm_trace.h"

#include <gtest/gtest.h>

namespace {

class TraceTest : public ::testing::Test {
protected:
    void SetUp() override {
        m_trace = new RocProfVis::DataModel::Trace();
    }

    void TearDown() override {
        delete m_trace;
        m_trace = nullptr;
    }

    RocProfVis::DataModel::Trace* m_trace;
};

TEST_F(TraceTest, CreateTrace) {
    ASSERT_NE(m_trace, nullptr);
}

TEST_F(TraceTest, TraceHasNoTracksInitially) {
    EXPECT_EQ(m_trace->NumberOfTracks(), 0u);
}

TEST_F(TraceTest, TraceHasNoTablesInitially) {
    EXPECT_EQ(m_trace->NumberOfTables(), 0u);
}

TEST_F(TraceTest, TraceStartTimeIsZeroInitially) {
    EXPECT_EQ(m_trace->StartTime(), 0u);
}

TEST_F(TraceTest, TraceEndTimeIsZeroInitially) {
    EXPECT_EQ(m_trace->EndTime(), 0u);
}

TEST_F(TraceTest, TraceDatabaseIsNullInitially) {
    EXPECT_EQ(m_trace->Database(), nullptr);
}

TEST_F(TraceTest, TraceMutexIsNotNull) {
    EXPECT_NE(m_trace->Mutex(), nullptr);
}

TEST_F(TraceTest, BindingInfoIsNotNull) {
    EXPECT_NE(m_trace->BindingInfo(), nullptr);
}

TEST_F(TraceTest, DeleteAllSlicesSucceedsOnEmptyTrace) {
    auto result = m_trace->DeleteAllSlices();
    EXPECT_EQ(result, kRocProfVisDmResultSuccess);
}

TEST_F(TraceTest, DeleteAllTablesSucceedsOnEmptyTrace) {
    auto result = m_trace->DeleteAllTables();
    EXPECT_EQ(result, kRocProfVisDmResultSuccess);
}

TEST_F(TraceTest, DeleteAllEventPropertiesForFlowTraceSucceeds) {
    auto result = m_trace->DeleteAllEventPropertiesFor(kRPVDMEventFlowTrace);
    EXPECT_EQ(result, kRocProfVisDmResultSuccess);
}

TEST_F(TraceTest, DeleteAllEventPropertiesForStackTraceSucceeds) {
    auto result = m_trace->DeleteAllEventPropertiesFor(kRPVDMEventStackTrace);
    EXPECT_EQ(result, kRocProfVisDmResultSuccess);
}

TEST_F(TraceTest, DeleteAllEventPropertiesForExtDataSucceeds) {
    auto result = m_trace->DeleteAllEventPropertiesFor(kRPVDMEventExtData);
    EXPECT_EQ(result, kRocProfVisDmResultSuccess);
}

} // namespace
