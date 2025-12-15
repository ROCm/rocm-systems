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

#include "rocprofvis_db_remote.h"

#include <gtest/gtest.h>

namespace {

class RemoteDatabaseTest : public ::testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override {}
};

TEST_F(RemoteDatabaseTest, ConstructWithUrl) {
    RocProfVis::DataModel::RemoteDatabase db("http://localhost:8080");
    EXPECT_EQ(db.GetServerUrl(), "http://localhost:8080");
    EXPECT_FALSE(db.IsConnected());
}

TEST_F(RemoteDatabaseTest, OpenFailsWhenServerNotRunning) {
    RocProfVis::DataModel::RemoteDatabase db("http://localhost:59999");
    auto result = db.Open();
    EXPECT_NE(result, kRocProfVisDmResultSuccess);
    EXPECT_FALSE(db.IsConnected());
}

TEST_F(RemoteDatabaseTest, CloseSucceedsEvenWhenNotConnected) {
    RocProfVis::DataModel::RemoteDatabase db("http://localhost:8080");
    auto result = db.Close();
    EXPECT_EQ(result, kRocProfVisDmResultSuccess);
    EXPECT_FALSE(db.IsConnected());
}

TEST_F(RemoteDatabaseTest, SaveTrimmedDataNotSupported) {
    RocProfVis::DataModel::RemoteDatabase db("http://localhost:8080");
    auto result = db.SaveTrimmedData(0, 1000, "/tmp/test.db", nullptr);
    EXPECT_EQ(result, kRocProfVisDmResultNotSupported);
}

// Integration tests - disabled by default, run with --gtest_also_run_disabled_tests
TEST_F(RemoteDatabaseTest, DISABLED_ConnectToRunningServer) {
    RocProfVis::DataModel::RemoteDatabase db("http://localhost:8080");
    auto result = db.Open();
    if (result == kRocProfVisDmResultSuccess) {
        EXPECT_TRUE(db.IsConnected());
        db.Close();
        EXPECT_FALSE(db.IsConnected());
    } else {
        GTEST_SKIP() << "Server not running";
    }
}

}  // namespace