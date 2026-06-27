/*
Copyright (c) 2026 - Advanced Micro Devices, Inc. All rights reserved.

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

#include <gtest/gtest.h>

#include <string>

#include "common/rdc_utils.h"

// The unauthenticated (insecure) rdcd gRPC server must never be reachable over
// the network. rdcd refuses to start in unauthenticated mode unless the listen
// address is loopback or a UNIX domain socket; these tests pin the address
// classification that guard relies on.

TEST(RdcServerSecurity, RejectsNonLoopbackInsecureBind) {
  // The default (0.0.0.0) and any routable address must be rejected.
  EXPECT_FALSE(amd::rdc::IsLoopbackOrUnixAddress("0.0.0.0:50051"));
  EXPECT_FALSE(amd::rdc::IsLoopbackOrUnixAddress("10.0.0.5:50051"));
  EXPECT_FALSE(amd::rdc::IsLoopbackOrUnixAddress("192.168.1.10:50051"));
  EXPECT_FALSE(amd::rdc::IsLoopbackOrUnixAddress("172.16.0.1:50051"));
  EXPECT_FALSE(amd::rdc::IsLoopbackOrUnixAddress("[2001:db8::1]:50051"));
  EXPECT_FALSE(amd::rdc::IsLoopbackOrUnixAddress("::"));
  EXPECT_FALSE(amd::rdc::IsLoopbackOrUnixAddress("example.com:50051"));
  EXPECT_FALSE(amd::rdc::IsLoopbackOrUnixAddress(""));
}

TEST(RdcServerSecurity, AllowsLoopbackAndUnixInsecureBind) {
  EXPECT_TRUE(amd::rdc::IsLoopbackOrUnixAddress("127.0.0.1:50051"));
  EXPECT_TRUE(amd::rdc::IsLoopbackOrUnixAddress("127.5.5.5:50051"));  // 127.0.0.0/8
  EXPECT_TRUE(amd::rdc::IsLoopbackOrUnixAddress("[::1]:50051"));
  EXPECT_TRUE(amd::rdc::IsLoopbackOrUnixAddress("localhost:50051"));
  EXPECT_TRUE(amd::rdc::IsLoopbackOrUnixAddress("unix:/var/run/rdc.sock"));
  EXPECT_TRUE(amd::rdc::IsLoopbackOrUnixAddress("unix-abstract:rdc"));
}
