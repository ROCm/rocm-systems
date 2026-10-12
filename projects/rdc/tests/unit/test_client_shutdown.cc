/*
Copyright (c) 2026 - present Advanced Micro Devices, Inc. All rights reserved.

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

// rdc_shutdown() closes librdc_client.so once the client is done with it. gRPC
// inside that library keeps detached worker threads running after the channel is
// destroyed, so the library has to stay mapped: unmapping it under those threads
// crashed rdci with SIGSEGV on exit, after it had printed its results.
//
// glibc never unloads a library that has an STB_GNU_UNIQUE symbol. GCC emits
// those for static variables in inline functions and templates, so a client
// library with GCC-compiled code in it usually passes this test with or
// without the fix. A clang-only build, as TheRock ships, needs the fix.

#include <dlfcn.h>
#include <gtest/gtest.h>

#include <chrono>
#include <thread>

#include "rdc/rdc.h"

namespace {

// Nothing listens here, so RPCs fail at once without touching the network.
constexpr char kNoServer[] = "unix:/nonexistent/rdc_client_tests.sock";

// gRPC's thread-pool watchdog wakes at least once a second. One wakeup after
// the library is unmapped is enough to crash the process.
constexpr auto kThreadWakeupWait = std::chrono::milliseconds(1500);

}  // namespace

TEST(RdcClientShutdown, KeepsClientLibraryLoaded) {
  // The second pass reconnects after rdc_shutdown(), as rdctst does between tests.
  for (int pass = 0; pass < 2; ++pass) {
    SCOPED_TRACE(pass);
    ASSERT_EQ(rdc_init(0), RDC_ST_OK);

    rdc_handle_t handle = nullptr;
    ASSERT_EQ(rdc_connect(kNoServer, &handle, nullptr, nullptr, nullptr), RDC_ST_OK)
        << "librdc_client.so is not on the library search path";
    ASSERT_NE(handle, nullptr);

    // A failed RPC is enough to start gRPC's worker threads.
    uint32_t gpu_index_list[RDC_MAX_NUM_DEVICES] = {};
    uint32_t count = 0;
    EXPECT_EQ(rdc_device_get_all(handle, gpu_index_list, &count), RDC_ST_CLIENT_ERROR);

    ASSERT_EQ(rdc_disconnect(handle), RDC_ST_OK);
    ASSERT_EQ(rdc_shutdown(), RDC_ST_OK);

    void* client = dlopen("librdc_client.so", RTLD_LAZY | RTLD_NOLOAD);
    EXPECT_NE(client, nullptr) << "rdc_shutdown() unloaded librdc_client.so";
    if (client != nullptr) {
      dlclose(client);
    }
  }

  std::this_thread::sleep_for(kThreadWakeupWait);
}
