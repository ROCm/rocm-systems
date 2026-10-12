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
// inside that library keeps worker threads running after the channel is
// destroyed, so the library has to stay mapped: unmapping it under those threads
// crashed rdci with SIGSEGV on exit, after it had printed its results.

#include <dlfcn.h>
#include <gtest/gtest.h>

#include <chrono>
#include <thread>

#include "rdc/rdc.h"

namespace {

// Nothing listens here, so RPCs fail at once without touching the network.
constexpr char kNoServer[] = "unix:/nonexistent/rdc_client_tests.sock";

}  // namespace

TEST(RdcClientShutdown, KeepsClientLibraryLoaded) {
  ASSERT_EQ(rdc_init(0), RDC_ST_OK);

  rdc_handle_t handle = nullptr;
  rdc_status_t status = rdc_connect(kNoServer, &handle, nullptr, nullptr, nullptr);
  if (status == RDC_ST_FAIL_LOAD_MODULE) {
    GTEST_SKIP() << "librdc_client.so is not on the library search path";
  }
  ASSERT_EQ(status, RDC_ST_OK);
  ASSERT_NE(handle, nullptr);

  // A failed RPC is enough to start gRPC's worker threads.
  uint32_t gpu_index_list[RDC_MAX_NUM_DEVICES];
  uint32_t count = 0;
  EXPECT_EQ(rdc_device_get_all(handle, gpu_index_list, &count), RDC_ST_CLIENT_ERROR);

  ASSERT_EQ(rdc_disconnect(handle), RDC_ST_OK);
  ASSERT_EQ(rdc_shutdown(), RDC_ST_OK);

  void* client = dlopen("librdc_client.so", RTLD_LAZY | RTLD_NOLOAD);
  EXPECT_NE(client, nullptr) << "rdc_shutdown() unloaded librdc_client.so";
  if (client != nullptr) {
    dlclose(client);
  }

  // gRPC's thread-pool watchdog wakes at least once a second; if the library
  // were gone, the process would crash during this wait.
  std::this_thread::sleep_for(std::chrono::milliseconds(1500));
}
