/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/
#include <gtest/gtest.h>
#include <rccl/rccl.h>
#include <hip/hip_runtime.h>

#include <chrono>
#include <string>
#include <thread>

#include "common/ErrCode.hpp"
#include "common/ProcessIsolatedTestRunner.hpp"
#include "common/RasClientHelpers.hpp"

namespace RcclUnitTesting {

// Verifies the RAS subsystem supports JSON output and that switching to JSON
// actually returns a structured document rather than the human-readable text
// banner.
TEST(RasJson, JsonFormatIsSupportedAndDistinctFromText) {
  RUN_ISOLATED_TEST_WITH_ENV(
      "RasJson_JsonFormatIsSupportedAndDistinctFromText",
      []() {
        int devCount = 0;
        if (hipGetDeviceCount(&devCount) != hipSuccess || devCount < 1) {
          GTEST_SKIP() << "No HIP-visible GPU; skipping";
        }
        ASSERT_EQ(hipSetDevice(0), hipSuccess);

        // Pick the RAS port and publish NCCL_RAS_ADDR *inside* this process,
        // which is the same process that runs ncclCommInitRank() (so the RAS
        // server binds here) and rasRequest() (so the client connects here).
        // The process-isolation harness re-execs the test binary, which would
        // re-run any port selection placed in the TEST() body in a *second*
        // process; the server and client would then disagree on the port and
        // every connect() would get ECONNREFUSED.  Owning the port here keeps
        // them in lockstep.
        //
        // Use 127.0.0.1 explicitly (not "localhost") so the RAS server binds on
        // the same IPv4 address pickFreePort() probed: on hosts where
        // getaddrinfo("localhost", AF_UNSPEC) returns ::1 first, the bind would
        // otherwise target an IPv6 port that was never checked for being free.
        int port = pickFreePort();
        ASSERT_GT(port, 0) << "Could not allocate a free TCP port for RAS";
        std::string rasAddr = "127.0.0.1:" + std::to_string(port);
        ASSERT_EQ(setenv("NCCL_RAS_ENABLE", "1", /*overwrite=*/1), 0);
        ASSERT_EQ(setenv("NCCL_RAS_ADDR", rasAddr.c_str(), /*overwrite=*/1), 0);

        ncclUniqueId id;
        ASSERT_EQ(ncclGetUniqueId(&id), ncclSuccess);
        ncclComm_t comm = nullptr;
        ASSERT_EQ(ncclCommInitRank(&comm, /*nranks=*/1, id, /*rank=*/0),
                  ncclSuccess);

        // Give the RAS listener a brief moment to come up after init.
        std::this_thread::sleep_for(std::chrono::milliseconds(250));

        std::string textBody = rasRequest(port, "STATUS");
        std::string jsonBody = rasRequest(port, "STATUS", "json");

        ASSERT_FALSE(textBody.empty()) << "RAS returned no text response";
        ASSERT_FALSE(jsonBody.empty())
            << "RAS did not respond to 'SET FORMAT json' -- JSON support missing";

        EXPECT_NE(textBody.find("RCCL version"), std::string::npos)
            << "Text mode should contain the RCCL version banner";

        EXPECT_EQ(jsonBody.front(), '{')
            << "JSON mode should return a JSON object, not text. Got:\n"
            << jsonBody.substr(0, 200);
        EXPECT_NE(jsonBody.find("\"communicators\""), std::string::npos)
            << "JSON output missing the 'communicators' field";
        EXPECT_EQ(jsonBody.find("RCCL version"), std::string::npos)
            << "JSON output must not contain the human-readable text banner";

        ASSERT_EQ(ncclCommDestroy(comm), ncclSuccess);
      },
      {{"NCCL_RAS_ENABLE", "1"}});
}

}  // namespace RcclUnitTesting
