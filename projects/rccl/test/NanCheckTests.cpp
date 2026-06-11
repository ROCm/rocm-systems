/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Validates the opt-in device-side NaN/Inf detector (RCCL_ENABLE_NAN_CHECK).
// We inject a NaN into one rank's AllReduce input and assert the detector's
// device-side report ("[RCCL NaN/Inf] ...") is emitted.
//
// Why a subprocess: HIP device printf is flushed to the stdout fd the runtime
// caches at init, so gtest's per-test CaptureStdout (which redirects fd 1 later)
// does NOT capture it. RCCL's reduce kernels are compiled device-only, so there
// is also no host-registered symbol for hipMemcpyFromSymbol. The reliable
// signal is the device printf itself, captured by running the collective in a
// fresh child process whose stdout is a pipe from the start. We re-exec this
// same test binary filtered to the helper test, so no extra binary is needed
// and the detector requires no changes.

#include <gtest/gtest.h>
#include <rccl/rccl.h>
#include <hip/hip_runtime.h>

#include <unistd.h>

#include <cmath>
#include <cstdio>
#include <iostream>
#include <string>
#include <vector>

#include "common/StandaloneUtils.hpp"   // HIPCALL, NCCLCHECK

namespace RcclUnitTesting
{
#ifdef RCCL_ENABLE_NAN_CHECK

  // Runs a 2-GPU AllReduce(sum) with a NaN injected into GPU 0's input.
  // Returns: 0 on success (NaN propagated), 1 on logic error, 77 to indicate skip.
  static int RunNanAllReduce()
  {
    int numDevices = 0;
    if (hipGetDeviceCount(&numDevices) != hipSuccess || numDevices < 2) return 77;

    const int    ngpu = 2;
    const size_t N    = 1 << 20;
    std::vector<ncclComm_t> comms(ngpu);
    if (ncclCommInitAll(comms.data(), ngpu, /*devlist=*/nullptr) != ncclSuccess) return 1;

    std::vector<float>       host(N, 1.0f);
    std::vector<float*>      sbuf(ngpu, nullptr), rbuf(ngpu, nullptr);
    std::vector<hipStream_t> st(ngpu);
    for (int i = 0; i < ngpu; i++) {
      if (hipSetDevice(i) != hipSuccess) return 1;
      hipStreamCreate(&st[i]);
      hipMalloc(&sbuf[i], N * sizeof(float));
      hipMalloc(&rbuf[i], N * sizeof(float));
      hipMemcpy(sbuf[i], host.data(), N * sizeof(float), hipMemcpyHostToDevice);
    }
    const size_t kIdx = 123456;
    float nan = std::nanf("");
    hipSetDevice(0);
    hipMemcpy(&sbuf[0][kIdx], &nan, sizeof(float), hipMemcpyHostToDevice);

    ncclGroupStart();
    for (int i = 0; i < ngpu; i++)
      ncclAllReduce(sbuf[i], rbuf[i], N, ncclFloat, ncclSum, comms[i], st[i]);
    ncclGroupEnd();
    for (int i = 0; i < ngpu; i++) { hipSetDevice(i); hipStreamSynchronize(st[i]); hipDeviceSynchronize(); }

    float result = 0.0f;
    hipSetDevice(1);
    hipMemcpy(&result, &rbuf[1][kIdx], sizeof(float), hipMemcpyDeviceToHost);

    for (int i = 0; i < ngpu; i++) { hipSetDevice(i); hipFree(sbuf[i]); hipFree(rbuf[i]); hipStreamDestroy(st[i]); }
    for (int i = 0; i < ngpu; i++) ncclCommDestroy(comms[i]);
    return std::isnan(result) ? 0 : 1;
  }

  // Helper "test": just runs the collective so its device printf hits this
  // process's stdout. Invoked in-process by the suite and re-exec'd as a child
  // by the assertion test below.
  TEST(NanCheck, HelperRunCollectiveWithNaN)
  {
    int rc = RunNanAllReduce();
    if (rc == 77) GTEST_SKIP() << "This test requires at least 2 GPUs.";
    EXPECT_EQ(rc, 0) << "Injected NaN did not propagate through AllReduce.";
  }

  // Drives the bias / accumulate reduce path (reduceCopyPacksWithBias, USE_ACC=1)
  // via the RCCL ncclAllReduceWithBias extension, with a NaN in GPU 0's input.
  static int RunNanAllReduceWithBias()
  {
    int numDevices = 0;
    if (hipGetDeviceCount(&numDevices) != hipSuccess || numDevices < 2) return 77;

    const int    ngpu = 2;
    const size_t N    = 1 << 20;
    std::vector<ncclComm_t> comms(ngpu);
    if (ncclCommInitAll(comms.data(), ngpu, /*devlist=*/nullptr) != ncclSuccess) return 1;

    std::vector<float>       host(N, 1.0f);
    std::vector<float*>      sbuf(ngpu, nullptr), rbuf(ngpu, nullptr), abuf(ngpu, nullptr);
    std::vector<hipStream_t> st(ngpu);
    for (int i = 0; i < ngpu; i++) {
      if (hipSetDevice(i) != hipSuccess) return 1;
      hipStreamCreate(&st[i]);
      hipMalloc(&sbuf[i], N * sizeof(float));
      hipMalloc(&rbuf[i], N * sizeof(float));
      hipMalloc(&abuf[i], N * sizeof(float));            // bias / accumulator buffer
      hipMemcpy(sbuf[i], host.data(), N * sizeof(float), hipMemcpyHostToDevice);
      hipMemcpy(abuf[i], host.data(), N * sizeof(float), hipMemcpyHostToDevice);
    }
    const size_t kIdx = 123456;
    float nan = std::nanf("");
    hipSetDevice(0);
    hipMemcpy(&sbuf[0][kIdx], &nan, sizeof(float), hipMemcpyHostToDevice);

    ncclGroupStart();
    for (int i = 0; i < ngpu; i++)
      ncclAllReduceWithBias(sbuf[i], rbuf[i], N, ncclFloat, ncclSum, comms[i], st[i], abuf[i]);
    ncclGroupEnd();
    for (int i = 0; i < ngpu; i++) { hipSetDevice(i); hipStreamSynchronize(st[i]); hipDeviceSynchronize(); }

    float result = 0.0f;
    hipSetDevice(1);
    hipMemcpy(&result, &rbuf[1][kIdx], sizeof(float), hipMemcpyDeviceToHost);

    for (int i = 0; i < ngpu; i++) { hipSetDevice(i); hipFree(sbuf[i]); hipFree(rbuf[i]); hipFree(abuf[i]); hipStreamDestroy(st[i]); }
    for (int i = 0; i < ngpu; i++) ncclCommDestroy(comms[i]);
    return std::isnan(result) ? 0 : 1;
  }

  TEST(NanCheck, HelperRunBiasCollectiveWithNaN)
  {
    int rc = RunNanAllReduceWithBias();
    if (rc == 77) GTEST_SKIP() << "This test requires at least 2 GPUs.";
    EXPECT_EQ(rc, 0) << "Injected NaN did not propagate through AllReduceWithBias.";
  }

  // Re-exec this binary filtered to a helper test in a child process (optionally
  // forcing a protocol) and return the child's combined stdout/stderr. A fresh
  // child is required because HIP device printf only reaches the stdout fd the
  // runtime cached at process start (gtest's CaptureStdout cannot intercept it).
  static std::string RunHelper(const char* filter, const char* proto)
  {
    char self[4096] = {0};
    ssize_t n = readlink("/proc/self/exe", self, sizeof(self) - 1);
    if (n <= 0) return "";
    self[n] = '\0';

    std::string cmd;
    if (proto) cmd += std::string("NCCL_PROTO=") + proto + " ";
    cmd += "'" + std::string(self) + "' --gtest_filter='" + filter + "' --gtest_color=no 2>&1";
    FILE* p = popen(cmd.c_str(), "r");
    if (!p) return "";
    std::string out;
    char buf[4096];
    size_t r;
    while ((r = fread(buf, 1, sizeof(buf), p)) > 0) out.append(buf, r);
    pclose(p);
    return out;
  }

  static void ExpectDetects(const char* filter, const char* proto, const char* what)
  {
    int numDevices = 0;
    if (hipGetDeviceCount(&numDevices) != hipSuccess || numDevices < 2)
      GTEST_SKIP() << "This test requires at least 2 GPUs.";
    std::string out = RunHelper(filter, proto);
    std::cout << out;   // surface in CI logs
    EXPECT_NE(out.find("[RCCL NaN/Inf]"), std::string::npos)
        << "Detector did not report the injected NaN for " << what
        << ". Child output:\n" << out;
  }

  // Simple / LL / LL128 reduce paths (common_kernel.h, prims_ll.h, prims_ll128.h).
  TEST(NanCheck, AllReduceDetectsInjectedNaN_Simple) { ExpectDetects("NanCheck.HelperRunCollectiveWithNaN", "Simple", "AllReduce/Simple"); }
  TEST(NanCheck, AllReduceDetectsInjectedNaN_LL)     { ExpectDetects("NanCheck.HelperRunCollectiveWithNaN", "LL", "AllReduce/LL"); }
  TEST(NanCheck, AllReduceDetectsInjectedNaN_LL128)  { ExpectDetects("NanCheck.HelperRunCollectiveWithNaN", "LL128", "AllReduce/LL128"); }
  // Bias / accumulate reduce path (reduceCopyPacksWithBias).
  TEST(NanCheck, AllReduceWithBiasDetectsInjectedNaN) { ExpectDetects("NanCheck.HelperRunBiasCollectiveWithNaN", nullptr, "AllReduceWithBias"); }

#else  // !RCCL_ENABLE_NAN_CHECK

  TEST(NanCheck, DISABLED_AllReduceDetectsInjectedNaN)
  {
    GTEST_SKIP() << "Built without RCCL_ENABLE_NAN_CHECK; the NaN detector is a no-op.";
  }

#endif
}  // namespace RcclUnitTesting
