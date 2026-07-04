/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <hip_test_common.hh>

// Both the RPC transport and the GPU sanitizer ABI ship with the LLVM libc
// shared headers. Skip the test when the compiler predates them.
#if __has_include(<shared/rpc.h>) && __has_include(<sanitizer/gpu_sanitizer.h>)

#include <shared/rpc.h>
#include <shared/rpc_opcodes.h>
#include <sanitizer/gpu_sanitizer.h>

#include <unistd.h>

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <string>
#include <thread>

namespace rpc {
[[gnu::visibility("protected")]] __device__ Client client asm("__llvm_rpc_client");
}  // namespace rpc

// Fabricate a race report on the device and ship it to the host exactly like
// the GPU ConcurrencySanitizer runtime would. The host decodes the packet and
// prints a "==CSAN==" report through amd::reportGpuCSanRace.
__global__ void rpcCSanReportKernel() {
  __tsan_gpu_race race = {};
  race.pc = 0x1000;
  race.peer_pc = 0x2000;
  race.addr = 0xdeadbeefULL;
  race.size = 4;
  race.access_type = TSAN_GPU_ACCESS_WRITE;
  race.kind = TSAN_GPU_DATA_RACE;
  race.block[0] = 1;
  race.thread[0] = 2;
  race.lane = 3;

  static_assert(sizeof(race) <= sizeof(rpc::Buffer), "Report must fit in one packet");
  rpc::client.open<TSAN_GPU_REPORT_OPCODE>().send(
      [&](rpc::Buffer* buf, uint32_t) { __builtin_memcpy(buf->data, &race, sizeof(race)); });
}

namespace {
std::string slurp(const char* path) {
  std::string out;
  if (std::FILE* f = std::fopen(path, "rb")) {
    char chunk[512];
    size_t n;
    while ((n = std::fread(chunk, 1, sizeof(chunk), f)) > 0) out.append(chunk, n);
    std::fclose(f);
  }
  return out;
}
}  // namespace

TEST_CASE("Unit_Rpc_CSan_Report") {
  char path[] = "/tmp/csan_report_XXXXXX";
  int fd = mkstemp(path);
  REQUIRE(fd != -1);

  // Redirect stderr so we can scrape the host-side report.
  fflush(stderr);
  int saved = dup(fileno(stderr));
  REQUIRE(saved != -1);
  REQUIRE(dup2(fd, fileno(stderr)) != -1);

  rpcCSanReportKernel<<<1, 1>>>();
  HIP_CHECK(hipDeviceSynchronize());

  // The report is delivered asynchronously by the RPC listener thread, so poll
  // the captured output until it lands (or we give up).
  std::string out;
  for (int i = 0; i < 500; ++i) {
    fflush(stderr);
    out = slurp(path);
    if (out.find("==CSAN==") != std::string::npos) break;
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }

  fflush(stderr);
  dup2(saved, fileno(stderr));
  close(saved);
  close(fd);
  std::remove(path);

  INFO("captured report:\n" << out);
  REQUIRE(out.find("==CSAN==") != std::string::npos);
  REQUIRE(out.find("ConcurrencySanitizer: data race") != std::string::npos);
  REQUIRE(out.find("Write of size 4") != std::string::npos);
}

#endif  // __has_include(<shared/rpc.h>) && __has_include(<sanitizer/gpu_sanitizer.h>)
