/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

/**
 * First stream of a new process.
 * The first stream a process creates on a device also creates the runtime's
 * device queue object, which has 64-byte aligned members. Runtime builds with
 * AVX2 or AVX-512 code faulted in its constructor in about half of all
 * processes when it was allocated without that alignment (ROCm/TheRock#10048).
 * Each child here is this executable started anew, so it initializes HIP itself
 * and runs that path once, whatever HIP state the parent has. The heap starts
 * at a page boundary in every new process, so each child first allocates a pad
 * of a different size to start the runtime's allocations at different heap
 * offsets.
 */

#include <hip_test_common.hh>
#include <hip_test_process.hh>

#include <cstdlib>
#include <string>

namespace {

constexpr int kProcesses = 16;

// Tell the child which API creates the stream and which heap pad it allocates.
constexpr const char* kApiEnv = "HIP_TEST_FIRST_STREAM_API";
constexpr const char* kPadEnv = "HIP_TEST_FIRST_STREAM_PAD";

enum class CreateApi { Default, NonBlocking, Priority };

const char* ApiName(CreateApi api) {
  switch (api) {
    case CreateApi::Default:
      return "hipStreamCreate";
    case CreateApi::NonBlocking:
      return "hipStreamCreateWithFlags(hipStreamNonBlocking)";
    case CreateApi::Priority:
      return "hipStreamCreateWithPriority";
  }
  return "?";
}

int EnvInt(const char* name) {
  const char* value = std::getenv(name);
  return value == nullptr ? -1 : std::atoi(value);
}

// Kept until exit: shifts the heap offsets of everything the runtime allocates after it. A
// global the compiler must store to, so the allocation is not optimized away.
void* volatile g_heapPad = nullptr;

}  // namespace

/**
 * Test Description
 * ------------------------
 *  - Creates and destroys the first stream of kProcesses new processes with
 *    hipStreamCreate, hipStreamCreateWithFlags(hipStreamNonBlocking) and
 *    hipStreamCreateWithPriority, each child after a heap pad of a different
 *    size. Every child must exit normally with success.
 * Test source
 * ------------------------
 *  - catch/multiproc/hipStreamCreateMproc.cc
 * Test requirements
 * ------------------------
 *  - None
 */
HIP_TEST_CASE(Unit_hipStreamCreate_FirstStreamInNewProcess) {
  const CreateApi api = GENERATE(CreateApi::Default, CreateApi::NonBlocking, CreateApi::Priority);

  for (int i = 0; i < kProcesses; ++i) {
    hip::SpawnProc child(getSelfExePath());
    child.setEnv(kApiEnv, std::to_string(static_cast<int>(api)));
    child.setEnv(kPadEnv, std::to_string(i));
    INFO(ApiName(api) << ", child " << i << " (exit codes above 128 are 128 + signal)");
    REQUIRE(child.run("Unit_hipStreamCreate_FirstStreamInNewProcess_Child") == 0);
  }
}

// Runs in the child: the first HIP calls of the process. Hidden ("[.]"): it runs only when
// Unit_hipStreamCreate_FirstStreamInNewProcess selects it by name.
TEST_CASE("Unit_hipStreamCreate_FirstStreamInNewProcess_Child", "[.]") {
  const int api = EnvInt(kApiEnv);
  const int pad = EnvInt(kPadEnv);
  REQUIRE(api >= static_cast<int>(CreateApi::Default));
  REQUIRE(api <= static_cast<int>(CreateApi::Priority));
  REQUIRE(pad >= 0);

  g_heapPad = std::malloc(16 * pad + 1);
  REQUIRE(g_heapPad != nullptr);

  HIP_CHECK(hipSetDevice(0));
  hipStream_t stream = nullptr;
  switch (static_cast<CreateApi>(api)) {
    case CreateApi::Default:
      HIP_CHECK(hipStreamCreate(&stream));
      break;
    case CreateApi::NonBlocking:
      HIP_CHECK(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));
      break;
    case CreateApi::Priority: {
      int least = 0, greatest = 0;
      HIP_CHECK(hipDeviceGetStreamPriorityRange(&least, &greatest));
      HIP_CHECK(hipStreamCreateWithPriority(&stream, hipStreamDefault, greatest));
      break;
    }
  }
  REQUIRE(stream != nullptr);
  HIP_CHECK(hipStreamDestroy(stream));
}
