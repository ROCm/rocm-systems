/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#pragma once
#ifdef __linux__
#include <unistd.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
#include <errno.h>
#include <sys/socket.h>
#include <memory.h>
#include <sys/un.h>
#endif
#include "hip_test_context.hh"

#define checkVMMSupported(device)                                                                  \
  {                                                                                                \
    int value = 0;                                                                                 \
    hipDeviceAttribute_t attr = hipDeviceAttributeVirtualMemoryManagementSupported;                \
    HIP_CHECK(hipDeviceGetAttribute(&value, attr, device));                                        \
    if (value == 0) {                                                                              \
      HIP_SKIP_TEST(HipTest::SkipReason::kVmmUnsupported);                                         \
    }                                                                                              \
  }

#define checkDmaBufSupported(device)                                                               \
  {                                                                                                \
    int value = 0;                                                                                 \
    hipDeviceAttribute_t attr = hipDeviceAttributeDmaBufSupported;                                 \
    HIP_CHECK(hipDeviceGetAttribute(&value, attr, device));                                        \
    if (value == 0) {                                                                              \
      HIP_SKIP_TEST("DMA-BUF is not supported for this test on this device.");                     \
    }                                                                                              \
  }
#define checkHostAllocDmaBufSupported(device)                                                      \
  {                                                                                                \
    int value = 0;                                                                                 \
    hipDeviceAttribute_t attr = hipDeviceAttributeHostAllocDmaBufSupported;                        \
    HIP_CHECK(hipDeviceGetAttribute(&value, attr, device));                                        \
    if (value == 0) {                                                                              \
      HIP_SKIP_TEST("Host-alloc DMA-BUF is not supported for this test on this device.");          \
    }                                                                                              \
  }
#define checkFabricHandleSupported(device)                                                         \
  {                                                                                                \
    int value = 0;                                                                                 \
    hipDeviceAttribute_t attr = hipDeviceAttributeHandleTypeFabricSupported;                       \
    HIP_CHECK(hipDeviceGetAttribute(&value, attr, device));                                        \
    if (value == 0) {                                                                              \
      HIP_SKIP_TEST("Fabric Handle Support not found. Skipping Test..");                  \
      return;                                                                                      \
    }                                                                                              \
  }

#ifdef __linux__
#define checkSysCallErrors(result)                                                                 \
  if (result == -1) {                                                                              \
    fprintf(stderr, "Failure at %u %s\n", __LINE__, __FILE__); exit(EXIT_FAILURE);                 \
  }

typedef pid_t Process;
#endif
constexpr int threadsPerBlk = 64;
