/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include <hip_test_common.hh>

#if __has_include(<shared/rpc.h>)
#define HIP_TEST_HAS_RPC 1

#include <shared/rpc_util.h>
#include <shared/rpc.h>
#include <shared/rpc_opcodes.h>

namespace rpc {
[[gnu::visibility("protected")]] __device__ Client client asm("__llvm_rpc_client");
}  // namespace rpc

static void requireRpcSupport() {
  void* client = nullptr;
  if (hipGetSymbolAddress(&client, HIP_SYMBOL(rpc::client)) != hipSuccess) {
    (void)hipGetLastError();
    HIP_SKIP_TEST("Runtime RPC support is unavailable");
  }
}

#else
#define HIP_TEST_HAS_RPC 0
#endif
