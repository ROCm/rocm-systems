/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Makes a std::function seam fail the calls `when` picks and pass the rest to
// its current behaviour. Restored by the owning fakes file's Reset*().

#ifndef RCCL_TEST_HOST_FAILWHEN_H_
#define RCCL_TEST_HOST_FAILWHEN_H_

#include <functional>

template <typename R, typename... Args, typename When>
void FailWhen(std::function<R(Args...)>& slot, When when, R fail) {
  auto pass = slot;
  slot = [pass, when, fail](Args... args) mutable -> R { return when(args...) ? fail : pass(args...); };
}

#endif  // RCCL_TEST_HOST_FAILWHEN_H_
