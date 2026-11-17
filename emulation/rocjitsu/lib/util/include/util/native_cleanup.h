// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include <optional>
#include <pthread.h>
#include <type_traits>
#include <utility>

namespace util {

/// @brief Run an operation with cleanup on native thread exit or cancellation.
/// @details Cleanup must not throw and is not called on normal return. Its state
/// and the resources it releases must outlive the operation. Keeping registration
/// in this active frame also supports builds where C++ destructors do not run on
/// native thread exit. The operation must return void or a value, not a reference.
template <typename Cleanup, typename Function>
auto with_native_cleanup(Cleanup &cleanup, Function &&function) {
  using Result = std::invoke_result_t<Function &&>;
  static_assert(!std::is_reference_v<Result>);
  // A void operation needs no result; use an inert storage type for that case.
  [[maybe_unused]] std::optional<std::conditional_t<std::is_void_v<Result>, bool, Result>> result;
  pthread_cleanup_push(+[](void *state) { (*static_cast<Cleanup *>(state))(); }, &cleanup);
  if constexpr (std::is_void_v<Result>)
    std::forward<Function>(function)();
  else
    result.emplace(std::forward<Function>(function)());
  pthread_cleanup_pop(0);
  if constexpr (!std::is_void_v<Result>)
    return std::move(*result);
}

} // namespace util
