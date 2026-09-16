/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef RCCL_TEST_HOST_SIGNATURE_DRIFT_H_
#define RCCL_TEST_HOST_SIGNATURE_DRIFT_H_

// Compile-time watchdog against signature drift between a std::function seam
// hook and the production symbol it stands in for. std::function accepts any
// compatible callable, so a hook whose signature no longer matches the real
// symbol would drift silently; ASSERT_HOOK_MATCHES_PROD turns that into a
// compile error. Use it in a fakes .cc once the hooks and the production
// declarations are both visible:
//   ASSERT_HOOK_MATCHES_PROD(g_someHook, someProdSymbol);

#include <functional>
#include <type_traits>

namespace rccl_test_host {
template <typename F> struct FnSigOf;
template <typename R, typename... A>
struct FnSigOf<std::function<R(A...)>> { using type = R(A...); };
template <typename F> using FnSigOf_t = typename FnSigOf<F>::type;

template <typename Hook, typename ProdFn>
constexpr bool HookMatchesProd() {
    return std::is_same_v<FnSigOf_t<Hook>, std::remove_pointer_t<ProdFn>>;
}
}  // namespace rccl_test_host

#define ASSERT_HOOK_MATCHES_PROD(hook, prod)                                  \
    static_assert(::rccl_test_host::HookMatchesProd<decltype(hook),           \
                                                    decltype(&prod)>(),       \
                  "signature drift: " #hook " no longer matches " #prod       \
                  " -- update the std::function hook signature to match")

#endif  // RCCL_TEST_HOST_SIGNATURE_DRIFT_H_
