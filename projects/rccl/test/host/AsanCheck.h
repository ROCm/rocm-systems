/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// kAsan is true in an AddressSanitizer build; tests that pin a production leak or overflow skip under it.
#ifndef RCCL_TEST_HOST_ASANCHECK_H_
#define RCCL_TEST_HOST_ASANCHECK_H_

#if defined(__SANITIZE_ADDRESS__)
#define RCCL_TEST_ASAN 1
#elif defined(__has_feature)
#if __has_feature(address_sanitizer)
#define RCCL_TEST_ASAN 1
#endif
#endif
#ifndef RCCL_TEST_ASAN
#define RCCL_TEST_ASAN 0
#endif

constexpr bool kAsan = RCCL_TEST_ASAN;

#endif  // RCCL_TEST_HOST_ASANCHECK_H_
