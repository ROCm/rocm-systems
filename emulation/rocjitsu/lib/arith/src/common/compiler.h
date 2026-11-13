// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Compiler hints for fixed-size arithmetic loops whose generated code is inspected.
#pragma once

// Use a literal upper bound for the trip count: GCC 13 rejects template
// parameters here. This is a performance hint, never a correctness requirement.
#define RJ_ARITH_PRAGMA_IMPL(directive) _Pragma(#directive)
#if defined(__clang__)
#define RJ_ARITH_UNROLL(count) RJ_ARITH_PRAGMA_IMPL(unroll count)
#elif defined(__GNUC__)
#define RJ_ARITH_UNROLL(count) RJ_ARITH_PRAGMA_IMPL(GCC unroll count)
#else
#define RJ_ARITH_UNROLL(count)
#endif
