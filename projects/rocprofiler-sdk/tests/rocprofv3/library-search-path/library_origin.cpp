// Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

#include <cstdio>

// The app links against this symbol directly, so it has to stay in the dynamic symbol
// table even when the toolchain defaults to hidden visibility.
extern "C" __attribute__((visibility("default"))) void
print_library_origin()
{
    std::puts("from " LIBRARY_LOCATION);
}
