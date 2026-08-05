/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Force-included (via -include) at the top of the standalone init.cc microtest
// TU. init.cc's include chain pulls headers that rely on transitive standard
// headers the host clang++ chain doesn't otherwise provide in order:
//   - <cstdint>: fixed-width ints used pervasively (matches the p2p target).
//   - <ostream>: rccl_float8.h defines operator<<(std::ostream&, ...) but only
//     forward-declares ostream via its transitive includes, so the free
//     operator<< for float isn't visible without a complete <ostream>.
// Using an absolute-path -include header (rather than `-include ostream`)
// guarantees resolution regardless of the bare-name include search behavior.

#pragma once
#include <cstdint>
#include <ostream>
