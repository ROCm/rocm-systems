// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// The build selects one implementation namespace together with its CPU options.
// Shared inline definitions must not be interchangeable across ISA domains.
#pragma once

#ifndef RJ_ARITH_IMPL_NAMESPACE
#error "Compile implementation files with an explicit rj-arith ISA domain"
#endif

namespace rocjitsu::arith {
namespace RJ_ARITH_IMPL_NAMESPACE {}
// Keep call sites independent of the domain selected for this translation unit.
using namespace RJ_ARITH_IMPL_NAMESPACE;
} // namespace rocjitsu::arith
