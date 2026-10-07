// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef ROCJITSU_ISA_DETAIL_REGISTER_SET_TYPES_H_
#define ROCJITSU_ISA_DETAIL_REGISTER_SET_TYPES_H_

#include <cstddef>
#include <cstdint>

namespace rocjitsu {

/// @brief Storage implementation used by a register set.
enum class RegisterSetWordType { StandardUint64, Avx2M256 };

namespace register_set_detail {
template <RegisterSetWordType wordType> struct WordOps {};
} // namespace register_set_detail

} // namespace rocjitsu

#endif // ROCJITSU_ISA_DETAIL_REGISTER_SET_TYPES_H_
