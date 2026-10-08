// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef ROCJITSU_ISA_DETAIL_REGISTER_SET_SELECT_H_
#define ROCJITSU_ISA_DETAIL_REGISTER_SET_SELECT_H_

#include "rocjitsu/isa/detail/register_set/portable.h"

#if defined(__AVX2__)
#include "rocjitsu/isa/detail/register_set/x86_avx2.h"
namespace rocjitsu::register_set_detail {
inline constexpr auto default_word_type = RegisterSetWordType::Avx2M256;
}
#else
namespace rocjitsu::register_set_detail {
inline constexpr auto default_word_type = RegisterSetWordType::StandardUint64;
}
#endif

#endif // ROCJITSU_ISA_DETAIL_REGISTER_SET_SELECT_H_
