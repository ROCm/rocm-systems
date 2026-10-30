// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/vm/amdgpu/image_addressing.h"
#include "rocjitsu/isa/isa_traits.h"
#include "rocjitsu/kmd/linux/amdgpu_properties.h"
#include "util/bit.h"

#include <algorithm>
#include <bit>
#include <cassert>
#include <cstdlib>

namespace rocjitsu::amdgpu {
namespace {
// Single-sample thin-image coordinate equations for the device identities RJ
// exposes: RDNA1 (0, 0x44), RDNA2 (0, 0x444), GFX11 (0, 0x545).
// X/Y/Z occupy successive 16-bit fields; each row gives an XOR mask per byte
// address bit. Only these configurations are supported; new device identities
// require new qualified rows. Swizzle kinds are 22, 26, 27, 24, and 31 (GFX11 only).
// Derived from ROCr image/addrlib at a10dca087065a82f15ea89dc7d6fdd3c6bb2c246,
// gfx10/gfx11SwizzlePattern.h via GetSwizzlePatternInfo and
// ConvertSwizzlePatternToEquation, for single-sample thin 2D views. RDNA2 uses
// Navi21 revision 0x28 (RB+), including NUM_PKRS. X byte-coordinate indices
// are reduced by log2(element_bytes). Identical rows share an index.
constexpr uint64_t kXorSelectors[87][18] = {
    {0x1ull, 0x2ull, 0x4ull, 0x20000ull, 0x10000ull, 0x40000ull, 0x8ull, 0x80000ull, 0x100000ull,
     0x10ull, 0x200000ull, 0x20ull, 0x0ull, 0x0ull, 0x0ull, 0x0ull, 0x0ull, 0x0ull},
    {0x0ull, 0x1ull, 0x2ull, 0x4ull, 0x10000ull, 0x20000ull, 0x40000ull, 0x8ull, 0x80000ull,
     0x10ull, 0x100000ull, 0x20ull, 0x0ull, 0x0ull, 0x0ull, 0x0ull, 0x0ull, 0x0ull},
    {0x0ull, 0x0ull, 0x1ull, 0x2ull, 0x10000ull, 0x20000ull, 0x40000ull, 0x4ull, 0x80000ull, 0x8ull,
     0x100000ull, 0x10ull, 0x0ull, 0x0ull, 0x0ull, 0x0ull, 0x0ull, 0x0ull},
    {0x0ull, 0x0ull, 0x0ull, 0x1ull, 0x10000ull, 0x2ull, 0x4ull, 0x20000ull, 0x40000ull, 0x8ull,
     0x80000ull, 0x10ull, 0x0ull, 0x0ull, 0x0ull, 0x0ull, 0x0ull, 0x0ull},
    {0x0ull, 0x0ull, 0x0ull, 0x0ull, 0x1ull, 0x10000ull, 0x2ull, 0x20000ull, 0x40000ull, 0x4ull,
     0x80000ull, 0x8ull, 0x0ull, 0x0ull, 0x0ull, 0x0ull, 0x0ull, 0x0ull},
    {0x1ull, 0x2ull, 0x4ull, 0x20000ull, 0x10000ull, 0x40000ull, 0x8ull, 0x80000ull, 0x100000ull,
     0x10ull, 0x200000ull, 0x20ull, 0x400000ull, 0x40ull, 0x800000ull, 0x80ull, 0x0ull, 0x0ull},
    {0x0ull, 0x1ull, 0x2ull, 0x4ull, 0x10000ull, 0x20000ull, 0x40000ull, 0x8ull, 0x80000ull,
     0x10ull, 0x100000ull, 0x20ull, 0x200000ull, 0x40ull, 0x400000ull, 0x80ull, 0x0ull, 0x0ull},
    {0x0ull, 0x0ull, 0x1ull, 0x2ull, 0x10000ull, 0x20000ull, 0x40000ull, 0x4ull, 0x80000ull, 0x8ull,
     0x100000ull, 0x10ull, 0x200000ull, 0x20ull, 0x400000ull, 0x40ull, 0x0ull, 0x0ull},
    {0x0ull, 0x0ull, 0x0ull, 0x1ull, 0x10000ull, 0x2ull, 0x4ull, 0x20000ull, 0x40000ull, 0x8ull,
     0x80000ull, 0x10ull, 0x100000ull, 0x20ull, 0x200000ull, 0x40ull, 0x0ull, 0x0ull},
    {0x0ull, 0x0ull, 0x0ull, 0x0ull, 0x1ull, 0x10000ull, 0x2ull, 0x20000ull, 0x40000ull, 0x4ull,
     0x80000ull, 0x8ull, 0x100000ull, 0x10ull, 0x200000ull, 0x20ull, 0x0ull, 0x0ull},
    {0x1ull, 0x10000ull, 0x2ull, 0x20000ull, 0x4ull, 0x40000ull, 0x8ull, 0x80000ull, 0x100000ull,
     0x10ull, 0x200000ull, 0x20ull, 0x400000ull, 0x40ull, 0x800000ull, 0x80ull, 0x0ull, 0x0ull},
    {0x0ull, 0x1ull, 0x10000ull, 0x2ull, 0x20000ull, 0x4ull, 0x40000ull, 0x8ull, 0x80000ull,
     0x10ull, 0x100000ull, 0x20ull, 0x200000ull, 0x40ull, 0x400000ull, 0x80ull, 0x0ull, 0x0ull},
    {0x0ull, 0x0ull, 0x1ull, 0x10000ull, 0x2ull, 0x20000ull, 0x4ull, 0x40000ull, 0x80000ull, 0x8ull,
     0x100000ull, 0x10ull, 0x200000ull, 0x20ull, 0x400000ull, 0x40ull, 0x0ull, 0x0ull},
    {0x0ull, 0x0ull, 0x0ull, 0x1ull, 0x10000ull, 0x2ull, 0x20000ull, 0x4ull, 0x40000ull, 0x8ull,
     0x80000ull, 0x10ull, 0x100000ull, 0x20ull, 0x200000ull, 0x40ull, 0x0ull, 0x0ull},
    {0x1ull, 0x2ull, 0x4ull, 0x20000ull, 0x10000ull, 0x40000ull, 0x8ull, 0x80000ull, 0x800100080ull,
     0x400800010ull, 0x200200040ull, 0x100400020ull, 0x0ull, 0x0ull, 0x0ull, 0x0ull, 0x0ull,
     0x0ull},
    {0x0ull, 0x1ull, 0x2ull, 0x4ull, 0x10000ull, 0x20000ull, 0x40000ull, 0x8ull, 0x800080080ull,
     0x400400010ull, 0x200100040ull, 0x100200020ull, 0x0ull, 0x0ull, 0x0ull, 0x0ull, 0x0ull,
     0x0ull},
    {0x0ull, 0x0ull, 0x1ull, 0x2ull, 0x10000ull, 0x20000ull, 0x40000ull, 0x4ull, 0x800080040ull,
     0x400400008ull, 0x200100020ull, 0x100200010ull, 0x0ull, 0x0ull, 0x0ull, 0x0ull, 0x0ull,
     0x0ull},
    {0x0ull, 0x0ull, 0x0ull, 0x1ull, 0x10000ull, 0x2ull, 0x4ull, 0x20000ull, 0x800040040ull,
     0x400200008ull, 0x200080020ull, 0x100100010ull, 0x0ull, 0x0ull, 0x0ull, 0x0ull, 0x0ull,
     0x0ull},
    {0x0ull, 0x0ull, 0x0ull, 0x0ull, 0x1ull, 0x10000ull, 0x2ull, 0x20000ull, 0x800040020ull,
     0x400200004ull, 0x200080010ull, 0x100100008ull, 0x0ull, 0x0ull, 0x0ull, 0x0ull, 0x0ull,
     0x0ull},
    {0x1ull, 0x2ull, 0x4ull, 0x20000ull, 0x10000ull, 0x40000ull, 0x8ull, 0x80000ull, 0x800100080ull,
     0x400800010ull, 0x200200040ull, 0x100400020ull, 0x400000ull, 0x40ull, 0x800000ull, 0x80ull,
     0x0ull, 0x0ull},
    {0x0ull, 0x1ull, 0x2ull, 0x4ull, 0x10000ull, 0x20000ull, 0x40000ull, 0x8ull, 0x800080080ull,
     0x400400010ull, 0x200100040ull, 0x100200020ull, 0x200000ull, 0x40ull, 0x400000ull, 0x80ull,
     0x0ull, 0x0ull},
    {0x0ull, 0x0ull, 0x1ull, 0x2ull, 0x10000ull, 0x20000ull, 0x40000ull, 0x4ull, 0x800080040ull,
     0x400400008ull, 0x200100020ull, 0x100200010ull, 0x200000ull, 0x20ull, 0x400000ull, 0x40ull,
     0x0ull, 0x0ull},
    {0x0ull, 0x0ull, 0x0ull, 0x1ull, 0x10000ull, 0x2ull, 0x4ull, 0x20000ull, 0x800040040ull,
     0x400200008ull, 0x200080020ull, 0x100100010ull, 0x100000ull, 0x20ull, 0x200000ull, 0x40ull,
     0x0ull, 0x0ull},
    {0x0ull, 0x0ull, 0x0ull, 0x0ull, 0x1ull, 0x10000ull, 0x2ull, 0x20000ull, 0x800040020ull,
     0x400200004ull, 0x200080010ull, 0x100100008ull, 0x100000ull, 0x10ull, 0x200000ull, 0x20ull,
     0x0ull, 0x0ull},
    {0x1ull, 0x2ull, 0x4ull, 0x20000ull, 0x10000ull, 0x40000ull, 0x8ull, 0x100000ull,
     0x800080008ull, 0x400100010ull, 0x200200040ull, 0x100400020ull, 0x400000ull, 0x40ull,
     0x800000ull, 0x80ull, 0x0ull, 0x0ull},
    {0x0ull, 0x1ull, 0x2ull, 0x4ull, 0x10000ull, 0x20000ull, 0x40000ull, 0x8ull, 0x800080008ull,
     0x400100010ull, 0x200200040ull, 0x100400020ull, 0x100000ull, 0x40ull, 0x400000ull, 0x80ull,
     0x0ull, 0x0ull},
    {0x0ull, 0x0ull, 0x1ull, 0x2ull, 0x10000ull, 0x20000ull, 0x40000ull, 0x4ull, 0x800080008ull,
     0x400100010ull, 0x200200040ull, 0x100400020ull, 0x80000ull, 0x10ull, 0x400000ull, 0x40ull,
     0x0ull, 0x0ull},
    {0x0ull, 0x0ull, 0x0ull, 0x1ull, 0x10000ull, 0x2ull, 0x4ull, 0x20000ull, 0x800080008ull,
     0x400100010ull, 0x200200040ull, 0x100400020ull, 0x40000ull, 0x8ull, 0x100000ull, 0x40ull,
     0x0ull, 0x0ull},
    {0x0ull, 0x0ull, 0x0ull, 0x0ull, 0x1ull, 0x10000ull, 0x2ull, 0x20000ull, 0x800080008ull,
     0x400100010ull, 0x200200040ull, 0x100400020ull, 0x40000ull, 0x4ull, 0x80000ull, 0x10ull,
     0x0ull, 0x0ull},
    {0x1ull, 0x10000ull, 0x2ull, 0x20000ull, 0x4ull, 0x40000ull, 0x8ull, 0x100000ull,
     0x800080008ull, 0x400100010ull, 0x200200040ull, 0x100400020ull, 0x400000ull, 0x40ull,
     0x800000ull, 0x80ull, 0x0ull, 0x0ull},
    {0x0ull, 0x1ull, 0x10000ull, 0x2ull, 0x20000ull, 0x4ull, 0x40000ull, 0x8ull, 0x800080008ull,
     0x400100010ull, 0x200200040ull, 0x100400020ull, 0x100000ull, 0x40ull, 0x400000ull, 0x80ull,
     0x0ull, 0x0ull},
    {0x0ull, 0x0ull, 0x1ull, 0x10000ull, 0x2ull, 0x20000ull, 0x4ull, 0x40000ull, 0x800080008ull,
     0x400100010ull, 0x200200040ull, 0x100400020ull, 0x80000ull, 0x10ull, 0x400000ull, 0x40ull,
     0x0ull, 0x0ull},
    {0x0ull, 0x0ull, 0x0ull, 0x1ull, 0x10000ull, 0x2ull, 0x20000ull, 0x4ull, 0x800080008ull,
     0x400100010ull, 0x200200040ull, 0x100400020ull, 0x40000ull, 0x8ull, 0x100000ull, 0x40ull,
     0x0ull, 0x0ull},
    {0x0ull, 0x0ull, 0x1ull, 0x2ull, 0x10000ull, 0x20000ull, 0x4ull, 0x40000ull, 0x80000ull, 0x8ull,
     0x100000ull, 0x10ull, 0x0ull, 0x0ull, 0x0ull, 0x0ull, 0x0ull, 0x0ull},
    {0x0ull, 0x0ull, 0x1ull, 0x2ull, 0x10000ull, 0x20000ull, 0x4ull, 0x40000ull, 0x80000ull, 0x8ull,
     0x100000ull, 0x10ull, 0x200000ull, 0x20ull, 0x400000ull, 0x40ull, 0x0ull, 0x0ull},
    {0x1ull, 0x2ull, 0x4ull, 0x8ull, 0x10000ull, 0x20000ull, 0x40000ull, 0x80000ull, 0x10ull,
     0x100000ull, 0x4000020ull, 0x200400ull, 0x2000040ull, 0x400200ull, 0x1000080ull, 0x800100ull,
     0x0ull, 0x0ull},
    {0x0ull, 0x1ull, 0x2ull, 0x4ull, 0x10000ull, 0x20000ull, 0x40000ull, 0x8ull, 0x80000ull,
     0x10ull, 0x100400ull, 0x2000020ull, 0x200200ull, 0x1000040ull, 0x400100ull, 0x800080ull,
     0x0ull, 0x0ull},
    {0x0ull, 0x0ull, 0x1ull, 0x2ull, 0x10000ull, 0x20000ull, 0x4ull, 0x40000ull, 0x8ull, 0x80000ull,
     0x2000010ull, 0x100200ull, 0x1000020ull, 0x200100ull, 0x800040ull, 0x400080ull, 0x0ull,
     0x0ull},
    {0x0ull, 0x0ull, 0x0ull, 0x1ull, 0x10000ull, 0x2ull, 0x4ull, 0x20000ull, 0x40000ull, 0x8ull,
     0x80200ull, 0x1000010ull, 0x100100ull, 0x800020ull, 0x200080ull, 0x400040ull, 0x0ull, 0x0ull},
    {0x0ull, 0x0ull, 0x0ull, 0x0ull, 0x1ull, 0x10000ull, 0x2ull, 0x20000ull, 0x4ull, 0x40000ull,
     0x1000008ull, 0x80100ull, 0x800010ull, 0x100080ull, 0x400020ull, 0x200040ull, 0x0ull, 0x0ull},
    {0x1ull, 0x10000ull, 0x2ull, 0x20000ull, 0x4ull, 0x40000ull, 0x8ull, 0x80000ull, 0x10ull,
     0x100000ull, 0x4000020ull, 0x200400ull, 0x2000040ull, 0x400200ull, 0x1000080ull, 0x800100ull,
     0x0ull, 0x0ull},
    {0x0ull, 0x1ull, 0x10000ull, 0x2ull, 0x20000ull, 0x4ull, 0x40000ull, 0x8ull, 0x80000ull,
     0x10ull, 0x100400ull, 0x2000020ull, 0x200200ull, 0x1000040ull, 0x400100ull, 0x800080ull,
     0x0ull, 0x0ull},
    {0x0ull, 0x0ull, 0x1ull, 0x10000ull, 0x2ull, 0x20000ull, 0x4ull, 0x40000ull, 0x8ull, 0x80000ull,
     0x2000010ull, 0x100200ull, 0x1000020ull, 0x200100ull, 0x800040ull, 0x400080ull, 0x0ull,
     0x0ull},
    {0x0ull, 0x0ull, 0x0ull, 0x1ull, 0x10000ull, 0x2ull, 0x20000ull, 0x4ull, 0x40000ull, 0x8ull,
     0x80200ull, 0x1000010ull, 0x100100ull, 0x800020ull, 0x200080ull, 0x400040ull, 0x0ull, 0x0ull},
    {0x0ull, 0x0ull, 0x1ull, 0x2ull, 0x10000ull, 0x20000ull, 0x4ull, 0x40000ull, 0x800080040ull,
     0x400400008ull, 0x200100020ull, 0x100200010ull, 0x0ull, 0x0ull, 0x0ull, 0x0ull, 0x0ull,
     0x0ull},
    {0x0ull, 0x0ull, 0x1ull, 0x2ull, 0x10000ull, 0x20000ull, 0x4ull, 0x40000ull, 0x800080040ull,
     0x400400008ull, 0x200100020ull, 0x100200010ull, 0x200000ull, 0x20ull, 0x400000ull, 0x40ull,
     0x0ull, 0x0ull},
    {0x1ull, 0x2ull, 0x4ull, 0x8ull, 0x10000ull, 0x20000ull, 0x40000ull, 0x80000ull, 0x1100100ull,
     0x800100010ull, 0x400200080ull, 0x200800020ull, 0x40ull, 0x80ull, 0x800100ull, 0x100400040ull,
     0x0ull, 0x0ull},
    {0x0ull, 0x1ull, 0x2ull, 0x4ull, 0x10000ull, 0x20000ull, 0x40000ull, 0x8ull, 0x1100100ull,
     0x800100010ull, 0x400200080ull, 0x200800020ull, 0x80000ull, 0x40ull, 0x800080ull,
     0x100400040ull, 0x0ull, 0x0ull},
    {0x0ull, 0x0ull, 0x1ull, 0x2ull, 0x10000ull, 0x20000ull, 0x4ull, 0x40000ull, 0x1100100ull,
     0x800100010ull, 0x400200080ull, 0x200800020ull, 0x8ull, 0x80000ull, 0x800040ull,
     0x100400040ull, 0x0ull, 0x0ull},
    {0x0ull, 0x0ull, 0x0ull, 0x1ull, 0x10000ull, 0x2ull, 0x4ull, 0x20000ull, 0x1100100ull,
     0x800100010ull, 0x400200080ull, 0x200800020ull, 0x40000ull, 0x8ull, 0x80080ull, 0x100400040ull,
     0x0ull, 0x0ull},
    {0x0ull, 0x0ull, 0x0ull, 0x0ull, 0x1ull, 0x10000ull, 0x2ull, 0x20000ull, 0x1100100ull,
     0x400100010ull, 0x200200080ull, 0x100800020ull, 0x4ull, 0x8ull, 0x80040ull, 0x440040ull,
     0x0ull, 0x0ull},
    {0x1ull, 0x10000ull, 0x2ull, 0x20000ull, 0x4ull, 0x40000ull, 0x8ull, 0x80000ull, 0x1100100ull,
     0x200100010ull, 0x100200080ull, 0x800020ull, 0x40ull, 0x80ull, 0x800100ull, 0x400040ull,
     0x0ull, 0x0ull},
    {0x0ull, 0x1ull, 0x10000ull, 0x2ull, 0x20000ull, 0x4ull, 0x40000ull, 0x8ull, 0x1100100ull,
     0x200100010ull, 0x100200080ull, 0x800020ull, 0x80000ull, 0x40ull, 0x800080ull, 0x400040ull,
     0x0ull, 0x0ull},
    {0x0ull, 0x0ull, 0x1ull, 0x10000ull, 0x2ull, 0x20000ull, 0x4ull, 0x40000ull, 0x1100100ull,
     0x200100010ull, 0x100200080ull, 0x800020ull, 0x8ull, 0x80000ull, 0x800040ull, 0x400040ull,
     0x0ull, 0x0ull},
    {0x0ull, 0x0ull, 0x0ull, 0x1ull, 0x10000ull, 0x2ull, 0x20000ull, 0x4ull, 0x1100100ull,
     0x800100010ull, 0x400200080ull, 0x200800020ull, 0x40000ull, 0x8ull, 0x80080ull, 0x100400040ull,
     0x0ull, 0x0ull},
    {0x1ull, 0x2ull, 0x10000ull, 0x4ull, 0x20000ull, 0x40000ull, 0x8ull, 0x80000ull, 0x100000ull,
     0x10ull, 0x200000ull, 0x20ull, 0x0ull, 0x0ull, 0x0ull, 0x0ull, 0x0ull, 0x0ull},
    {0x0ull, 0x1ull, 0x10000ull, 0x2ull, 0x20000ull, 0x4ull, 0x40000ull, 0x8ull, 0x80000ull,
     0x10ull, 0x100000ull, 0x20ull, 0x0ull, 0x0ull, 0x0ull, 0x0ull, 0x0ull, 0x0ull},
    {0x0ull, 0x0ull, 0x1ull, 0x10000ull, 0x2ull, 0x20000ull, 0x4ull, 0x40000ull, 0x80000ull, 0x8ull,
     0x100000ull, 0x10ull, 0x0ull, 0x0ull, 0x0ull, 0x0ull, 0x0ull, 0x0ull},
    {0x1ull, 0x2ull, 0x10000ull, 0x4ull, 0x20000ull, 0x40000ull, 0x8ull, 0x80000ull, 0x100000ull,
     0x10ull, 0x200000ull, 0x20ull, 0x400000ull, 0x40ull, 0x800000ull, 0x80ull, 0x0ull, 0x0ull},
    {0x1ull, 0x2ull, 0x10000ull, 0x4ull, 0x20000ull, 0x40000ull, 0x8ull, 0x80000ull, 0x10ull,
     0x100000ull, 0x1000020ull, 0x200100ull, 0x800040ull, 0x400080ull, 0x80ull, 0x800000ull, 0x0ull,
     0x0ull},
    {0x0ull, 0x1ull, 0x10000ull, 0x2ull, 0x20000ull, 0x4ull, 0x40000ull, 0x8ull, 0x80000ull,
     0x10ull, 0x100100ull, 0x800020ull, 0x200080ull, 0x400040ull, 0x400000ull, 0x80ull, 0x0ull,
     0x0ull},
    {0x0ull, 0x0ull, 0x1ull, 0x10000ull, 0x2ull, 0x20000ull, 0x4ull, 0x40000ull, 0x8ull, 0x80000ull,
     0x800010ull, 0x100080ull, 0x400020ull, 0x200040ull, 0x40ull, 0x400000ull, 0x0ull, 0x0ull},
    {0x0ull, 0x0ull, 0x0ull, 0x1ull, 0x10000ull, 0x2ull, 0x4ull, 0x20000ull, 0x40000ull, 0x8ull,
     0x80080ull, 0x800010ull, 0x100040ull, 0x400020ull, 0x200000ull, 0x40ull, 0x0ull, 0x0ull},
    {0x0ull, 0x0ull, 0x0ull, 0x0ull, 0x1ull, 0x10000ull, 0x2ull, 0x20000ull, 0x4ull, 0x40000ull,
     0x800008ull, 0x80040ull, 0x400010ull, 0x100020ull, 0x20ull, 0x200000ull, 0x0ull, 0x0ull},
    {0x1ull, 0x2ull, 0x10000ull, 0x4ull, 0x20000ull, 0x40000ull, 0x8ull, 0x80000ull, 0x10ull,
     0x100000ull, 0x1000020ull, 0x200100ull, 0x800040ull, 0x400080ull, 0x80ull, 0x800000ull,
     0x100ull, 0x1000000ull},
    {0x0ull, 0x1ull, 0x10000ull, 0x2ull, 0x20000ull, 0x4ull, 0x40000ull, 0x8ull, 0x80000ull,
     0x10ull, 0x100100ull, 0x800020ull, 0x200080ull, 0x400040ull, 0x400000ull, 0x80ull, 0x800000ull,
     0x100ull},
    {0x0ull, 0x0ull, 0x1ull, 0x10000ull, 0x2ull, 0x20000ull, 0x4ull, 0x40000ull, 0x8ull, 0x80000ull,
     0x800010ull, 0x100080ull, 0x400020ull, 0x200040ull, 0x40ull, 0x400000ull, 0x80ull,
     0x800000ull},
    {0x0ull, 0x0ull, 0x0ull, 0x1ull, 0x10000ull, 0x2ull, 0x4ull, 0x20000ull, 0x40000ull, 0x8ull,
     0x80080ull, 0x800010ull, 0x100040ull, 0x400020ull, 0x200000ull, 0x40ull, 0x400000ull, 0x80ull},
    {0x0ull, 0x0ull, 0x0ull, 0x0ull, 0x1ull, 0x10000ull, 0x2ull, 0x20000ull, 0x4ull, 0x40000ull,
     0x800008ull, 0x80040ull, 0x400010ull, 0x100020ull, 0x20ull, 0x200000ull, 0x40ull, 0x400000ull},
    {0x1ull, 0x2ull, 0x10000ull, 0x4ull, 0x20000ull, 0x40000ull, 0x8ull, 0x80000ull, 0x800100080ull,
     0x400800010ull, 0x200200040ull, 0x100400020ull, 0x0ull, 0x0ull, 0x0ull, 0x0ull, 0x0ull,
     0x0ull},
    {0x0ull, 0x1ull, 0x10000ull, 0x2ull, 0x20000ull, 0x4ull, 0x40000ull, 0x8ull, 0x800080080ull,
     0x400400010ull, 0x200100040ull, 0x100200020ull, 0x0ull, 0x0ull, 0x0ull, 0x0ull, 0x0ull,
     0x0ull},
    {0x0ull, 0x0ull, 0x1ull, 0x10000ull, 0x2ull, 0x20000ull, 0x4ull, 0x40000ull, 0x800080040ull,
     0x400400008ull, 0x200100020ull, 0x100200010ull, 0x0ull, 0x0ull, 0x0ull, 0x0ull, 0x0ull,
     0x0ull},
    {0x1ull, 0x2ull, 0x10000ull, 0x4ull, 0x20000ull, 0x40000ull, 0x8ull, 0x80000ull,
     0x1000100100ull, 0x801000010ull, 0x400200080ull, 0x200800020ull, 0x100400040ull, 0x40ull,
     0x800000ull, 0x80ull, 0x0ull, 0x0ull},
    {0x0ull, 0x1ull, 0x10000ull, 0x2ull, 0x20000ull, 0x4ull, 0x40000ull, 0x8ull, 0x1000080100ull,
     0x800800010ull, 0x400100080ull, 0x200400020ull, 0x100200040ull, 0x40ull, 0x400000ull, 0x80ull,
     0x0ull, 0x0ull},
    {0x0ull, 0x0ull, 0x1ull, 0x10000ull, 0x2ull, 0x20000ull, 0x4ull, 0x40000ull, 0x1000080080ull,
     0x800800008ull, 0x400100040ull, 0x200400010ull, 0x100200020ull, 0x20ull, 0x400000ull, 0x40ull,
     0x0ull, 0x0ull},
    {0x0ull, 0x0ull, 0x0ull, 0x1ull, 0x10000ull, 0x2ull, 0x4ull, 0x20000ull, 0x1000040080ull,
     0x800400008ull, 0x400080040ull, 0x200200010ull, 0x100100020ull, 0x20ull, 0x200000ull, 0x40ull,
     0x0ull, 0x0ull},
    {0x0ull, 0x0ull, 0x0ull, 0x0ull, 0x1ull, 0x10000ull, 0x2ull, 0x20000ull, 0x1000040040ull,
     0x800400004ull, 0x400080020ull, 0x200200008ull, 0x100100010ull, 0x10ull, 0x200000ull, 0x20ull,
     0x0ull, 0x0ull},
    {0x1ull, 0x2ull, 0x10000ull, 0x4ull, 0x20000ull, 0x40000ull, 0x8ull, 0x80000ull, 0x2100200ull,
     0x200100010ull, 0x100200100ull, 0x1000020ull, 0x400080ull, 0x80ull, 0x800000ull, 0x800040ull,
     0x0ull, 0x0ull},
    {0x0ull, 0x1ull, 0x10000ull, 0x2ull, 0x20000ull, 0x4ull, 0x40000ull, 0x8ull, 0x2100200ull,
     0x200100010ull, 0x100200100ull, 0x1000020ull, 0x400080ull, 0x80000ull, 0x80ull, 0x800040ull,
     0x0ull, 0x0ull},
    {0x0ull, 0x0ull, 0x1ull, 0x10000ull, 0x2ull, 0x20000ull, 0x4ull, 0x40000ull, 0x2100200ull,
     0x200100010ull, 0x100200100ull, 0x1000020ull, 0x400080ull, 0x8ull, 0x80000ull, 0x800040ull,
     0x0ull, 0x0ull},
    {0x0ull, 0x0ull, 0x0ull, 0x1ull, 0x10000ull, 0x2ull, 0x4ull, 0x20000ull, 0x2100200ull,
     0x800100010ull, 0x400200100ull, 0x201000020ull, 0x440080ull, 0x8ull, 0x80000ull,
     0x100800040ull, 0x0ull, 0x0ull},
    {0x0ull, 0x0ull, 0x0ull, 0x0ull, 0x1ull, 0x10000ull, 0x2ull, 0x20000ull, 0x2100200ull,
     0x400100010ull, 0x200200100ull, 0x101000020ull, 0x440080ull, 0x8ull, 0x80000ull, 0x800044ull,
     0x0ull, 0x0ull},
    {0x1ull, 0x2ull, 0x10000ull, 0x4ull, 0x20000ull, 0x40000ull, 0x8ull, 0x80000ull, 0x2100200ull,
     0x200100010ull, 0x100200100ull, 0x1000020ull, 0x400080ull, 0x80ull, 0x800000ull, 0x2000100ull,
     0x1000200ull, 0x800040ull},
    {0x0ull, 0x1ull, 0x10000ull, 0x2ull, 0x20000ull, 0x4ull, 0x40000ull, 0x8ull, 0x2100200ull,
     0x200100010ull, 0x100200100ull, 0x1000020ull, 0x400080ull, 0x80000ull, 0x80ull, 0x1000100ull,
     0x800200ull, 0x800040ull},
    {0x0ull, 0x0ull, 0x1ull, 0x10000ull, 0x2ull, 0x20000ull, 0x4ull, 0x40000ull, 0x2100200ull,
     0x200100010ull, 0x100200100ull, 0x1000020ull, 0x400080ull, 0x8ull, 0x80000ull, 0x1000080ull,
     0x800100ull, 0x800040ull},
    {0x0ull, 0x0ull, 0x0ull, 0x1ull, 0x10000ull, 0x2ull, 0x4ull, 0x20000ull, 0x2100200ull,
     0x1000100010ull, 0x800200100ull, 0x401000020ull, 0x200400080ull, 0x40000ull, 0x8ull,
     0x80100ull, 0x800080ull, 0x100800040ull},
    {0x0ull, 0x0ull, 0x0ull, 0x0ull, 0x1ull, 0x10000ull, 0x2ull, 0x20000ull, 0x2100200ull,
     0x1000100010ull, 0x800200100ull, 0x401000020ull, 0x200400080ull, 0x4ull, 0x40000ull,
     0x80080ull, 0x800008ull, 0x100800040ull},
};
constexpr uint16_t kDeviceSelectors[6][5][5] = {
    {{0, 1, 2, 3, 4},
     {5, 6, 7, 8, 9},
     {5, 6, 7, 8, 9},
     {10, 11, 12, 13, 65535},
     {65535, 65535, 65535, 65535, 65535}},
    {{14, 15, 16, 17, 18},
     {19, 20, 21, 22, 23},
     {24, 25, 26, 27, 28},
     {29, 30, 31, 32, 65535},
     {65535, 65535, 65535, 65535, 65535}},
    {{0, 1, 33, 3, 4},
     {5, 6, 34, 8, 9},
     {35, 36, 37, 38, 39},
     {40, 41, 42, 43, 65535},
     {65535, 65535, 65535, 65535, 65535}},
    {{14, 15, 44, 17, 18},
     {19, 20, 45, 22, 23},
     {46, 47, 48, 49, 50},
     {51, 52, 53, 54, 65535},
     {65535, 65535, 65535, 65535, 65535}},
    {{55, 56, 57, 3, 4},
     {58, 11, 12, 8, 9},
     {59, 60, 61, 62, 63},
     {59, 60, 61, 62, 65535},
     {64, 65, 66, 67, 68}},
    {{69, 70, 71, 17, 18},
     {72, 73, 74, 75, 76},
     {77, 78, 79, 80, 81},
     {77, 78, 79, 80, 65535},
     {82, 83, 84, 85, 86}},
};

// Positive selectors name X bits, negative selectors name Y bits (one-based).
// Zero reserves a byte-within-element bit, supplied by the memory access.
constexpr int8_t kMicrotiles[4][5][8] = {{{1, 2, 3, -2, -1, -3, 4, -4},
                                          {0, 1, 2, 3, -1, -2, -3, 4},
                                          {0, 0, 1, 2, -1, 3, -2, -3},
                                          {0, 0, 0, 1, -1, 2, 3, -2},
                                          {0, 0, 0, 0, 1, -1, 2, -2}},
                                         {{1, 2, 3, -2, -1, -3, 4, -4},
                                          {0, 1, 2, 3, -1, -2, -3, 4},
                                          {0, 0, 1, 2, -1, -2, -3, 3},
                                          {0, 0, 0, 1, -1, 2, 3, -2},
                                          {0, 0, 0, 0, 1, -1, 2, -2}},
                                         {{1, 2, -1, 3, -2, -3, 4, -4},
                                          {0, 1, -1, 2, -2, 3, -3, 4},
                                          {0, 0, 1, -1, 2, -2, 3, -3},
                                          {0, 0, 0, 1, -1, 2, 3, -2},
                                          {0, 0, 0, 0, 1, -1, 2, -2}},
                                         {{1, 2, 3, 4, -1, -2, -3, -4},
                                          {0, 1, 2, 3, -1, -2, -3, 4},
                                          {0, 0, 1, 2, -1, -2, -3, 3},
                                          {0, 0, 0, 1, -1, -2, 2, 3},
                                          {0, 0, 0, 0, -1, -2, 1, 2}}};

// GFX12 single-sample 2D ordering, also used for thin 3D image views. These
// are native coordinate-bit permutations for the supported thin layouts.
// This XY selector space addresses thin slices; graphics thick-volume tables
// encode all three spatial axes with a different selector representation.
constexpr int8_t kGfx12Tiles[5][18] = {
    {1, 2, -1, 3, -2, -3, 4, -4, -5, 5, -6, 6, -7, 7, -8, 8, -9, 9},
    {0, 1, -1, 2, -2, 3, -3, 4, -4, 5, -5, 6, -6, 7, -7, 8, -8, 9},
    {0, 0, 1, -1, 2, -2, 3, -3, -4, 4, -5, 5, -6, 6, -7, 7, -8, 8},
    {0, 0, 0, 1, -1, 2, 3, -2, -3, 4, -4, 5, -5, 6, -6, 7, -7, 8},
    {0, 0, 0, 0, 1, -1, 2, -2, -3, 3, -4, 4, -5, 5, -6, 6, -7, 7}};

uint32_t permutation(const int8_t *selectors, uint32_t bits, uint32_t x, uint32_t y) {
  uint32_t result = 0;
  for (uint32_t bit = 0; bit < bits; ++bit) {
    const int selector = selectors[bit];
    if (selector != 0)
      result |= (((selector > 0 ? x : y) >> (std::abs(selector) - 1)) & 1u) << bit;
  }
  return result;
}

// The exposed GFX9 topology has no pipe/bank XOR bits. Thin surfaces
// interleave 256-byte microtiles; array slices are handled by their stride.
uint32_t gfx9_tile_offset(uint32_t swizzle_mode, uint32_t element, uint32_t block_bits, uint32_t x,
                          uint32_t y, bool volume) {
  const bool standard = volume || (swizzle_mode & 3) == 1;
  uint32_t offset = permutation(kMicrotiles[standard ? 3 : 0][element], 8, x, y);
  const uint32_t x_bits = (9 - element) / 2;
  const uint32_t y_bits = (8 - element) / 2;
  for (uint32_t i = 0; i < 12; ++i) {
    offset |= ((y >> (y_bits + i)) & 1u) << (8 + 2 * i);
    offset |= ((x >> (x_bits + i)) & 1u) << (9 + 2 * i);
  }
  return offset & ((1u << block_bits) - 1);
}

std::optional<uint32_t> tile_offset(const ImageAddressConfig &config, const ImageLayout &layout,
                                    uint32_t block_bits, uint32_t element, uint32_t x, uint32_t y,
                                    uint32_t z) {
  const uint32_t swizzle_mode = layout.swizzle_mode;
  if (arch_is_cdna_4_or_lower(config.arch)) {
    // The shipped CDNA image targets expose this topology; other register
    // configurations require independent layout qualification.
    if (config.gb_addr_config != 0)
      return std::nullopt;
    return gfx9_tile_offset(swizzle_mode, element, block_bits, x, y, layout.dimensions == 3);
  }
  if (config.arch == ROCJITSU_CODE_ARCH_RDNA4)
    return permutation(kGfx12Tiles[element], block_bits, x, y);
  const bool rdna1 = config.arch == ROCJITSU_CODE_ARCH_RDNA1;
  const bool rdna2 = config.arch == ROCJITSU_CODE_ARCH_RDNA2;
  // RB+ keeps the GFX10 display ordering except for the 32-bit microtile.
  constexpr int8_t kRbPlus32[8] = {0, 0, 1, 2, -1, -2, 3, -3};
  const auto *microtile =
      rdna2 && element == 2 ? kRbPlus32 : kMicrotiles[rdna1 || rdna2 ? 1 : 2][element];
  uint32_t offset = permutation(microtile, 8, x, y);
  if (swizzle_mode < 16) {
    const uint32_t x_bits = (9 - element) / 2, y_bits = (8 - element) / 2;
    for (uint32_t bit = 8; bit < block_bits; ++bit)
      offset |= (((bit & 1) ? x >> (x_bits + (bit - 8) / 2) : y >> (y_bits + (bit - 8) / 2)) & 1u)
                << bit;
    return offset;
  }
  const uint32_t expected_config = rdna1 ? 0x44u : rdna2 ? 0x444u : 0x545u;
  if (config.gb_addr_config != 0 && config.gb_addr_config != expected_config)
    return std::nullopt;
  const uint32_t index = (rdna1 ? 0u : rdna2 ? 2u : 4u) + (config.gb_addr_config != 0);
  const uint32_t kind = swizzle_mode == 22   ? 0
                        : swizzle_mode == 26 ? 1
                        : swizzle_mode == 27 ? 2
                        : swizzle_mode == 24 ? 3
                                             : 4;
  const uint32_t row = kDeviceSelectors[index][kind][element];
  if (row == 65535)
    return std::nullopt;
  const uint64_t coordinates = uint64_t{x} | (uint64_t{y} << 16) | (uint64_t{z} << 32);
  offset = 0;
  for (uint32_t bit = 0; bit < block_bits; ++bit)
    offset |= (std::popcount(coordinates & kXorSelectors[row][bit]) & 1u) << bit;
  return offset;
}

struct MipAddress {
  uint64_t offset = 0;
  uint64_t slice = 0;
  uint32_t pitch = 0;
  uint32_t tail_x = 0;
  uint32_t tail_y = 0;
  uint32_t tail_bits = 0;
};

uint32_t mip_extent(uint32_t extent, uint32_t level) {
  return (extent + (1u << level) - 1) >> level;
}

// GFX9 arranges large mips in a two-dimensional block atlas. The remaining
// levels occupy fixed 256-byte tail slots, applied after the swizzle equation.
MipAddress gfx9_mip_address(const ImageLayout &layout, uint32_t block_bits, uint32_t block_width,
                            uint32_t block_height, uint32_t mip) {
  const uint32_t width = util::align_up<uint64_t>(layout.width, block_width);
  const uint32_t height = util::align_up<uint64_t>(layout.height, block_height);
  uint32_t first_tail = layout.mip_levels;
  uint32_t mip_width = layout.width, mip_height = layout.height;
  for (uint32_t i = 0; i < layout.mip_levels; ++i) {
    if (mip_width <= block_width / 2 && mip_height <= block_height) {
      first_tail = i;
      break;
    }
    mip_width =
        std::max(1u, static_cast<uint32_t>(util::align_up<uint64_t>(mip_width, block_width)) / 2);
    mip_height =
        std::max(1u, static_cast<uint32_t>(util::align_up<uint64_t>(mip_height, block_height)) / 2);
  }
  const bool y_major = width / block_width < height / block_height;
  uint32_t chain_width = width, chain_height = height;
  const uint32_t ending = std::min(first_tail, layout.mip_levels - 1);
  if (ending != 0) {
    if (y_major) {
      uint32_t extra = (width / block_width + 1) / 2;
      if (extra == 1 && ending > 2)
        extra = 2;
      chain_width += extra * block_width;
    } else {
      uint32_t extra = (height / block_height + 1) / 2;
      if (extra == 1 && ending > 2)
        extra = 2;
      chain_height += extra * block_height;
    }
  }
  MipAddress result;
  result.pitch = chain_width;
  result.slice = uint64_t{chain_width} * chain_height * layout.element_bytes;
  uint32_t block_x = 0, block_y = 0;
  uint32_t block_count_x = width / block_width, block_count_y = height / block_height;
  uint32_t tail_level = 0;
  bool in_tail = first_tail == 0;
  if (!in_tail) {
    for (uint32_t i = 1; i <= mip; ++i) {
      if (i == 1 || i == 3) {
        if (y_major)
          block_x += block_count_x;
        else
          block_y += block_count_y;
      } else {
        if (y_major)
          block_y += block_count_y;
        else
          block_x += block_count_x;
      }
      if (block_count_x == 1 && block_count_y <= 2) {
        in_tail = true;
        tail_level = i;
        break;
      }
      block_count_x = (block_count_x + 1) / 2;
      block_count_y = (block_count_y + 1) / 2;
    }
  }
  result.offset = (uint64_t{block_y} * (chain_width / block_width) + block_x) << block_bits;
  if (in_tail) {
    constexpr uint32_t slots[]{2048, 1024, 512, 256, 128, 64, 32, 16, 8, 6, 5, 4, 3, 2, 1, 0};
    const uint32_t index = 20 - block_bits + mip - tail_level;
    if (index < std::size(slots))
      result.tail_bits = slots[index] << 8;
  }
  return result;
}

// Native thin-image mip packing. Small
// mips precede large ones; a packed tail occupies one complete storage block.
MipAddress thin_mip_address(const ImageAddressConfig &config, const ImageLayout &layout,
                            uint32_t block_bits, uint32_t element, uint32_t block_width,
                            uint32_t block_height, uint32_t mip) {
  if (arch_is_cdna_4_or_lower(config.arch))
    return gfx9_mip_address(layout, block_bits, block_width, block_height, mip);
  MipAddress result;
  uint32_t tail_width = block_width / 2, tail_height = block_height;
  const bool gfx11 =
      config.arch == ROCJITSU_CODE_ARCH_RDNA3 || config.arch == ROCJITSU_CODE_ARCH_RDNA3_5;
  const uint32_t micro_width = 1u << ((9 - element) / 2);
  const uint32_t micro_height = 1u << ((8 - element) / 2);
  // The projected RDNA1 device is Navi10, which retains the larger Z-mode tail.
  if (config.arch != ROCJITSU_CODE_ARCH_RDNA1 &&
      (layout.swizzle_mode == 24 || layout.swizzle_mode == 28) && element < 2) {
    tail_width /= micro_width / 8;
    tail_height /= (gfx11 ? micro_width : micro_height) / 8;
  }
  const uint32_t max_tail = block_bits > 8 ? block_bits - 4 : 0;
  uint32_t first_tail = layout.mip_levels;
  for (uint32_t i = 0; i < layout.mip_levels; ++i) {
    if (mip_extent(layout.width, i) <= tail_width && mip_extent(layout.height, i) <= tail_height &&
        layout.mip_levels - i <= max_tail) {
      first_tail = i;
      break;
    }
  }
  result.pitch = util::align_up<uint64_t>(mip_extent(layout.width, mip), block_width);
  result.slice = first_tail < layout.mip_levels ? uint64_t{1} << block_bits : 0;
  for (uint32_t i = 0; i < first_tail; ++i) {
    const uint64_t size = util::align_up<uint64_t>(mip_extent(layout.width, i), block_width) *
                          util::align_up<uint64_t>(mip_extent(layout.height, i), block_height) *
                          layout.element_bytes;
    result.slice += size;
    if (i > mip)
      result.offset += size;
  }
  if (mip >= first_tail) {
    const uint32_t slot = max_tail - 1 - (mip - first_tail);
    const uint32_t offset = slot > 6 ? 16u << slot : slot << 8;
    for (uint32_t bit = 0; bit < 6; ++bit) {
      result.tail_x |= ((offset >> (9 + 2 * bit)) & 1u) << bit;
      result.tail_y |= ((offset >> (8 + 2 * bit)) & 1u) << bit;
    }
    result.tail_x *= micro_width;
    result.tail_y *= micro_height;
    result.pitch = block_width;
  } else if (first_tail < layout.mip_levels) {
    result.offset += uint64_t{1} << block_bits;
  }
  return result;
}

MipAddress linear_mip_address(const ImageAddressConfig &config, const ImageLayout &layout,
                              uint32_t mip) {
  const bool gfx9 = arch_is_cdna_4_or_lower(config.arch);
  const bool gfx12 = config.arch == ROCJITSU_CODE_ARCH_RDNA4;
  const uint32_t element_bytes = layout.element_bytes;
  uint64_t offset = 0, slice = 0;
  uint32_t row_pitch = 0;
  for (uint32_t i = 0; i < layout.mip_levels; ++i) {
    const uint32_t width = gfx9 ? layout.width : mip_extent(layout.width, i);
    const uint32_t height = mip_extent(layout.height, i);
    const bool trim = gfx12 && i == 0 && layout.dimensions != 3 && layout.depth == 1;
    const uint32_t allocated_pitch =
        util::align_up<uint64_t>(width, (trim ? 128 : 256) / element_bytes);
    const uint64_t size = uint64_t{allocated_pitch} * height * element_bytes;
    slice += size;
    if (gfx9 ? i < mip : i > mip)
      offset += size;
    // GFX12 addresses 128-byte rows even when slice allocation rounds to 256.
    if (i == mip)
      row_pitch = gfx12 ? util::align_up<uint64_t>(width, 128 / element_bytes) : allocated_pitch;
  }
  return {.offset = offset, .slice = slice, .pitch = row_pitch};
}

} // namespace

ImageAddressConfig default_image_address_config(rj_code_arch_t arch) {
  return {arch, kmd::gb_addr_config_for_arch(arch)};
}

std::optional<ImageAddressing> ImageAddressing::prepare(const ImageAddressConfig &config,
                                                        const ImageLayout &layout, uint32_t mip) {
  const bool supported_arch =
      config.arch == ROCJITSU_CODE_ARCH_CDNA1 || config.arch == ROCJITSU_CODE_ARCH_CDNA2 ||
      config.arch == ROCJITSU_CODE_ARCH_RDNA1 || config.arch == ROCJITSU_CODE_ARCH_RDNA2 ||
      config.arch == ROCJITSU_CODE_ARCH_RDNA3 || config.arch == ROCJITSU_CODE_ARCH_RDNA3_5 ||
      config.arch == ROCJITSU_CODE_ARCH_RDNA4;
  if (!supported_arch || layout.dimensions < 1 || layout.dimensions > 3 || !layout.width ||
      !layout.height || !layout.depth || layout.width > 65536 || layout.height > 65536 ||
      layout.depth > 16384 || !layout.mip_levels || layout.mip_levels > 17 ||
      mip >= layout.mip_levels ||
      layout.mip_levels >
          static_cast<uint32_t>(std::bit_width(std::max(
              {layout.width, layout.height, layout.dimensions == 3 ? layout.depth : 1u}))) ||
      !std::has_single_bit(layout.element_bytes) || layout.element_bytes > 16)
    return std::nullopt;
  ImageAddressing result;
  result.config_ = config;
  result.layout_ = layout;
  result.width_ = std::max(1u, layout.width >> mip);
  result.height_ = std::max(1u, layout.height >> mip);
  result.depth_ = layout.dimensions == 3 ? std::max(1u, layout.depth >> mip) : layout.depth;
  const uint32_t element_bytes = layout.element_bytes, swizzle_mode = layout.swizzle_mode;
  MipAddress allocation;
  if (swizzle_mode == 0) {
    if (layout.mip_levels > 1 || !layout.pitch) {
      allocation = linear_mip_address(config, layout, mip);
    } else {
      if (layout.pitch && layout.pitch < layout.width)
        return std::nullopt;
      // Patched descriptors on GFX11 and GFX12 establish rounding to the row
      // quantum. Keep unqualified misaligned pitches unsupported elsewhere.
      const uint32_t row_alignment =
          config.arch == ROCJITSU_CODE_ARCH_RDNA4 ? 128 / element_bytes : 256 / element_bytes;
      const bool rounds_pitch =
          config.arch == ROCJITSU_CODE_ARCH_RDNA3 || config.arch == ROCJITSU_CODE_ARCH_RDNA4;
      if (!rounds_pitch && layout.pitch % row_alignment != 0)
        return std::nullopt;
      allocation.pitch = util::align_up<uint64_t>(layout.pitch, row_alignment);
      allocation.slice = uint64_t{layout.height} * allocation.pitch * element_bytes;
    }
  } else {
    const bool gfx12 = config.arch == ROCJITSU_CODE_ARCH_RDNA4;
    const bool gfx9 = arch_is_cdna_4_or_lower(config.arch);
    const bool supported =
        gfx12  ? swizzle_mode <= 4
        : gfx9 ? swizzle_mode == 1 || swizzle_mode == 2 || swizzle_mode == 5 || swizzle_mode == 6 ||
                     swizzle_mode == 9 || swizzle_mode == 10 || swizzle_mode == 21 ||
                     swizzle_mode == 22 || swizzle_mode == 25 || swizzle_mode == 26
               : swizzle_mode == 2 || swizzle_mode == 6 || swizzle_mode == 10 ||
                     swizzle_mode == 22 || swizzle_mode == 24 || swizzle_mode == 26 ||
                     swizzle_mode == 27 || swizzle_mode == 28 || swizzle_mode == 31;
    if (!supported || (layout.dimensions == 1 && !gfx12) ||
        (!gfx12 && layout.dimensions == 3 &&
         ((swizzle_mode & 3) == 1 || (!gfx9 && (swizzle_mode & 3) == 2))) ||
        (!gfx12 && (swizzle_mode == 24 || swizzle_mode == 28) && element_bytes == 16))
      return std::nullopt;
    result.block_bits_ =
        gfx12                                      ? (swizzle_mode == 4 ? 18 : 4 * swizzle_mode + 4)
        : (swizzle_mode == 1 || swizzle_mode == 2) ? 8
        : (swizzle_mode == 5 || swizzle_mode == 6 || swizzle_mode == 21 || swizzle_mode == 22) ? 12
        : swizzle_mode >= 28                                                                   ? 18
                                                                                               : 16;
    const uint32_t element = std::countr_zero(element_bytes);
    result.block_width_ = 1u << ((result.block_bits_ - element + 1) / 2);
    result.block_height_ = 1u << ((result.block_bits_ - element) / 2);
    if (layout.mip_levels > 1) {
      if (gfx9 && result.block_bits_ == 8)
        return std::nullopt;
      allocation = thin_mip_address(config, layout, result.block_bits_, element,
                                    result.block_width_, result.block_height_, mip);
    } else {
      allocation.pitch = gfx9 && layout.pitch
                             ? layout.pitch
                             : util::align_up<uint64_t>(layout.width, result.block_width_);
      allocation.slice = uint64_t{allocation.pitch} *
                         util::align_up<uint64_t>(layout.height, result.block_height_) *
                         element_bytes;
    }
    if (allocation.pitch < (layout.mip_levels == 1 ? layout.width : 1) ||
        allocation.pitch % result.block_width_ != 0 ||
        !tile_offset(config, layout, result.block_bits_, element, 0, 0, 0))
      return std::nullopt;
  }
  result.offset_ = allocation.offset;
  result.slice_ = allocation.slice;
  result.pitch_ = allocation.pitch;
  result.tail_x_ = allocation.tail_x;
  result.tail_y_ = allocation.tail_y;
  result.tail_bits_ = allocation.tail_bits;
  return result;
}

bool ImageAddressing::contains(uint32_t x, uint32_t y, uint32_t z) const {
  return x < width_ && y < height_ && z < depth_;
}

uint64_t ImageAddressing::byte_offset(uint32_t x, uint32_t y, uint32_t z) const {
  assert(contains(x, y, z));
  const uint32_t element_bytes = layout_.element_bytes;
  if (layout_.swizzle_mode == 0)
    return uint64_t{z} * slice_ + offset_ + (uint64_t{y} * pitch_ + x) * element_bytes;
  const bool gfx9 = arch_is_cdna_4_or_lower(config_.arch);
  const uint32_t element = std::countr_zero(element_bytes);
  const uint32_t within =
      *tile_offset(config_, layout_, block_bits_, element, x + tail_x_, y + tail_y_, gfx9 ? 0 : z);
  const uint64_t block = uint64_t{y / block_height_} * (pitch_ / block_width_) + x / block_width_;
  const uint32_t pixel_offset = within | tail_bits_;
  return uint64_t{z} * slice_ + offset_ + (block << block_bits_) + pixel_offset;
}

} // namespace rocjitsu::amdgpu
