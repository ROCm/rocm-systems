// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file mtype_flags_test.cpp
/// @brief Phase D unit tests for per-family mtype_from_flags_* functions.

#include "rocjitsu/isa/arch/amdgpu/shared/gfx10_cache_flags.h"
#include "rocjitsu/isa/arch/amdgpu/shared/gfx11_cache_flags.h"
#include "rocjitsu/isa/arch/amdgpu/shared/gfx12_cache_flags.h"
#include "rocjitsu/isa/arch/amdgpu/shared/gfx940_cache_flags.h"
#include "rocjitsu/isa/arch/amdgpu/shared/gfx9_cache_flags.h"

#include <gtest/gtest.h>

namespace {

using namespace rocjitsu::amdgpu;

// ---------------------------------------------------------------------------
// GFX9 (CDNA1/2) — GLC only
// ---------------------------------------------------------------------------

TEST(MtypeFlagsTest, Gfx9GlcOff) { EXPECT_EQ(mtype_from_flags_gfx9(false), Mtype::RW); }
TEST(MtypeFlagsTest, Gfx9GlcOn) { EXPECT_EQ(mtype_from_flags_gfx9(true), Mtype::CC); }

// ---------------------------------------------------------------------------
// GFX940 (CDNA3/4) — SC0/SC1 + NT
// ---------------------------------------------------------------------------

TEST(MtypeFlagsTest, Gfx940WaveScope) {
  EXPECT_EQ(mtype_from_flags_gfx940(false, false, false), Mtype::RW);
}

TEST(MtypeFlagsTest, Gfx940DeviceScope) {
  EXPECT_EQ(mtype_from_flags_gfx940(true, false, false), Mtype::CC);
}

TEST(MtypeFlagsTest, Gfx940SystemScope) {
  EXPECT_EQ(mtype_from_flags_gfx940(false, true, false), Mtype::UC);
}

TEST(MtypeFlagsTest, Gfx940SystemScopeBoth) {
  EXPECT_EQ(mtype_from_flags_gfx940(true, true, false), Mtype::UC);
}

TEST(MtypeFlagsTest, Gfx940NonTemporal) {
  EXPECT_EQ(mtype_from_flags_gfx940(false, false, true), Mtype::NT);
}

TEST(MtypeFlagsTest, Gfx940NtOverridesScope) {
  EXPECT_EQ(mtype_from_flags_gfx940(true, true, true), Mtype::NT);
}

// ---------------------------------------------------------------------------
// GFX10 (RDNA1/2) — GLC + DLC + SLC
// ---------------------------------------------------------------------------

TEST(MtypeFlagsTest, Gfx10AllOff) {
  EXPECT_EQ(mtype_from_flags_gfx10(false, false, false), Mtype::RW);
}

TEST(MtypeFlagsTest, Gfx10GlcOnly) {
  EXPECT_EQ(mtype_from_flags_gfx10(true, false, false), Mtype::CC);
}

TEST(MtypeFlagsTest, Gfx10DlcOnly) {
  EXPECT_EQ(mtype_from_flags_gfx10(false, true, false), Mtype::CC);
}

TEST(MtypeFlagsTest, Gfx10GlcAndDlc) {
  EXPECT_EQ(mtype_from_flags_gfx10(true, true, false), Mtype::UC);
}

TEST(MtypeFlagsTest, Gfx10SlcNonTemporal) {
  EXPECT_EQ(mtype_from_flags_gfx10(false, false, true), Mtype::NT);
}

TEST(MtypeFlagsTest, Gfx10SlcOverrides) {
  EXPECT_EQ(mtype_from_flags_gfx10(true, true, true), Mtype::NT);
}

// ---------------------------------------------------------------------------
// GFX11 (RDNA3/3.5) — GLC→SC0, SLC→SC1, DLC→NT
// ---------------------------------------------------------------------------

TEST(MtypeFlagsTest, Gfx11CuScope) {
  EXPECT_EQ(mtype_from_flags_gfx11(false, false, false), Mtype::RW);
}

TEST(MtypeFlagsTest, Gfx11DeviceScope) {
  // GLC=1 (SC0=1), DLC=0, SLC=0 → scope=1 → CC
  EXPECT_EQ(mtype_from_flags_gfx11(true, false, false), Mtype::CC);
}

TEST(MtypeFlagsTest, Gfx11SystemScope) {
  // GLC=0, DLC=0, SLC=1 (SC1=1) → scope=2 → UC
  EXPECT_EQ(mtype_from_flags_gfx11(false, false, true), Mtype::UC);
}

TEST(MtypeFlagsTest, Gfx11NonTemporal) {
  // DLC=1 → NT
  EXPECT_EQ(mtype_from_flags_gfx11(false, true, false), Mtype::NT);
}

TEST(MtypeFlagsTest, Gfx11NtOverridesScope) {
  EXPECT_EQ(mtype_from_flags_gfx11(true, true, true), Mtype::NT);
}

// ---------------------------------------------------------------------------
// GFX12 (RDNA4) — SCOPE + TH
// ---------------------------------------------------------------------------

TEST(MtypeFlagsTest, Gfx12CuScope) { EXPECT_EQ(mtype_from_flags_gfx12(0, 0), Mtype::RW); }

TEST(MtypeFlagsTest, Gfx12SeScope) { EXPECT_EQ(mtype_from_flags_gfx12(1, 0), Mtype::CC); }

TEST(MtypeFlagsTest, Gfx12DeviceScope) { EXPECT_EQ(mtype_from_flags_gfx12(2, 0), Mtype::UC); }

TEST(MtypeFlagsTest, Gfx12SystemScope) { EXPECT_EQ(mtype_from_flags_gfx12(3, 0), Mtype::UC); }

TEST(MtypeFlagsTest, Gfx12NonTemporal) {
  EXPECT_EQ(mtype_from_flags_gfx12(0, GFX12_TH_NT), Mtype::NT);
}

TEST(MtypeFlagsTest, Gfx12NtOverridesScope) {
  EXPECT_EQ(mtype_from_flags_gfx12(3, GFX12_TH_NT), Mtype::NT);
}

// ---------------------------------------------------------------------------
// Shared helper: mtype_from_scope_nt
// ---------------------------------------------------------------------------

TEST(MtypeFlagsTest, ScopeNtHelper) {
  EXPECT_EQ(mtype_from_scope_nt(0, false), Mtype::RW);
  EXPECT_EQ(mtype_from_scope_nt(1, false), Mtype::CC);
  EXPECT_EQ(mtype_from_scope_nt(2, false), Mtype::UC);
  EXPECT_EQ(mtype_from_scope_nt(3, false), Mtype::UC);
  EXPECT_EQ(mtype_from_scope_nt(0, true), Mtype::NT);
  EXPECT_EQ(mtype_from_scope_nt(3, true), Mtype::NT);
}

} // namespace
