/*
 * SPDX-License-Identifier: MIT
 */
#include "suites/image/mipmap_array.h"
#include "gtest/gtest.h"
#include "common/base_rocr_utils.h"

using namespace rocrtst;

// Helper to ensure agent enumeration occurs once per test case group
static void EnsureAgents(BaseRocR *t) {
  if(!t->gpu_device1()) {
    ASSERT_EQ(HSA_STATUS_SUCCESS, rocrtst::SetDefaultAgents(t));
  }
}

TEST(MipmapArrayBasic, CreateDestroy1D) {
  MipmapArrayTest t; EnsureAgents(&t); t.SetUp(); t.MipmapCreate1DArrayTest(); t.MipmapDestroy1DArrayTest(); }
TEST(MipmapArrayBasic, GetLevels1D) {
  MipmapArrayTest t; EnsureAgents(&t); t.SetUp(); t.MipmapGetLevel1DArrayTest(); }

TEST(MipmapArrayBasic, CreateDestroy2D) {
  MipmapArrayTest t; EnsureAgents(&t); t.SetUp(); t.MipmapCreate2DArrayTest(); t.MipmapDestroy2DArrayTest(); }
TEST(MipmapArrayBasic, GetLevels2D) {
  MipmapArrayTest t; EnsureAgents(&t); t.SetUp(); t.MipmapGetLevel2DArrayTest(); }

TEST(MipmapArrayBasic, CreateDestroy3D) {
  MipmapArrayTest t; EnsureAgents(&t); t.SetUp(); t.MipmapCreate3DArrayTest(); t.MipmapDestroy3DArrayTest(); }
TEST(MipmapArrayBasic, GetLevels3D) {
  MipmapArrayTest t; EnsureAgents(&t); t.SetUp(); t.MipmapGetLevel3DArrayTest(); }

TEST(MipmapArrayDimensions, Sweep1D) {
  Mipmap1DArrayTest t; EnsureAgents(&t); t.SetUp(); t.TestVariousDimensions1D(); }
TEST(MipmapArrayDimensions, Sweep2D) {
  Mipmap2DArrayTest t; EnsureAgents(&t); t.SetUp(); t.TestVariousDimensions2D(); }
TEST(MipmapArrayDimensions, Sweep3D) {
  Mipmap3DArrayTest t; EnsureAgents(&t); t.SetUp(); t.TestVariousDimensions3D(); }

TEST(MipmapArrayNegative, Errors1D) { Mipmap1DArrayTest t; EnsureAgents(&t); t.SetUp(); t.TestErrorConditions1D(); }
TEST(MipmapArrayNegative, Errors2D) { Mipmap2DArrayTest t; EnsureAgents(&t); t.SetUp(); t.TestErrorConditions2D(); }
TEST(MipmapArrayNegative, Errors3D) { Mipmap3DArrayTest t; EnsureAgents(&t); t.SetUp(); t.TestErrorConditions3D(); }

