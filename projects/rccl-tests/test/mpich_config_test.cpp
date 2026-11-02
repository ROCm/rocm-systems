/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Host unit tests for mpichConfiguredWith() in mpich_config.h.
// No GPU or MPI required.

#include <gtest/gtest.h>

#include <string>

#include "mpich_config.h"

using rccl_tests_mpi::mpichConfiguredWith;

TEST(MpichConfiguredWith, BareFlagAndValue) {
  EXPECT_TRUE(mpichConfiguredWith("--with-hip", "--with-hip"));
  EXPECT_TRUE(mpichConfiguredWith("configure: --with-hip --prefix=/usr", "--with-hip"));
  EXPECT_TRUE(mpichConfiguredWith("configure: --with-hip=/opt/rocm", "--with-hip"));
  EXPECT_TRUE(mpichConfiguredWith("configure: --with-hip=yes", "--with-hip"));
  EXPECT_TRUE(mpichConfiguredWith("--with-hip\n--prefix=/usr", "--with-hip"));
  EXPECT_TRUE(mpichConfiguredWith("--with-hip\t--prefix=/usr", "--with-hip"));
}

TEST(MpichConfiguredWith, WithoutAndNegativeValue) {
  // "--without-hip" contains the letters of a with-flag but is not one.
  EXPECT_FALSE(mpichConfiguredWith("configure: --without-hip", "--with-hip"));
  // Autoconf's spelling of --without-hip.
  EXPECT_FALSE(mpichConfiguredWith("configure: --with-hip=no", "--with-hip"));
  EXPECT_FALSE(mpichConfiguredWith("configure: --with-hip=no --prefix=/usr", "--with-hip"));
  EXPECT_FALSE(mpichConfiguredWith("--with-cuda=no\n--with-ze=no", "--with-cuda"));
  EXPECT_FALSE(mpichConfiguredWith("--with-cuda=no\n--with-ze=no", "--with-ze"));
}

TEST(MpichConfiguredWith, DoesNotMatchLongerFlag) {
  EXPECT_FALSE(mpichConfiguredWith("configure: --with-zelda", "--with-ze"));
  EXPECT_TRUE(mpichConfiguredWith("configure: --with-ze", "--with-ze"));
  EXPECT_TRUE(mpichConfiguredWith("configure: --with-ze=/opt/level-zero", "--with-ze"));
  EXPECT_FALSE(mpichConfiguredWith("configure: --with-ze=no", "--with-ze"));
}

TEST(MpichConfiguredWith, LaterEnablingFlagStillMatches) {
  EXPECT_TRUE(mpichConfiguredWith("--with-hip=no --with-hip=/opt/rocm", "--with-hip"));
  EXPECT_TRUE(mpichConfiguredWith("--with-hip=no --with-cuda=/usr", "--with-cuda"));
  EXPECT_FALSE(mpichConfiguredWith("--with-hip=no --with-cuda=/usr", "--with-hip"));
}

TEST(MpichConfiguredWith, RequiresATokenBoundary) {
  EXPECT_FALSE(mpichConfiguredWith("xx--with-hip", "--with-hip"));
  EXPECT_FALSE(mpichConfiguredWith("", "--with-hip"));
  EXPECT_FALSE(mpichConfiguredWith("MPICH Version: 4.0", "--with-hip"));
}
