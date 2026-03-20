/*************************************************************************
 * Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "archinfo.h"
#include <gtest/gtest.h>
#include <cstring>

namespace RcclUnitTesting {

// ---------------------------------------------------------------------------
// IsArchMatch
// ---------------------------------------------------------------------------

TEST(IsArchMatch, ExactMatchReturnsTrue) {
    EXPECT_TRUE(IsArchMatch("gfx942", "gfx942"));
}

TEST(IsArchMatch, PrefixMatchReturnsTrue) {
    // strlen("gfx94") == 5, strncmp("gfx942", "gfx94", 5) == 0
    EXPECT_TRUE(IsArchMatch("gfx942", "gfx94"));
}

TEST(IsArchMatch, ShortPrefixMatchReturnsTrue) {
    EXPECT_TRUE(IsArchMatch("gfx906", "gfx"));
}

TEST(IsArchMatch, DifferentArchReturnsFalse) {
    EXPECT_FALSE(IsArchMatch("gfx942", "gfx906"));
}

TEST(IsArchMatch, DifferentFamilyReturnsFalse) {
    EXPECT_FALSE(IsArchMatch("gfx1100", "gfx9"));
}

TEST(IsArchMatch, TargetLongerThanArchReturnsFalse) {
    // "gfx942" vs "gfx9421" — target is longer, so strncmp of 7 chars will differ
    EXPECT_FALSE(IsArchMatch("gfx942", "gfx9421"));
}

TEST(IsArchMatch, GfxFamilies) {
    // RDNA3 gfx1100 should not match MI300 gfx942
    EXPECT_FALSE(IsArchMatch("gfx1100", "gfx942"));
    // gfx90a should match prefix gfx90
    EXPECT_TRUE(IsArchMatch("gfx90a", "gfx90"));
    // gfx90a should not match gfx906
    EXPECT_FALSE(IsArchMatch("gfx90a", "gfx906"));
}

TEST(IsArchMatch, GfxWithSuffix) {
    // arch strings from device properties may have xnack/sramecc stripped by
    // GcnArchNameFormat, but IsArchMatch itself does a plain strncmp
    EXPECT_TRUE(IsArchMatch("gfx942", "gfx942"));
    EXPECT_FALSE(IsArchMatch("gfx942xnack+", "gfx950"));
}

// ---------------------------------------------------------------------------
// convertGcnArchToGcnArchName
// ---------------------------------------------------------------------------

TEST(ConvertGcnArch, Gfx906From906) {
    const char* name = nullptr;
    convertGcnArchToGcnArchName("906", &name);
    EXPECT_STREQ(name, "gfx906");
}

TEST(ConvertGcnArch, Gfx908From908) {
    const char* name = nullptr;
    convertGcnArchToGcnArchName("908", &name);
    EXPECT_STREQ(name, "gfx908");
}

TEST(ConvertGcnArch, Gfx90aFrom910) {
    const char* name = nullptr;
    convertGcnArchToGcnArchName("910", &name);
    EXPECT_STREQ(name, "gfx90a");
}

TEST(ConvertGcnArch, Gfx942From942) {
    const char* name = nullptr;
    convertGcnArchToGcnArchName("942", &name);
    EXPECT_STREQ(name, "gfx942");
}

TEST(ConvertGcnArch, Gfx950From950) {
    const char* name = nullptr;
    convertGcnArchToGcnArchName("950", &name);
    EXPECT_STREQ(name, "gfx950");
}

TEST(ConvertGcnArch, UnknownArchPassthrough) {
    // An unrecognized gcnArch should be returned as-is
    const char* name = nullptr;
    convertGcnArchToGcnArchName("gfx1100", &name);
    EXPECT_STREQ(name, "gfx1100");
}

TEST(ConvertGcnArch, EmptyStringPassthrough) {
    const char* name = nullptr;
    convertGcnArchToGcnArchName("", &name);
    EXPECT_STREQ(name, "");
}

} // namespace RcclUnitTesting
