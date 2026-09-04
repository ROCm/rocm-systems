// Modification Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// Unit tests for the public ncclParam* C API (handle-based and key-based
// parameter access). Grounded in the implementation at src/param/c_api.cc and
// src/include/param/param.h.
//
#include <gtest/gtest.h>
#include <rccl/rccl.h>

#include <cstdint>
#include <cstring>
#include <string>

#include "common/ProcessIsolatedTestRunner.hpp"

// Internal param subsystem: DEFINE_NCCL_PARAM + parser factories. Uses the
// hipify-staged "_tmp" header names, matching src/param/*.cc.
#include "param/param_tmp.h"
#include "param/parsers.h"

namespace RcclUnitTesting {

// ---------------------------------------------------------------------------
// Test-only registered parameters. Registered at static-init time and thus
// visible in every re-exec'd isolated child process.
// ---------------------------------------------------------------------------

// Published integer params, one per width, used for typed-getter and boundary
// coverage. Defaults chosen so "unset" reads are unambiguous.
DEFINE_NCCL_PARAM(testParamI8, int8_t, NCCL_TEST_PARAM_I8, 7, NCCL_PARAM_FLAG_PUBLISHED,
                  NCCL_PARAM_DEFAULT, "test-only i8 param");
DEFINE_NCCL_PARAM(testParamI16, int16_t, NCCL_TEST_PARAM_I16, 7, NCCL_PARAM_FLAG_PUBLISHED,
                  NCCL_PARAM_DEFAULT, "test-only i16 param");
DEFINE_NCCL_PARAM(testParamI32, int32_t, NCCL_TEST_PARAM_I32, 7, NCCL_PARAM_FLAG_PUBLISHED,
                  NCCL_PARAM_DEFAULT, "test-only i32 param");
DEFINE_NCCL_PARAM(testParamI64, int64_t, NCCL_TEST_PARAM_I64, 7, NCCL_PARAM_FLAG_PUBLISHED,
                  NCCL_PARAM_DEFAULT, "test-only i64 param");
DEFINE_NCCL_PARAM(testParamU8, uint8_t, NCCL_TEST_PARAM_U8, 7, NCCL_PARAM_FLAG_PUBLISHED,
                  NCCL_PARAM_DEFAULT, "test-only u8 param");
DEFINE_NCCL_PARAM(testParamU16, uint16_t, NCCL_TEST_PARAM_U16, 7, NCCL_PARAM_FLAG_PUBLISHED,
                  NCCL_PARAM_DEFAULT, "test-only u16 param");
DEFINE_NCCL_PARAM(testParamU32, uint32_t, NCCL_TEST_PARAM_U32, 7, NCCL_PARAM_FLAG_PUBLISHED,
                  NCCL_PARAM_DEFAULT, "test-only u32 param");
DEFINE_NCCL_PARAM(testParamU64, uint64_t, NCCL_TEST_PARAM_U64, 7, NCCL_PARAM_FLAG_PUBLISHED,
                  NCCL_PARAM_DEFAULT, "test-only u64 param");

// Cached integer param for caching behavior tests.
DEFINE_NCCL_PARAM(testParamCached, int32_t, NCCL_TEST_PARAM_CACHED, 0,
                  NCCL_PARAM_FLAG_PUBLISHED | NCCL_PARAM_FLAG_CACHED, NCCL_PARAM_DEFAULT,
                  "test-only cached i32 param");

// Private (non-published) param for published-vs-all filtering tests.
DEFINE_NCCL_PARAM(testParamPrivate, int32_t, NCCL_TEST_PARAM_PRIVATE, 0, NCCL_PARAM_FLAG_NONE,
                  NCCL_PARAM_DEFAULT, "test-only private i32 param");

// Params covering the flag combinations ncclParamCheckFlag branches on. PUBLISHED alone and no
// flags at all are already covered by testParamI32 and testParamPrivate above.
DEFINE_NCCL_PARAM(testParamUnused, int32_t, NCCL_TEST_PARAM_UNUSED, 0, NCCL_PARAM_FLAG_UNUSED,
                  NCCL_PARAM_DEFAULT, "test-only unused i32 param");
DEFINE_NCCL_PARAM(testParamUnusedPub, int32_t, NCCL_TEST_PARAM_UNUSED_PUB, 0,
                  NCCL_PARAM_FLAG_UNUSED | NCCL_PARAM_FLAG_PUBLISHED, NCCL_PARAM_DEFAULT,
                  "test-only unused published i32 param");
DEFINE_NCCL_PARAM(testParamDeprecated, int32_t, NCCL_TEST_PARAM_DEPRECATED, 0,
                  NCCL_PARAM_FLAG_DEPRECATED, NCCL_PARAM_DEFAULT,
                  "test-only deprecated i32 param");
DEFINE_NCCL_PARAM(testParamDeprecatedPub, int32_t, NCCL_TEST_PARAM_DEPRECATED_PUB, 0,
                  NCCL_PARAM_FLAG_DEPRECATED | NCCL_PARAM_FLAG_PUBLISHED, NCCL_PARAM_DEFAULT,
                  "test-only deprecated published i32 param");
DEFINE_NCCL_PARAM(testParamUnusedDep, int32_t, NCCL_TEST_PARAM_UNUSED_DEP, 0,
                  NCCL_PARAM_FLAG_UNUSED | NCCL_PARAM_FLAG_DEPRECATED, NCCL_PARAM_DEFAULT,
                  "test-only unused+deprecated i32 param");

namespace {

// Keys used across tests.
constexpr const char* kI32Key = "NCCL_TEST_PARAM_I32";
constexpr const char* kI8Key = "NCCL_TEST_PARAM_I8";
constexpr const char* kU8Key = "NCCL_TEST_PARAM_U8";
constexpr const char* kCachedKey = "NCCL_TEST_PARAM_CACHED";
constexpr const char* kPrivateKey = "NCCL_TEST_PARAM_PRIVATE";
constexpr const char* kDumpAllKey = "NCCL_PARAM_DUMP_ALL";
constexpr const char* kNoCacheKey = "NCCL_NO_CACHE";
constexpr const char* kUnusedKey = "NCCL_TEST_PARAM_UNUSED";
constexpr const char* kUnusedPubKey = "NCCL_TEST_PARAM_UNUSED_PUB";
constexpr const char* kDeprecatedKey = "NCCL_TEST_PARAM_DEPRECATED";
constexpr const char* kDeprecatedPubKey = "NCCL_TEST_PARAM_DEPRECATED_PUB";
constexpr const char* kUnusedDepKey = "NCCL_TEST_PARAM_UNUSED_DEP";

// Returns true if `table` (length `len`) contains `key`.
bool tableContains(const char** table, int len, const char* key) {
  for (int i = 0; i < len; ++i) {
    if (table[i] && std::strcmp(table[i], key) == 0) return true;
  }
  return false;
}

} // namespace

// ===========================================================================
// ncclParamBind
// ===========================================================================

TEST(ParameterApiTests, Bind_KnownKey_Succeeds) {
  RUN_ISOLATED_TEST("Bind_KnownKey_Succeeds", []() {
    ncclParamHandle_t h = nullptr;
    ASSERT_EQ(ncclParamBind(&h, kI32Key), ncclSuccess);
    ASSERT_NE(h, nullptr);
  });
}

TEST(ParameterApiTests, Bind_UnknownKey_ReturnsInvalidArgAndLeavesOutUntouched) {
  RUN_ISOLATED_TEST("Bind_UnknownKey", []() {
    auto sentinel = reinterpret_cast<ncclParamHandle_t>(0xdeadbeef);
    ncclParamHandle_t h = sentinel;
    ASSERT_EQ(ncclParamBind(&h, "NCCL_DEFINITELY_NOT_A_PARAM"), ncclInvalidArgument);
    ASSERT_EQ(h, sentinel) << "*out must be left untouched on unknown key";
  });
}

// Legacy NCCL_PARAM(...) knobs are not registered in the new registry.
// NCCL_DEBUG was migrated to DEFINE_NCCL_PARAM in debug.cc; NVLS_ENABLE still
// uses the legacy macro in src/transport/nvls.cc.
TEST(ParameterApiTests, Bind_LegacyParamNotRegistered) {
  RUN_ISOLATED_TEST("Bind_LegacyParamNotRegistered", []() {
    ncclParamHandle_t h = nullptr;
    ASSERT_EQ(ncclParamBind(&h, "NVLS_ENABLE"), ncclInvalidArgument);
  });
}

TEST(ParameterApiTests, Bind_NullArgs) {
  RUN_ISOLATED_TEST("Bind_NullArgs", []() {
    ncclParamHandle_t h = nullptr;
    ASSERT_EQ(ncclParamBind(nullptr, kI32Key), ncclInvalidArgument);
    ASSERT_EQ(ncclParamBind(&h, nullptr), ncclInvalidArgument);
  });
}

TEST(ParameterApiTests, Bind_BothArgsNull_ReturnsInvalidArg) {
  RUN_ISOLATED_TEST("Bind_BothArgsNull", []() {
    ASSERT_EQ(ncclParamBind(nullptr, nullptr), ncclInvalidArgument);
  });
}

TEST(ParameterApiTests, Bind_KnownKey_ReturnsSameHandleOnRebind) {
  RUN_ISOLATED_TEST("Bind_KnownKey_SameHandle", []() {
    ncclParamHandle_t first = nullptr;
    ncclParamHandle_t second = nullptr;
    ASSERT_EQ(ncclParamBind(&first, kI32Key), ncclSuccess);
    ASSERT_EQ(ncclParamBind(&second, kI32Key), ncclSuccess);
    ASSERT_EQ(first, second) << "bind must resolve to the one registry entry for the key";
  });
}

// ncclParamBind must succeed for every flag combination. Flags only steer the advisory messages
// ncclParamCheckFlag emits; they never change the return code or the handle.
TEST(ParameterApiTests, Bind_AllFlagCombinations_Succeed) {
  RUN_ISOLATED_TEST("Bind_AllFlagCombinations", []() {
    constexpr const char* kKeys[] = {
        kI32Key,            // PUBLISHED
        kPrivateKey,        // no flags
        kUnusedKey,         // UNUSED
        kUnusedPubKey,      // UNUSED | PUBLISHED
        kDeprecatedKey,     // DEPRECATED
        kDeprecatedPubKey,  // DEPRECATED | PUBLISHED
        kUnusedDepKey,      // UNUSED | DEPRECATED
    };
    for (const char* key : kKeys) {
      ncclParamHandle_t h = nullptr;
      EXPECT_EQ(ncclParamBind(&h, key), ncclSuccess) << "key: " << key;
      EXPECT_NE(h, nullptr) << "key: " << key;
    }
  });
}

// Withholding NCCL_PARAM_FLAG_PUBLISHED only hides a param from ncclParamGetAllParameterKeys and
// ncclParamDumpAll. A caller that knows the name can still bind it and read its value.
TEST(ParameterApiTests, Bind_PrivateKey_HandleReadsValue) {
  RUN_ISOLATED_TEST_WITH_ENV(
      "Bind_PrivateKey_HandleReadsValue",
      []() {
        ncclParamHandle_t h = nullptr;
        ASSERT_EQ(ncclParamBind(&h, kPrivateKey), ncclSuccess);
        int32_t v = 0;
        ASSERT_EQ(ncclParamGetI32(h, &v), ncclSuccess);
        EXPECT_EQ(v, 13) << "private param must resolve from the environment like any other";
      },
      {{"NCCL_TEST_PARAM_PRIVATE", "13"}});
}

// ===========================================================================
// Typed integer accessors
// ===========================================================================

TEST(ParameterApiTests, GetI32_MatchingType_ReturnsValue) {
  RUN_ISOLATED_TEST_WITH_ENV(
      "GetI32_MatchingType",
      []() {
        ncclParamHandle_t h = nullptr;
        ASSERT_EQ(ncclParamBind(&h, kI32Key), ncclSuccess);
        int32_t v = 0;
        ASSERT_EQ(ncclParamGetI32(h, &v), ncclSuccess);
        ASSERT_EQ(v, 42);
      },
      {{"NCCL_TEST_PARAM_I32", "42"}});
}

TEST(ParameterApiTests, GetI32_Unset_ReturnsDefault) {
  RUN_ISOLATED_TEST("GetI32_Unset_ReturnsDefault", []() {
    ncclParamHandle_t h = nullptr;
    ASSERT_EQ(ncclParamBind(&h, kI32Key), ncclSuccess);
    int32_t v = 0;
    ASSERT_EQ(ncclParamGetI32(h, &v), ncclSuccess);
    ASSERT_EQ(v, 7) << "default value from DEFINE_NCCL_PARAM";
  });
}

// A handle bound to a bool param cannot be read as I32 (typeId guard).
TEST(ParameterApiTests, GetI32_TypeMismatch_ReturnsInvalidArg) {
  RUN_ISOLATED_TEST("GetI32_TypeMismatch", []() {
    ncclParamHandle_t h = nullptr;
    ASSERT_EQ(ncclParamBind(&h, kDumpAllKey), ncclSuccess); // bool param
    int32_t v = 0;
    ASSERT_EQ(ncclParamGetI32(h, &v), ncclInvalidArgument);
  });
}

// Reading an I8 param through the I32 accessor is a typeId mismatch.
TEST(ParameterApiTests, GetI32_CrossWidth_ReturnsInvalidArg) {
  RUN_ISOLATED_TEST("GetI32_CrossWidth", []() {
    ncclParamHandle_t h = nullptr;
    ASSERT_EQ(ncclParamBind(&h, kI8Key), ncclSuccess);
    int32_t v = 0;
    ASSERT_EQ(ncclParamGetI32(h, &v), ncclInvalidArgument);
  });
}

TEST(ParameterApiTests, GetI32_NullArgs) {
  RUN_ISOLATED_TEST("GetI32_NullArgs", []() {
    ncclParamHandle_t h = nullptr;
    ASSERT_EQ(ncclParamBind(&h, kI32Key), ncclSuccess);
    int32_t v = 0;
    ASSERT_EQ(ncclParamGetI32(nullptr, &v), ncclInvalidArgument);
    ASSERT_EQ(ncclParamGetI32(h, nullptr), ncclInvalidArgument);
  });
}

TEST(ParameterApiTests, GetU32_MatchingType_ReturnsValue) {
  RUN_ISOLATED_TEST_WITH_ENV(
      "GetU32_MatchingType",
      []() {
        ncclParamHandle_t h = nullptr;
        ASSERT_EQ(ncclParamBind(&h, "NCCL_TEST_PARAM_U32"), ncclSuccess);
        uint32_t v = 0;
        ASSERT_EQ(ncclParamGetU32(h, &v), ncclSuccess);
        ASSERT_EQ(v, 4000000000u);
      },
      {{"NCCL_TEST_PARAM_U32", "4000000000"}});
}

TEST(ParameterApiTests, GetI64_MatchingType_ReturnsValue) {
  RUN_ISOLATED_TEST_WITH_ENV(
      "GetI64_MatchingType",
      []() {
        ncclParamHandle_t h = nullptr;
        ASSERT_EQ(ncclParamBind(&h, "NCCL_TEST_PARAM_I64"), ncclSuccess);
        int64_t v = 0;
        ASSERT_EQ(ncclParamGetI64(h, &v), ncclSuccess);
        ASSERT_EQ(v, static_cast<int64_t>(9000000000LL));
      },
      {{"NCCL_TEST_PARAM_I64", "9000000000"}});
}

TEST(ParameterApiTests, GetI16_MatchingType_ReturnsValue) {
  RUN_ISOLATED_TEST_WITH_ENV(
      "GetI16_MatchingType",
      []() {
        ncclParamHandle_t h = nullptr;
        ASSERT_EQ(ncclParamBind(&h, "NCCL_TEST_PARAM_I16"), ncclSuccess);
        int16_t v = 0;
        ASSERT_EQ(ncclParamGetI16(h, &v), ncclSuccess);
        ASSERT_EQ(v, static_cast<int16_t>(32767));
      },
      {{"NCCL_TEST_PARAM_I16", "32767"}});
}

TEST(ParameterApiTests, GetI16_Unset_ReturnsDefault) {
  RUN_ISOLATED_TEST("GetI16_Unset_ReturnsDefault", []() {
    ncclParamHandle_t h = nullptr;
    ASSERT_EQ(ncclParamBind(&h, "NCCL_TEST_PARAM_I16"), ncclSuccess);
    int16_t v = 0;
    ASSERT_EQ(ncclParamGetI16(h, &v), ncclSuccess);
    ASSERT_EQ(v, 7) << "default value from DEFINE_NCCL_PARAM";
  });
}

TEST(ParameterApiTests, GetI16_NullArgs) {
  RUN_ISOLATED_TEST("GetI16_NullArgs", []() {
    ncclParamHandle_t h = nullptr;
    ASSERT_EQ(ncclParamBind(&h, "NCCL_TEST_PARAM_I16"), ncclSuccess);
    int16_t v = 0;
    ASSERT_EQ(ncclParamGetI16(nullptr, &v), ncclInvalidArgument);
    ASSERT_EQ(ncclParamGetI16(h, nullptr), ncclInvalidArgument);
  });
}

// Reading an I16 param through a different width is a typeId mismatch.
TEST(ParameterApiTests, GetI16_CrossWidth_ReturnsInvalidArg) {
  RUN_ISOLATED_TEST("GetI16_CrossWidth", []() {
    ncclParamHandle_t h = nullptr;
    ASSERT_EQ(ncclParamBind(&h, kI32Key), ncclSuccess);
    int16_t v = 0;
    ASSERT_EQ(ncclParamGetI16(h, &v), ncclInvalidArgument);
  });
}

TEST(ParameterApiTests, GetU16_MatchingType_ReturnsValue) {
  RUN_ISOLATED_TEST_WITH_ENV(
      "GetU16_MatchingType",
      []() {
        ncclParamHandle_t h = nullptr;
        ASSERT_EQ(ncclParamBind(&h, "NCCL_TEST_PARAM_U16"), ncclSuccess);
        uint16_t v = 0;
        ASSERT_EQ(ncclParamGetU16(h, &v), ncclSuccess);
        ASSERT_EQ(v, static_cast<uint16_t>(65535));
      },
      {{"NCCL_TEST_PARAM_U16", "65535"}});
}

TEST(ParameterApiTests, GetU16_NullArgs) {
  RUN_ISOLATED_TEST("GetU16_NullArgs", []() {
    ncclParamHandle_t h = nullptr;
    ASSERT_EQ(ncclParamBind(&h, "NCCL_TEST_PARAM_U16"), ncclSuccess);
    uint16_t v = 0;
    ASSERT_EQ(ncclParamGetU16(nullptr, &v), ncclInvalidArgument);
    ASSERT_EQ(ncclParamGetU16(h, nullptr), ncclInvalidArgument);
  });
}

TEST(ParameterApiTests, GetU64_MatchingType_ReturnsValue) {
  RUN_ISOLATED_TEST_WITH_ENV(
      "GetU64_MatchingType",
      []() {
        ncclParamHandle_t h = nullptr;
        ASSERT_EQ(ncclParamBind(&h, "NCCL_TEST_PARAM_U64"), ncclSuccess);
        uint64_t v = 0;
        ASSERT_EQ(ncclParamGetU64(h, &v), ncclSuccess);
        ASSERT_EQ(v, 18446744073709551615ULL) << "ULLONG_MAX must round-trip exactly";
      },
      {{"NCCL_TEST_PARAM_U64", "18446744073709551615"}});
}

TEST(ParameterApiTests, GetU64_NullArgs) {
  RUN_ISOLATED_TEST("GetU64_NullArgs", []() {
    ncclParamHandle_t h = nullptr;
    ASSERT_EQ(ncclParamBind(&h, "NCCL_TEST_PARAM_U64"), ncclSuccess);
    uint64_t v = 0;
    ASSERT_EQ(ncclParamGetU64(nullptr, &v), ncclInvalidArgument);
    ASSERT_EQ(ncclParamGetU64(h, nullptr), ncclInvalidArgument);
  });
}

// ===========================================================================
// Boundary / overflow behavior
//
// The integer parser uses strtoll/strtoull. Values that overflow (unsigned)
// long long set errno==ERANGE and are rejected -> fall back to default. Values
// that fit long long but overflow the narrow target width are silently
// truncated by static_cast, and validate() runs on the truncated value, so
// they are accepted (wrapped), not rejected.
// ===========================================================================

TEST(ParameterApiTests, GetI64_WordOverflow_FallsBackToDefault) {
  RUN_ISOLATED_TEST_WITH_ENV(
      "GetI64_WordOverflow",
      []() {
        ncclParamHandle_t h = nullptr;
        ASSERT_EQ(ncclParamBind(&h, "NCCL_TEST_PARAM_I64"), ncclSuccess);
        int64_t v = 0;
        ASSERT_EQ(ncclParamGetI64(h, &v), ncclSuccess);
        ASSERT_EQ(v, 7) << "ERANGE overflow -> default";
      },
      {{"NCCL_TEST_PARAM_I64", "99999999999999999999"}});
}

TEST(ParameterApiTests, GetU64_WordOverflow_FallsBackToDefault) {
  RUN_ISOLATED_TEST_WITH_ENV(
      "GetU64_WordOverflow",
      []() {
        ncclParamHandle_t h = nullptr;
        ASSERT_EQ(ncclParamBind(&h, "NCCL_TEST_PARAM_U64"), ncclSuccess);
        uint64_t v = 0;
        ASSERT_EQ(ncclParamGetU64(h, &v), ncclSuccess);
        ASSERT_EQ(v, 7u) << "ERANGE overflow -> default";
      },
      {{"NCCL_TEST_PARAM_U64", "99999999999999999999999999"}});
}

// "32768" fits in long long, then truncates to int16_t(-32768); validate() sees the already
// wrapped value and passes, so the wrapped result is returned.
TEST(ParameterApiTests, GetI16_SubWordOverflow_Wraps) {
  RUN_ISOLATED_TEST_WITH_ENV(
      "GetI16_SubWordOverflow",
      []() {
        ncclParamHandle_t h = nullptr;
        ASSERT_EQ(ncclParamBind(&h, "NCCL_TEST_PARAM_I16"), ncclSuccess);
        int16_t v = 0;
        ASSERT_EQ(ncclParamGetI16(h, &v), ncclSuccess);
        ASSERT_EQ(v, static_cast<int16_t>(-32768)) << "32768 wraps to -32768 for int16_t";
      },
      {{"NCCL_TEST_PARAM_I16", "32768"}});
}

TEST(ParameterApiTests, GetU16_Negative_Wraps) {
  RUN_ISOLATED_TEST_WITH_ENV(
      "GetU16_Negative",
      []() {
        ncclParamHandle_t h = nullptr;
        ASSERT_EQ(ncclParamBind(&h, "NCCL_TEST_PARAM_U16"), ncclSuccess);
        uint16_t v = 0;
        // strtoull("-1") wraps to ULLONG_MAX (no ERANGE), truncates to 0xFFFF.
        ASSERT_EQ(ncclParamGetU16(h, &v), ncclSuccess);
        ASSERT_EQ(v, static_cast<uint16_t>(0xFFFF));
      },
      {{"NCCL_TEST_PARAM_U16", "-1"}});
}

TEST(ParameterApiTests, GetI8_InRange_ReturnsValue) {
  RUN_ISOLATED_TEST_WITH_ENV(
      "GetI8_InRange",
      []() {
        ncclParamHandle_t h = nullptr;
        ASSERT_EQ(ncclParamBind(&h, kI8Key), ncclSuccess);
        int8_t v = 0;
        ASSERT_EQ(ncclParamGetI8(h, &v), ncclSuccess);
        ASSERT_EQ(v, static_cast<int8_t>(127));
      },
      {{"NCCL_TEST_PARAM_I8", "127"}});
}

// "128" fits in long long, then truncates to int8_t(-128); validate() sees the
// already-wrapped value and passes, so the wrapped result is returned.
TEST(ParameterApiTests, GetI8_SubWordOverflow_Wraps) {
  RUN_ISOLATED_TEST_WITH_ENV(
      "GetI8_SubWordOverflow",
      []() {
        ncclParamHandle_t h = nullptr;
        ASSERT_EQ(ncclParamBind(&h, kI8Key), ncclSuccess);
        int8_t v = 0;
        ASSERT_EQ(ncclParamGetI8(h, &v), ncclSuccess);
        ASSERT_EQ(v, static_cast<int8_t>(-128)) << "128 wraps to -128 for int8_t";
      },
      {{"NCCL_TEST_PARAM_I8", "128"}});
}

TEST(ParameterApiTests, GetU8_Negative_Wraps) {
  RUN_ISOLATED_TEST_WITH_ENV(
      "GetU8_Negative",
      []() {
        ncclParamHandle_t h = nullptr;
        ASSERT_EQ(ncclParamBind(&h, kU8Key), ncclSuccess);
        uint8_t v = 0;
        // strtoull("-1") wraps to ULLONG_MAX (no ERANGE), truncates to 0xFF.
        ASSERT_EQ(ncclParamGetU8(h, &v), ncclSuccess);
        ASSERT_EQ(v, static_cast<uint8_t>(0xFF));
      },
      {{"NCCL_TEST_PARAM_U8", "-1"}});
}

TEST(ParameterApiTests, GetI32_NonNumeric_FallsBackToDefault) {
  RUN_ISOLATED_TEST_WITH_ENV(
      "GetI32_NonNumeric",
      []() {
        ncclParamHandle_t h = nullptr;
        ASSERT_EQ(ncclParamBind(&h, kI32Key), ncclSuccess);
        int32_t v = 0;
        ASSERT_EQ(ncclParamGetI32(h, &v), ncclSuccess);
        ASSERT_EQ(v, 7) << "unparseable value -> default";
      },
      {{"NCCL_TEST_PARAM_I32", "not_a_number"}});
}

// ===========================================================================
// ncclParamGetStr
// ===========================================================================

TEST(ParameterApiTests, GetStr_StringParam_ReturnsValue) {
  RUN_ISOLATED_TEST_WITH_ENV(
      "GetStr_StringParam",
      []() {
        ncclParamHandle_t h = nullptr;
        ASSERT_EQ(ncclParamBind(&h, kNoCacheKey), ncclSuccess);
        const char* s = nullptr;
        ASSERT_EQ(ncclParamGetStr(h, &s), ncclSuccess);
        ASSERT_NE(s, nullptr);
        ASSERT_STREQ(s, "NCCL_TEST_PARAM_I32");
      },
      {{"NCCL_NO_CACHE", "NCCL_TEST_PARAM_I32"}});
}

// A non-cstr (bool) param cannot be read via GetStr.
TEST(ParameterApiTests, GetStr_TypeMismatch_ReturnsInvalidArg) {
  RUN_ISOLATED_TEST("GetStr_TypeMismatch", []() {
    ncclParamHandle_t h = nullptr;
    ASSERT_EQ(ncclParamBind(&h, kDumpAllKey), ncclSuccess); // bool param
    const char* s = nullptr;
    ASSERT_EQ(ncclParamGetStr(h, &s), ncclInvalidArgument);
  });
}

TEST(ParameterApiTests, GetStr_NullArgs) {
  RUN_ISOLATED_TEST("GetStr_NullArgs", []() {
    ncclParamHandle_t h = nullptr;
    ASSERT_EQ(ncclParamBind(&h, kNoCacheKey), ncclSuccess);
    const char* s = nullptr;
    ASSERT_EQ(ncclParamGetStr(nullptr, &s), ncclInvalidArgument);
    ASSERT_EQ(ncclParamGetStr(h, nullptr), ncclInvalidArgument);
  });
}

// ===========================================================================
// ncclParamGet (raw)
// ===========================================================================

TEST(ParameterApiTests, GetRaw_SufficientBuffer_WritesLenBytes) {
  RUN_ISOLATED_TEST_WITH_ENV(
      "GetRaw_SufficientBuffer",
      []() {
        ncclParamHandle_t h = nullptr;
        ASSERT_EQ(ncclParamBind(&h, kI32Key), ncclSuccess);
        int32_t buf = 0;
        int len = -1;
        ASSERT_EQ(ncclParamGet(h, &buf, sizeof(buf), &len), ncclSuccess);
        ASSERT_EQ(len, static_cast<int>(sizeof(int32_t)));
        ASSERT_EQ(buf, 55);
      },
      {{"NCCL_TEST_PARAM_I32", "55"}});
}

// A too-small buffer returns ncclInvalidArgument and sets *len = 0 (NOT the
// required size that the NVIDIA doc describes).
TEST(ParameterApiTests, GetRaw_MaxLenTooSmall_ReturnsInvalidArgAndZeroLen) {
  RUN_ISOLATED_TEST("GetRaw_MaxLenTooSmall", []() {
    ncclParamHandle_t h = nullptr;
    ASSERT_EQ(ncclParamBind(&h, kI32Key), ncclSuccess);
    int8_t tiny = 0;
    int len = -1;
    ASSERT_EQ(ncclParamGet(h, &tiny, sizeof(tiny), &len), ncclInvalidArgument);
    ASSERT_EQ(len, 0);
  });
}

TEST(ParameterApiTests, GetRaw_NullArgs) {
  RUN_ISOLATED_TEST("GetRaw_NullArgs", []() {
    ncclParamHandle_t h = nullptr;
    ASSERT_EQ(ncclParamBind(&h, kI32Key), ncclSuccess);
    int32_t buf = 0;
    int len = 0;
    ASSERT_EQ(ncclParamGet(nullptr, &buf, sizeof(buf), &len), ncclInvalidArgument);
    ASSERT_EQ(ncclParamGet(h, nullptr, sizeof(buf), &len), ncclInvalidArgument);
  });
}

// ===========================================================================
// ncclParamGetParameter (key-based)
// ===========================================================================

TEST(ParameterApiTests, GetParameter_KnownKey_ReturnsStringAndLen) {
  RUN_ISOLATED_TEST_WITH_ENV(
      "GetParameter_KnownKey",
      []() {
        const char* value = nullptr;
        int len = -1;
        ASSERT_EQ(ncclParamGetParameter(kI32Key, &value, &len), ncclSuccess);
        ASSERT_NE(value, nullptr);
        ASSERT_STREQ(value, "123");
        ASSERT_EQ(len, static_cast<int>(std::strlen(value)));
      },
      {{"NCCL_TEST_PARAM_I32", "123"}});
}

TEST(ParameterApiTests, GetParameter_UnknownKey_SetsNullZeroAndInvalidArg) {
  RUN_ISOLATED_TEST("GetParameter_UnknownKey", []() {
    const char* value = reinterpret_cast<const char*>(0x1);
    int len = -1;
    ASSERT_EQ(ncclParamGetParameter("NCCL_DEFINITELY_NOT_A_PARAM", &value, &len),
              ncclInvalidArgument);
    ASSERT_EQ(value, nullptr);
    ASSERT_EQ(len, 0);
  });
}

TEST(ParameterApiTests, GetParameter_NullArgs) {
  RUN_ISOLATED_TEST("GetParameter_NullArgs", []() {
    const char* value = nullptr;
    int len = 0;
    ASSERT_EQ(ncclParamGetParameter(nullptr, &value, &len), ncclInvalidArgument);
    ASSERT_EQ(ncclParamGetParameter(kI32Key, nullptr, &len), ncclInvalidArgument);
    ASSERT_EQ(ncclParamGetParameter(kI32Key, &value, nullptr), ncclInvalidArgument);
  });
}

// ===========================================================================
// ncclParamGetAllParameterKeys
// ===========================================================================

TEST(ParameterApiTests, GetAllKeys_DefaultPublishedOnly) {
  RUN_ISOLATED_TEST("GetAllKeys_DefaultPublishedOnly", []() {
    const char** table = nullptr;
    int len = 0;
    ASSERT_EQ(ncclParamGetAllParameterKeys(&table, &len), ncclSuccess);
    ASSERT_GT(len, 0);
    EXPECT_TRUE(tableContains(table, len, kI32Key)) << "published test key expected";
    EXPECT_FALSE(tableContains(table, len, kPrivateKey))
        << "private key must be excluded by default";
  });
}

TEST(ParameterApiTests, GetAllKeys_DumpAllIncludesPrivate) {
  RUN_ISOLATED_TEST_WITH_ENV(
      "GetAllKeys_DumpAllIncludesPrivate",
      []() {
        const char** table = nullptr;
        int len = 0;
        ASSERT_EQ(ncclParamGetAllParameterKeys(&table, &len), ncclSuccess);
        EXPECT_TRUE(tableContains(table, len, kPrivateKey))
            << "NCCL_PARAM_DUMP_ALL=1 must include private keys";
      },
      {{"NCCL_PARAM_DUMP_ALL", "1"}});
}

TEST(ParameterApiTests, GetAllKeys_NullArgs) {
  RUN_ISOLATED_TEST("GetAllKeys_NullArgs", []() {
    const char** table = nullptr;
    int len = 0;
    ASSERT_EQ(ncclParamGetAllParameterKeys(nullptr, &len), ncclInvalidArgument);
    ASSERT_EQ(ncclParamGetAllParameterKeys(&table, nullptr), ncclInvalidArgument);
  });
}

// ===========================================================================
// ncclParamDumpAll
// ===========================================================================

TEST(ParameterApiTests, DumpAll_WritesRegistryToStdout) {
  RUN_ISOLATED_TEST("DumpAll_WritesRegistryToStdout", []() {
    testing::internal::CaptureStdout();
    ncclParamDumpAll();
    std::string out = testing::internal::GetCapturedStdout();
    EXPECT_NE(out.find("=== ncclParam Registry Dump ==="), std::string::npos);
    EXPECT_NE(out.find("NCCL_TEST_PARAM_I32"), std::string::npos);
  });
}

TEST(ParameterApiTests, DumpAll_PrivateHiddenByDefault) {
  RUN_ISOLATED_TEST("DumpAll_PrivateHiddenByDefault", []() {
    testing::internal::CaptureStdout();
    ncclParamDumpAll();
    std::string out = testing::internal::GetCapturedStdout();
    EXPECT_EQ(out.find("NCCL_TEST_PARAM_PRIVATE"), std::string::npos)
        << "private key must be hidden without NCCL_PARAM_DUMP_ALL";
  });
}

TEST(ParameterApiTests, DumpAll_PrivateShownWhenDumpAll) {
  RUN_ISOLATED_TEST_WITH_ENV(
      "DumpAll_PrivateShownWhenDumpAll",
      []() {
        testing::internal::CaptureStdout();
        ncclParamDumpAll();
        std::string out = testing::internal::GetCapturedStdout();
        EXPECT_NE(out.find("NCCL_TEST_PARAM_PRIVATE"), std::string::npos);
      },
      {{"NCCL_PARAM_DUMP_ALL", "1"}});
}

// ===========================================================================
// Caching behavior (NCCL_NO_CACHE)
// ===========================================================================

// A CACHED param resolves once; a later env change is ignored.
TEST(ParameterApiTests, Cache_DefaultCachesValue) {
  RUN_ISOLATED_TEST_WITH_ENV(
      "Cache_DefaultCachesValue",
      []() {
        ncclParamHandle_t h = nullptr;
        ASSERT_EQ(ncclParamBind(&h, kCachedKey), ncclSuccess);
        int32_t v = 0;
        ASSERT_EQ(ncclParamGetI32(h, &v), ncclSuccess);
        ASSERT_EQ(v, 100);

        setenv("NCCL_TEST_PARAM_CACHED", "200", 1);
        ASSERT_EQ(ncclParamGetI32(h, &v), ncclSuccess);
        ASSERT_EQ(v, 100) << "cached value must not change when NCCL_NO_CACHE is unset";
      },
      {{"NCCL_TEST_PARAM_CACHED", "100"}});
}

// Listing the key in NCCL_NO_CACHE forces a re-read on the next access.
TEST(ParameterApiTests, Cache_PerKeyForcesReRead) {
  RUN_ISOLATED_TEST_WITH_ENV(
      "Cache_PerKeyForcesReRead",
      []() {
        ncclParamHandle_t h = nullptr;
        ASSERT_EQ(ncclParamBind(&h, kCachedKey), ncclSuccess);
        int32_t v = 0;
        ASSERT_EQ(ncclParamGetI32(h, &v), ncclSuccess);
        ASSERT_EQ(v, 100);

        setenv("NCCL_TEST_PARAM_CACHED", "200", 1);
        ASSERT_EQ(ncclParamGetI32(h, &v), ncclSuccess);
        ASSERT_EQ(v, 200) << "value should be re-read when key is listed in NCCL_NO_CACHE";
      },
      {{"NCCL_NO_CACHE", "NCCL_TEST_PARAM_CACHED"}, {"NCCL_TEST_PARAM_CACHED", "100"}});
}

// NCCL_NO_CACHE=ALL disables caching for every key.
TEST(ParameterApiTests, Cache_AllForcesReRead) {
  RUN_ISOLATED_TEST_WITH_ENV(
      "Cache_AllForcesReRead",
      []() {
        ncclParamHandle_t h = nullptr;
        ASSERT_EQ(ncclParamBind(&h, kCachedKey), ncclSuccess);
        int32_t v = 0;
        ASSERT_EQ(ncclParamGetI32(h, &v), ncclSuccess);
        ASSERT_EQ(v, 100);

        setenv("NCCL_TEST_PARAM_CACHED", "200", 1);
        ASSERT_EQ(ncclParamGetI32(h, &v), ncclSuccess);
        ASSERT_EQ(v, 200) << "NCCL_NO_CACHE=ALL should disable caching for every key";
      },
      {{"NCCL_NO_CACHE", "ALL"}, {"NCCL_TEST_PARAM_CACHED", "100"}});
}

} // namespace RcclUnitTesting
