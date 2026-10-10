/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Host-only microtests for src/param/param_registry.cc: the process-wide
// ncclParamRegistry singleton (add/find/remove) that every DEFINE_NCCL_PARAM
// registers into and the C API (src/param/c_api.cc) looks parameters up
// through. Third PR in the AICOMRCCL-2820 stacked chain.
//
// A minimal FakeParam stands in for ncclParam<T> here: the registry only
// ever touches its entries through the ncclParamInterface* pointer, so a
// template-free double is enough to test add/find/remove in isolation from
// param.h's env-plugin-dependent ncclParam<T> (covered by a later PR in this
// chain, where c_api.cc's own NCCL_PARAM_DUMP_ALL dependency is covered too).
// Every registration uses a key derived from the running test's own name, so
// tests can run in any order, repeat, or shuffle without colliding with a
// leftover registration from another test.

#include <gtest/gtest.h>

#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

#include "../common/LogCapture.hpp"

#include PARAM_REGISTRY_CC_PATH

namespace {

using RcclUnitTesting::CaptureLog;
using RcclUnitTesting::LogHas;

class FakeParam : public ncclParamInterface {
 public:
  explicit FakeParam(int32_t value) : value_(value) {}

  ncclResult_t getRawData(void* out, int maxLen, int* len) override {
    if (!out || !len || maxLen <= 0) return ncclInvalidArgument;
    if (static_cast<int>(sizeof(value_)) > maxLen) {
      *len = 0;
      return ncclInvalidArgument;
    }
    std::memcpy(out, &value_, sizeof(value_));
    *len = static_cast<int>(sizeof(value_));
    return ncclSuccess;
  }

  std::string toString() override {
    return std::to_string(value_);
  }

  std::string dump() override {
    return "fake:" + std::to_string(value_);
  }

 private:
  int32_t value_;
};

class ParamRegistryMicrotest : public ::testing::Test {
 protected:
  void TearDown() override {
    for (const auto& key : registeredKeys_) ncclParamRegistry::remove(key);
    registeredKeys_.clear();
  }

  // Keys are scoped to the current test name so re-running, repeating, or
  // shuffling tests never collides with a registration left behind -- or
  // still live -- from another test.
  std::string Key(const char* suffix = "") {
    return std::string("PARAM_REGISTRY_TEST_") +
           ::testing::UnitTest::GetInstance()->current_test_info()->name() + suffix;
  }

  ncclParamInfo_t MakeInfo(const char* key, uint64_t flags = NCCL_PARAM_FLAG_NONE) {
    return {NCCL_PARAM_TYPE_I32, flags, "int32_t", key, "test param"};
  }

  // Registers fake under key and arranges for TearDown to remove it.
  ncclResult_t Register(const std::string& key, FakeParam* fake, uint64_t flags = NCCL_PARAM_FLAG_NONE) {
    registeredKeys_.push_back(key);
    return ncclParamRegistry::add(key, MakeInfo(key.c_str(), flags), fake);
  }

  std::vector<std::string> registeredKeys_;
};

TEST_F(ParamRegistryMicrotest, Add_RegistersSoFindReturnsTheSameEntry) {
  std::string key = Key();
  FakeParam fake(42);
  ASSERT_EQ(ncclSuccess,
            Register(key, &fake, NCCL_PARAM_FLAG_PUBLISHED | NCCL_PARAM_FLAG_CACHED));

  auto* entry = ncclParamRegistry::find(key);
  ASSERT_NE(nullptr, entry);
  EXPECT_EQ(&fake, entry->param);
  EXPECT_STREQ(key.c_str(), entry->info.key);
  EXPECT_EQ(NCCL_PARAM_TYPE_I32, entry->info.typeId);
  EXPECT_EQ(static_cast<uint64_t>(NCCL_PARAM_FLAG_PUBLISHED | NCCL_PARAM_FLAG_CACHED), entry->info.flags);
  EXPECT_STREQ("int32_t", entry->info.typeStr);
  EXPECT_STREQ("test param", entry->info.desc);
}

TEST_F(ParamRegistryMicrotest, Find_UnknownKeyReturnsNullptr) {
  EXPECT_EQ(nullptr, ncclParamRegistry::find(Key("_NEVER_REGISTERED")));
}

TEST_F(ParamRegistryMicrotest, Add_DuplicateKeyWarnsAndLeavesTheFirstRegistrationIntact) {
  std::string key = Key();
  FakeParam first(1);
  FakeParam second(2);
  ASSERT_EQ(ncclSuccess, Register(key, &first));

  std::string log = CaptureLog([&]() {
    EXPECT_EQ(ncclInternalError, ncclParamRegistry::add(key, MakeInfo(key.c_str()), &second));
  });
  EXPECT_TRUE(LogHas(log, "Duplicate registration"));
  EXPECT_TRUE(LogHas(log, key.c_str()));

  // The duplicate add() left the original registration (and pointer) in place.
  auto* entry = ncclParamRegistry::find(key);
  ASSERT_NE(nullptr, entry);
  EXPECT_EQ(&first, entry->param);
}

TEST_F(ParamRegistryMicrotest, Remove_UnregistersSoFindReturnsNullptrAfterward) {
  std::string key = Key();
  FakeParam fake(7);
  ASSERT_EQ(ncclSuccess, Register(key, &fake));
  ASSERT_NE(nullptr, ncclParamRegistry::find(key));

  EXPECT_EQ(ncclSuccess, ncclParamRegistry::remove(key));
  EXPECT_EQ(nullptr, ncclParamRegistry::find(key));
}

TEST_F(ParamRegistryMicrotest, Remove_OfAnUnknownKeyIsANoOpThatStillSucceeds) {
  // Asserts more than the return code: remove() unconditionally returns
  // ncclSuccess, so a map.clear() in its place would pass a return-code-only
  // check too. A live entry surviving proves this really was a no-op.
  std::string key = Key();
  FakeParam fake(5);
  ASSERT_EQ(ncclSuccess, Register(key, &fake));
  const auto sizeBefore = ncclParamRegistry::instance().size();

  EXPECT_EQ(ncclSuccess, ncclParamRegistry::remove(Key("_NEVER_REGISTERED")));

  EXPECT_EQ(sizeBefore, ncclParamRegistry::instance().size());
  EXPECT_NE(nullptr, ncclParamRegistry::find(key));
}

TEST_F(ParamRegistryMicrotest, Remove_ThenReRegisteringTheSameKeySucceeds) {
  std::string key = Key();
  FakeParam first(1);
  ASSERT_EQ(ncclSuccess, Register(key, &first));
  ASSERT_EQ(ncclSuccess, ncclParamRegistry::remove(key));

  FakeParam second(2);
  EXPECT_EQ(ncclSuccess, Register(key, &second));
  auto* entry = ncclParamRegistry::find(key);
  ASSERT_NE(nullptr, entry);
  EXPECT_EQ(&second, entry->param);
}

TEST_F(ParamRegistryMicrotest, Instance_ReflectsEveryAddedEntryByKey) {
  std::string key = Key();
  FakeParam fake(9);
  ASSERT_EQ(ncclSuccess, Register(key, &fake));

  auto& map = ncclParamRegistry::instance();
  auto it = map.find(key);
  ASSERT_NE(map.end(), it);
  EXPECT_EQ(&fake, it->second.param);
}

TEST_F(ParamRegistryMicrotest, State_IsASingletonSharedAcrossAllAccessors) {
  // ncclParamRegistryInstance() is the C-linkage accessor every DSO resolves
  // to the same symbol through; state() casts its return, and instance()/
  // mutex() hand out the map and mutex living inside that one RegistryState.
  // Cross-comparing them (not each against itself) proves they share state
  // rather than each independently happening to be self-consistent.
  EXPECT_EQ(ncclParamRegistryInstance(), static_cast<void*>(&ncclParamRegistry::state()));
  EXPECT_EQ(&ncclParamRegistry::state().map, &ncclParamRegistry::instance());
  EXPECT_EQ(&ncclParamRegistry::state().mtx, &ncclParamRegistry::mutex());
}

}  // namespace
