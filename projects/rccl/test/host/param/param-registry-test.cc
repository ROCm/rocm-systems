/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Host-only microtests for src/param/param_registry.cc's ncclParamRegistry singleton. FakeParam
// stands in for ncclParam<T>, since the registry only uses the ncclParamInterface pointer.

#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <functional>
#include <mutex>
#include <string>
#include <thread>
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

  // Scoped to the test name, so repeats and shuffles never collide.
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
  // Registered first, so a find() that ignored its key can't pass on an empty map. Bound to a local:
  // the map keeps key.c_str(), so a temporary would dangle.
  std::string key = Key();
  FakeParam live(3);
  ASSERT_EQ(ncclSuccess, Register(key, &live));
  EXPECT_EQ(nullptr, ncclParamRegistry::find(Key("_NEVER_REGISTERED")));
}

TEST_F(ParamRegistryMicrotest, Add_DuplicateKeyWarnsAndLeavesTheFirstRegistrationIntact) {
  std::string key = Key();
  FakeParam first(1);
  FakeParam second(2);
  ASSERT_EQ(ncclSuccess, Register(key, &first));

  // Different flags, so overwriting info on a rejected add() shows below.
  std::string log = CaptureLog([&]() {
    EXPECT_EQ(ncclInternalError,
              ncclParamRegistry::add(key, MakeInfo(key.c_str(), NCCL_PARAM_FLAG_PUBLISHED), &second));
  });
  EXPECT_TRUE(LogHas(log, "Duplicate registration"));
  EXPECT_TRUE(LogHas(log, key.c_str()));

  // The duplicate add() left the original registration (info and pointer) in place.
  auto* entry = ncclParamRegistry::find(key);
  ASSERT_NE(nullptr, entry);
  EXPECT_EQ(&first, entry->param);
  EXPECT_EQ(static_cast<uint64_t>(NCCL_PARAM_FLAG_NONE), entry->info.flags);
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
  // remove() always returns ncclSuccess, so check a live entry survived too.
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
  // find() returns a pointer into the map, which the C API keeps as a long-lived handle.
  EXPECT_EQ(&it->second, ncclParamRegistry::find(key));
}

TEST_F(ParamRegistryMicrotest, State_IsASingletonSharedAcrossAllAccessors) {
  // Cross-compared, so they share one RegistryState rather than each being self-consistent.
  EXPECT_EQ(ncclParamRegistryInstance(), static_cast<void*>(&ncclParamRegistry::state()));
  EXPECT_EQ(&ncclParamRegistry::state().map, &ncclParamRegistry::instance());
  EXPECT_EQ(&ncclParamRegistry::state().mtx, &ncclParamRegistry::mutex());
}

// None of the other tests run concurrently, so they pass without the lock_guards; holding the
// mutex here and checking a concurrent call blocks pins each one.
void ExpectBlocksOnRegistryMutex(const std::function<void()>& call) {
  std::atomic<bool> done{false};
  std::unique_lock<std::mutex> lock(ncclParamRegistry::mutex());
  std::thread worker([&] {
    call();
    done.store(true, std::memory_order_release);
  });
  std::this_thread::sleep_for(std::chrono::milliseconds(50));
  EXPECT_FALSE(done.load(std::memory_order_acquire));

  lock.unlock();
  worker.join();
  EXPECT_TRUE(done.load(std::memory_order_acquire));
}

TEST_F(ParamRegistryMicrotest, AddLocksTheSharedMutexForTheDurationOfTheCall) {
  std::string key = Key();
  FakeParam fake(7);
  ncclResult_t result = ncclInternalError;
  ExpectBlocksOnRegistryMutex([&] { result = Register(key, &fake); });
  EXPECT_EQ(ncclSuccess, result);
}

TEST_F(ParamRegistryMicrotest, FindLocksTheSharedMutexForTheDurationOfTheCall) {
  std::string key = Key();
  FakeParam fake(7);
  ASSERT_EQ(ncclSuccess, Register(key, &fake));
  ncclParamRegistry::mapEntry* entry = nullptr;
  ExpectBlocksOnRegistryMutex([&] { entry = ncclParamRegistry::find(key); });
  ASSERT_NE(nullptr, entry);
  EXPECT_EQ(&fake, entry->param);
}

TEST_F(ParamRegistryMicrotest, RemoveLocksTheSharedMutexForTheDurationOfTheCall) {
  std::string key = Key();
  FakeParam fake(7);
  ASSERT_EQ(ncclSuccess, Register(key, &fake));
  ncclResult_t result = ncclInternalError;
  ExpectBlocksOnRegistryMutex([&] { result = ncclParamRegistry::remove(key); });
  EXPECT_EQ(ncclSuccess, result);
  EXPECT_EQ(nullptr, ncclParamRegistry::find(key));
}

}  // namespace
