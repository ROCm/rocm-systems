// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/result.h"
#include "util/diagnostic.h"
#include "util/result.h"

#include <gtest/gtest.h>

#include <cstddef>
#include <iomanip>
#include <memory>
#include <string>
#include <string_view>
#include <type_traits>
#include <vector>

struct CustomNicheValue {
  int value;
};

class FancyPointer {
public:
  using element_type = int;

  constexpr FancyPointer() = default;
  constexpr FancyPointer(std::nullptr_t) {}
  explicit constexpr FancyPointer(int state) : state_(state) {}

  explicit constexpr operator bool() const { return state_ != 0; }
  friend constexpr bool operator==(FancyPointer, FancyPointer) = default;
  friend constexpr bool operator==(FancyPointer pointer, std::nullptr_t) {
    return pointer.state_ == 0;
  }
  [[nodiscard]] constexpr int state() const { return state_; }

private:
  int state_ = 0;
};

struct FancyPointerDeleter {
  using pointer = FancyPointer;

  void operator()(FancyPointer) const noexcept {}
};

template <> struct util::failure_or_storage_traits<CustomNicheValue> {
  static constexpr bool has_niche = true;

  [[nodiscard]] static constexpr CustomNicheValue failure_value() noexcept { return {-1}; }
  [[nodiscard]] static constexpr bool is_failure(const CustomNicheValue &value) noexcept {
    return value.value == -1;
  }
};

template <> struct util::failure_or_storage_traits<FancyPointer> {
  static constexpr bool has_niche = true;

  [[nodiscard]] static constexpr FancyPointer failure_value() noexcept { return FancyPointer(1); }
  [[nodiscard]] static constexpr bool is_failure(FancyPointer pointer) noexcept {
    return pointer.state() == 1;
  }
};

namespace {

struct PotentiallyThrowingValue {
  PotentiallyThrowingValue() = default;
  PotentiallyThrowingValue(PotentiallyThrowingValue &&) noexcept(false) {}
  PotentiallyThrowingValue &operator=(PotentiallyThrowingValue &&) noexcept(false) { return *this; }
};

struct CountingDeleter {
  static inline int calls = 0;

  void operator()(int *value) const noexcept {
    ++calls;
    delete value;
  }
};

struct FormattingProbe {
  bool &formatted;
};

struct ConstCollector {
  std::vector<std::string> &messages;

  void operator()(std::string_view message) const { messages.emplace_back(message); }
};

using DiagnosticFunction = void(std::string_view);

template <typename T>
concept HasRvalueEmitter = requires { std::declval<T &&>().emitter(); };

std::ostream &operator<<(std::ostream &stream, const FormattingProbe &probe) {
  probe.formatted = true;
  return stream;
}

TEST(ResultTest, RepresentsSuccessAndFailure) {
  constexpr util::Result default_failure;
  constexpr auto success = util::Result::success();
  constexpr util::Result failure = util::Result::failure();
  static_assert(default_failure.failed());
  static_assert(success.succeeded() && !success.failed());
  static_assert(failure.failed() && !failure.succeeded());
  static_assert(noexcept(success.succeeded()) && noexcept(failure.failed()));
  static_assert(std::is_same_v<decltype(util::Result::failure()), util::Result>);
  static_assert(std::is_same_v<rocjitsu::Result, util::Result>);
}

TEST(FailureOrTest, RepresentsValueOrFailureWithoutThrowing) {
  using IntResult = util::FailureOr<int>;
  constexpr IntResult value(42);
  constexpr IntResult failure(util::Result::failure());
  static_assert(value.succeeded() && value.value() == 42);
  static_assert(failure.failed());
  static_assert(std::is_nothrow_copy_constructible_v<IntResult>);
  static_assert(std::is_nothrow_copy_assignable_v<IntResult>);
  static_assert(std::is_nothrow_move_constructible_v<IntResult>);
  static_assert(std::is_nothrow_move_assignable_v<IntResult>);
  static_assert(std::is_nothrow_destructible_v<IntResult>);
  static_assert(std::is_nothrow_constructible_v<IntResult, util::Result>);
  static_assert(std::is_nothrow_move_constructible_v<util::FailureOr<std::unique_ptr<int>>>);
  static_assert(sizeof(util::FailureOr<std::unique_ptr<int>>) == sizeof(std::unique_ptr<int>));
  static_assert(sizeof(util::FailureOr<std::unique_ptr<int[]>>) == sizeof(std::unique_ptr<int[]>));
  using TerminatingResult = util::FailureOr<PotentiallyThrowingValue>;
  static_assert(std::is_nothrow_constructible_v<TerminatingResult, PotentiallyThrowingValue>);
  static_assert(std::is_nothrow_move_constructible_v<TerminatingResult>);
  static_assert(std::is_nothrow_move_assignable_v<TerminatingResult>);
  static_assert(noexcept(value.succeeded()) && noexcept(failure.failed()) &&
                noexcept(value.value()));
  static_assert(std::is_same_v<rocjitsu::FailureOr<int>, IntResult>);
}

TEST(FailureOrTest, DeducesValueType) {
  constexpr auto int_result = util::FailureOr(42);
  auto pointer_result = util::FailureOr(std::make_unique<int>(42));

  static_assert(std::is_same_v<std::remove_cv_t<decltype(int_result)>, util::FailureOr<int>>);
  static_assert(std::is_same_v<decltype(pointer_result), util::FailureOr<std::unique_ptr<int>>>);
  static_assert(int_result.succeeded() && int_result.value() == 42);
  ASSERT_TRUE(pointer_result.succeeded());
  EXPECT_EQ(*pointer_result.value(), 42);
}

TEST(FailureOrTest, RejectsSuccessfulResult) {
  EXPECT_DEBUG_DEATH(
      { [[maybe_unused]] util::FailureOr<int> result(util::Result::success()); }, "");
}

TEST(FailureOrTest, UsesNullAsPointerFailure) {
  constexpr util::FailureOr<int *> pointer_failure(util::Result::failure());
  constexpr util::FailureOr<char *> byte_pointer_failure(util::Result::failure());
  constexpr util::FailureOr<void *> void_pointer_failure(util::Result::failure());
  constexpr util::FailureOr<void (*)()> function_pointer_failure(util::Result::failure());

  static_assert(pointer_failure.failed());
  static_assert(sizeof(util::FailureOr<char *>) == sizeof(char *));
  static_assert(byte_pointer_failure.failed());
  static_assert(void_pointer_failure.failed());
  static_assert(function_pointer_failure.failed());
}

TEST(FailureOrTest, SupportsCustomNicheTraits) {
  constexpr util::FailureOr<CustomNicheValue> value(CustomNicheValue{42});
  constexpr util::FailureOr<CustomNicheValue> failure(util::Result::failure());

  static_assert(value.succeeded() && value.value().value == 42);
  static_assert(failure.failed());
  static_assert(sizeof(util::FailureOr<CustomNicheValue>) == sizeof(CustomNicheValue));
}

TEST(FailureOrTest, UsesNullUniquePointerAsFailure) {
  util::FailureOr<std::unique_ptr<int>> value(std::make_unique<int>(42));
  util::FailureOr<std::unique_ptr<int>> failure(util::Result::failure());

  ASSERT_TRUE(value.succeeded());
  EXPECT_EQ(*value.value(), 42);
  EXPECT_TRUE(failure.failed());
}

TEST(FailureOrTest, FallsBackToGenericStorageForFancyUniquePointer) {
  using Pointer = std::unique_ptr<int, FancyPointerDeleter>;
  static_assert(!util::failure_or_storage_traits<Pointer>::has_niche);

  util::FailureOr<Pointer> failure(util::Result::failure());
  util::FailureOr<Pointer> null_value(Pointer{});

  EXPECT_TRUE(failure.failed());
  EXPECT_TRUE(null_value.succeeded());
  EXPECT_FALSE(static_cast<bool>(null_value.value()));
}

TEST(FailureOrTest, RejectsTheFailureRepresentationOnTheSuccessPath) {
  EXPECT_DEBUG_DEATH(
      { [[maybe_unused]] util::FailureOr<int *> result(static_cast<int *>(nullptr)); }, "");
  EXPECT_DEBUG_DEATH(
      { [[maybe_unused]] util::FailureOr<std::unique_ptr<int>> result(std::unique_ptr<int>{}); },
      "");
}

TEST(FailureOrTest, DoesNotDeleteTheNullUniquePointerFailure) {
  using Pointer = std::unique_ptr<int, CountingDeleter>;
  using PointerResult = util::FailureOr<Pointer>;
  static_assert(sizeof(PointerResult) == sizeof(Pointer));

  CountingDeleter::calls = 0;
  {
    PointerResult result(util::Result::failure());
    result = PointerResult(Pointer(new int(42)));
    ASSERT_TRUE(result.succeeded());
    EXPECT_EQ(*result.value(), 42);

    result = PointerResult(util::Result::failure());
    EXPECT_TRUE(result.failed());
    EXPECT_EQ(CountingDeleter::calls, 1);
  }
  EXPECT_EQ(CountingDeleter::calls, 1);
}

TEST(InFlightDiagnosticTest, EmitsExactlyOnceAfterMove) {
  std::vector<std::string> messages;
  auto collect = [&](std::string_view message) { messages.emplace_back(message); };
  util::DiagnosticEmitter emitter(collect);

  {
    auto first = emitter.emit();
    first << "invalid opcode 0x" << std::hex << 42 << std::endl;
    auto second = std::move(first);
    EXPECT_TRUE(second.emit().failed());
  }

  ASSERT_EQ(messages.size(), 1u);
  EXPECT_EQ(messages.front(), "invalid opcode 0x2a\n");
}

TEST(DiagnosticSinkTest, SupportsConstCallableObjects) {
  std::vector<std::string> messages;
  const ConstCollector collect{messages};
  util::DiagnosticEmitter emitter(collect);

  EXPECT_TRUE((emitter.emit() << "const callable").emit().failed());
  ASSERT_EQ(messages.size(), 1u);
  EXPECT_EQ(messages.front(), "const callable");
  static_assert(!std::is_constructible_v<util::DiagnosticEmitter, DiagnosticFunction &>);
}

TEST(InFlightDiagnosticTest, EmitsOnDestruction) {
  std::vector<std::string> messages;
  auto collect = [&](std::string_view message) { messages.emplace_back(message); };
  util::DiagnosticEmitter emitter(collect);

  { emitter.emit() << "implicit emit"; }

  ASSERT_EQ(messages.size(), 1u);
  EXPECT_EQ(messages.front(), "implicit emit");
}

TEST(InFlightDiagnosticTest, RejectsFragmentsAfterEmission) {
  auto diagnostic = util::DiagnosticEmitter{}.emit();
  EXPECT_TRUE(diagnostic.emit().failed());
  EXPECT_DEBUG_DEATH(diagnostic << "late fragment", "");
}

TEST(InFlightDiagnosticTest, MoveAssignmentEmitsTheOverwrittenMessage) {
  std::vector<std::string> messages;
  auto collect = [&](std::string_view message) { messages.emplace_back(message); };
  util::DiagnosticEmitter emitter(collect);

  {
    auto first = emitter.emit();
    first << "first";
    auto second = emitter.emit();
    second << "second";
    first = std::move(second);
  }

  ASSERT_EQ(messages.size(), 2u);
  EXPECT_EQ(messages[0], "first");
  EXPECT_EQ(messages[1], "second");
}

TEST(InFlightDiagnosticTest, ConvertsToFailedValueResult) {
  std::vector<std::string> messages;
  auto collect = [&](std::string_view message) { messages.emplace_back(message); };
  util::DiagnosticEmitter emitter(collect);

  auto fail = [&]() -> util::FailureOr<std::unique_ptr<int>> {
    return emitter.emit() << "bad operand";
  };

  auto result = fail();
  EXPECT_TRUE(result.failed());
  ASSERT_EQ(messages.size(), 1u);
  EXPECT_EQ(messages.front(), "bad operand");
}

TEST(InFlightDiagnosticTest, NoOpEmitterStillReturnsFailure) {
  util::DiagnosticEmitter emitter;
  bool formatted = false;

  auto fail = [&]() -> util::Result { return emitter.emit() << FormattingProbe{formatted}; };

  EXPECT_TRUE(fail().failed());
  EXPECT_TRUE(emitter.ignores_messages());
  EXPECT_FALSE(formatted);
}

TEST(StringDiagnosticTest, RetainsLatestMessage) {
  static_assert(!std::is_copy_constructible_v<util::StringDiagnostic>);
  static_assert(!std::is_move_constructible_v<util::StringDiagnostic>);
  static_assert(!HasRvalueEmitter<util::StringDiagnostic>);

  util::StringDiagnostic diagnostic;
  auto emitter = diagnostic.emitter();

  EXPECT_TRUE((emitter.emit() << "invalid operand " << 7).emit().failed());
  EXPECT_EQ(diagnostic.message(), "invalid operand 7");
}

} // namespace
