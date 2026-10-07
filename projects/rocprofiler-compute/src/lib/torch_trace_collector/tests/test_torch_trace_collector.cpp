// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT

#include "../argument_capture.h"
#include "../torch_trace_collector.h"
#include "../wire_format.h"
#include "roctx_range_intercept.h"

// Include only the constants by path, so the shim cannot shadow real headers.
#include "../torch_abi/torch_abi.h"

#include <ATen/ATen.h>
#include <ATen/ThreadLocalState.h>
#include <ATen/core/dispatch/Dispatcher.h>
#include <ATen/record_function.h>
#include <c10/util/ThreadLocalDebugInfo.h>
#include <gtest/gtest.h>
#include <torch/library.h>
#include <torch/version.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <exception>
#include <functional>
#include <future>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

static_assert(sizeof(at::RecordFunction) == torch_abi::kRecordFunctionSize);
static_assert(alignof(at::RecordFunction) == torch_abi::kRecordFunctionAlignment);
static_assert(static_cast<std::size_t>(at::RecordScope::NUM_SCOPES) == torch_abi::kScopeCount);
static_assert(sizeof(at::RecordFunctionCallback) == torch_abi::kCallbackSize);
static_assert(alignof(at::RecordFunctionCallback) == torch_abi::kCallbackAlignment);
static_assert(sizeof(at::ObserverContext) == torch_abi::kObserverContextSize);
static_assert(alignof(at::ObserverContext) == torch_abi::kObserverContextAlignment);
static_assert(sizeof(c10::DebugInfoKind) == torch_abi::kDebugInfoKindSize);
static_assert(alignof(c10::DebugInfoKind) == torch_abi::kDebugInfoKindAlignment);
static_assert(sizeof(c10::DebugInfoBase) == torch_abi::kDebugInfoBaseSize);
static_assert(alignof(c10::DebugInfoBase) == torch_abi::kDebugInfoBaseAlignment);
static_assert(sizeof(std::shared_ptr<c10::DebugInfoBase>) == torch_abi::kSharedDebugInfoSize);
static_assert(alignof(std::shared_ptr<c10::DebugInfoBase>) == torch_abi::kSharedDebugInfoAlignment);
static_assert(sizeof(c10::IValue) == torch_abi::kIValueSize);
static_assert(alignof(c10::IValue) == torch_abi::kIValueAlignment);
static_assert(sizeof(at::Tensor) == torch_abi::kTensorSize);
static_assert(sizeof(c10::OperatorName) == torch_abi::kOperatorNameSize);
static_assert(alignof(c10::OperatorName) == torch_abi::kOperatorNameAlignment);
static_assert(offsetof(c10::OperatorName, name) == torch_abi::kOperatorNameNameOff);
static_assert(offsetof(c10::OperatorName, overload_name) == torch_abi::kOperatorNameOverloadNameOff);
static_assert(sizeof(std::optional<c10::OperatorName>) == torch_abi::kOptionalOperatorNameSize);
static_assert(alignof(std::optional<c10::OperatorName>) == torch_abi::kOptionalOperatorNameAlignment);
static_assert(sizeof(c10::OperatorHandle) == torch_abi::kOperatorHandleSize);
static_assert(alignof(c10::OperatorHandle) == torch_abi::kOperatorHandleAlignment);
static_assert(sizeof(c10::OpRegistrationListener) == torch_abi::kOpRegistrationListenerSize);
static_assert(alignof(c10::OpRegistrationListener) == torch_abi::kOpRegistrationListenerAlignment);
static_assert(sizeof(c10::RegistrationHandleRAII) == torch_abi::kRegistrationHandleSize);
static_assert(alignof(c10::RegistrationHandleRAII) == torch_abi::kRegistrationHandleAlignment);
static_assert(sizeof(c10::ArrayRef<const c10::IValue>) == torch_abi::kInputArrayViewSize);
static_assert(alignof(c10::ArrayRef<const c10::IValue>) == torch_abi::kInputArrayViewAlignment);

namespace
{

template<typename Value, typename Object>
Value read_at_offset(const Object& object, std::size_t offset)
{
    Value value{};
    std::memcpy(&value, reinterpret_cast<const std::byte*>(&object) + offset, sizeof(value));
    return value;
}

void emit_record(std::string_view                name,
                 const std::vector<c10::IValue>& inputs = {},
                 at::RecordScope                 scope  = at::RecordScope::FUNCTION)
{
    at::RecordFunction record(scope);
    record.before(name, &inputs);
}

std::string captured_arguments(const std::vector<c10::IValue>& inputs)
{
    at::RecordFunction record(at::RecordScope::FUNCTION);
    record.before("test.arguments", &inputs);
    std::array<char, torch_trace_collector::detail::kMaxEncodedArgumentsSize> output{};
    torch_trace_collector::detail::capture_args(record, output.data(), output.size());
    return output.data();
}

std::string captured_schema_arguments(const char*                     name,
                                      const char*                     overload,
                                      const std::vector<c10::IValue>& inputs)
{
    const auto         handle = c10::Dispatcher::singleton().findSchemaOrThrow(name, overload);
    at::RecordFunction record(at::RecordScope::FUNCTION);
    record.before(std::cref(handle.schema()), &inputs);
    std::array<char, torch_trace_collector::detail::kMaxEncodedArgumentsSize> output{};
    torch_trace_collector::detail::capture_args(record, output.data(), output.size());
    return output.data();
}

class TorchTraceCollectorTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        ASSERT_EQ(torch_trace_collector_install(), 0);
        roctx_range_intercept::start_recording();
    }

    void TearDown() override
    {
        (void)roctx_range_intercept::stop_recording();
        while (torch_trace_collector_pop_launcher_tid() == 0)
        {
        }
    }
};

}  // namespace

// CTest runs this group separately because installation lasts for the process.
TEST(TorchTraceCollectorBeforeInstall, RejectsLauncherUntilConcurrentInstallSucceeds)
{
    EXPECT_EQ(torch_trace_collector_abi_revision(), TORCH_TRACE_COLLECTOR_ABI_REVISION);
    EXPECT_NE(torch_trace_collector_push_launcher_tid(7), 0);
    EXPECT_NE(torch_trace_collector_pop_launcher_tid(), 0);
    std::atomic<int>         failures{0};
    std::atomic<bool>        start{false};
    std::vector<std::thread> workers;
    for (int index = 0; index < 8; ++index)
    {
        workers.emplace_back(
            [&]
            {
                while (!start.load())
                    std::this_thread::yield();
                if (torch_trace_collector_install() != 0)
                    ++failures;
            });
    }
    start.store(true);
    for (auto& worker : workers)
        worker.join();
    EXPECT_EQ(failures.load(), 0);
    ASSERT_EQ(torch_trace_collector_push_launcher_tid(7), 0);
    roctx_range_intercept::start_recording();
    emit_record("initial-install");
    const auto captured = roctx_range_intercept::stop_recording();
    EXPECT_EQ(captured.messages.size(), 1u);
    EXPECT_EQ(captured.pops, 1u);
    EXPECT_EQ(torch_trace_collector_pop_launcher_tid(), 0);
}

TEST_F(TorchTraceCollectorTest, RecordFunctionOffsetsMatchRealAccessors)
{
    const std::vector<c10::IValue> inputs{std::int64_t{9}, true};
    at::RecordFunction             record(at::RecordScope::USER_SCOPE);
    record.setForwardThreadId(1234);
    record.before("layout", &inputs, 987654321);
    EXPECT_EQ(read_at_offset<at::RecordScope>(record, torch_abi::kRecordFunctionScopeOff),
              record.scope());
    EXPECT_EQ(read_at_offset<std::int64_t>(record, torch_abi::kRecordFunctionSeqNrOff), record.seqNr());
    EXPECT_EQ(read_at_offset<std::uint64_t>(record, torch_abi::kRecordFunctionFwdThreadOff),
              record.forwardThreadId());
    const auto view = read_at_offset<c10::ArrayRef<const c10::IValue>>(record,
                                                                       torch_abi::kRecordFunctionInputsOff);
    EXPECT_EQ(view.data(), inputs.data());
    EXPECT_EQ(view.size(), inputs.size());
    EXPECT_EQ(record.seqNr(), 987654321);
    EXPECT_EQ(record.forwardThreadId(), 1234u);
}

TEST(TorchAbiLayout, CallbackOffsetsMatchRealAccessors)
{
    const auto start = +[](const at::RecordFunction&) -> std::unique_ptr<at::ObserverContext>
    {
        return nullptr;
    };
    const auto end = +[](const at::RecordFunction&, at::ObserverContext*) {
    };
    at::RecordFunctionCallback callback(start, end);
    callback.needsInputs(true).samplingProb(0.5).scopes({at::RecordScope::USER_SCOPE});
    EXPECT_EQ(read_at_offset<at::RecordFunctionCallback::StartCallback>(callback,
                                                                        torch_abi::kCallbackStartOff),
              callback.start());
    EXPECT_EQ(read_at_offset<at::RecordFunctionCallback::EndCallback>(callback, torch_abi::kCallbackEndOff),
              callback.end());
    EXPECT_EQ(read_at_offset<double>(callback, torch_abi::kCallbackProbabilityOff),
              callback.samplingProb());
    EXPECT_EQ(read_at_offset<bool>(callback, torch_abi::kCallbackNeedsInputsOff),
              callback.needsInputs());
    const auto scopes = read_at_offset<std::array<bool, torch_abi::kScopeCount>>(callback,
                                                                                 torch_abi::kCallbackScopesOff);
    for (std::size_t index = 0; index < scopes.size(); ++index)
    {
        EXPECT_EQ(scopes[index], index == static_cast<std::size_t>(at::RecordScope::USER_SCOPE));
    }
}

TEST(TorchAbiLayout, IValueTensorPayloadAndTagMatchRealTensor)
{
    const auto        tensor = at::empty({2, 3});
    const c10::IValue value(tensor);
    EXPECT_EQ(read_at_offset<std::uint32_t>(value, torch_abi::kIValueTagOff), torch_abi::kIValueTensorTag);
    EXPECT_EQ(read_at_offset<const void*>(value, torch_abi::kIValuePayloadOff),
              tensor.unsafeGetTensorImpl());
}

TEST(TorchAbiLayout, TensorListElementOffsetsMatchRealStorage)
{
    const auto tensor        = at::empty({2, 3});
    const auto check_storage = [](const c10::IValue& value)
    {
        ASSERT_TRUE(value.isTensorList());
        const auto* impl = static_cast<const c10::detail::ListImpl*>(value.internalToPointer());
        ASSERT_NE(impl, nullptr);
        EXPECT_EQ(read_at_offset<const void*>(value, torch_abi::kIValuePayloadOff), impl);
        const auto elements = value.toListRef();
        const auto* begin = read_at_offset<const c10::IValue*>(*impl,
                                                               torch_abi::kListImplElementsBeginOff);
        const auto* end = read_at_offset<const c10::IValue*>(*impl, torch_abi::kListImplElementsEndOff);
        EXPECT_EQ(begin, elements.data());
        EXPECT_EQ(end, elements.empty() ? elements.data() : &elements.back() + 1);
    };
    for (const std::size_t size : {0, 1, 9})
    {
        SCOPED_TRACE(size);
        check_storage(c10::IValue(std::vector<at::Tensor>(size, tensor)));
    }
    check_storage(c10::IValue(std::vector<at::Tensor>{tensor, at::Tensor{}, tensor}));
    c10::List<at::Tensor> reserved;
    reserved.reserve(8);
    const c10::IValue reserved_value(reserved);
    ASSERT_NE(reserved_value.toListRef().data(), nullptr);
    check_storage(reserved_value);
}

TEST_F(TorchTraceCollectorTest, ConcurrentInstallationKeepsOneCallback)
{
    std::atomic<int>         failures{0};
    std::vector<std::thread> workers;
    for (int index = 0; index < 8; ++index)
    {
        workers.emplace_back(
            [&failures]
            {
                for (int iteration = 0; iteration < 16; ++iteration)
                {
                    if (torch_trace_collector_install() != 0)
                        ++failures;
                }
            });
    }
    for (auto& worker : workers)
        worker.join();
    emit_record("one-callback");
    const auto captured = roctx_range_intercept::stop_recording();
    EXPECT_EQ(failures.load(), 0);
    ASSERT_EQ(captured.messages.size(), 1u);
    EXPECT_EQ(captured.pops, 1u);
}

TEST_F(TorchTraceCollectorTest, EveryRecordScopeProducesOneBalancedRange)
{
    constexpr std::array<const char*, 10> names = {"FUNCTION",
                                                   "BACKWARD_FUNCTION",
                                                   "TORCHSCRIPT_FUNCTION",
                                                   "KERNEL_FUNCTION_DTYPE",
                                                   "CUSTOM_CLASS",
                                                   "BUILD_FEATURE",
                                                   "LITE_INTERPRETER",
                                                   "USER_SCOPE",
                                                   "STATIC_RUNTIME_OP",
                                                   "STATIC_RUNTIME_MODEL"};
    for (std::size_t index = 0; index < names.size(); ++index)
        emit_record("scope/%", {}, static_cast<at::RecordScope>(index));
    const auto captured = roctx_range_intercept::stop_recording();
    ASSERT_EQ(captured.messages.size(), names.size());
    EXPECT_EQ(captured.pops, names.size());
    for (std::size_t index = 0; index < names.size(); ++index)
    {
        EXPECT_EQ(captured.messages[index].find("scope%2F%25:n/a|seqNr=n/a|"), 0u);
        EXPECT_NE(captured.messages[index].find(std::string{"|scope="} + names[index] + "|args=n/a|torch"),
                  std::string::npos);
    }
}

TEST_F(TorchTraceCollectorTest, NestedRecordsStayBalancedWhenWorkloadThrows)
{
    EXPECT_THROW(
        {
            at::RecordFunction outer(at::RecordScope::USER_SCOPE);
            outer.before("outer");
            at::RecordFunction inner(at::RecordScope::FUNCTION);
            inner.before("inner");
            throw std::runtime_error("workload failure");
        },
        std::runtime_error);
    const auto captured = roctx_range_intercept::stop_recording();
    EXPECT_EQ(captured.messages.size(), 2u);
    EXPECT_EQ(captured.pops, 2u);
}

TEST_F(TorchTraceCollectorTest, FailedPushDoesNotPopTheParentRange)
{
    {
        at::RecordFunction parent(at::RecordScope::USER_SCOPE);
        parent.before("parent");
        roctx_range_intercept::fail_next_push();
        emit_record("failed-child");
    }
    const auto captured = roctx_range_intercept::stop_recording();
    EXPECT_EQ(captured.messages.size(), 1u);
    EXPECT_EQ(captured.pops, 1u);
}

TEST_F(TorchTraceCollectorTest, LauncherIdsNestAndRestore)
{
    EXPECT_NE(torch_trace_collector_pop_launcher_tid(), 0);
    ASSERT_EQ(torch_trace_collector_push_launcher_tid(101), 0);
    emit_record("outer-launcher");
    ASSERT_EQ(torch_trace_collector_push_launcher_tid(202), 0);
    emit_record("inner-launcher");
    ASSERT_EQ(torch_trace_collector_pop_launcher_tid(), 0);
    emit_record("restored-launcher");
    ASSERT_EQ(torch_trace_collector_pop_launcher_tid(), 0);
    emit_record("no-launcher");
    EXPECT_NE(torch_trace_collector_pop_launcher_tid(), 0);
    const auto captured = roctx_range_intercept::stop_recording();
    ASSERT_EQ(captured.messages.size(), 4u);
    const std::array<const char*, 4> expected = {"101", "202", "101", "n/a"};
    for (std::size_t index = 0; index < expected.size(); ++index)
        EXPECT_NE(captured.messages[index].find(std::string{"|ltid="} + expected[index] + "|"),
                  std::string::npos);
    EXPECT_EQ(captured.pops, 4u);
}

TEST_F(TorchTraceCollectorTest, LauncherLifetimeSurvivesPropagationToWorker)
{
    ASSERT_EQ(torch_trace_collector_push_launcher_tid(12345), 0);
    const at::ThreadLocalState launcher_state;
    ASSERT_EQ(torch_trace_collector_pop_launcher_tid(), 0);
    std::thread worker(
        [&launcher_state]
        {
            emit_record("before-guard");
            {
                at::ThreadLocalStateGuard guard(launcher_state);
                emit_record("worker", {}, at::RecordScope::BACKWARD_FUNCTION);
            }
            emit_record("after-guard");
        });
    worker.join();
    const auto captured = roctx_range_intercept::stop_recording();
    ASSERT_EQ(captured.messages.size(), 3u);
    EXPECT_NE(captured.messages[0].find("|ltid=n/a|"), std::string::npos);
    EXPECT_NE(captured.messages[1].find("|ltid=12345|"), std::string::npos);
    EXPECT_NE(captured.messages[2].find("|ltid=n/a|"), std::string::npos);
    EXPECT_EQ(captured.pops, 3u);
}

TEST_F(TorchTraceCollectorTest, ConcurrentLaunchersRemainIsolated)
{
    std::atomic<int>         failures{0};
    std::vector<std::thread> workers;
    for (std::uint64_t index = 1; index <= 4; ++index)
    {
        workers.emplace_back(
            [index, &failures]
            {
                if (torch_trace_collector_push_launcher_tid(index) != 0)
                    ++failures;
                emit_record("worker-" + std::to_string(index));
                if (torch_trace_collector_pop_launcher_tid() != 0)
                    ++failures;
            });
    }
    for (auto& worker : workers)
        worker.join();
    const auto captured = roctx_range_intercept::stop_recording();
    EXPECT_EQ(failures.load(), 0);
    ASSERT_EQ(captured.messages.size(), 4u);
    for (std::uint64_t index = 1; index <= 4; ++index)
    {
        EXPECT_TRUE(std::any_of(captured.messages.begin(),
                                captured.messages.end(),
                                [index](const std::string& message)
                                {
                                    return message.find("worker-" + std::to_string(index) + ":") == 0 &&
                                           message.find("|ltid=" + std::to_string(index) + "|") !=
                                               std::string::npos;
                                }));
    }
    EXPECT_EQ(captured.pops, 4u);
}

TEST_F(TorchTraceCollectorTest, CapturesTensorShapesDtypesAndScalarTags)
{
    const auto matrix = at::empty({2, 3}, at::TensorOptions().dtype(at::kFloat));
    const auto scalar = at::empty({}, at::TensorOptions().dtype(at::kLong));
    const auto empty  = at::empty({0, 7}, at::TensorOptions().dtype(at::kBFloat16));
    EXPECT_EQ(captured_arguments({matrix, scalar, empty, at::Tensor{}, std::int64_t{4}, true, "text"}),
              "(float32[2x3], int64[], bfloat16[0x7], None, Int, Bool, String)");
    EXPECT_EQ(captured_arguments({}), "n/a");
}

TEST_F(TorchTraceCollectorTest, CapturesRealSchemaArgumentNamesAndOverloads)
{
    const auto tensor = at::empty({2, 3});
    EXPECT_EQ(captured_schema_arguments("aten::add", "Tensor", {tensor, tensor, std::int64_t{1}}),
              "(self=float32[2x3], other=float32[2x3], alpha=Int)");
    EXPECT_EQ(captured_schema_arguments("aten::add", "Scalar", {tensor, std::int64_t{1}, std::int64_t{1}}),
              "(self=float32[2x3], other=Int, alpha=Int)");
    EXPECT_EQ(captured_schema_arguments("aten::to",
                                        "dtype",
                                        {tensor, std::int64_t{6}, false, false, c10::IValue{}}),
              "(self=float32[2x3], dtype=Int, non_blocking=Bool, copy=Bool, memory_format=None)");
}

TEST_F(TorchTraceCollectorTest, CapturesReloadedSchemaNamesOnWarmAndColdThreads)
{
    constexpr auto       name = "schema_cache_regression::reload";
    constexpr std::array argument_names{"original_input", "replacement_input", "final_input"};
    torch::Library library(torch::Library::FRAGMENT, "schema_cache_regression", std::nullopt, __FILE__, __LINE__);
    torch::Library implementation(torch::Library::IMPL, "schema_cache_regression", std::nullopt, __FILE__, __LINE__);
    implementation.impl("reload", [](std::int64_t input) { return input; });
    auto original_handle = c10::Dispatcher::singleton().findOp({name, ""});
    ASSERT_TRUE(original_handle.has_value());
    const std::vector<c10::IValue> inputs{std::int64_t{7}};

    std::array<std::promise<std::string>, argument_names.size()> worker_captured;
    std::array<std::future<std::string>, argument_names.size()>  worker_arguments;
    std::future<void>                                            worker;
    // Destruction releases the worker before its async future can wait for it.
    std::array<std::promise<void>, argument_names.size()> schema_ready;
    std::array<std::future<void>, argument_names.size()>  readiness;
    for (std::size_t index = 0; index < argument_names.size(); ++index)
    {
        worker_arguments[index] = worker_captured[index].get_future();
        readiness[index]        = schema_ready[index].get_future();
    }
    worker = std::async(std::launch::async,
                        [&, ready = std::move(readiness)]() mutable
                        {
                            for (std::size_t index = 0; index < argument_names.size(); ++index)
                            {
                                try
                                {
                                    ready[index].get();
                                    worker_captured[index].set_value(
                                        captured_schema_arguments(name, "", inputs));
                                }
                                catch (...)
                                {
                                    worker_captured[index].set_exception(std::current_exception());
                                    return;
                                }
                            }
                        });

    for (std::size_t index = 0; index < argument_names.size(); ++index)
    {
        SCOPED_TRACE(argument_names[index]);
        library.reset();
        EXPECT_FALSE(c10::Dispatcher::singleton().findSchema({name, ""}).has_value());
        if (index + 1 == argument_names.size())
        {
            original_handle.reset();
            implementation.reset();
            EXPECT_FALSE(c10::Dispatcher::singleton().findOp({name, ""}).has_value());
        }
        const auto schema = "reload(int " + std::string{argument_names[index]} + ") -> int";
        library.def(schema.c_str());
        const auto handle = c10::Dispatcher::singleton().findSchemaOrThrow(name, "");
        EXPECT_EQ(handle.schema().arguments()[0].name(), argument_names[index]);
        if (original_handle.has_value())
        {
            EXPECT_EQ(handle, *original_handle);
        }

        const auto expected = "(" + std::string{argument_names[index]} + "=Int)";
        EXPECT_EQ(captured_schema_arguments(name, "", inputs), expected);
        schema_ready[index].set_value();
        EXPECT_EQ(worker_arguments[index].get(), expected);
        EXPECT_EQ(std::async(std::launch::async,
                             [&] { return captured_schema_arguments(name, "", inputs); })
                      .get(),
                  expected);
    }
    worker.get();
}

TEST_F(TorchTraceCollectorTest, RealForwardBackwardDispatchCapturesArgumentsAndBalances)
{
    const auto input = at::randn({2, 3}).requires_grad_(true);
    roctx_range_intercept::start_recording();
    const auto loss = (input * 2).sum();
    loss.backward();
    const auto captured = roctx_range_intercept::stop_recording();
    ASSERT_FALSE(captured.messages.empty());
    EXPECT_EQ(captured.messages.size(), captured.pops);
    EXPECT_TRUE(std::any_of(captured.messages.begin(),
                            captured.messages.end(),
                            [](const std::string& message)
                            {
                                return message.find("aten::mul:") == 0 &&
                                       message.find("|args=(self=float32[2x3],") != std::string::npos;
                            }));
    EXPECT_TRUE(
        std::any_of(captured.messages.begin(),
                    captured.messages.end(),
                    [](const std::string& message)
                    { return message.find("|scope=BACKWARD_FUNCTION|") != std::string::npos; }));
}

TEST_F(TorchTraceCollectorTest, TensorListsAndTopLevelItemsRespectBounds)
{
    const auto                 tensor = at::empty({2}, at::TensorOptions().dtype(at::kHalf));
    constexpr std::string_view seven_elements =
        "([float16[2], float16[2], float16[2], float16[2], float16[2], float16[2], float16[2]])";
    constexpr std::string_view eight_elements = "([float16[2], float16[2], float16[2], float16[2], "
                                                "float16[2], float16[2], float16[2], float16[2]])";
    constexpr std::array<std::pair<std::size_t, std::string_view>, 6> cases = {{
        {0, "([])"},
        {1, "([float16[2]])"},
        {7, seven_elements},
        {8, eight_elements},
        {9, eight_elements},
        {1048577, eight_elements},
    }};
    for (const auto& [size, expected] : cases)
    {
        SCOPED_TRACE(size);
        EXPECT_EQ(captured_arguments({c10::IValue(std::vector<at::Tensor>(size, tensor))}), expected);
    }
    c10::List<at::Tensor> reserved;
    reserved.reserve(8);
    EXPECT_EQ(captured_arguments({c10::IValue(reserved)}), "([])");
    EXPECT_EQ(captured_arguments({c10::IValue(std::vector<at::Tensor>{at::Tensor{}})}), "([None])");
    const auto matrix = at::empty({3, 4});
    const auto scalar = at::empty({}, at::TensorOptions().dtype(at::kLong));
    const auto empty  = at::empty({0, 7}, at::TensorOptions().dtype(at::kBFloat16));
    const std::vector<at::Tensor>
        mixed{at::Tensor{}, tensor, matrix, at::Tensor{}, scalar, empty, tensor, at::Tensor{}, matrix};
    EXPECT_EQ(captured_arguments({c10::IValue(mixed)}),
              "([None, float16[2], float32[3x4], None, int64[], bfloat16[0x7], float16[2], None])");
    const std::vector<c10::IValue> inputs(33, true);
    const auto                     arguments = captured_arguments(inputs);
    EXPECT_EQ(std::count(arguments.begin(), arguments.end(), ','), 31);
    EXPECT_EQ(arguments.substr(arguments.size() - 5), "Bool)");
}

TEST_F(TorchTraceCollectorTest, LongArgumentOutputIsBoundedAndTerminated)
{
    const std::vector<std::int64_t> dimensions(64, 1);
    const auto                      tensor = at::empty(dimensions);
    const std::vector<c10::IValue>  inputs(32, tensor);
    const auto                      arguments = captured_arguments(inputs);
    EXPECT_EQ(arguments.size(), torch_trace_collector::detail::kMaxArgsLength + 4);
    EXPECT_EQ(arguments.substr(arguments.size() - 4), "...)");

    std::string tensor_text = "float32[1";
    for (std::size_t index = 1; index < dimensions.size(); ++index)
        tensor_text += "x1";
    tensor_text += ']';
    std::string list_text = "[";
    for (int index = 0; index < 8; ++index)
        list_text += (index == 0 ? "" : ", ") + tensor_text;
    list_text += ']';
    const c10::IValue list(std::vector<at::Tensor>(100000, tensor));
    const auto        truncate = [](std::string expected)
    {
        expected.resize(torch_trace_collector::detail::kMaxArgsLength);
        return expected + "...)";
    };
    EXPECT_EQ(captured_arguments({list}), truncate("(" + list_text + ")"));
    EXPECT_EQ(captured_arguments({tensor, tensor, tensor, list}),
              truncate("(" + tensor_text + ", " + tensor_text + ", " + tensor_text + ", " +
                       list_text + ")"));

    at::RecordFunction record(at::RecordScope::FUNCTION);
    record.before("small-output", &inputs);
    std::array<char, 5> short_output{'?', '?', '?', '?', '!'};
    const auto required = torch_trace_collector::detail::capture_args(record, short_output.data(), 4);
    EXPECT_EQ(required, arguments.size() + 1);
    EXPECT_EQ(short_output[3], '\0');
    EXPECT_EQ(short_output[4], '!');
    EXPECT_EQ(torch_trace_collector::detail::capture_args(record, nullptr, 0), required);
}

TEST_F(TorchTraceCollectorTest, ExtendedDtypesMatchTheRuntimeMinor)
{
    const auto tensor = at::empty({2}, at::TensorOptions().dtype(at::ScalarType::UInt16));
    EXPECT_EQ(captured_arguments({tensor}), "(uint16[2])");
#if TORCH_VERSION_MAJOR == 2 && TORCH_VERSION_MINOR >= 14
    static_assert(static_cast<std::int32_t>(at::ScalarType::BComplex32) ==
                  torch_abi::kBComplex32ScalarType);
    const auto complex_tensor = at::empty({2}, at::TensorOptions().dtype(at::ScalarType::BComplex32));
    EXPECT_EQ(captured_arguments({complex_tensor}), "(bcomplex32[2])");
#else
    static_assert(static_cast<std::int32_t>(at::ScalarType::Undefined) == torch_abi::kBComplex32ScalarType);
#endif
}

TEST(MarkerEncoding, EncodesNamesAndPreservesEncodedArgumentField)
{
    using torch_trace_collector::detail::format_range_name;
    using torch_trace_collector::detail::RangeNameFields;
    const RangeNameFields fields{"aten::add/%", "n/a", 7, 11, 13, 17, "FUNCTION", "(value=%25%7C%3B%0D%0A)", "torch"};
    const std::string expected =
        "aten::add%2F%25:n/"
        "a|seqNr=7|tid=11|ftid=13|ltid=17|scope=FUNCTION|args=(value=%25%7C%3B%0D%0A)|torch";
    std::array<char, 256> output{};
    EXPECT_EQ(format_range_name(output.data(), output.size(), fields), expected.size() + 1);
    EXPECT_EQ(std::string{output.data()}, expected);
    std::array<char, 12> short_output{};
    EXPECT_EQ(format_range_name(short_output.data(), short_output.size(), fields), expected.size() + 1);
    EXPECT_EQ(std::string{short_output.data()}, expected.substr(0, short_output.size() - 1));
    EXPECT_EQ(short_output.back(), '\0');
    EXPECT_EQ(format_range_name(nullptr, 0, fields), expected.size() + 1);
}
