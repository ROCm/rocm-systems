// Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

#include <hip_test_common.hh>
#include <resource_guards.hh>

#include <array>
#include <limits>

namespace {
enum class Submission { Eager, Batch, Capture, CaptureBatch, ExplicitGraph };

struct Input {
  uint16_t payload[256];
  uint32_t indices[4];
  uint32_t coefficients[4];
};
using Output = std::array<uint16_t, 256>;
struct SharedIO {
  Input input;
  Output output;
};

Input MakeInput(unsigned iteration) {
  Input input{};
  for (unsigned i = 0; i < 256; ++i) input.payload[i] = iteration * 107 + i * 13 + 1;
  for (unsigned i = 0; i < 4; ++i) {
    input.indices[i] = (iteration * 3 + i) % 97;
    input.coefficients[i] = iteration * 7 + i * 11;
  }
  return input;
}

Output Transform(const Input& input) {
  Output output{};
  for (unsigned i = 0; i < output.size(); ++i) {
    output[i] = input.payload[i] ^ (input.indices[i % 4] * 31) ^ input.coefficients[i % 4];
  }
  return output;
}

template <typename T> T Load(T* ptr) { return __atomic_load_n(ptr, __ATOMIC_ACQUIRE); }
template <typename T> void Store(T* ptr, T value) {
  __atomic_store_n(ptr, value, __ATOMIC_RELEASE);
}

struct SignalMemory {
  uint64_t* ptr = nullptr;
  SignalMemory() {
    HIP_CHECK(
        hipExtMallocWithFlags(reinterpret_cast<void**>(&ptr), sizeof(*ptr), hipMallocSignalMemory));
    Store(ptr, uint64_t{0});
  }
  ~SignalMemory() { static_cast<void>(hipFree(ptr)); }
  SignalMemory(const SignalMemory&) = delete;
  SignalMemory& operator=(const SignalMemory&) = delete;
};

struct GraphResources {
  hipGraph_t graph = nullptr;
  hipGraphExec_t exec = nullptr;
  ~GraphResources() {
    if (exec) static_cast<void>(hipGraphExecDestroy(exec));
    if (graph) static_cast<void>(hipGraphDestroy(graph));
  }
};

// Keep the responder alive through exceptional teardown, so a queued WAIT is
// released before destroying the stream or freeing its signal memory.
template <typename T> struct Responder {
  hipStream_t stream;
  T* ready;
  T* done;
  SharedIO* shared;
  std::atomic<unsigned> iteration{0};
  std::atomic<bool> stop{false}, timedOut{false};
  unsigned served = 0, badInputs = 0;
  std::thread worker;

  Responder(hipStream_t stream, T* ready, T* done, SharedIO* shared)
      : stream(stream), ready(ready), done(done), shared(shared), worker([this] {
          auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
          while (!stop.load()) {
            if (timedOut.load()) {
              Store(this->done, T{1});
            } else if (Load(this->ready) != 0) {
              Store(this->ready, T{0});
              const Input expected = MakeInput(iteration.load());
              const Input actual = this->shared->input;
              badInputs += std::memcmp(&actual, &expected, sizeof(Input)) != 0;
              this->shared->output = Transform(actual);
              ++served;
              Store(this->done, T{1});
              deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
            } else if (std::chrono::steady_clock::now() > deadline) {
              timedOut = true;
            }
            std::this_thread::yield();
          }
        }) {}

  void Join() {
    stop = true;
    if (worker.joinable()) worker.join();
  }
  ~Responder() {
    if (worker.joinable()) {
      timedOut = true;
      static_cast<void>(hipStreamSynchronize(stream));
      Join();
    }
  }
};

template <typename T> void CheckPublication(Submission submission) {
  constexpr unsigned iterations = 2000;
  const T published = sizeof(T) == 8 ? static_cast<T>(0x100000001ULL) : T{1};
  SignalMemory readyMemory, doneMemory;
  auto* ready = reinterpret_cast<T*>(readyMemory.ptr);
  auto* done = reinterpret_cast<T*>(doneMemory.ptr);
  LinearAllocGuard<SharedIO> sharedMemory(LinearAllocs::hipHostMalloc, sizeof(SharedIO),
                                          hipHostMallocMapped | hipHostMallocCoherent);
  auto* shared = sharedMemory.host_ptr();
  std::memset(shared, 0xa5, sizeof(*shared));
  LinearAllocGuard<Input> input(LinearAllocs::hipMalloc, sizeof(Input));
  LinearAllocGuard<Output> output(LinearAllocs::hipMalloc, sizeof(Output));
  StreamGuard streamGuard(Streams::withFlags, hipStreamNonBlocking);
  const auto stream = streamGuard.stream();
  GraphResources graph;

  hipStreamBatchMemOpParams ops[2]{};
  ops[0].operation = sizeof(T) == 4 ? hipStreamMemOpWriteValue32 : hipStreamMemOpWriteValue64;
  ops[0].writeValue.address = reinterpret_cast<hipDeviceptr_t>(ready);
  ops[0].writeValue.value64 = published;
  ops[0].writeValue.flags = hipStreamWriteValueDefault;
  ops[1].operation = sizeof(T) == 4 ? hipStreamMemOpWaitValue32 : hipStreamMemOpWaitValue64;
  ops[1].waitValue.address = reinterpret_cast<hipDeviceptr_t>(done);
  ops[1].waitValue.value64 = 1;
  ops[1].waitValue.flags = hipStreamWaitValueGte;

  auto enqueue = [&] {
    HIP_CHECK(hipMemcpyAsync(shared->input.payload, input.ptr()->payload, sizeof(Input::payload),
                             hipMemcpyDeviceToHost, stream));
    HIP_CHECK(hipMemcpyAsync(shared->input.indices, input.ptr()->indices, sizeof(Input::indices),
                             hipMemcpyDeviceToHost, stream));
    HIP_CHECK(hipMemcpyAsync(shared->input.coefficients, input.ptr()->coefficients,
                             sizeof(Input::coefficients), hipMemcpyDeviceToHost, stream));
    if (submission == Submission::Batch || submission == Submission::CaptureBatch) {
      HIP_CHECK(hipStreamBatchMemOp(stream, 2, ops, 0));
    } else if constexpr (sizeof(T) == 4) {
      HIP_CHECK(hipStreamWriteValue32(stream, ready, published, hipStreamWriteValueDefault));
      HIP_CHECK(hipStreamWaitValue32(stream, done, 1, hipStreamWaitValueGte,
                                     std::numeric_limits<T>::max()));
    } else {
      HIP_CHECK(hipStreamWriteValue64(stream, ready, published, hipStreamWriteValueDefault));
      HIP_CHECK(hipStreamWaitValue64(stream, done, 1, hipStreamWaitValueGte,
                                     std::numeric_limits<T>::max()));
    }
    HIP_CHECK(hipMemcpyAsync(output.ptr(), &shared->output, sizeof(Output), hipMemcpyHostToDevice,
                             stream));
  };

  if (submission == Submission::ExplicitGraph) {
    HIP_CHECK(hipGraphCreate(&graph.graph, 0));
    hipGraphNode_t previous = nullptr;
    auto copy = [&](void* dst, const void* src, size_t bytes, hipMemcpyKind kind) {
      hipGraphNode_t node;
      HIP_CHECK(hipGraphAddMemcpyNode1D(&node, graph.graph, previous ? &previous : nullptr,
                                        previous ? 1 : 0, dst, src, bytes, kind));
      previous = node;
    };
    copy(shared->input.payload, input.ptr()->payload, sizeof(Input::payload),
         hipMemcpyDeviceToHost);
    copy(shared->input.indices, input.ptr()->indices, sizeof(Input::indices),
         hipMemcpyDeviceToHost);
    copy(shared->input.coefficients, input.ptr()->coefficients, sizeof(Input::coefficients),
         hipMemcpyDeviceToHost);
    hipBatchMemOpNodeParams params{};
    HIP_CHECK(hipCtxGetCurrent(&params.ctx));
    params.count = 2;
    params.paramArray = ops;
    hipGraphNode_t node;
    HIP_CHECK(hipGraphAddBatchMemOpNode(&node, graph.graph, &previous, 1, &params));
    previous = node;
    copy(output.ptr(), &shared->output, sizeof(Output), hipMemcpyHostToDevice);
  } else if (submission == Submission::Capture || submission == Submission::CaptureBatch) {
    HIP_CHECK(hipStreamBeginCapture(stream, hipStreamCaptureModeGlobal));
    enqueue();
    HIP_CHECK(hipStreamEndCapture(stream, &graph.graph));
  }
  if (graph.graph) HIP_CHECK(hipGraphInstantiate(&graph.exec, graph.graph, nullptr, nullptr, 0));
  REQUIRE(Load(ready) == 0);  // Capture/instantiation must not publish a notification.

  Responder<T> responder(stream, ready, done, shared);
  unsigned badOutputs = 0;
  for (unsigned iteration = 1; iteration <= iterations && !responder.timedOut; ++iteration) {
    const Input expectedInput = MakeInput(iteration);
    responder.iteration = iteration;
    // Only the host re-arms done. A GPU reset WRITE before the publication can
    // accidentally provide the missing release and hide the bug under test.
    Store(done, T{0});
    HIP_CHECK(
        hipMemcpyAsync(input.ptr(), &expectedInput, sizeof(Input), hipMemcpyHostToDevice, stream));
    HIP_CHECK(hipStreamSynchronize(stream));
    if (graph.exec) {
      HIP_CHECK(hipGraphLaunch(graph.exec, stream));
    } else {
      enqueue();
    }
    HIP_CHECK(hipStreamSynchronize(stream));
    Output actual{};
    HIP_CHECK(hipMemcpy(&actual, output.ptr(), sizeof(Output), hipMemcpyDeviceToHost));
    const Output expected = Transform(expectedInput);
    badOutputs += std::memcmp(actual.data(), expected.data(), sizeof(Output)) != 0;
  }
  responder.Join();
  INFO("submission=" << static_cast<int>(submission) << ", bad inputs=" << responder.badInputs
                     << ", bad outputs=" << badOutputs << ", served=" << responder.served);
  REQUIRE_FALSE(responder.timedOut.load());  // Diagnostic releases never count as success.
  CHECK(responder.served == iterations);
  CHECK(responder.badInputs == 0);
  CHECK(badOutputs == 0);
}
}  // namespace

/**
 * Test Description
 * ------------------------
 * - A single WRITE must publish three earlier D2H copies before the host sees
 *   its notification. Check the actual input and the result copied back to the
 *   GPU against independent references, without a preceding reset WRITE.
 * - Cover 32/64-bit writes, eager single and batch operations, native capture,
 *   captured batches, and an explicit batch memop graph node across replays.
 * Test source
 * ------------------------
 * - unit/stream/hipStreamMemOpPublication.cc
 * Test requirements
 * ------------------------
 * - HIP_VERSION >= 6.4
 */
HIP_TEMPLATE_TEST_CASE(Unit_hipStreamMemOp_PublishPriorCopies, uint32_t, uint64_t) {
  int device = 0, supported = 0;
  HIP_CHECK(hipGetDevice(&device));
  HIP_CHECK(hipDeviceGetAttribute(&supported, hipDeviceAttributeCanUseStreamWaitValue, device));
  if (!supported) HIP_SKIP_TEST(HipTest::SkipReason::kStreamWaitValueUnsupported);
  const auto submission = GENERATE(Submission::Eager, Submission::Batch, Submission::Capture,
                                   Submission::CaptureBatch, Submission::ExplicitGraph);
  CheckPublication<TestType>(submission);
}
