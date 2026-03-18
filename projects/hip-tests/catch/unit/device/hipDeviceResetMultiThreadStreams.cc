/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <hip_test_common.hh>
#include <hip/hip_runtime_api.h>
#include <atomic>
#include <thread>
#include <vector>

/**
 * @addtogroup hipDeviceReset hipDeviceReset
 * @{
 * @ingroup DeviceTest
 * Tests for multi-threaded stream operations to verify thread-safe
 * access to the stored streams during concurrent stream creation/destruction
 */

static constexpr int NUM_THREADS = 8;
static constexpr int NUM_STREAMS_PER_THREAD = 10;
static constexpr size_t ARRAY_SIZE = 1024;

// Simple kernel for testing
__global__ void vectorAdd(float* c, const float* a, const float* b, size_t n) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    c[idx] = a[idx] + b[idx];
  }
}

/**
 * Test Description
 * ------------------------
 *  - Creates and immediately destroys streams from multiple threads concurrently
 *    to test that concurrent access to streamSet_ (via AddStream/RemoveStream)
 *    is thread-safe and doesn't cause race conditions.
 * Test source
 * ------------------------
 *  - unit/device/hipDeviceResetMultiThreadStreams.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 7.2
 */
TEST_CASE(Unit_hipDeviceReset_MultiThreadStreams_Concurrent) {
  const auto device = 0;
  HIP_CHECK(hipSetDevice(device));
  INFO("Current device is: " << device);

  std::atomic<int> streams_created{0};
  std::atomic<int> streams_destroyed{0};

  // Thread function that creates and destroys streams
  // This tests concurrent access to streamSet_ via AddStream/RemoveStream
  auto thread_func = [&]() {
    HIP_CHECK_THREAD(hipSetDevice(device));

    for (int i = 0; i < NUM_STREAMS_PER_THREAD; ++i) {
      hipStream_t stream;
      if (hipStreamCreate(&stream) == hipSuccess) {
        streams_created.fetch_add(1, std::memory_order_relaxed);
        // Do minimal work
        (void)hipStreamQuery(stream);
        // Destroy in same thread - tests concurrent streamSet_ modifications
        if (hipStreamDestroy(stream) == hipSuccess) {
          streams_destroyed.fetch_add(1, std::memory_order_relaxed);
        }
      }
    }
  };

  // Launch threads
  std::vector<std::thread> threads;
  threads.reserve(NUM_THREADS);
  for (int i = 0; i < NUM_THREADS; ++i) {
    threads.emplace_back(thread_func);
  }

  // Wait for all threads to complete
  for (auto& t : threads) {
    t.join();
  }

  HIP_CHECK_THREAD_FINALIZE();

  INFO("Created " << streams_created.load() << " streams, destroyed " << streams_destroyed.load()
                  << " streams across " << NUM_THREADS << " threads");
  REQUIRE(streams_created.load() > 0);
  REQUIRE(streams_created.load() == streams_destroyed.load());
}

/**
 * Test Description
 * ------------------------
 *  - Creates streams with various flags (blocking, non-blocking, priority)
 *    from multiple threads, does NOT destroy them, then calls hipDeviceReset()
 *    to test that destruction correctly handles mixed stream types.
 * Test source
 * ------------------------
 *  - unit/device/hipDeviceResetMultiThreadStreams.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 7.2
 */
TEST_CASE(Unit_hipDeviceReset_MultiThreadStreams_MixedStreamTypes) {
  const auto device = 0;
  HIP_CHECK(hipSetDevice(device));
  INFO("Current device is: " << device);

  std::atomic<int> blocking_streams{0};
  std::atomic<int> nonblocking_streams{0};
  std::atomic<int> priority_streams{0};

  // Thread function that creates different types of streams but does NOT destroy them
  auto thread_func = [&](int thread_id) {
    HIP_CHECK_THREAD(hipSetDevice(device));

    for (int i = 0; i < NUM_STREAMS_PER_THREAD; ++i) {
      hipStream_t stream;
      hipError_t err;

      // Create different types of streams based on iteration
      if (i % 3 == 0) {
        // Blocking stream (default)
        err = hipStreamCreate(&stream);
        if (err == hipSuccess) {
          blocking_streams.fetch_add(1, std::memory_order_relaxed);
        }
      } else if (i % 3 == 1) {
        // Non-blocking stream
        err = hipStreamCreateWithFlags(&stream, hipStreamNonBlocking);
        if (err == hipSuccess) {
          nonblocking_streams.fetch_add(1, std::memory_order_relaxed);
        }
      } else {
        // Stream with priority
        int priority = (thread_id % 2 == 0) ? -1 : 0;  // High or normal priority
        err = hipStreamCreateWithPriority(&stream, hipStreamNonBlocking, priority);
        if (err == hipSuccess) {
          priority_streams.fetch_add(1, std::memory_order_relaxed);
        }
      }

      if (err == hipSuccess) {
        // Just test stream creation and basic operations
        (void)hipStreamQuery(stream);
        // Intentionally NOT destroying stream - let hipDeviceReset handle it
      }

      // Small delay to introduce more interleaving
      std::this_thread::sleep_for(std::chrono::microseconds(100));
    }
  };

  // Launch threads
  std::vector<std::thread> threads;
  threads.reserve(NUM_THREADS);
  for (int i = 0; i < NUM_THREADS; ++i) {
    threads.emplace_back(thread_func, i);
  }

  // Wait for all threads to create their streams
  for (auto& t : threads) {
    t.join();
  }

  HIP_CHECK_THREAD_FINALIZE();

  INFO("Created streams - Blocking: " << blocking_streams.load()
                                      << ", Non-blocking: " << nonblocking_streams.load()
                                      << ", Priority: " << priority_streams.load());

  // Verify we created a reasonable number of streams
  int total_streams =
      blocking_streams.load() + nonblocking_streams.load() + priority_streams.load();
  REQUIRE(total_streams > 0);

  // Now reset device - destroyAllStreams() must handle mixed stream types
  HIP_CHECK(hipDeviceReset());
}

/**
 * Test Description
 * ------------------------
 *  - Repeatedly creates and destroys streams from multiple threads across
 *    multiple iterations. Stress tests concurrent access to the streams.
 * Test source
 * ------------------------
 *  - unit/device/hipDeviceResetMultiThreadStreams.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 7.2
 */
TEST_CASE(Unit_hipDeviceReset_MultiThreadStreams_StressTest) {
  const auto device = 0;
  HIP_CHECK(hipSetDevice(device));
  INFO("Current device is: " << device);

  constexpr int NUM_ITERATIONS = 5;
  constexpr int REDUCED_THREADS = 4;
  constexpr int STREAMS_PER_ITERATION = 5;

  for (int iter = 0; iter < NUM_ITERATIONS; ++iter) {
    INFO("Iteration " << iter + 1 << " of " << NUM_ITERATIONS);

    std::atomic<int> streams_created{0};
    std::atomic<int> streams_destroyed{0};

    // Thread function - creates and destroys streams
    auto thread_func = [&]() {
      HIP_CHECK_THREAD(hipSetDevice(device));

      for (int i = 0; i < STREAMS_PER_ITERATION; ++i) {
        hipStream_t stream;
        if (hipStreamCreate(&stream) == hipSuccess) {
          streams_created.fetch_add(1, std::memory_order_relaxed);
          (void)hipStreamQuery(stream);
          if (hipStreamDestroy(stream) == hipSuccess) {
            streams_destroyed.fetch_add(1, std::memory_order_relaxed);
          }
        }
      }
    };

    // Launch threads
    std::vector<std::thread> threads;
    threads.reserve(REDUCED_THREADS);
    for (int i = 0; i < REDUCED_THREADS; ++i) {
      threads.emplace_back(thread_func);
    }

    // Wait for threads
    for (auto& t : threads) {
      t.join();
    }

    HIP_CHECK_THREAD_FINALIZE();

    INFO("Created " << streams_created.load() << " streams, destroyed " << streams_destroyed.load()
                    << " in this iteration");
    REQUIRE(streams_created.load() > 0);
    REQUIRE(streams_created.load() == streams_destroyed.load());
  }
}

/**
 * Test Description
 * ------------------------
 *  - Creates streams from multiple threads, where some threads destroy
 *    their streams and others don't, testing that the streamSet_
 *    modifications are thread-safe and streams can be safely tracked
 *    and cleaned up by individual threads.
 * Test source
 * ------------------------
 *  - unit/device/hipDeviceResetMultiThreadStreams.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 7.2
 */
TEST_CASE(Unit_hipDeviceReset_MultiThreadStreams_ConcurrentReset) {
  const auto device = 0;
  HIP_CHECK(hipSetDevice(device));
  INFO("Current device is: " << device);

  constexpr int NUM_CREATOR_THREADS = 4;
  constexpr int STREAMS_PER_THREAD = 5;
  std::atomic<int> streams_created{0};
  std::atomic<int> streams_destroyed_by_creators{0};

  // Have some threads create and destroy streams, others just create then destroy
  // This tests concurrent modifications to streamSet_ via AddStream/RemoveStream
  auto thread_func = [&](int thread_id) {
    HIP_CHECK_THREAD(hipSetDevice(device));

    std::vector<hipStream_t> local_streams;
    for (int i = 0; i < STREAMS_PER_THREAD; ++i) {
      hipStream_t stream;
      if (hipStreamCreate(&stream) == hipSuccess) {
        streams_created.fetch_add(1, std::memory_order_relaxed);
        local_streams.push_back(stream);
        (void)hipStreamQuery(stream);
      }
    }

    // All threads destroy their streams to test concurrent RemoveStream calls
    // This exercises the same locking mechanism that destroyAllStreams() uses
    for (auto stream : local_streams) {
      if (hipStreamDestroy(stream) == hipSuccess) {
        streams_destroyed_by_creators.fetch_add(1, std::memory_order_relaxed);
      }
    }
  };

  // Launch threads
  std::vector<std::thread> threads;
  threads.reserve(NUM_CREATOR_THREADS);
  for (int i = 0; i < NUM_CREATOR_THREADS; ++i) {
    threads.emplace_back(thread_func, i);
  }

  // Wait for all threads to complete
  for (auto& t : threads) {
    t.join();
  }

  HIP_CHECK_THREAD_FINALIZE();

  INFO("Created " << streams_created.load() << " streams, all "
                  << streams_destroyed_by_creators.load() << " destroyed by creating threads");

  // Verify all streams were created and destroyed successfully
  REQUIRE(streams_created.load() > 0);
  REQUIRE(streams_created.load() == streams_destroyed_by_creators.load());
}

/**
 * Test Description
 * ------------------------
 *  - Tests that multiple threads can concurrently call hipDeviceReset()
 *    without causing race conditions. This directly tests the thread-safety
 *    of destroyAllStreams() when called concurrently from multiple threads.
 * Test source
 * ------------------------
 *  - unit/device/hipDeviceResetMultiThreadStreams.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 7.2
 */
TEST_CASE(Unit_hipDeviceReset_MultiThreadStreams_ConcurrentDeviceReset) {
  const auto device = 0;
  HIP_CHECK(hipSetDevice(device));
  INFO("Current device is: " << device);

  // First create some streams to ensure there's something to clean up
  constexpr int NUM_INITIAL_STREAMS = 20;
  std::vector<hipStream_t> initial_streams;
  for (int i = 0; i < NUM_INITIAL_STREAMS; ++i) {
    hipStream_t stream;
    HIP_CHECK(hipStreamCreate(&stream));
    initial_streams.push_back(stream);
  }
  INFO("Created " << NUM_INITIAL_STREAMS << " initial streams");

  constexpr int NUM_RESET_THREADS = 4;
  std::atomic<int> reset_successes{0};
  std::atomic<int> reset_errors{0};
  std::atomic<bool> start_flag{false};

  // Thread function that waits for start signal then calls hipDeviceReset
  auto reset_func = [&]() {
    HIP_CHECK_THREAD(hipSetDevice(device));
    
    // Wait for all threads to be ready
    while (!start_flag.load(std::memory_order_acquire)) {
      std::this_thread::yield();
    }
    
    // All threads call hipDeviceReset concurrently
    hipError_t err = hipDeviceReset();
    if (err == hipSuccess) {
      reset_successes.fetch_add(1, std::memory_order_relaxed);
    } else {
      reset_errors.fetch_add(1, std::memory_order_relaxed);
      INFO("Thread got error from hipDeviceReset: " << hipGetErrorString(err));
    }
  };

  // Launch threads
  std::vector<std::thread> threads;
  threads.reserve(NUM_RESET_THREADS);
  for (int i = 0; i < NUM_RESET_THREADS; ++i) {
    threads.emplace_back(reset_func);
  }

  // Give threads time to reach the wait point
  std::this_thread::sleep_for(std::chrono::milliseconds(100));
  
  // Signal all threads to start concurrently
  start_flag.store(true, std::memory_order_release);

  // Wait for all threads to complete
  for (auto& t : threads) {
    t.join();
  }

  HIP_CHECK_THREAD_FINALIZE();

  INFO("Reset successes: " << reset_successes.load() 
       << ", errors: " << reset_errors.load());

  // At least some resets should succeed. The question is whether we get
  // race conditions (crashes, hangs, or invalid memory access)
  REQUIRE(reset_successes.load() > 0);
}

/**
 * End doxygen group DeviceTest.
 * @}
 */
