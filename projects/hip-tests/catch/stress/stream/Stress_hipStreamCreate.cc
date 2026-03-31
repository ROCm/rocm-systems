/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <hip_test_common.hh>
#include <cstdio>
#include <cassert>

#define NUM_ITER 100000
#define TOTALSEQ 18

namespace hipStreamCreateStressTest {
__global__ void kernel_do_nothing() {
  // do nothing
}

int stream_seq[TOTALSEQ][4] = {
    {0, 1, 2, 0},  // Launch0->Launch1->Launch2->Sync0
    {0, 2, 1, 0},  // Launch0->Launch2->Launch1->Sync0
    {1, 0, 2, 0},  // Launch1->Launch0->Launch2->Sync0
    {1, 2, 0, 0},  // Launch1->Launch2->Launch0->Sync0
    {2, 0, 1, 0},  // Launch2->Launch0->Launch1->Sync0
    {2, 1, 0, 0},  // Launch2->Launch1->Launch0->Sync0
    {0, 1, 2, 1},  // Launch0->Launch1->Launch2->Sync1
    {0, 2, 1, 1},  // Launch0->Launch2->Launch1->Sync1
    {1, 0, 2, 1},  // Launch1->Launch0->Launch2->Sync1
    {1, 2, 0, 1},  // Launch1->Launch2->Launch0->Sync1
    {2, 0, 1, 1},  // Launch2->Launch0->Launch1->Sync1
    {2, 1, 0, 1},  // Launch2->Launch1->Launch0->Sync1
    {0, 1, 2, 2},  // Launch0->Launch1->Launch2->Sync2
    {0, 2, 1, 2},  // Launch0->Launch2->Launch1->Sync2
    {1, 0, 2, 2},  // Launch1->Launch0->Launch2->Sync2
    {1, 2, 0, 2},  // Launch1->Launch2->Launch0->Sync2
    {2, 0, 1, 2},  // Launch2->Launch0->Launch1->Sync2
    {2, 1, 0, 2}   // Launch2->Launch1->Launch0->Sync2
};

/**
 * Scenario: This test extends the DTEST introduced for SWDEV-238360 to test
 * all the possible scenarios mentioned under comments section
 * in SWDEV-237846.
 */

void testhipStreamCreate(int* stream_sequence) {
  printf("%s: Testing sequence %d->%d->%d->sync(%d) \n", __func__, stream_sequence[0],
         stream_sequence[1], stream_sequence[2], stream_sequence[3]);
  // Streams
  hipStream_t stream[3];
  stream[0] = 0;
  HIP_CHECK(hipStreamCreate(&stream[1]));
  HIP_CHECK(hipStreamCreate(&stream[2]));
  // Run test loop
  for (int k = 0; k < NUM_ITER; ++k) {
    // Sync
    HIP_CHECK(hipDeviceSynchronize());
    HIP_CHECK(hipLaunchKernel((const void*)kernel_do_nothing, dim3(1, 1, 1), dim3(1, 1, 1), NULL, 0,
                              stream[stream_sequence[0]]));
    HIP_CHECK(hipLaunchKernel((const void*)kernel_do_nothing, dim3(1, 1, 1), dim3(1, 1, 1), NULL, 0,
                              stream[stream_sequence[1]]));
    HIP_CHECK(hipLaunchKernel((const void*)kernel_do_nothing, dim3(1, 1, 1), dim3(1, 1, 1), NULL, 0,
                              stream[stream_sequence[2]]));
    // Sync stream 1
    HIP_CHECK(hipStreamSynchronize(stream[stream_sequence[3]]));
  }
  HIP_CHECK(hipDeviceSynchronize());
  // Clean up
  HIP_CHECK(hipStreamDestroy(stream[1]));
  HIP_CHECK(hipStreamDestroy(stream[2]));
}
/**
 * Scenario: This test extends the above test by using 2 streams
 * (of highest and lowest priority) created using hipStreamCreateWithPriority
 * along with the default stream.
 */
void testhipStreamCreatePriority(int* stream_sequence, unsigned int flag) {
  printf("%s: Testing sequence %d->%d->%d->sync(%d) \n", __func__, stream_sequence[0],
         stream_sequence[1], stream_sequence[2], stream_sequence[3]);
  // Streams
  hipStream_t stream[3];
  stream[0] = 0;
  int priority_low = 0;
  int priority_high = 0;
  HIP_CHECK(hipDeviceGetStreamPriorityRange(&priority_low, &priority_high));
  if (priority_low == priority_high) {
    printf("Exiting test since priorities are not supported \n");
    return;
  }
  HIP_CHECK(hipStreamCreateWithPriority(&stream[1], flag, priority_high));
  HIP_CHECK(hipStreamCreateWithPriority(&stream[2], flag, priority_low));
  // Run test loop
  for (int k = 0; k < NUM_ITER; ++k) {
    // Sync
    HIP_CHECK(hipDeviceSynchronize());
    HIP_CHECK(hipLaunchKernel((const void*)kernel_do_nothing, dim3(1, 1, 1), dim3(1, 1, 1), NULL, 0,
                              stream[stream_sequence[0]]));
    HIP_CHECK(hipLaunchKernel((const void*)kernel_do_nothing, dim3(1, 1, 1), dim3(1, 1, 1), NULL, 0,
                              stream[stream_sequence[1]]));
    HIP_CHECK(hipLaunchKernel((const void*)kernel_do_nothing, dim3(1, 1, 1), dim3(1, 1, 1), NULL, 0,
                              stream[stream_sequence[2]]));
    // Sync stream 1
    HIP_CHECK(hipStreamSynchronize(stream[stream_sequence[3]]));
  }
  HIP_CHECK(hipDeviceSynchronize());
  // Clean up
  HIP_CHECK(hipStreamDestroy(stream[1]));
  HIP_CHECK(hipStreamDestroy(stream[2]));
}
/**
 * Scenario: This test extends the above test by using 2 streams
 * created using hipStreamCreateWithFlags along with the default stream.
 */
void testhipStreamCreateFlags(int* stream_sequence, unsigned int flag) {
  printf("%s: Testing sequence %d->%d->%d->sync(%d) \n", __func__, stream_sequence[0],
         stream_sequence[1], stream_sequence[2], stream_sequence[3]);
  // Streams
  hipStream_t stream[3];
  stream[0] = 0;
  HIP_CHECK(hipStreamCreateWithFlags(&stream[1], flag));
  HIP_CHECK(hipStreamCreateWithFlags(&stream[2], flag));
  // Run test loop
  for (int k = 0; k < NUM_ITER; ++k) {
    // Sync
    HIP_CHECK(hipDeviceSynchronize());
    HIP_CHECK(hipLaunchKernel((const void*)kernel_do_nothing, dim3(1, 1, 1), dim3(1, 1, 1), NULL, 0,
                              stream[stream_sequence[0]]));
    HIP_CHECK(hipLaunchKernel((const void*)kernel_do_nothing, dim3(1, 1, 1), dim3(1, 1, 1), NULL, 0,
                              stream[stream_sequence[1]]));
    HIP_CHECK(hipLaunchKernel((const void*)kernel_do_nothing, dim3(1, 1, 1), dim3(1, 1, 1), NULL, 0,
                              stream[stream_sequence[2]]));
    // Sync stream 1
    HIP_CHECK(hipStreamSynchronize(stream[stream_sequence[3]]));
  }
  HIP_CHECK(hipDeviceSynchronize());
  // Clean up
  HIP_CHECK(hipStreamDestroy(stream[1]));
  HIP_CHECK(hipStreamDestroy(stream[2]));
}
}  // namespace hipStreamCreateStressTest

HIP_TEST_CASE(Stress_hipStreamCreate_SyncTest) {
  printf("hipStreamCreate stress test:\n");
  for (int i = 0; i < TOTALSEQ; i++) {
    hipStreamCreateStressTest::testhipStreamCreate(hipStreamCreateStressTest::stream_seq[i]);
  }
}

HIP_TEST_CASE(Stress_hipStreamCreatePriority_SyncTest) {
  printf("hipStreamCreateWithPriority(hipStreamDefault) stress test:\n");
  for (int i = 0; i < TOTALSEQ; i++) {
    hipStreamCreateStressTest::testhipStreamCreatePriority(hipStreamCreateStressTest::stream_seq[i],
                                                           hipStreamDefault);
  }
  printf("hipStreamCreateWithPriority(hipStreamNonBlocking) stress test:\n");
  for (int i = 0; i < TOTALSEQ; i++) {
    hipStreamCreateStressTest::testhipStreamCreatePriority(hipStreamCreateStressTest::stream_seq[i],
                                                           hipStreamNonBlocking);
  }
}

HIP_TEST_CASE(Stress_hipStreamCreateWithFlags_SyncTest) {
  printf("hipStreamCreateWithFlags(hipStreamDefault) stress test:\n");
  for (int i = 0; i < TOTALSEQ; i++) {
    hipStreamCreateStressTest::testhipStreamCreateFlags(hipStreamCreateStressTest::stream_seq[i],
                                                        hipStreamDefault);
  }
  printf("hipStreamCreateWithFlags(hipStreamNonBlocking) stress test:\n");
  for (int i = 0; i < TOTALSEQ; i++) {
    hipStreamCreateStressTest::testhipStreamCreateFlags(hipStreamCreateStressTest::stream_seq[i],
                                                        hipStreamNonBlocking);
  }
}
