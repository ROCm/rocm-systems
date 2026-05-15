/******************************************************************************
 * Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 *****************************************************************************/

#include "atomic_wf_queue_gtest.hpp"

using namespace rocshmem;

/*****************************************************************************
 ******************************* Fixture Tests *******************************
 *****************************************************************************/

TEST_F(AtomicWFQueueTestFixture, active_logical_lane_ids_4) {
  get_thread_lane_ids(4);
}

TEST_F(AtomicWFQueueTestFixture, active_logical_lane_ids_10) {
  get_thread_lane_ids(10);
}

TEST_F(AtomicWFQueueTestFixture, active_logical_lane_ids_16) {
  get_thread_lane_ids(16);
}

TEST_F(AtomicWFQueueTestFixture, active_logical_lane_ids_17) {
  get_thread_lane_ids(17);
}

TEST_F(AtomicWFQueueTestFixture, active_logical_lane_ids_64) {
  get_thread_lane_ids(64);
}

TEST_F(AtomicWFQueueTestFixture, active_logical_lane_ids_97) {
  get_thread_lane_ids(97);
}

TEST_F(AtomicWFQueueTestFixture, active_logical_lane_ids_256) {
  get_thread_lane_ids(256);
}

TEST_F(AtomicWFQueueTestFixture, active_logical_lane_ids_183) {
  get_thread_lane_ids(183);
}

TEST_F(AtomicWFQueueTestFixture, active_logical_lane_ids_1024) {
  get_thread_lane_ids(1024);
}

TEST_F(AtomicWFQueueTestFixture, init_2_64) {
  init_queue(2, 64);
}

TEST_F(AtomicWFQueueTestFixture, init_64_256) {
  init_queue(64, 256);
}

TEST_F(AtomicWFQueueTestFixture, init_1024_96) {
  init_queue(1024, 96);
}

TEST_F(AtomicWFQueueTestFixture, dequeue_enqueue_threads_eq_qsize_2_32) {
  int num_blocks {2};
  int block_size {32};
  int wf_size {this->wf_size};
  int queue_size {num_blocks * ((block_size - 1) / wf_size + 1)};
  dequeue_enqueue(num_blocks, block_size, queue_size);
}

TEST_F(AtomicWFQueueTestFixture, dequeue_enqueue_threads_eq_qsize_32_192) {
  int num_blocks {32};
  int block_size {192};
  int wf_size {this->wf_size};
  int queue_size {num_blocks * ((block_size - 1) / wf_size + 1)};
  dequeue_enqueue(num_blocks, block_size, queue_size);
}

TEST_F(AtomicWFQueueTestFixture, dequeue_enqueue_threads_eq_qsize_64_256) {
  int num_blocks {64};
  int block_size {256};
  int wf_size {this->wf_size};
  int queue_size {num_blocks * ((block_size - 1) / wf_size + 1)};
  dequeue_enqueue(num_blocks, block_size, queue_size);
}

TEST_F(AtomicWFQueueTestFixture, dequeue_enqueue_threads_ge_qsize_16_32_4) {
  int num_blocks {16};
  int block_size {32};
  int queue_size {4};
  dequeue_enqueue(num_blocks, block_size, queue_size);
}

TEST_F(AtomicWFQueueTestFixture, dequeue_enqueue_threads_ge_qsize_16_96_8) {
  int num_blocks {16};
  int block_size {96};
  int queue_size {8};
  dequeue_enqueue(num_blocks, block_size, queue_size);
}

TEST_F(AtomicWFQueueTestFixture, dequeue_enqueue_threads_ge_qsize_16_320_8) {
  int num_blocks {16};
  int block_size {320};
  int queue_size {8};
  dequeue_enqueue(num_blocks, block_size, queue_size);
}

TEST_F(AtomicWFQueueTestFixture, dequeue_enqueue_threads_ge_qsize_32_192_16) {
  int num_blocks {32};
  int block_size {192};
  int queue_size {16};
  dequeue_enqueue(num_blocks, block_size, queue_size);
}

TEST_F(AtomicWFQueueTestFixture, dequeue_enqueue_threads_ge_qsize_64_256_64) {
  int num_blocks {64};
  int block_size {256};
  int queue_size {64};
  dequeue_enqueue(num_blocks, block_size, queue_size);
}

TEST_F(AtomicWFQueueTestFixture, dequeue_enqueue_threads_ge_qsize_64_576_64) {
  int num_blocks {64};
  int block_size {576};
  int queue_size {64};
  dequeue_enqueue(num_blocks, block_size, queue_size);
}

TEST_F(AtomicWFQueueTestFixture, dequeue_enqueue_qsize_ge_threads_4_8_16) {
  int num_blocks {4};
  int block_size {8};
  int queue_size {16};
  dequeue_enqueue(num_blocks, block_size, queue_size);
}

TEST_F(AtomicWFQueueTestFixture, dequeue_enqueue_qsize_ge_threads_4_96_16) {
  int num_blocks {4};
  int block_size {96};
  int queue_size {16};
  dequeue_enqueue(num_blocks, block_size, queue_size);
}

TEST_F(AtomicWFQueueTestFixture, dequeue_enqueue_qsize_ge_threads_32_192_128) {
  int num_blocks {32};
  int block_size {192};
  int queue_size {128};
  dequeue_enqueue(num_blocks, block_size, queue_size);
}

TEST_F(AtomicWFQueueTestFixture, dequeue_enqueue_qsize_ge_threads_32_320_256) {
  int num_blocks {32};
  int block_size {320};
  int queue_size {256};
  dequeue_enqueue(num_blocks, block_size, queue_size);
}

TEST_F(AtomicWFQueueTestFixture, dequeue_enqueue_qsize_ge_threads_64_16_128) {
  int num_blocks {64};
  int block_size {16};
  int queue_size {128};
  dequeue_enqueue(num_blocks, block_size, queue_size);
}

TEST_F(AtomicWFQueueTestFixture, dequeue_enqueue_qsize_ge_threads_64_96_256) {
  int num_blocks {64};
  int block_size {96};
  int queue_size {256};
  dequeue_enqueue(num_blocks, block_size, queue_size);
}

TEST_F(AtomicWFQueueTestFixture, dequeue_enqueue_qsize_ge_threads_64_320_512) {
  int num_blocks {64};
  int block_size {320};
  int queue_size {512};
  dequeue_enqueue(num_blocks, block_size, queue_size);
}

TEST_F(AtomicWFQueueTestFixture, dequeue_enqueue_qsize_ge_threads_64_576_1024) {
  int num_blocks {64};
  int block_size {576};
  int queue_size {1024};
  dequeue_enqueue(num_blocks, block_size, queue_size);
}