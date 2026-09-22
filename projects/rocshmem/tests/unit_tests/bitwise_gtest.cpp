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

#include "bitwise_gtest.hpp"

using namespace rocshmem;

/*****************************************************************************
 *************************** Kernel Definitions ******************************
 *****************************************************************************/
__global__ void lowest_active_lane_kernel(BitwiseDeviceMethods *device_methods,
                                          WarpMatrix *warp_matrix,
                                          uint64_t activate_lanes_bitfield) {
  device_methods->lowest_active_lane(warp_matrix, activate_lanes_bitfield);
}

__global__ void is_lowest_active_lane_kernel(
    BitwiseDeviceMethods *device_methods, WarpMatrix *warp_matrix,
    uint64_t activate_lanes_bitfield) {
  device_methods->is_lowest_active_lane(warp_matrix, activate_lanes_bitfield);
}

__global__ void active_logical_lane_id_2_kernel(
    BitwiseDeviceMethods *device_methods, WarpMatrix *warp_matrix,
    uint64_t activate_lanes_bitfield) {
  device_methods->active_logical_lane_id_2(warp_matrix,
                                           activate_lanes_bitfield);
}

__global__ void lane_id_kernel(BitwiseDeviceMethods *device_methods,
                               WarpMatrix *warp_matrix,
                               uint64_t activate_lanes_bitfield) {
  device_methods->lane_id(warp_matrix, activate_lanes_bitfield);
}

__global__ void number_active_lanes_kernel(BitwiseDeviceMethods *device_methods,
                                           WarpMatrix *warp_matrix,
                                           uint64_t activate_lanes_bitfield) {
  device_methods->number_active_lanes(warp_matrix, activate_lanes_bitfield);
}

__global__ void broadcast_up_value_42_kernel(
    BitwiseDeviceMethods *device_methods, WarpMatrix *warp_matrix,
    uint64_t activate_lanes_bitfield) {
  device_methods->broadcast_up_value_42(warp_matrix, activate_lanes_bitfield);
}

__global__ void fetch_incr_lowest_active_lane_kernel(
    BitwiseDeviceMethods *device_methods, WarpMatrix *warp_matrix,
    uint64_t activate_lanes_bitfield) {
  device_methods->fetch_incr_lowest_active_lane(warp_matrix,
                                                activate_lanes_bitfield);
}

__global__ void fetch_incr_active_logical_lane_1_kernel(
    BitwiseDeviceMethods *device_methods, WarpMatrix *warp_matrix,
    uint64_t activate_lanes_bitfield) {
  device_methods->fetch_incr_active_logical_lane_1(warp_matrix,
                                                   activate_lanes_bitfield);
}

__global__ void activate_lane_helper_kernel(
    BitwiseDeviceMethods *device_methods, WarpMatrix *warp_matrix,
    uint64_t activate_lanes_bitfield) {
  bool is_an_active_lane =
      device_methods->activate_lane_helper(activate_lanes_bitfield);

  /*
   * Index into warp matrix to save return value to be read by host.
   */
  if (is_an_active_lane) {
    size_t warp_index = hipThreadIdx_x / WF_SIZE;
    size_t block_index = hipBlockIdx_x;
    auto *elem = warp_matrix->access(warp_index, block_index);
    *elem = hipThreadIdx_x;
  }
}

/*****************************************************************************
 ****************************** Fixture Tests ********************************
 *****************************************************************************/

/*****************************************************************************
 ******************************* Setup Tests *********************************
 *****************************************************************************/
TEST_F(BitwiseTestFixture, setup_fixture_only_1_1) {
  setup_fixture({1, 1, 1}, {1, 1, 1});
}

TEST_F(BitwiseTestFixture, setup_fixture_check_matrix_size_1_1) {
  setup_fixture({1, 1, 1}, {1, 1, 1});

  ASSERT_EQ(_warp_matrix->rows(), 1);
  ASSERT_EQ(_warp_matrix->columns(), 1);
}

TEST_F(BitwiseTestFixture, setup_fixture_check_matrix_size_64_1) {
  setup_fixture({64, 1, 1}, {1, 1, 1});

  ASSERT_EQ(_warp_matrix->rows(), 1);
  ASSERT_EQ(_warp_matrix->columns(), 1);
}

TEST_F(BitwiseTestFixture, setup_fixture_check_matrix_size_128_1) {
  setup_fixture({128, 1, 1}, {1, 1, 1});

  ASSERT_EQ(_warp_matrix->rows(), 2);
  ASSERT_EQ(_warp_matrix->columns(), 1);
}

TEST_F(BitwiseTestFixture, setup_fixture_check_matrix_size_128_2) {
  setup_fixture({128, 1, 1}, {2, 1, 1});

  ASSERT_EQ(_warp_matrix->rows(), 2);
  ASSERT_EQ(_warp_matrix->columns(), 2);
}

TEST_F(BitwiseTestFixture, verify_host_warp_matrix_init_1024_8) {
  setup_fixture({1024, 1, 1}, {8, 1, 1});

  zero_warp_matrix();

  verify_zeroed_warp_matrix();
}

/*****************************************************************************
 ************************** Activate Lane Helper******************************
 *****************************************************************************/
TEST_F(BitwiseTestFixture, activate_lane_helper_64_1_lane_0) {
  setup_fixture({64, 1, 1}, {1, 1, 1});

  zero_warp_matrix();

  /*
   * index (tens):    6554443322211000
   *       (ones):    0628406284062840
   */
  uint64_t activate_lanes_bitfield = 0x0000000000000001;

  host_run_device_kernel(activate_lane_helper_kernel, activate_lanes_bitfield);

  auto *elem = _warp_matrix->access(0, 0);
  ASSERT_EQ(*elem, 0);
}

TEST_F(BitwiseTestFixture, activate_lane_helper_64_1_lane_1) {
  setup_fixture({64, 1, 1}, {1, 1, 1});

  zero_warp_matrix();

  /*
   * index (tens):    6554443322211000
   *       (ones):    0628406284062840
   */
  uint64_t activate_lanes_bitfield = 0x0000000000000002;

  host_run_device_kernel(activate_lane_helper_kernel, activate_lanes_bitfield);

  auto *elem = _warp_matrix->access(0, 0);
  ASSERT_EQ(*elem, 1);
}

TEST_F(BitwiseTestFixture, activate_lane_helper_64_1_lane_2) {
  setup_fixture({64, 1, 1}, {1, 1, 1});

  zero_warp_matrix();

  /*
   * index (tens):    6554443322211000
   *       (ones):    0628406284062840
   */
  uint64_t activate_lanes_bitfield = 0x0000000000000004;

  host_run_device_kernel(activate_lane_helper_kernel, activate_lanes_bitfield);

  auto *elem = _warp_matrix->access(0, 0);
  ASSERT_EQ(*elem, 2);
}

TEST_F(BitwiseTestFixture, activate_lane_helper_64_1_lane_3) {
  setup_fixture({64, 1, 1}, {1, 1, 1});

  zero_warp_matrix();

  /*
   * index (tens):    6554443322211000
   *       (ones):    0628406284062840
   */
  uint64_t activate_lanes_bitfield = 0x0000000000000008;

  host_run_device_kernel(activate_lane_helper_kernel, activate_lanes_bitfield);

  auto *elem = _warp_matrix->access(0, 0);
  ASSERT_EQ(*elem, 3);
}

TEST_F(BitwiseTestFixture, activate_lane_helper_64_1_lane_4) {
  setup_fixture({64, 1, 1}, {1, 1, 1});

  zero_warp_matrix();

  /*
   * index (tens):    6554443322211000
   *       (ones):    0628406284062840
   */
  uint64_t activate_lanes_bitfield = 0x0000000000000010;

  host_run_device_kernel(activate_lane_helper_kernel, activate_lanes_bitfield);

  auto *elem = _warp_matrix->access(0, 0);
  ASSERT_EQ(*elem, 4);
}

TEST_F(BitwiseTestFixture, activate_lane_helper_64_1_lane_8) {
  setup_fixture({64, 1, 1}, {1, 1, 1});

  zero_warp_matrix();

  /*
   * index (tens):    6554443322211000
   *       (ones):    0628406284062840
   */
  uint64_t activate_lanes_bitfield = 0x0000000000000100;

  host_run_device_kernel(activate_lane_helper_kernel, activate_lanes_bitfield);

  auto *elem = _warp_matrix->access(0, 0);
  ASSERT_EQ(*elem, 8);
}

TEST_F(BitwiseTestFixture, activate_lane_helper_64_1_lane_12) {
  setup_fixture({64, 1, 1}, {1, 1, 1});

  zero_warp_matrix();

  /*
   * index (tens):    6554443322211000
   *       (ones):    0628406284062840
   */
  uint64_t activate_lanes_bitfield = 0x0000000000001000;

  host_run_device_kernel(activate_lane_helper_kernel, activate_lanes_bitfield);

  auto *elem = _warp_matrix->access(0, 0);
  ASSERT_EQ(*elem, 12);
}

TEST_F(BitwiseTestFixture, activate_lane_helper_64_1_lane_28) {
  setup_fixture({64, 1, 1}, {1, 1, 1});

  zero_warp_matrix();

  /*
   * index (tens):    6554443322211000
   *       (ones):    0628406284062840
   */
  uint64_t activate_lanes_bitfield = 0x0000000010000000;

  host_run_device_kernel(activate_lane_helper_kernel, activate_lanes_bitfield);

  auto *elem = _warp_matrix->access(0, 0);
  ASSERT_EQ(*elem, 28);
}

TEST_F(BitwiseTestFixture, activate_lane_helper_64_1_lane_29) {
  setup_fixture({64, 1, 1}, {1, 1, 1});

  zero_warp_matrix();

  /*
   * index (tens):    6554443322211000
   *       (ones):    0628406284062840
   */
  uint64_t activate_lanes_bitfield = 0x0000000020000000;

  host_run_device_kernel(activate_lane_helper_kernel, activate_lanes_bitfield);

  auto *elem = _warp_matrix->access(0, 0);
  ASSERT_EQ(*elem, 29);
}

TEST_F(BitwiseTestFixture, activate_lane_helper_64_1_lane_30) {
  setup_fixture({64, 1, 1}, {1, 1, 1});

  zero_warp_matrix();

  /*
   * index (tens):    6554443322211000
   *       (ones):    0628406284062840
   */
  uint64_t activate_lanes_bitfield = 0x0000000040000000;

  host_run_device_kernel(activate_lane_helper_kernel, activate_lanes_bitfield);

  auto *elem = _warp_matrix->access(0, 0);
  ASSERT_EQ(*elem, 30);
}

TEST_F(BitwiseTestFixture, activate_lane_helper_64_1_lane_31) {
  setup_fixture({64, 1, 1}, {1, 1, 1});

  zero_warp_matrix();

  /*
   * index (tens):    6554443322211000
   *       (ones):    0628406284062840
   */
  uint64_t activate_lanes_bitfield = 0x0000000080000000;

  host_run_device_kernel(activate_lane_helper_kernel, activate_lanes_bitfield);

  auto *elem = _warp_matrix->access(0, 0);
  ASSERT_EQ(*elem, 31);
}

TEST_F(BitwiseTestFixture, activate_lane_helper_64_1_lane_32) {
  setup_fixture({64, 1, 1}, {1, 1, 1});

  zero_warp_matrix();

  /*
   * index (tens):    6554443322211000
   *       (ones):    0628406284062840
   */
  uint64_t activate_lanes_bitfield = 0x0000000100000000;

  host_run_device_kernel(activate_lane_helper_kernel, activate_lanes_bitfield);

  auto *elem = _warp_matrix->access(0, 0);
  ASSERT_EQ(*elem, 32);
}

TEST_F(BitwiseTestFixture, activate_lane_helper_64_1_lane_33) {
  setup_fixture({64, 1, 1}, {1, 1, 1});

  zero_warp_matrix();

  /*
   * index (tens):    6554443322211000
   *       (ones):    0628406284062840
   */
  uint64_t activate_lanes_bitfield = 0x0000000200000000;

  host_run_device_kernel(activate_lane_helper_kernel, activate_lanes_bitfield);

  auto *elem = _warp_matrix->access(0, 0);
  ASSERT_EQ(*elem, 33);
}

TEST_F(BitwiseTestFixture, activate_lane_helper_64_1_lane_34) {
  setup_fixture({64, 1, 1}, {1, 1, 1});

  zero_warp_matrix();

  /*
   * index (tens):    6554443322211000
   *       (ones):    0628406284062840
   */
  uint64_t activate_lanes_bitfield = 0x0000000400000000;

  host_run_device_kernel(activate_lane_helper_kernel, activate_lanes_bitfield);

  auto *elem = _warp_matrix->access(0, 0);
  ASSERT_EQ(*elem, 34);
}

TEST_F(BitwiseTestFixture, activate_lane_helper_64_1_lane_35) {
  setup_fixture({64, 1, 1}, {1, 1, 1});

  zero_warp_matrix();

  /*
   * index (tens):    6554443322211000
   *       (ones):    0628406284062840
   */
  uint64_t activate_lanes_bitfield = 0x0000000800000000;

  host_run_device_kernel(activate_lane_helper_kernel, activate_lanes_bitfield);

  auto *elem = _warp_matrix->access(0, 0);
  ASSERT_EQ(*elem, 35);
}

TEST_F(BitwiseTestFixture, activate_lane_helper_64_1_lane_44) {
  setup_fixture({64, 1, 1}, {1, 1, 1});

  zero_warp_matrix();

  /*
   * index (tens):    6554443322211000
   *       (ones):    0628406284062840
   */
  uint64_t activate_lanes_bitfield = 0x0000100000000000;

  host_run_device_kernel(activate_lane_helper_kernel, activate_lanes_bitfield);

  auto *elem = _warp_matrix->access(0, 0);
  ASSERT_EQ(*elem, 44);
}

TEST_F(BitwiseTestFixture, activate_lane_helper_64_1_lane_48) {
  setup_fixture({64, 1, 1}, {1, 1, 1});

  zero_warp_matrix();

  /*
   * index (tens):    6554443322211000
   *       (ones):    0628406284062840
   */
  uint64_t activate_lanes_bitfield = 0x0001000000000000;

  host_run_device_kernel(activate_lane_helper_kernel, activate_lanes_bitfield);

  auto *elem = _warp_matrix->access(0, 0);
  ASSERT_EQ(*elem, 48);
}

TEST_F(BitwiseTestFixture, activate_lane_helper_64_1_lane_52) {
  setup_fixture({64, 1, 1}, {1, 1, 1});

  zero_warp_matrix();

  /*
   * index (tens):    6554443322211000
   *       (ones):    0628406284062840
   */
  uint64_t activate_lanes_bitfield = 0x0010000000000000;

  host_run_device_kernel(activate_lane_helper_kernel, activate_lanes_bitfield);

  auto *elem = _warp_matrix->access(0, 0);
  ASSERT_EQ(*elem, 52);
}

TEST_F(BitwiseTestFixture, activate_lane_helper_64_1_lane_56) {
  setup_fixture({64, 1, 1}, {1, 1, 1});

  zero_warp_matrix();

  /*
   * index (tens):    6554443322211000
   *       (ones):    0628406284062840
   */
  uint64_t activate_lanes_bitfield = 0x0100000000000000;

  host_run_device_kernel(activate_lane_helper_kernel, activate_lanes_bitfield);

  auto *elem = _warp_matrix->access(0, 0);
  ASSERT_EQ(*elem, 56);
}

TEST_F(BitwiseTestFixture, activate_lane_helper_64_1_lane_60) {
  setup_fixture({64, 1, 1}, {1, 1, 1});

  zero_warp_matrix();

  /*
   * index (tens):    6554443322211000
   *       (ones):    0628406284062840
   */
  uint64_t activate_lanes_bitfield = 0x1000000000000000;

  host_run_device_kernel(activate_lane_helper_kernel, activate_lanes_bitfield);

  auto *elem = _warp_matrix->access(0, 0);
  ASSERT_EQ(*elem, 60);
}

TEST_F(BitwiseTestFixture, activate_lane_helper_64_1_lane_61) {
  setup_fixture({64, 1, 1}, {1, 1, 1});

  zero_warp_matrix();

  /*
   * index (tens):    6554443322211000
   *       (ones):    0628406284062840
   */
  uint64_t activate_lanes_bitfield = 0x2000000000000000;

  host_run_device_kernel(activate_lane_helper_kernel, activate_lanes_bitfield);

  auto *elem = _warp_matrix->access(0, 0);
  ASSERT_EQ(*elem, 61);
}

TEST_F(BitwiseTestFixture, activate_lane_helper_64_1_lane_62) {
  setup_fixture({64, 1, 1}, {1, 1, 1});

  zero_warp_matrix();

  /*
   * index (tens):    6554443322211000
   *       (ones):    0628406284062840
   */
  uint64_t activate_lanes_bitfield = 0x4000000000000000;

  host_run_device_kernel(activate_lane_helper_kernel, activate_lanes_bitfield);

  auto *elem = _warp_matrix->access(0, 0);
  ASSERT_EQ(*elem, 62);
}

TEST_F(BitwiseTestFixture, activate_lane_helper_64_1_lane_63) {
  setup_fixture({64, 1, 1}, {1, 1, 1});

  zero_warp_matrix();

  /*
   * index (tens):    6554443322211000
   *       (ones):    0628406284062840
   */
  uint64_t activate_lanes_bitfield = 0x8000000000000000;

  host_run_device_kernel(activate_lane_helper_kernel, activate_lanes_bitfield);

  auto *elem = _warp_matrix->access(0, 0);
  ASSERT_EQ(*elem, 63);
}

/*****************************************************************************
 *************************** Lowest Active Lane ******************************
 *****************************************************************************/
TEST_F(BitwiseTestFixture, lowest_active_lane_64_1_lane_0) {
  setup_fixture({64, 1, 1}, {1, 1, 1});

  zero_warp_matrix();

  /*
   * index (tens):    6554443322211000
   *       (ones):    0628406284062840
   */
  uint64_t activate_lanes_bitfield = 0xffffffffffffffff;

  host_run_device_kernel(lowest_active_lane_kernel, activate_lanes_bitfield);

  auto *elem = _warp_matrix->access(0, 0);
  ASSERT_EQ(*elem, 0);
}

TEST_F(BitwiseTestFixture, lowest_active_lane_64_1_lane_1) {
  setup_fixture({64, 1, 1}, {1, 1, 1});

  zero_warp_matrix();

  /*
   * index (tens):    6554443322211000
   *       (ones):    0628406284062840
   */
  uint64_t activate_lanes_bitfield = 0xfffffffffffffffe;

  host_run_device_kernel(lowest_active_lane_kernel, activate_lanes_bitfield);

  auto *elem = _warp_matrix->access(0, 0);
  ASSERT_EQ(*elem, 1);
}

TEST_F(BitwiseTestFixture, lowest_active_lane_64_1_lane_2) {
  setup_fixture({64, 1, 1}, {1, 1, 1});

  zero_warp_matrix();

  /*
   * index (tens):    6554443322211000
   *       (ones):    0628406284062840
   */
  uint64_t activate_lanes_bitfield = 0xfffffffffffffffc;

  host_run_device_kernel(lowest_active_lane_kernel, activate_lanes_bitfield);

  auto *elem = _warp_matrix->access(0, 0);
  ASSERT_EQ(*elem, 2);
}

TEST_F(BitwiseTestFixture, lowest_active_lane_64_1_lane_3) {
  setup_fixture({64, 1, 1}, {1, 1, 1});

  zero_warp_matrix();

  /*
   * index (tens):    6554443322211000
   *       (ones):    0628406284062840
   */
  uint64_t activate_lanes_bitfield = 0xfffffffffffffff8;

  host_run_device_kernel(lowest_active_lane_kernel, activate_lanes_bitfield);

  auto *elem = _warp_matrix->access(0, 0);
  ASSERT_EQ(*elem, 3);
}

TEST_F(BitwiseTestFixture, lowest_active_lane_64_1_lane_4) {
  setup_fixture({64, 1, 1}, {1, 1, 1});

  zero_warp_matrix();

  /*
   * index (tens):    6554443322211000
   *       (ones):    0628406284062840
   */
  uint64_t activate_lanes_bitfield = 0xfffffffffffffff0;

  host_run_device_kernel(lowest_active_lane_kernel, activate_lanes_bitfield);

  auto *elem = _warp_matrix->access(0, 0);
  ASSERT_EQ(*elem, 4);
}

TEST_F(BitwiseTestFixture, lowest_active_lane_64_1_lane_8) {
  setup_fixture({64, 1, 1}, {1, 1, 1});

  zero_warp_matrix();

  /*
   * index (tens):    6554443322211000
   *       (ones):    0628406284062840
   */
  uint64_t activate_lanes_bitfield = 0xffffffffffffff00;

  host_run_device_kernel(lowest_active_lane_kernel, activate_lanes_bitfield);

  auto *elem = _warp_matrix->access(0, 0);
  ASSERT_EQ(*elem, 8);
}

TEST_F(BitwiseTestFixture, lowest_active_lane_64_1_lane_16) {
  setup_fixture({64, 1, 1}, {1, 1, 1});

  zero_warp_matrix();

  /*
   * index (tens):    6554443322211000
   *       (ones):    0628406284062840
   */
  uint64_t activate_lanes_bitfield = 0xffffffffffff0000;

  host_run_device_kernel(lowest_active_lane_kernel, activate_lanes_bitfield);

  auto *elem = _warp_matrix->access(0, 0);
  ASSERT_EQ(*elem, 16);
}

TEST_F(BitwiseTestFixture, lowest_active_lane_16_1_lane_15) {
  setup_fixture({16, 1, 1}, {1, 1, 1});

  zero_warp_matrix();

  /*
   * index (tens):    6554443322211000
   *       (ones):    0628406284062840
   */
  uint64_t activate_lanes_bitfield = 0xffffffffffff8000;

  host_run_device_kernel(lowest_active_lane_kernel, activate_lanes_bitfield);

  auto *elem = _warp_matrix->access(0, 0);
  ASSERT_EQ(*elem, 15);
}

TEST_F(BitwiseTestFixture, lowest_active_lane_64_1_lane_31) {
  setup_fixture({64, 1, 1}, {1, 1, 1});

  zero_warp_matrix();

  /*
   * index (tens):    6554443322211000
   *       (ones):    0628406284062840
   */
  uint64_t activate_lanes_bitfield = 0xffffffff80000000;

  host_run_device_kernel(lowest_active_lane_kernel, activate_lanes_bitfield);

  auto *elem = _warp_matrix->access(0, 0);
  ASSERT_EQ(*elem, 31);
}

TEST_F(BitwiseTestFixture, lowest_active_lane_32_1_lane_31) {
  setup_fixture({32, 1, 1}, {1, 1, 1});

  zero_warp_matrix();

  /*
   * index (tens):    6554443322211000
   *       (ones):    0628406284062840
   */
  uint64_t activate_lanes_bitfield = 0xffffffff80000000;

  host_run_device_kernel(lowest_active_lane_kernel, activate_lanes_bitfield);

  auto *elem = _warp_matrix->access(0, 0);
  ASSERT_EQ(*elem, 31);
}

TEST_F(BitwiseTestFixture, lowest_active_lane_64_1_lane_32) {
  setup_fixture({64, 1, 1}, {1, 1, 1});

  zero_warp_matrix();

  /*
   * index (tens):    6554443322211000
   *       (ones):    0628406284062840
   */
  uint64_t activate_lanes_bitfield = 0xffffffff00000000;

  host_run_device_kernel(lowest_active_lane_kernel, activate_lanes_bitfield);

  auto *elem = _warp_matrix->access(0, 0);
  ASSERT_EQ(*elem, 32);
}

TEST_F(BitwiseTestFixture, lowest_active_lane_64_1_lane_36) {
  setup_fixture({64, 1, 1}, {1, 1, 1});

  zero_warp_matrix();

  /*
   * index (tens):    6554443322211000
   *       (ones):    0628406284062840
   */
  uint64_t activate_lanes_bitfield = 0xfffffff000000000;

  host_run_device_kernel(lowest_active_lane_kernel, activate_lanes_bitfield);

  auto *elem = _warp_matrix->access(0, 0);
  ASSERT_EQ(*elem, 36);
}

TEST_F(BitwiseTestFixture, lowest_active_lane_64_1_lane_47) {
  setup_fixture({64, 1, 1}, {1, 1, 1});

  zero_warp_matrix();

  /*
   * index (tens):    6554443322211000
   *       (ones):    0628406284062840
   */
  uint64_t activate_lanes_bitfield = 0xffff800000000000;

  host_run_device_kernel(lowest_active_lane_kernel, activate_lanes_bitfield);

  auto *elem = _warp_matrix->access(0, 0);
  ASSERT_EQ(*elem, 47);
}

TEST_F(BitwiseTestFixture, lowest_active_lane_64_1_lane_48) {
  setup_fixture({64, 1, 1}, {1, 1, 1});

  zero_warp_matrix();

  /*
   * index (tens):    6554443322211000
   *       (ones):    0628406284062840
   */
  uint64_t activate_lanes_bitfield = 0xffff000000000000;

  host_run_device_kernel(lowest_active_lane_kernel, activate_lanes_bitfield);

  auto *elem = _warp_matrix->access(0, 0);
  ASSERT_EQ(*elem, 48);
}

TEST_F(BitwiseTestFixture, lowest_active_lane_64_1_lane_49) {
  setup_fixture({64, 1, 1}, {1, 1, 1});

  zero_warp_matrix();

  /*
   * index (tens):    6554443322211000
   *       (ones):    0628406284062840
   */
  uint64_t activate_lanes_bitfield = 0xfffe000000000000;

  host_run_device_kernel(lowest_active_lane_kernel, activate_lanes_bitfield);

  auto *elem = _warp_matrix->access(0, 0);
  ASSERT_EQ(*elem, 49);
}

TEST_F(BitwiseTestFixture, lowest_active_lane_64_1_lane_60) {
  setup_fixture({64, 1, 1}, {1, 1, 1});

  zero_warp_matrix();

  /*
   * index (tens):    6554443322211000
   *       (ones):    0628406284062840
   */
  uint64_t activate_lanes_bitfield = 0xf000000000000000;

  host_run_device_kernel(lowest_active_lane_kernel, activate_lanes_bitfield);

  auto *elem = _warp_matrix->access(0, 0);
  ASSERT_EQ(*elem, 60);
}

TEST_F(BitwiseTestFixture, lowest_active_lane_128_1) {
  setup_fixture({128, 1, 1}, {1, 1, 1});

  /*
   * index (tens):    6554443322211000
   *       (ones):    0628406284062840
   */
  uint64_t activate_lanes_bitfield = 0x0000000080000000;

  host_run_device_kernel(lowest_active_lane_kernel, activate_lanes_bitfield);

  auto *elem = _warp_matrix->access(0, 0);
  ASSERT_EQ(*elem, 31);

  elem = _warp_matrix->access(1, 0);
  ASSERT_EQ(*elem, 31);
}

TEST_F(BitwiseTestFixture, lowest_active_lane_1024_80) {
  setup_fixture({1024, 1, 1}, {80, 1, 1});

  zero_warp_matrix();

  /*
   * index (tens):    6554443322211000
   *       (ones):    0628406284062840
   */
  uint64_t activate_lanes_bitfield = 0x0000000000000010;

  host_run_device_kernel(lowest_active_lane_kernel, activate_lanes_bitfield);

  for (size_t i = 0; i < _warp_matrix->rows(); i++) {
    for (size_t j = 0; j < _warp_matrix->columns(); j++) {
      auto *elem = _warp_matrix->access(i, j);
      ASSERT_EQ(*elem, 4);
    }
  }
}

/*****************************************************************************
 ************************** Is Lowest Active Lane ****************************
 *****************************************************************************/
TEST_F(BitwiseTestFixture, is_lowest_active_lane_64_1_lane_1) {
  setup_fixture({64, 1, 1}, {1, 1, 1});

  zero_warp_matrix();

  /*
   * index (tens):    6554443322211000
   *       (ones):    0628406284062840
   */
  uint64_t activate_lanes_bitfield = 0xf0f0f10000000002;

  host_run_device_kernel(is_lowest_active_lane_kernel, activate_lanes_bitfield);

  for (size_t i = 0; i < _warp_matrix->rows(); i++) {
    for (size_t j = 0; j < _warp_matrix->columns(); j++) {
      auto *elem = _warp_matrix->access(i, j);
      ASSERT_EQ(*elem, 1);
    }
  }
}

TEST_F(BitwiseTestFixture, is_lowest_active_lane_64_1_lane_2) {
  setup_fixture({64, 1, 1}, {1, 1, 1});

  zero_warp_matrix();

  /*
   * index (tens):    6554443322211000
   *       (ones):    0628406284062840
   */
  uint64_t activate_lanes_bitfield = 0xf0f0f10000000004;

  host_run_device_kernel(is_lowest_active_lane_kernel, activate_lanes_bitfield);

  for (size_t i = 0; i < _warp_matrix->rows(); i++) {
    for (size_t j = 0; j < _warp_matrix->columns(); j++) {
      auto *elem = _warp_matrix->access(i, j);
      ASSERT_EQ(*elem, 2);
    }
  }
}

TEST_F(BitwiseTestFixture, is_lowest_active_lane_64_1_lane_3) {
  setup_fixture({64, 1, 1}, {1, 1, 1});

  zero_warp_matrix();

  /*
   * index (tens):    6554443322211000
   *       (ones):    0628406284062840
   */
  uint64_t activate_lanes_bitfield = 0xf0f0f10000000008;

  host_run_device_kernel(is_lowest_active_lane_kernel, activate_lanes_bitfield);

  for (size_t i = 0; i < _warp_matrix->rows(); i++) {
    for (size_t j = 0; j < _warp_matrix->columns(); j++) {
      auto *elem = _warp_matrix->access(i, j);
      ASSERT_EQ(*elem, 3);
    }
  }
}

TEST_F(BitwiseTestFixture, is_lowest_active_lane_64_1_lane_4) {
  setup_fixture({64, 1, 1}, {1, 1, 1});

  zero_warp_matrix();

  /*
   * index (tens):    6554443322211000
   *       (ones):    0628406284062840
   */
  uint64_t activate_lanes_bitfield = 0x00f0fd8f350f0010;

  host_run_device_kernel(is_lowest_active_lane_kernel, activate_lanes_bitfield);

  for (size_t i = 0; i < _warp_matrix->rows(); i++) {
    for (size_t j = 0; j < _warp_matrix->columns(); j++) {
      auto *elem = _warp_matrix->access(i, j);
      ASSERT_EQ(*elem, 4);
    }
  }
}

TEST_F(BitwiseTestFixture, is_lowest_active_lane_256_8_lane_40) {
  setup_fixture({256, 1, 1}, {8, 1, 1});

  zero_warp_matrix();

  /*
   * index (tens):    6554443322211000
   *       (ones):    0628406284062840
   */
  uint64_t activate_lanes_bitfield = 0xf0f0f10000000000;

  host_run_device_kernel(is_lowest_active_lane_kernel, activate_lanes_bitfield);

  for (size_t i = 0; i < _warp_matrix->rows(); i++) {
    for (size_t j = 0; j < _warp_matrix->columns(); j++) {
      auto *elem = _warp_matrix->access(i, j);
      ASSERT_EQ(*elem, 40);
    }
  }
}

/*****************************************************************************
 ********************************* Lane ID ***********************************
 *****************************************************************************/
TEST_F(BitwiseTestFixture, lane_id_1_1_lane_15_inactive_left_zero) {
  setup_fixture({1, 1, 1}, {1, 1, 1});

  zero_warp_matrix();

  /*
   * index (tens):    6554443322211000
   *       (ones):    0628406284062840
   */
  uint64_t activate_lanes_bitfield = 0x0000000000008000;

  host_run_device_kernel(lane_id_kernel, activate_lanes_bitfield);

  for (size_t i = 0; i < _warp_matrix->rows(); i++) {
    for (size_t j = 0; j < _warp_matrix->columns(); j++) {
      auto *elem = _warp_matrix->access(i, j);
      ASSERT_EQ(*elem, 0);
    }
  }
}

TEST_F(BitwiseTestFixture, lane_id_32_1_lane_15) {
  setup_fixture({32, 1, 1}, {1, 1, 1});

  zero_warp_matrix();

  /*
   * index (tens):    6554443322211000
   *       (ones):    0628406284062840
   */
  uint64_t activate_lanes_bitfield = 0x0000000000008000;

  host_run_device_kernel(lane_id_kernel, activate_lanes_bitfield);

  for (size_t i = 0; i < _warp_matrix->rows(); i++) {
    for (size_t j = 0; j < _warp_matrix->columns(); j++) {
      auto *elem = _warp_matrix->access(i, j);
      ASSERT_EQ(*elem, 15);
    }
  }
}

TEST_F(BitwiseTestFixture, lane_id_64_1_lane_32) {
  setup_fixture({64, 1, 1}, {1, 1, 1});

  zero_warp_matrix();

  /*
   * index (tens):    6554443322211000
   *       (ones):    0628406284062840
   */
  uint64_t activate_lanes_bitfield = 0x0000000100000000;

  host_run_device_kernel(lane_id_kernel, activate_lanes_bitfield);

  for (size_t i = 0; i < _warp_matrix->rows(); i++) {
    for (size_t j = 0; j < _warp_matrix->columns(); j++) {
      auto *elem = _warp_matrix->access(i, j);
      ASSERT_EQ(*elem, 32);
    }
  }
}

TEST_F(BitwiseTestFixture, lane_id_64_1_lane_49) {
  setup_fixture({64, 1, 1}, {1, 1, 1});

  zero_warp_matrix();

  /*
   * index (tens):    6554443322211000
   *       (ones):    0628406284062840
   */
  uint64_t activate_lanes_bitfield = 0x0002000000000000;

  host_run_device_kernel(lane_id_kernel, activate_lanes_bitfield);

  for (size_t i = 0; i < _warp_matrix->rows(); i++) {
    for (size_t j = 0; j < _warp_matrix->columns(); j++) {
      auto *elem = _warp_matrix->access(i, j);
      ASSERT_EQ(*elem, 49);
    }
  }
}

TEST_F(BitwiseTestFixture, lane_id_128_2_lane_2) {
  setup_fixture({128, 1, 1}, {2, 1, 1});

  zero_warp_matrix();

  /*
   * index (tens):    6554443322211000
   *       (ones):    0628406284062840
   */
  uint64_t activate_lanes_bitfield = 0x0000000000000002;

  host_run_device_kernel(lane_id_kernel, activate_lanes_bitfield);

  for (size_t i = 0; i < _warp_matrix->rows(); i++) {
    for (size_t j = 0; j < _warp_matrix->columns(); j++) {
      auto *elem = _warp_matrix->access(i, j);
      ASSERT_EQ(*elem, 1);
    }
  }
}

TEST_F(BitwiseTestFixture, lane_id_128_2_lane_61) {
  setup_fixture({128, 1, 1}, {2, 1, 1});

  zero_warp_matrix();

  /*
   * index (tens):    6554443322211000
   *       (ones):    0628406284062840
   */
  uint64_t activate_lanes_bitfield = 0x2000000000000000;

  host_run_device_kernel(lane_id_kernel, activate_lanes_bitfield);

  for (size_t i = 0; i < _warp_matrix->rows(); i++) {
    for (size_t j = 0; j < _warp_matrix->columns(); j++) {
      auto *elem = _warp_matrix->access(i, j);
      ASSERT_EQ(*elem, 61);
    }
  }
}

TEST_F(BitwiseTestFixture, lane_id_128_2_lane_63) {
  setup_fixture({128, 1, 1}, {2, 1, 1});

  zero_warp_matrix();

  /*
   * index (tens):    6554443322211000
   *       (ones):    0628406284062840
   */
  uint64_t activate_lanes_bitfield = 0x8000000000000000;

  host_run_device_kernel(lane_id_kernel, activate_lanes_bitfield);

  for (size_t i = 0; i < _warp_matrix->rows(); i++) {
    for (size_t j = 0; j < _warp_matrix->columns(); j++) {
      auto *elem = _warp_matrix->access(i, j);
      ASSERT_EQ(*elem, 63);
    }
  }
}

/*****************************************************************************
 *************************** Number Active Lanes *****************************
 *****************************************************************************/
TEST_F(BitwiseTestFixture, number_active_lanes_kernel_128_2_num_active_1) {
  setup_fixture({256, 1, 1}, {2, 1, 1});

  zero_warp_matrix();

  /*
   * index (tens):    6554443322211000
   *       (ones):    0628406284062840
   */
  uint64_t activate_lanes_bitfield = 0x0004000000000000;

  host_run_device_kernel(number_active_lanes_kernel, activate_lanes_bitfield);

  for (size_t i = 0; i < _warp_matrix->rows(); i++) {
    for (size_t j = 0; j < _warp_matrix->columns(); j++) {
      auto *elem = _warp_matrix->access(i, j);
      ASSERT_EQ(*elem, 1);
    }
  }
}

TEST_F(BitwiseTestFixture, number_active_lanes_kernel_128_2_num_active_52) {
  setup_fixture({256, 1, 1}, {2, 1, 1});

  zero_warp_matrix();

  /*
   * index (tens):    6554443322211000
   *       (ones):    0628406284062840
   */
  uint64_t activate_lanes_bitfield = 0xfff0ffff0ffff0ff;

  host_run_device_kernel(number_active_lanes_kernel, activate_lanes_bitfield);

  for (size_t i = 0; i < _warp_matrix->rows(); i++) {
    for (size_t j = 0; j < _warp_matrix->columns(); j++) {
      auto *elem = _warp_matrix->access(i, j);
      ASSERT_EQ(*elem, 52);
    }
  }
}

TEST_F(BitwiseTestFixture, number_active_lanes_kernel_128_2_num_active_64) {
  setup_fixture({128, 1, 1}, {2, 1, 1});

  zero_warp_matrix();

  /*
   * index (tens):    6554443322211000
   *       (ones):    0628406284062840
   */
  uint64_t activate_lanes_bitfield = 0xffffffffffffffff;

  host_run_device_kernel(number_active_lanes_kernel, activate_lanes_bitfield);

  for (size_t i = 0; i < _warp_matrix->rows(); i++) {
    for (size_t j = 0; j < _warp_matrix->columns(); j++) {
      auto *elem = _warp_matrix->access(i, j);
      ASSERT_EQ(*elem, 64);
    }
  }
}

/*****************************************************************************
 ************************* Active Logical Lane ID ****************************
 *****************************************************************************/
TEST_F(BitwiseTestFixture, active_logical_lane_id_kernel_64_1_logic_2_is_29) {
  setup_fixture({64, 1, 1}, {1, 1, 1});

  zero_warp_matrix();

  /*
   * index (tens):    6554443322211000
   *       (ones):    0628406284062840
   */
  uint64_t activate_lanes_bitfield = 0x0FC0000020002004;

  host_run_device_kernel(active_logical_lane_id_2_kernel,
                         activate_lanes_bitfield);

  for (size_t i = 0; i < _warp_matrix->rows(); i++) {
    for (size_t j = 0; j < _warp_matrix->columns(); j++) {
      auto *elem = _warp_matrix->access(i, j);
      ASSERT_EQ(*elem, 29);
    }
  }
}

TEST_F(BitwiseTestFixture, active_logical_lane_id_kernel_64_1_logic_2_is_62) {
  setup_fixture({64, 1, 1}, {1, 1, 1});

  zero_warp_matrix();

  /*
   * index (tens):    6554443322211000
   *       (ones):    0628406284062840
   */
  uint64_t activate_lanes_bitfield = 0x4000000000000024;

  host_run_device_kernel(active_logical_lane_id_2_kernel,
                         activate_lanes_bitfield);

  for (size_t i = 0; i < _warp_matrix->rows(); i++) {
    for (size_t j = 0; j < _warp_matrix->columns(); j++) {
      auto *elem = _warp_matrix->access(i, j);
      ASSERT_EQ(*elem, 62);
    }
  }
}

TEST_F(BitwiseTestFixture, active_logical_lane_id_kernel_768_3_logic_2_is_32) {
  setup_fixture({768, 1, 1}, {3, 1, 1});

  zero_warp_matrix();

  /*
   * index (tens):    6554443322211000
   *       (ones):    0628406284062840
   */
  uint64_t activate_lanes_bitfield = 0xFFFF000100001100;

  host_run_device_kernel(active_logical_lane_id_2_kernel,
                         activate_lanes_bitfield);

  for (size_t i = 0; i < _warp_matrix->rows(); i++) {
    for (size_t j = 0; j < _warp_matrix->columns(); j++) {
      auto *elem = _warp_matrix->access(i, j);
      ASSERT_EQ(*elem, 32);
    }
  }
}

/*****************************************************************************
 ****************************** Broadcast Up *********************************
 *****************************************************************************/
TEST_F(BitwiseTestFixture, broadcast_up_value_42_kernel_64_1_lane_0) {
  setup_fixture({64, 1, 1}, {1, 1, 1});

  zero_warp_matrix();

  /*
   * index (tens):    6554443322211000
   *       (ones):    0628406284062840
   */
  uint64_t activate_lanes_bitfield = 0x0000000000000001;

  host_run_device_kernel(broadcast_up_value_42_kernel, activate_lanes_bitfield);

  for (size_t i = 0; i < _warp_matrix->rows(); i++) {
    for (size_t j = 0; j < _warp_matrix->columns(); j++) {
      auto *elem = _warp_matrix->access(i, j);
      ASSERT_EQ(*elem, 42);
    }
  }
}

TEST_F(BitwiseTestFixture, broadcast_up_value_42_kernel_64_1_lane_1) {
  setup_fixture({64, 1, 1}, {1, 1, 1});

  zero_warp_matrix();

  /*
   * index (tens):    6554443322211000
   *       (ones):    0628406284062840
   */
  uint64_t activate_lanes_bitfield = 0x0000000000000002;

  host_run_device_kernel(broadcast_up_value_42_kernel, activate_lanes_bitfield);

  for (size_t i = 0; i < _warp_matrix->rows(); i++) {
    for (size_t j = 0; j < _warp_matrix->columns(); j++) {
      auto *elem = _warp_matrix->access(i, j);
      ASSERT_EQ(*elem, 42);
    }
  }
}

TEST_F(BitwiseTestFixture, broadcast_up_value_42_kernel_64_1_lane_16) {
  setup_fixture({64, 1, 1}, {1, 1, 1});

  zero_warp_matrix();

  /*
   * index (tens):    6554443322211000
   *       (ones):    0628406284062840
   */
  uint64_t activate_lanes_bitfield = 0x0000000000010000;

  host_run_device_kernel(broadcast_up_value_42_kernel, activate_lanes_bitfield);

  for (size_t i = 0; i < _warp_matrix->rows(); i++) {
    for (size_t j = 0; j < _warp_matrix->columns(); j++) {
      auto *elem = _warp_matrix->access(i, j);
      ASSERT_EQ(*elem, 42);
    }
  }
}

TEST_F(BitwiseTestFixture, broadcast_up_value_42_kernel_64_1_odd_index) {
  setup_fixture({64, 1, 1}, {1, 1, 1});

  zero_warp_matrix();

  /*
   * index (tens):    6554443322211000
   *       (ones):    0628406284062840
   */
  uint64_t activate_lanes_bitfield = 0x0101010101010101;

  host_run_device_kernel(broadcast_up_value_42_kernel, activate_lanes_bitfield);

  for (size_t i = 0; i < _warp_matrix->rows(); i++) {
    for (size_t j = 0; j < _warp_matrix->columns(); j++) {
      auto *elem = _warp_matrix->access(i, j);
      ASSERT_EQ(*elem, 42);
    }
  }
}

TEST_F(BitwiseTestFixture, broadcast_up_value_42_kernel_64_1_even_index) {
  setup_fixture({64, 1, 1}, {1, 1, 1});

  zero_warp_matrix();

  /*
   * index (tens):    6554443322211000
   *       (ones):    0628406284062840
   */
  uint64_t activate_lanes_bitfield = 0x1010101010101010;

  host_run_device_kernel(broadcast_up_value_42_kernel, activate_lanes_bitfield);

  for (size_t i = 0; i < _warp_matrix->rows(); i++) {
    for (size_t j = 0; j < _warp_matrix->columns(); j++) {
      auto *elem = _warp_matrix->access(i, j);
      ASSERT_EQ(*elem, 42);
    }
  }
}

TEST_F(BitwiseTestFixture, broadcast_up_value_42_kernel_64_1_all) {
  setup_fixture({64, 1, 1}, {1, 1, 1});

  zero_warp_matrix();

  /*
   * index (tens):    6554443322211000
   *       (ones):    0628406284062840
   */
  uint64_t activate_lanes_bitfield = 0xFFFFFFFFFFFFFFFF;

  host_run_device_kernel(broadcast_up_value_42_kernel, activate_lanes_bitfield);

  for (size_t i = 0; i < _warp_matrix->rows(); i++) {
    for (size_t j = 0; j < _warp_matrix->columns(); j++) {
      auto *elem = _warp_matrix->access(i, j);
      ASSERT_EQ(*elem, 42);
    }
  }
}

/*****************************************************************************
 ******************************* Fetch Incr **********************************
 *****************************************************************************/
TEST_F(BitwiseTestFixture, fetch_incr_kernel_4_1) {
  setup_fixture({4, 1, 1}, {1, 1, 1});

  zero_warp_matrix();

  /*
   * index (tens):    6554443322211000
   *       (ones):    0628406284062840
   */
  uint64_t activate_lanes_bitfield = 0x000000000000000A;

  host_run_device_kernel(fetch_incr_lowest_active_lane_kernel,
                         activate_lanes_bitfield);

  ASSERT_EQ(*_device_methods->_fetch_value, 2);

  for (size_t i = 0; i < _warp_matrix->rows(); i++) {
    for (size_t j = 0; j < _warp_matrix->columns(); j++) {
      auto *elem = _warp_matrix->access(i, j);
      ASSERT_EQ(*elem % this->_wf_size, 0);
    }
  }
}

TEST_F(BitwiseTestFixture, fetch_incr_kernel_64_1) {
  setup_fixture({64, 1, 1}, {1, 1, 1});

  zero_warp_matrix();

  /*
   * index (tens):    6554443322211000
   *       (ones):    0628406284062840
   */
  uint64_t activate_lanes_bitfield = 0xFFFFFFFFFFFFFFFF;

  host_run_device_kernel(fetch_incr_lowest_active_lane_kernel,
                         activate_lanes_bitfield);

  ASSERT_EQ(*_device_methods->_fetch_value, 64);

  for (size_t i = 0; i < _warp_matrix->rows(); i++) {
    for (size_t j = 0; j < _warp_matrix->columns(); j++) {
      auto *elem = _warp_matrix->access(i, j);
      ASSERT_EQ(*elem % this->_wf_size, 0);
    }
  }
}

TEST_F(BitwiseTestFixture, fetch_incr_kernel_64_1_top_half) {
  setup_fixture({64, 1, 1}, {1, 1, 1});

  zero_warp_matrix();

  /*
   * index (tens):    6554443322211000
   *       (ones):    0628406284062840
   */
  uint64_t activate_lanes_bitfield = 0xFFFFFFFF00000000;

  host_run_device_kernel(fetch_incr_lowest_active_lane_kernel,
                         activate_lanes_bitfield);

  ASSERT_EQ(*_device_methods->_fetch_value, 32);

  for (size_t i = 0; i < _warp_matrix->rows(); i++) {
    for (size_t j = 0; j < _warp_matrix->columns(); j++) {
      auto *elem = _warp_matrix->access(i, j);
      ASSERT_EQ(*elem % 32, 0);
    }
  }
}

TEST_F(BitwiseTestFixture, fetch_incr_kernel_64_1_alternating) {
  setup_fixture({64, 1, 1}, {1, 1, 1});

  zero_warp_matrix();

  /*
   * index (tens):    6554443322211000
   *       (ones):    0628406284062840
   */
  uint64_t activate_lanes_bitfield = 0xF0F0F0F0F0F0F0F0;

  host_run_device_kernel(fetch_incr_lowest_active_lane_kernel,
                         activate_lanes_bitfield);

  ASSERT_EQ(*_device_methods->_fetch_value, 32);

  for (size_t i = 0; i < _warp_matrix->rows(); i++) {
    for (size_t j = 0; j < _warp_matrix->columns(); j++) {
      auto *elem = _warp_matrix->access(i, j);
      ASSERT_EQ(*elem % 32, 0);
    }
  }
}

TEST_F(BitwiseTestFixture, fetch_incr_kernel_1024_1024) {
  setup_fixture({1024, 1, 1}, {1024, 1, 1});

  zero_warp_matrix();

  /*
   * index (tens):    6554443322211000
   *       (ones):    0628406284062840
   */
  uint64_t activate_lanes_bitfield = 0xFFFFFFFFFFFFFFFF;

  host_run_device_kernel(fetch_incr_lowest_active_lane_kernel,
                         activate_lanes_bitfield);

  ASSERT_EQ(*_device_methods->_fetch_value, 1024 * 1024);

  for (size_t i = 0; i < _warp_matrix->rows(); i++) {
    for (size_t j = 0; j < _warp_matrix->columns(); j++) {
      auto *elem = _warp_matrix->access(i, j);
      ASSERT_EQ(*elem % this->_wf_size, 0);
    }
  }
}

TEST_F(BitwiseTestFixture, fetch_incr_logical_1_kernel_1024_1024) {
  setup_fixture({1024, 1, 1}, {1024, 1, 1});

  zero_warp_matrix();

  /*
   * index (tens):    6554443322211000
   *       (ones):    0628406284062840
   */
  uint64_t activate_lanes_bitfield = 0xFFFFFFFFFFFFFFFF;

  host_run_device_kernel(fetch_incr_active_logical_lane_1_kernel,
                         activate_lanes_bitfield);

  ASSERT_EQ(*_device_methods->_fetch_value, 1024 * 1024);

  for (size_t i = 0; i < _warp_matrix->rows(); i++) {
    for (size_t j = 0; j < _warp_matrix->columns(); j++) {
      auto *elem = _warp_matrix->access(i, j);
      ASSERT_EQ(*elem % this->_wf_size, 1);
    }
  }
}
