/*
 * =============================================================================
 *   ROC Runtime Conformance Release License
 * =============================================================================
 * The University of Illinois/NCSA
 * Open Source License (NCSA)
 *
 * Copyright (c) 2018, Advanced Micro Devices, Inc.
 * All rights reserved.
 *
 * Developed by:
 *
 *                 AMD Research and AMD ROC Software Development
 *
 *                 Advanced Micro Devices, Inc.
 *
 *                 www.amd.com
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal with the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 *  - Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimers.
 *  - Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimers in
 *    the documentation and/or other materials provided with the distribution.
 *  - Neither the names of <Name of Development Group, Name of Institution>,
 *    nor the names of its contributors may be used to endorse or promote
 *    products derived from this Software without specific prior written
 *    permission.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
 * THE CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR
 * OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
 * ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS WITH THE SOFTWARE.
 *
 */

#ifndef ROCRTST_COMMON_CONCURRENT_UTILS_H_
#define ROCRTST_COMMON_CONCURRENT_UTILS_H_

#include <pthread.h>
#include <stdint.h>
#include <iostream>


namespace rocrtst {
/**
 * @enum TEST_STATUS
 * @brief This enum lists status of test pthread
 */
enum TEST_STATUS {TEST_NOT_STARTED, TEST_RUNNING,
                  TEST_STOP, TEST_FINISHED, TEST_ERROR};


typedef void (*func_ptr)(void *input);
/**
 * @struct test_aux
 * @brief This structure holds information for a test
 */
struct test_aux{
    // Pointer to the test function
    void *fun_prt;
    // Pointer to the data for the test function
    void *data;
    // status of the test listed in enum TEST_STATUS
    uint16_t status;
};

/**
 * @struct thread_aux
 * @brief This structure holds the data for a test thread.
 */
struct thread_aux {
    // Thread Id
    int tid;
    // Pointer to a test item
    test_aux *test;
    // Pointer to the run_flag shared in the test group
    volatile int *run_flag;
    // Pointer to the exit_flag shared in the test group
    volatile int *exit_flag;
    // Pointer to the pthread mutex shared in the test group
    pthread_mutex_t *test_mutex;
    // Pointer to the pthread condition shared in the test group
    pthread_cond_t *test_cond;
    // Pointer to the number of running tests
    volatile unsigned int *num_running_t;
};

/**
 * @struct test_group
 * @brief This structure holds data for a test group
 */
struct test_group {
    // test group size, i.e., size of test_list array
    size_t group_size;
    // number of test
    int num_test;
    // number of threads - since one test per thread, equal to num_test
    int n_threads;
    // a flag for telling all threads to run - 0: stop, 1: run
    volatile int run_flag;
    // a flag for telling all threads to finish - 1: exit
    volatile int exit_flag;
    // pthread tid
    pthread_t *tid;
    // pthread attr
    pthread_attr_t attr;
    // pthread mutex shared in a group
    pthread_mutex_t test_mutex;
    // pthread condition signal shared in a group
    pthread_cond_t test_cond;
    // the list of test info
    test_aux *test_list;
    // the list of thread info
    thread_aux *thread_list;
    // number of running tests
    volatile unsigned int num_running_t;
};

/**
 * @brief create a test group, and preallocate
 * test_list array with group_size
 * @return initialized struct test_group
 */
test_group* TestGroupCreate(size_t group_size);

/**
 * @brief resize the array of test_list
 * @return
 */
void TestGroupResize(test_group *t_group, size_t new_group_size);

/**
 * @brief add a new test into the specific test group
 * @param t_group Pointer to a test group
 * @param fun Pointer to the test function
 * @param data Pointer to data for the test function
 * @param num_copy Number of copies of the test
 */
void TestGroupAdd(test_group *t_group, func_ptr fun,
                    void *data, size_t num_copy);

/**
 * @brief create threads for tests in a test group
 * @param t_group Pointer to a test group
 */
void TestGroupThreadCreate(test_group *t_group);

/**
 * @brief return the number of tests in a test group
 * @param t_group Pointer to a test group
 */
int TestGroupNumTests(test_group *t_group);

/**
 * @brief run all threads/tests in a test group
 * @param t_group Pointer to a test group
 */
void TestGroupStart(test_group *t_group);

/**
 * @brief wait all threads/tests in a test group finish
 * The function is blocked until all threads are finished
 * @param t_group Pointer to a test group
 */
void TestGroupWait(test_group *t_group);

/**
 * @brief terminate all threads/tests in a test group by sending a signal
 * set exit_flag to 1, wait until all threads are finished
 * @param t_group Pointer to a test group
 */
void TestGroupExit(test_group *t_group);

/**
 * @brief destroy a test group, release all resources
 * @param t_group Pointer to a test group
 */
void TestGroupDestroy(test_group *t_group);

/**
 * @brief check the status of specific test in a test group
 * @param t_group Pointer to a test group
 * @param test_id Test No.
 * @return the status of the test listed in enum TEST_STATUS
 */
int TestGroupTestStatus(test_group *t_group, int test_id);

/**
 * @brief set affinity of the specific test
 * @param t_group Pointer to a test group
 * @param test_id Test No.
 * @param cpu_id CPU No. that the test is binded to
 */
void TestGroupThreadAffinity(test_group *t_group,
                                int test_id, int cpu_id);

/**
 * @brief force kill a test group
 * @param t_group Pointer to a test group
 */
void TestGroupKill(test_group *t_group);
}  // namespace rocrtst
#endif  // ROCRTST_COMMON_CONCURRENT_UTILS_H_
