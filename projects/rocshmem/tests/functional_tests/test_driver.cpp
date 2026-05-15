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

#include <rocshmem/rocshmem.hpp>
#include <vector>

#include "tester.hpp"
#include "tester_arguments.hpp"

#if defined(HAVE_PMIX)
#include <pmix.h>

static pmix_proc_t pmix_myproc;
static pmix_proc_t pmix_proc;

static void init_pmix(int *rank, int *nranks)
{
    pmix_status_t rc;
    pmix_value_t *val;

    if (PMIX_SUCCESS != (rc = PMIx_Init(&pmix_myproc, NULL, 0))) {
      std::cerr << "Rank " << pmix_myproc.rank << " PMIx_Init failed: " << rc << std::endl;
      abort();
    }
#ifdef VERBOSE
    printf("Client ns %s rank %d: Running\n", pmix_myproc.nspace, pmix_myproc.rank);
#endif
    PMIX_PROC_CONSTRUCT(&pmix_proc);
    PMIX_LOAD_PROCID(&pmix_proc, pmix_myproc.nspace, PMIX_RANK_WILDCARD);

    /* get our job size */
    if (PMIX_SUCCESS != (rc = PMIx_Get(&pmix_proc, PMIX_JOB_SIZE, NULL, 0, &val))) {
      std::cerr << "Rank " << pmix_myproc.rank << " PMIx_Get universe size failed: "
                <<  rc << std::endl;
        abort();
    }

    *nranks = val->data.uint32;
    *rank   = pmix_myproc.rank;

    PMIX_VALUE_RELEASE(val);
    return;
}

static void pmix_bcast(void *buf, size_t nbytes, char *key, int root)
{
    pmix_status_t rc;
    pmix_value_t value;
    pmix_value_t *val;
    pmix_info_t *info;
    bool flag;

    if (pmix_myproc.rank == root) {
      value.type = PMIX_BYTE_OBJECT;
      value.data.bo.bytes = (char *) (buf);
      value.data.bo.size = nbytes;

      rc = PMIx_Put(PMIX_GLOBAL, key, &value);
      if (PMIX_SUCCESS != rc) {
        std::cerr << "Rank " << pmix_myproc.rank << " PMIx_Put failed: " << rc << std::endl;
        abort();
      }

      /* push the data to our PMIx server */
      if (PMIX_SUCCESS != (rc = PMIx_Commit())) {
        std::cerr <<  "Rank " << pmix_myproc.rank << " PMIx_Commit failed: " << rc << std::endl;
        abort();
      }
    }

    /* call fence to synchronize with our peers - instruct
     * the fence operation to collect and return all "put"
     * data from our peers */
    PMIX_INFO_CREATE(info, 1);
    flag = true;
        PMIX_INFO_LOAD(info, PMIX_COLLECT_DATA, &flag, PMIX_BOOL);
    if (PMIX_SUCCESS != (rc = PMIx_Fence(&pmix_proc, 1, info, 1))) {
      std::cerr <<  "Rank " << pmix_myproc.rank << " PMIx_Fence failed: " << rc << std::endl;
      abort();
    }
    PMIX_INFO_FREE(info, 1);

    pmix_proc.rank = 0;
    if (PMIX_SUCCESS != (rc = PMIx_Get(&pmix_proc, key, NULL, 0, &val))) {
      std::cerr <<  "Rank " << pmix_myproc.rank << " PMIx_Get failed: " << rc << std::endl;
      abort();
    }
    if (PMIX_BYTE_OBJECT != val->type) {
      std::cerr <<  "Rank " << pmix_myproc.rank << " PMIx_Get returned wrong type: " << val->type  << std::endl;
      PMIX_VALUE_RELEASE(val);
      abort();
    }

    if (pmix_myproc.rank != root) {
      if (NULL == val->data.bo.bytes) {
        std::cerr <<  "Rank " << pmix_myproc.rank << " PMIx_Get %d returned NULL pointer\n";
        PMIX_VALUE_RELEASE(val);
        abort();
      }
      memcpy (buf, val->data.bo.bytes, val->data.bo.size);
    }
    PMIX_VALUE_RELEASE(val);

    return;
}
#endif

using namespace rocshmem;

int main(int argc, char *argv[]) {
  /**
   * Setup the tester arguments.
   */
  TesterArguments args(argc, argv);

  /***
   * Select a GPU
   */
  char* ompi_local_rank = getenv("OMPI_COMM_WORLD_LOCAL_RANK");
  if (nullptr == ompi_local_rank) {
    printf("Could not determine local rank, use Open MPI `mpiexec`\n");
    abort();
  }
  CHECK_HIP(hipSetDevice(atoi(ompi_local_rank)));

  /**
   * Must initialize rocshmem to access arguments needed by the tester.
   */
#ifdef HAVE_PMIX
  int test_uuid = 0;
  char *rocshmem_test_uuid = getenv("ROCSHMEM_TEST_UUID");
  if (rocshmem_test_uuid != nullptr) {
    test_uuid = atoi(rocshmem_test_uuid);
  }

  if (test_uuid) {
    int ret;
    int rank, nranks;
    rocshmem_uniqueid_t uid;
    rocshmem_init_attr_t attr;

    init_pmix(&rank, &nranks);
    if (rank == 0) {
      ret = rocshmem_get_uniqueid (&uid);
      if (ret != ROCSHMEM_SUCCESS) {
        std::cout << rank << ": Error in rocshmem_get_uniqueid. Aborting.\n";
        abort();
      }
    }

    char key[] = "rocshmem-uuid";
    pmix_bcast(&uid, sizeof(rocshmem_uniqueid_t), key, 0);

    // Close PMIx before potentially doing MPI_Init inside rocshmem_init
    PMIx_Finalize(NULL, 0);

    ret = rocshmem_set_attr_uniqueid_args(rank, nranks, &uid, &attr);
    if (ret != ROCSHMEM_SUCCESS) {
      std::cout << rank << ": Error in rocshmem_set_attr_uniqueid_args. Aborting.\n";
      abort();
    }

    ret = rocshmem_init_attr(ROCSHMEM_INIT_WITH_UNIQUEID, &attr);
    if (ret != ROCSHMEM_SUCCESS) {
      std::cout << rank << ": Error in rocshmem_init_attr. Aborting.\n";
      abort();
    }

#ifdef VERBOSE
    std::cout << rank << ": rocshmem_init_attr SUCCESS\n";
#endif
  } else {
    rocshmem_init();
  }
#else
  rocshmem_init();
#endif

  /**
   * Now grab the arguments from rocshmem.
   */
  args.get_arguments();

  /**
   * Using the arguments we just constructed, call the tester factory
   * method to get the tester (specified by the arguments).
   */
  std::vector<Tester *> tests = Tester::create(args);

  /**
   * Run the tests
   */
  for (auto test : tests) {
    test->execute();

    /**
     * The tester factory method news the tester to create it so we clean
     * up the memory here.
     */
    delete test;
  }

  /**
   * The rocshmem library needs to be cleaned up with this call. It pairs
   * with the init function above.
   */
  rocshmem_finalize();

  return 0;
}
