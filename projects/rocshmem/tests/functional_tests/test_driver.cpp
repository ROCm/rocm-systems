/******************************************************************************
 * Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *****************************************************************************/

#include <rocshmem/rocshmem.hpp>
#include <vector>

#include "tester.hpp"
#include "tester_arguments.hpp"
#include "slr.hpp"

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

    if (static_cast<int>(pmix_myproc.rank) == root) {
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

    if (static_cast<int>(pmix_myproc.rank) != root) {
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

// SLR helper functions (mimic PMIx pattern)
static void init_slr(int *rank, int *nranks)
{
    int ret = SLR::SLR_Init();
    if (ret != SLR::SLR_SUCCESS) {
        std::cerr << "SLR_Init failed: " << ret << std::endl;
        abort();
    }

    ret = SLR::SLR_Myrank(rank);
    if (ret != SLR::SLR_SUCCESS) {
        std::cerr << "SLR_Myrank failed: " << ret << std::endl;
        abort();
    }

    ret = SLR::SLR_Numprocs(nranks);
    if (ret != SLR::SLR_SUCCESS) {
        std::cerr << "SLR_Numprocs failed: " << ret << std::endl;
        abort();
    }

#ifdef VERBOSE
    printf("SLR rank %d of %d: Running\n", *rank, *nranks);
#endif
}

static void slr_bcast(void *buf, size_t nbytes)
{
    int ret = SLR::SLR_Broadcast(buf, nbytes);
    if (ret != SLR::SLR_SUCCESS) {
        int rank;
        SLR::SLR_Myrank(&rank);
        std::cerr << "Rank " << rank << " SLR_Broadcast failed: " << ret << std::endl;
        abort();
    }
}

static void finalize_slr()
{
    int ret = SLR::SLR_Finalize();
    if (ret != SLR::SLR_SUCCESS) {
        std::cerr << "SLR_Finalize failed: " << ret << std::endl;
        abort();
    }
}

using namespace rocshmem;

int main(int argc, char *argv[]) {
  /**
   * Setup the tester arguments.
   */
  TesterArguments args(argc, argv);

  /***
   * Determine which runtime to use and select a GPU.
   * Probe env vars in priority order; first match wins.
   */
  bool using_slr = false;

  struct { const char* var; bool is_slr; } providers[] = {
    {"OMPI_COMM_WORLD_LOCAL_RANK", false},  // Open MPI / prterun
    {"MPI_LOCALRANKID",            false},  // MPICH Hydra
    {"SLURM_LOCALID",              false},  // SLURM
    {"FLUX_TASK_LOCAL_ID",         false},  // Flux
    {"LOCAL_RANK",                 false},  // torchrun / generic
    {"ROCSHMEM_SLR_NP",            true},   // SLR runtime
  };

  int local_rank = -1;
  for (auto& p : providers) {
    const char* v = getenv(p.var);
    if (v) {
      using_slr = p.is_slr;
      if (p.is_slr) {
        int nranks;
        init_slr(&local_rank, &nranks);
      } else {
        local_rank = atoi(v);
      }
      break;
    }
  }

  if (local_rank < 0) {
    fprintf(stderr,
        "Could not determine local GPU rank.\n"
        "Supported env vars: OMPI_COMM_WORLD_LOCAL_RANK (Open MPI/prterun), "
        "MPI_LOCALRANKID (MPICH Hydra), SLURM_LOCALID (SLURM), "
        "FLUX_TASK_LOCAL_ID (Flux), LOCAL_RANK (torchrun), "
        "ROCSHMEM_SLR_NP (SLR)\n");
    abort();
  }

  int num_devices = 0;
  CHECK_HIP(hipGetDeviceCount(&num_devices));

  if (local_rank >= num_devices) {
    std::cerr << "Error: local rank " << local_rank
              << " exceeds the number of available GPUs (" << num_devices
              << "). Aborting.\n";
    abort();
  }
  CHECK_HIP(hipSetDevice(local_rank));

  /**
   * Must initialize rocshmem to access arguments needed by the tester.
   */
  if (using_slr) {
    // Using SLR runtime
    int ret;
    int rank, nranks;
    rocshmem_uniqueid_t uid;
    rocshmem_init_attr_t attr;

    // Get rank and nranks from SLR
    SLR::SLR_Myrank(&rank);
    SLR::SLR_Numprocs(&nranks);

    if (rank == 0) {
      ret = rocshmem_get_uniqueid(&uid);
      if (ret != ROCSHMEM_SUCCESS) {
        std::cout << rank << ": Error in rocshmem_get_uniqueid. Aborting.\n";
        abort();
      }
    }

    // Broadcast uniqueid from rank 0 to all ranks using SLR
    slr_bcast(&uid, sizeof(rocshmem_uniqueid_t));

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
  } else {
    // Using MPI or PMIx runtime
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
  if (getenv("ROCSHMEM_TEST_UUID") && atoi(getenv("ROCSHMEM_TEST_UUID")) != 0) {
    fprintf(stderr,
        "Warning: ROCSHMEM_TEST_UUID=1 requested but this binary was built without PMIx support.\n"
        "  UUID-based initialization is not available. Falling back to rocshmem_init().\n");
  }
  rocshmem_init();
#endif
  }

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

  /**
   * Finalize SLR if it was used
   */
  if (using_slr) {
    finalize_slr();
  }

  return 0;
}
