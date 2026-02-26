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

/**
 * @file rocshmem.cpp
 * @brief Public header for rocSHMEM device and host libraries.
 *
 * This is the implementation for the public rocshmem.hpp header file.  This
 * guy just extracts the transport from the opaque public handles and delegates
 * to the appropriate backend.
 */

#include "rocshmem/rocshmem.hpp"

#include "backend_bc.hpp"
#include "context_incl.hpp"
#include "envvar.hpp"
#if defined(USE_GDA)
#include "gda/backend_gda.hpp"
#include "gda/context_gda_tmpl_host.hpp"
#endif
#if defined(USE_RO)
#include "reverse_offload/backend_ro.hpp"
#include "reverse_offload/context_ro_tmpl_host.hpp"
#endif
#if defined(USE_IPC)
#include "ipc/backend_ipc.hpp"
#include "ipc/context_ipc_tmpl_host.hpp"
#endif
#include "mpi_instance.hpp"
#include "team.hpp"
#include "templates_host.hpp"
#include "util.hpp"
#include "bootstrap/bootstrap.hpp"

#include <cstdlib>
#include <cstring>
#include <functional>
#include <random>
#include <cassert>
#include <unistd.h>
#include <sys/time.h>
#include <sys/resource.h>

namespace rocshmem {

#define VERIFY_BACKEND()                                                      \
  {                                                                           \
    if (!backend) {                                                           \
      fprintf(stderr, "ROCSHMEM_ERROR: %s in file '%s' in line %d\n",         \
              "Call 'rocshmem_init'", __FILE__, __LINE__);                    \
      abort();                                                                \
    }                                                                         \
  }

Backend *backend = nullptr;
MPIInstance *mpi_instance = nullptr;
TcpBootstrap *bootstr = nullptr;
rocshmem_ctx_t ROCSHMEM_HOST_CTX_DEFAULT;

 /**
 * Begin Host Code
 **/

BackendType get_backend_type() { return backend->get_backend_type(); }

#if defined(USE_GDA) && defined(USE_RO) && defined(USE_IPC)
static BackendType select_backend_type(MPI_Comm comm, TcpBootstrap *bootstrap) {
  BackendType type;

  /* Check whether the user explicitely requests a particular backend type */
  std::string envstr = envvar::backend;
  std::transform(envstr.begin(), envstr.end(), envstr.begin(), ::tolower);
  if (!envstr.empty()) {
    DPRINTF("Found environment variable ROCSHMEM_BACKEND, value is %s\n", envstr.c_str());
    if (envstr.find("gda") != std::string::npos) {
      if (GDABackend::backend_can_run() != ROCSHMEM_SUCCESS) {
        fprintf(stderr, "Error: ROCSHMEM_BACKEND=gda requested but GDA backend cannot run.\n"
                        "No active RDMA interface found for the requested provider.\n"
                        "Check that the correct NIC hardware is present and the link is active.\n");
        exit(1);
      }
      return BackendType::GDA_BACKEND;
    }
    if (envstr.find("ro") != std::string::npos) {
      if (ROBackend::backend_can_run() != ROCSHMEM_SUCCESS) {
        fprintf(stderr, "Error: ROCSHMEM_BACKEND=ro requested but RO backend cannot run.\n"
                        "MPI library could not be loaded.\n"
                        "Check that MPI is properly installed and accessible.\n");
        exit(1);
      }
      return BackendType::RO_BACKEND;
    }
    if (envstr.find("ipc") != std::string::npos) {
      if (IPCBackend::backend_can_run(comm, bootstrap) != ROCSHMEM_SUCCESS) {
        fprintf(stderr, "Error: ROCSHMEM_BACKEND=ipc requested but IPC backend cannot run.\n"
                "Most likely cause is PEs distributed to more than one node.\n");
        exit(1);
      }
      return BackendType::IPC_BACKEND;
    }
  }

  if (IPCBackend::backend_can_run(comm, bootstrap) == ROCSHMEM_SUCCESS) {
    DPRINTF("IPCBackend::backend_can_run returned success\n");
    return BackendType::IPC_BACKEND;
  }
  if (GDABackend::backend_can_run() == ROCSHMEM_SUCCESS) {
    DPRINTF("GDABackend::backend_can_run returned success\n");
    return BackendType::GDA_BACKEND;
  }
  if (ROBackend::backend_can_run() == ROCSHMEM_SUCCESS) {
    DPRINTF("MPIInstance could dl_init MPI library\n");
    return BackendType::RO_BACKEND;
  }

  fprintf(stderr, "No backend capable of executing the job. This is most likely\n"
                  "a system or runtime configuration error. Aborting.\n");
  exit(1);

  // Return code left in to satisfy compiler.
  return BackendType::IPC_BACKEND;
}
#endif
static void setFilesLimit() {
  rlimit filesLimit;
  if (getrlimit(RLIMIT_NOFILE, &filesLimit) != 0) {
    DPRINTF("getrlimit failed\n");
    return;
  }
  filesLimit.rlim_cur = filesLimit.rlim_max;
  if (setrlimit(RLIMIT_NOFILE, &filesLimit) != 0) {
    DPRINTF("setrlimit failed\n");
    return;
  }
}

[[maybe_unused]] __host__ void inline library_init(MPI_Comm comm) {
  assert(!backend);
  int count = 0;
  CHECK_HIP(hipGetDeviceCount(&count));

  if (count == 0) {
    printf("No GPU found! \n");
    abort();
  }

  setFilesLimit();
  rocm_init();

  int ret;
  ret = MPIInstance::mpilib_dl_init();
  if (ret != ROCSHMEM_SUCCESS) {
    fprintf(stderr, "Could not initialize MPI library. This initialization method of "
            "rocSHMEM requires MPI library to be loaded at runtime. Aborting.\n");
    exit(1);
  }
  mpi_instance = new MPIInstance(comm);

#if defined(USE_GDA) && defined(USE_RO) && defined(USE_IPC)
  BackendType type = select_backend_type(comm, nullptr);
  switch (type) {
  case BackendType::GDA_BACKEND:
    DPRINTF("Initializing GDA backend using MPI\n");
    CHECK_HIP(hipHostMalloc(&backend, sizeof(GDABackend)));
    backend = new (backend) GDABackend(comm);
    break;
  case BackendType::RO_BACKEND:
    DPRINTF("Initializing RO backend using MPI\n");
    CHECK_HIP(hipHostMalloc(&backend, sizeof(ROBackend)));
    backend = new (backend) ROBackend(comm);
    break;
  case BackendType::IPC_BACKEND:
    DPRINTF("Initializing IPC backend using MPI\n");
    CHECK_HIP(hipHostMalloc(&backend, sizeof(IPCBackend)));
    backend = new (backend) IPCBackend(comm);
    break;
  }
#elif defined(USE_GDA)
  CHECK_HIP(hipHostMalloc(&backend, sizeof(GDABackend)));
  backend = new (backend) GDABackend(comm);
#elif defined(USE_RO)
  CHECK_HIP(hipHostMalloc(&backend, sizeof(ROBackend)));
  backend = new (backend) ROBackend(comm);
#elif defined(USE_IPC)
  CHECK_HIP(hipHostMalloc(&backend, sizeof(IPCBackend)));
  backend = new (backend) IPCBackend(comm);
#endif

  if (!backend) {
    printf("No Backend could be initialized! Aborting.\n");
    exit(1);
  }
}

[[maybe_unused]] __host__ static void inline library_init_subcomm(TcpBootstrap *bootstrap, int nranks, int rank) {
  int initialized;
  int world_size = -1;

  int ret;
  ret = MPIInstance::mpilib_dl_init();
  if (ret != ROCSHMEM_SUCCESS) {
    fprintf(stderr, "Could not initialize MPI library. This initialization method of "
            "rocSHMEM requires MPI library to be loaded at runtime. Aborting.\n");
    exit(1);
  }
  mpilib_ftable_.Initialized(&initialized);

  if (initialized) {
    mpilib_ftable_.Comm_size (MPI_COMM_WORLD, &world_size);
  } else {
    // This is an Open MPI specific solution to retrieve the number of
    // processes that have been started, value can be checked before MPI_Init
    char *value = getenv("OMPI_COMM_WORLD_SIZE");
    if (value != NULL) {
      world_size = atoi(value);
    }
    if (world_size != nranks) {
      // This solution will require MPI_Sessions. This is planned for the
      // future, but is not supported in the current version.
      fprintf(stderr, "Unsupported configuration to initialize rocSHMEM. Please "
              "initialize the MPI library using MPI_Init first, if you want to "
              "initialize rocSHMEM with a subset of the processes\n");
      exit(1);
    }
  }

  if (world_size == nranks) {
    library_init(MPI_COMM_WORLD);
  } else {
    MPI_Group world_group;
    int world_rank;

    mpilib_ftable_.Comm_rank (MPI_COMM_WORLD, &world_rank);
    mpilib_ftable_.Comm_group (MPI_COMM_WORLD, &world_group);

    int *inc_ranks = new int[nranks];
    inc_ranks[rank] = world_rank;

    bootstr->allGather (inc_ranks, sizeof(int));

    MPI_Group sub_group;
    MPI_Comm sub_comm;
    mpilib_ftable_.Group_incl (world_group, nranks, inc_ranks, &sub_group);
    mpilib_ftable_.Comm_create_group (MPI_COMM_WORLD, sub_group, 1234, &sub_comm);

    library_init(sub_comm);

    mpilib_ftable_.Group_free (&sub_group);
    mpilib_ftable_.Group_free (&world_group);
    mpilib_ftable_.Comm_free (&sub_comm);
    delete[] inc_ranks;
  }
}

[[maybe_unused]] __host__ void inline library_init(TcpBootstrap *bootstrap) {
  assert(!backend);
  int count = 0;
  CHECK_HIP(hipGetDeviceCount(&count));

  if (count == 0) {
    printf("No GPU found! \n");
    abort();
  }

  setFilesLimit();
  rocm_init();

#if defined(USE_GDA) && defined(USE_RO) && defined(USE_IPC)
  BackendType type = select_backend_type(MPI_COMM_NULL, bootstrap);
  switch (type) {
  case BackendType::GDA_BACKEND:
    DPRINTF("Initializing GDA backend with TCP bootstrapping\n");
    CHECK_HIP(hipHostMalloc(&backend, sizeof(GDABackend)));
    backend = new (backend) GDABackend(bootstrap);
    break;
  case BackendType::RO_BACKEND:
    DPRINTF("Initializing RO backend with TCP bootstrapping\n");
    library_init_subcomm(bootstr, bootstr->getNranks(), bootstr->getRank());
    break;
  case BackendType::IPC_BACKEND:
    DPRINTF("Initializing IPC backend with TCP bootstrapping\n");
    CHECK_HIP(hipHostMalloc(&backend, sizeof(IPCBackend)));
    backend = new (backend) IPCBackend(bootstrap);
    break;
  }
#elif defined(USE_GDA)
  CHECK_HIP(hipHostMalloc(&backend, sizeof(GDABackend)));
  backend = new (backend) GDABackend(bootstrap);
#elif defined(USE_RO)
  library_init_subcomm(bootstr, bootstr->getNranks(), bootstr->getRank());
#elif defined(USE_IPC)
  CHECK_HIP(hipHostMalloc(&backend, sizeof(IPCBackend)));
  backend = new (backend) IPCBackend(bootstrap);
#endif

  if (!backend) {
    printf("No Backend could be initialized! Aborting.\n");
    exit(1);
  }
}

[[maybe_unused]] __host__ int rocshmem_init_attr(unsigned int flags,
                                                 rocshmem_init_attr_t *attr) {
  MPI_Comm comm;

  if ((attr == nullptr) ||
      ((flags != ROCSHMEM_INIT_WITH_UNIQUEID) &&
       (flags != ROCSHMEM_INIT_WITH_MPI_COMM)) ) {
    fprintf(stderr, "ROCSHMEM_ERROR: %s in file '%s' in line %d\n",
            "Call 'rocshmem_init_attr: invalid input argument'",
            __FILE__, __LINE__);
    return ROCSHMEM_ERROR;
  }

  if (flags == ROCSHMEM_INIT_WITH_MPI_COMM) {
    comm = *(static_cast<MPI_Comm*>(attr->mpi_comm));
    library_init(comm);
    return ROCSHMEM_SUCCESS;
  }

  if (flags == ROCSHMEM_INIT_WITH_UNIQUEID) {
    assert (attr->nranks > 0);
    assert (attr->rank >= 0);
    assert (attr->rank < attr->nranks);

    bootstr = new TcpBootstrap(attr->rank, attr->nranks);
    bootstr->initialize(attr->uid, envvar::bootstrap::timeout);

    if (envvar::uniqueid_with_mpi) {
      library_init_subcomm(bootstr, attr->nranks, attr->rank);
    } else {
      library_init(bootstr);
    }
  }

  return ROCSHMEM_SUCCESS;
}

[[maybe_unused]] __host__ int rocshmem_set_attr_uniqueid_args(int rank, int nranks,
                                                              rocshmem_uniqueid_t *uid,
                                                              rocshmem_init_attr_t *attr) {
  if (uid == nullptr || attr == nullptr) {
      fprintf(stderr, "ROCSHMEM_ERROR: %s in file '%s' in line %d\n",
              "Call 'rocshmem_get_uniqueid: invalid input argument'",
              __FILE__, __LINE__);
      return ROCSHMEM_ERROR;
  }

  attr->rank = rank;
  attr->nranks = nranks;
  attr->uid = *uid;
  attr->mpi_comm = nullptr;

  return ROCSHMEM_SUCCESS;
}

// Note: this function will be called before rocshmem_init_*, so one
// cannot assume that a backend is already set
[[maybe_unused]] __host__ int rocshmem_get_uniqueid(rocshmem_uniqueid_t *uid) {
  rocshmem_uniqueid_t tuid;
  if (uid == nullptr) {
      fprintf(stderr, "ROCSHMEM_ERROR: %s in file '%s' in line %d\n",
              "Call 'rocshmem_get_uniqueid: invalid input argument'",
              __FILE__, __LINE__);
      return ROCSHMEM_ERROR;
  }

  tuid = TcpBootstrap::createUniqueId();
  *uid = tuid;

  return ROCSHMEM_SUCCESS;
}

#if defined(HAVE_EXTERNAL_MPI)
[[maybe_unused]] __host__ void rocshmem_init(MPI_Comm comm) {
  library_init(comm);
}
#endif

[[maybe_unused]] __host__ void rocshmem_init() {
  auto ret = MPIInstance::mpilib_dl_init();
  if (ret != ROCSHMEM_SUCCESS) {
    fprintf(stderr, "Could not initialize MPI library. This initialization method of "
            "rocSHMEM requires MPI library to be loaded at runtime. Aborting.\n");
    exit(1);
  }
  library_init(MPI_COMM_WORLD);
}

#if defined(HAVE_EXTERNAL_MPI)
[[maybe_unused]] __host__ int rocshmem_init_thread(
    [[maybe_unused]] int required, int *provided, MPI_Comm comm) {
  if (comm == static_cast<MPI_Comm>(0) || comm == MPI_COMM_NULL) {
    comm = MPI_COMM_WORLD;
  }
  library_init(comm);
  rocshmem_query_thread(provided);

  return ROCSHMEM_SUCCESS;
}
#endif

[[maybe_unused]] __host__ int rocshmem_my_pe() {
  if (backend != nullptr) {
    return backend->getMyPE();
  }

  fprintf(stderr, "[WARNING] rocshmem_init() has not been called\n");
  return -1;
}

[[maybe_unused]] __host__ int rocshmem_n_pes() {
  if (backend != nullptr) {
    return backend->getNumPEs();
  }

  fprintf(stderr, "[WARNING] rocshmem_init() has not been called\n");
  return -1;
}

[[maybe_unused]] __host__ void *rocshmem_malloc(size_t size) {
  VERIFY_BACKEND();

  void *ptr;
  backend->heap.malloc(&ptr, size);
  rocshmem_barrier_all();

  return ptr;
}

[[maybe_unused]] __host__ void rocshmem_free(void *ptr) {
  VERIFY_BACKEND();

  rocshmem_barrier_all();

  backend->heap.free(ptr);
}

__host__ void * rocshmem_ptr(const void * dest, int pe){

  Context *ctx = reinterpret_cast<Context *>(ROCSHMEM_HOST_CTX_DEFAULT.ctx_opaque);

  return ctx->shmem_ptr(dest, pe);
}

[[maybe_unused]] __host__ void rocshmem_reset_stats() {
  VERIFY_BACKEND();
  backend->reset_stats();
}

[[maybe_unused]] __host__ void rocshmem_dump_stats() {
  /** TODO: Many stats are backend independent! **/
  VERIFY_BACKEND();
  backend->dump_stats();
}

[[maybe_unused]] __host__ void rocshmem_finalize() {
  VERIFY_BACKEND();

  /*
   * Destroy all the ctxs that the user
   * created but did not manually destroy
   */
  backend->destroy_remaining_ctxs();

  /*
   * Destroy all the teams that the user
   * created but did not manually destroy
   */
  auto team_destroy{
      std::bind(&Backend::team_destroy, backend, std::placeholders::_1)};
  backend->team_tracker.destroy_all(team_destroy);

  backend->~Backend();
  CHECK_HIP(hipHostFree(backend));

  if (mpi_instance != nullptr)
    delete mpi_instance;

  if (bootstr != nullptr)
    delete bootstr;

  //TODO This crashes
  //MPIInstance::mpilib_dl_close();
}

__host__ void rocshmem_query_thread(int *provided) {
  /*
   * Host-facing functions always support full
   * thread flexibility i.e. THREAD_MULTIPLE.
   */
  *provided = ROCSHMEM_THREAD_MULTIPLE;
}

__host__ void rocshmem_global_exit(int status) {
  VERIFY_BACKEND();
  backend->global_exit(status);
}

/******************************************************************************
 ****************************** Teams Interface *******************************
 *****************************************************************************/

__host__ int rocshmem_team_n_pes(rocshmem_team_t team) {
  if (team == ROCSHMEM_TEAM_INVALID) {
    return -1;
  } else {
    return get_internal_team(team)->num_pes;
  }
}

__host__ int rocshmem_team_my_pe(rocshmem_team_t team) {
  if (team == ROCSHMEM_TEAM_INVALID) {
    return -1;
  } else {
    return get_internal_team(team)->my_pe;
  }
}

__host__ inline int pe_in_active_set(int start, int stride, int size, int pe) {
  /* Active set triplet is described with respect to team world */

  int translated_pe = (pe - start) / stride;

  if ((pe < start) || ((pe - start) % stride) || (translated_pe >= size)) {
    translated_pe = -1;
  }

  return translated_pe;
}

__host__ int rocshmem_team_split_strided(
    rocshmem_team_t parent_team, int start, int stride, int size,
    [[maybe_unused]] const rocshmem_team_config_t *config,
    [[maybe_unused]] long config_mask, rocshmem_team_t *new_team) {
  VERIFY_BACKEND();

  *new_team = ROCSHMEM_TEAM_INVALID;

  auto num_user_teams{backend->team_tracker.get_num_user_teams()};
  auto max_num_teams{backend->team_tracker.get_max_num_teams()};
  if (num_user_teams >= max_num_teams - 1) {
    /* Exceeded maximum number of teams */
    return -1;
  }

  if (parent_team == ROCSHMEM_TEAM_INVALID) {
    return 0;  // TODO(bpotter): is this the right return value?
  }

  Team *parent_team_obj = get_internal_team(parent_team);

  /* Santity check inputs */
  if (start < 0 || start >= parent_team_obj->num_pes || size < 1 ||
      size > parent_team_obj->num_pes || stride < 1) {
    return -1;
  }

  /* Calculate pe_start, stride, and pe_end wrt team world */
  int pe_start_in_world = parent_team_obj->get_pe_in_world(start);
  int stride_in_world = stride * parent_team_obj->tinfo_wrt_world->stride;
  int pe_end_in_world = pe_start_in_world + stride_in_world * (size - 1);

  /* Check if size is out of bounds */
  if (pe_end_in_world > backend->num_pes) {
    return -1;
  }

  /* Calculate my PE in the new team */
  int my_pe_in_world = backend->my_pe;
  int my_pe_in_new_team = pe_in_active_set(pe_start_in_world, stride_in_world,
                                           size, my_pe_in_world);

  /* Create team infos */
  TeamInfo *team_info_wrt_parent, *team_info_wrt_world;

  CHECK_HIP(hipMalloc(&team_info_wrt_parent, sizeof(TeamInfo)));
  new (team_info_wrt_parent) TeamInfo(parent_team_obj, start, stride, size);

  auto *team_world{backend->team_tracker.get_team_world()};
  CHECK_HIP(hipMalloc(&team_info_wrt_world, sizeof(TeamInfo)));
  new (team_info_wrt_world)
      TeamInfo(team_world, pe_start_in_world, stride_in_world, size);

  MPI_Comm team_comm{MPI_COMM_NULL};
  if (parent_team_obj->mpi_comm != MPI_COMM_NULL &&
      parent_team_obj->mpi_comm != static_cast<MPI_Comm>(0)) {
    /* Create a new MPI communicator for this team */
    int color;
    if (my_pe_in_new_team < 0) {
      color = MPI_UNDEFINED;
    } else {
      color = 1;
    }

    mpilib_ftable_.Comm_split(parent_team_obj->mpi_comm, color, my_pe_in_world, &team_comm);
}
  /**
   * Allocate new team for GPU-inittiated communication with backend-specific
   * objects
   * TODO: are there any backend specific objects?
   */

  if (my_pe_in_new_team < 0) {
    *new_team = ROCSHMEM_TEAM_INVALID;
  } else {
    backend->create_new_team(parent_team_obj, team_info_wrt_parent,
                             team_info_wrt_world, size, my_pe_in_new_team,
                             team_comm, new_team);

    /* Track the newly created team to destroy it in finalize if the user does
     * not */
    backend->team_tracker.track(*new_team);
  }

  if (team_comm != MPI_COMM_NULL && team_comm != static_cast<MPI_Comm>(0)) {
    mpilib_ftable_.Comm_free (&team_comm);
  }
  return 0;
}

__host__ void rocshmem_team_destroy(rocshmem_team_t team) {
  if (team == ROCSHMEM_TEAM_INVALID || team == ROCSHMEM_TEAM_WORLD) {
    /* Do nothing */
    return;
  }

  backend->team_tracker.untrack(team);

  backend->team_destroy(team);
}

__host__ int rocshmem_team_translate_pe(rocshmem_team_t src_team, int src_pe,
                                         rocshmem_team_t dst_team) {
  return team_translate_pe(src_team, src_pe, dst_team);
}

/******************************************************************************
 ************************** Default Context Wrappers **************************
 *****************************************************************************/

template <typename T>
__host__ void rocshmem_put(T *dest, const T *source, size_t nelems, int pe) {
  rocshmem_put(ROCSHMEM_HOST_CTX_DEFAULT, dest, source, nelems, pe);
}

__host__ void rocshmem_putmem(void *dest, const void *source, size_t nelems,
                               int pe) {
  rocshmem_ctx_putmem(ROCSHMEM_HOST_CTX_DEFAULT, dest, source, nelems, pe);
}

template <typename T>
__host__ void rocshmem_p(T *dest, T value, int pe) {
  rocshmem_p(ROCSHMEM_HOST_CTX_DEFAULT, dest, value, pe);
}

template <typename T>
__host__ void rocshmem_get(T *dest, const T *source, size_t nelems, int pe) {
  rocshmem_get(ROCSHMEM_HOST_CTX_DEFAULT, dest, source, nelems, pe);
}

__host__ void rocshmem_getmem(void *dest, const void *source, size_t nelems,
                               int pe) {
  rocshmem_ctx_getmem(ROCSHMEM_HOST_CTX_DEFAULT, dest, source, nelems, pe);
}

template <typename T>
__host__ T rocshmem_g(const T *source, int pe) {
  return rocshmem_g(ROCSHMEM_HOST_CTX_DEFAULT, source, pe);
}

template <typename T>
__host__ void rocshmem_put_nbi(T *dest, const T *source, size_t nelems,
                                int pe) {
  rocshmem_put_nbi(ROCSHMEM_HOST_CTX_DEFAULT, dest, source, nelems, pe);
}

__host__ void rocshmem_putmem_nbi(void *dest, const void *source,
                                   size_t nelems, int pe) {
  rocshmem_ctx_putmem_nbi(ROCSHMEM_HOST_CTX_DEFAULT, dest, source, nelems,
                           pe);
}

template <typename T>
__host__ void rocshmem_get_nbi(T *dest, const T *source, size_t nelems,
                                int pe) {
  rocshmem_get_nbi(ROCSHMEM_HOST_CTX_DEFAULT, dest, source, nelems, pe);
}

__host__ void rocshmem_getmem_nbi(void *dest, const void *source,
                                   size_t nelems, int pe) {
  rocshmem_ctx_getmem_nbi(ROCSHMEM_HOST_CTX_DEFAULT, dest, source, nelems,
                           pe);
}

template <typename T>
__host__ T rocshmem_atomic_fetch_add(T *dest, T val, int pe) {
  return rocshmem_atomic_fetch_add(ROCSHMEM_HOST_CTX_DEFAULT, dest, val, pe);
}

template <typename T>
__host__ T rocshmem_atomic_compare_swap(T *dest, T cond, T val, int pe) {
  return rocshmem_atomic_compare_swap(ROCSHMEM_HOST_CTX_DEFAULT, dest, cond,
                                       val, pe);
}

template <typename T>
__host__ T rocshmem_atomic_fetch_inc(T *dest, int pe) {
  return rocshmem_atomic_fetch_inc(ROCSHMEM_HOST_CTX_DEFAULT, dest, pe);
}

template <typename T>
__host__ T rocshmem_atomic_fetch(T *source, int pe) {
  return rocshmem_atomic_fetch(ROCSHMEM_HOST_CTX_DEFAULT, source, pe);
}

template <typename T>
__host__ void rocshmem_atomic_add(T *dest, T val, int pe) {
  rocshmem_atomic_add(ROCSHMEM_HOST_CTX_DEFAULT, dest, val, pe);
}

template <typename T>
__host__ void rocshmem_atomic_inc(T *dest, int pe) {
  rocshmem_atomic_inc(ROCSHMEM_HOST_CTX_DEFAULT, dest, pe);
}

template <typename T>
__host__ void rocshmem_atomic_set(T *dest, T val, int pe) {
  rocshmem_atomic_set(ROCSHMEM_HOST_CTX_DEFAULT, dest, val, pe);
}

template <typename T>
__host__ T rocshmem_atomic_swap(T *dest, T value, int pe) {
  return rocshmem_atomic_swap(ROCSHMEM_HOST_CTX_DEFAULT, dest, value, pe);
}

template <typename T>
__host__ T rocshmem_atomic_fetch_and(T *dest, T value, int pe) {
  return rocshmem_atomic_fetch_and(ROCSHMEM_HOST_CTX_DEFAULT, dest, value,
                                    pe);
}

template <typename T>
__host__ void rocshmem_atomic_and(T *dest, T value, int pe) {
  rocshmem_atomic_and(ROCSHMEM_HOST_CTX_DEFAULT, dest, value, pe);
}

template <typename T>
__host__ T rocshmem_atomic_fetch_or(T *dest, T value, int pe) {
  return rocshmem_atomic_fetch_or(ROCSHMEM_HOST_CTX_DEFAULT, dest, value, pe);
}

template <typename T>
__host__ void rocshmem_atomic_or(T *dest, T value, int pe) {
  rocshmem_atomic_or(ROCSHMEM_HOST_CTX_DEFAULT, dest, value, pe);
}

template <typename T>
__host__ T rocshmem_atomic_fetch_xor(T *dest, T value, int pe) {
  return rocshmem_atomic_fetch_xor(ROCSHMEM_HOST_CTX_DEFAULT, dest, value,
                                    pe);
}

template <typename T>
__host__ void rocshmem_atomic_xor(T *dest, T value, int pe) {
  rocshmem_atomic_xor(ROCSHMEM_HOST_CTX_DEFAULT, dest, value, pe);
}

__host__ void rocshmem_fence() {
  rocshmem_ctx_fence(ROCSHMEM_HOST_CTX_DEFAULT);
}

__host__ void rocshmem_quiet() {
  rocshmem_ctx_quiet(ROCSHMEM_HOST_CTX_DEFAULT);
}

/******************************************************************************
 ************************* Private Context Interfaces *************************
 *****************************************************************************/

__host__ Context *get_internal_ctx(rocshmem_ctx_t ctx) {
  return reinterpret_cast<Context *>(ctx.ctx_opaque);
}

__host__ int rocshmem_ctx_create(int64_t options, rocshmem_ctx_t *ctx) {
  DPRINTF("Host function: rocshmem_ctx_create\n");

  void *phys_ctx;
  backend->ctx_create(options, &phys_ctx);

  ctx->ctx_opaque = phys_ctx;
  /* This team in on TEAM_WORLD, no need for team info */
  ctx->team_opaque = nullptr;

  /* Track this context, if needed. */
  backend->track_ctx(reinterpret_cast<Context *>(phys_ctx));

  return 0;
}

__host__ void rocshmem_ctx_destroy(rocshmem_ctx_t ctx) {
  DPRINTF("Host function: rocshmem_ctx_destroy\n");

  /* TODO: Implicit quiet on this context */

  Context *phys_ctx = get_internal_ctx(ctx);

  backend->untrack_ctx(phys_ctx);

  backend->ctx_destroy(phys_ctx);
}

template <typename T>
__host__ void rocshmem_put(rocshmem_ctx_t ctx, T *dest, const T *source,
                            size_t nelems, int pe) {
  DPRINTF("Host function: rocshmem_put\n");

  get_internal_ctx(ctx)->put(dest, source, nelems, pe);
}

__host__ void rocshmem_ctx_putmem(rocshmem_ctx_t ctx, void *dest,
                                   const void *source, size_t nelems, int pe) {
  DPRINTF("Host function: rocshmem_ctx_putmem\n");

  get_internal_ctx(ctx)->putmem(dest, source, nelems, pe);
}

template <typename T>
__host__ void rocshmem_p(rocshmem_ctx_t ctx, T *dest, T value, int pe) {
  DPRINTF("Host function: rocshmem_p\n");

  get_internal_ctx(ctx)->p(dest, value, pe);
}

template <typename T>
__host__ void rocshmem_get(rocshmem_ctx_t ctx, T *dest, const T *source,
                            size_t nelems, int pe) {
  DPRINTF("Host function: rocshmem_get\n");

  get_internal_ctx(ctx)->get(dest, source, nelems, pe);
}

__host__ void rocshmem_ctx_getmem(rocshmem_ctx_t ctx, void *dest,
                                   const void *source, size_t nelems, int pe) {
  DPRINTF("Host function: rocshmem_ctx_getmem\n");

  get_internal_ctx(ctx)->getmem(dest, source, nelems, pe);
}

template <typename T>
__host__ T rocshmem_g(rocshmem_ctx_t ctx, const T *source, int pe) {
  DPRINTF("Host function: rocshmem_g\n");

  return get_internal_ctx(ctx)->g(source, pe);
}

template <typename T>
__host__ void rocshmem_put_nbi(rocshmem_ctx_t ctx, T *dest, const T *source,
                                size_t nelems, int pe) {
  DPRINTF("Host function: rocshmem_put_nbi\n");

  get_internal_ctx(ctx)->put_nbi(dest, source, nelems, pe);
}

__host__ void rocshmem_ctx_putmem_nbi(rocshmem_ctx_t ctx, void *dest,
                                       const void *source, size_t nelems,
                                       int pe) {
  DPRINTF("Host function: rocshmem_ctx_putmem_nbi\n");

  get_internal_ctx(ctx)->putmem_nbi(dest, source, nelems, pe);
}

template <typename T>
__host__ void rocshmem_get_nbi(rocshmem_ctx_t ctx, T *dest, const T *source,
                                size_t nelems, int pe) {
  DPRINTF("Host function: rocshmem_get_nbi\n");

  get_internal_ctx(ctx)->get_nbi(dest, source, nelems, pe);
}

__host__ void rocshmem_ctx_getmem_nbi(rocshmem_ctx_t ctx, void *dest,
                                       const void *source, size_t nelems,
                                       int pe) {
  DPRINTF("Host function: rocshmem_ctx_getmem_nbi\n");

  get_internal_ctx(ctx)->getmem_nbi(dest, source, nelems, pe);
}

template <typename T>
__host__ T rocshmem_atomic_fetch_add(rocshmem_ctx_t ctx, T *dest, T val,
                                      int pe) {
  DPRINTF("Host function: rocshmem_atomic_fetch_add\n");

  return get_internal_ctx(ctx)->amo_fetch_add<T>(dest, val, pe);
}

template <typename T>
__host__ T rocshmem_atomic_compare_swap(rocshmem_ctx_t ctx, T *dest, T cond,
                                         T val, int pe) {
  DPRINTF("Host function: rocshmem_atomic_compare_swap\n");

  return get_internal_ctx(ctx)->amo_fetch_cas(dest, val, cond, pe);
}

template <typename T>
__host__ T rocshmem_atomic_fetch_inc(rocshmem_ctx_t ctx, T *dest, int pe) {
  DPRINTF("Host function: rocshmem_atomic_fetch_inc\n");

  return get_internal_ctx(ctx)->amo_fetch_add<T>(dest, 1, pe);
}

template <typename T>
__host__ T rocshmem_atomic_fetch(rocshmem_ctx_t ctx, T *source, int pe) {
  DPRINTF("Host function: rocshmem_atomic_fetch\n");

  return get_internal_ctx(ctx)->amo_fetch_add<T>(source, 0, pe);
}

template <typename T>
__host__ void rocshmem_atomic_add(rocshmem_ctx_t ctx, T *dest, T val,
                                   int pe) {
  DPRINTF("Host function: rocshmem_atomic_add\n");

  get_internal_ctx(ctx)->amo_add<T>(dest, val, pe);
}

template <typename T>
__host__ void rocshmem_atomic_inc(rocshmem_ctx_t ctx, T *dest, int pe) {
  DPRINTF("Host function: rocshmem_atomic_inc\n");

  get_internal_ctx(ctx)->amo_add<T>(dest, 1, pe);
}

template <typename T>
__host__ void rocshmem_atomic_set(rocshmem_ctx_t ctx, T *dest, T val,
                                   int pe) {
  DPRINTF("Host function: rocshmem_atomic_set\n");

  get_internal_ctx(ctx)->amo_set(dest, val, pe);
}

template <typename T>
__host__ T rocshmem_atomic_swap(rocshmem_ctx_t ctx, T *dest, T val, int pe) {
  DPRINTF("Host function: rocshmem_atomic_set\n");

  return get_internal_ctx(ctx)->amo_swap(dest, val, pe);
}

template <typename T>
__host__ T rocshmem_atomic_fetch_and(rocshmem_ctx_t ctx, T *dest, T val,
                                      int pe) {
  DPRINTF("Host function: rocshmem_atomic_fetch_and\n");

  return get_internal_ctx(ctx)->amo_fetch_and(dest, val, pe);
}

template <typename T>
__host__ void rocshmem_atomic_and(rocshmem_ctx_t ctx, T *dest, T val,
                                   int pe) {
  DPRINTF("Host function: rocshmem_atomic_and\n");

  get_internal_ctx(ctx)->amo_and(dest, val, pe);
}

template <typename T>
__host__ T rocshmem_atomic_fetch_or(rocshmem_ctx_t ctx, T *dest, T val,
                                     int pe) {
  DPRINTF("Host function: rocshmem_atomic_fetch_or\n");

  return get_internal_ctx(ctx)->amo_fetch_or(dest, val, pe);
}

template <typename T>
__host__ void rocshmem_atomic_or(rocshmem_ctx_t ctx, T *dest, T val, int pe) {
  DPRINTF("Host function: rocshmem_atomic_or\n");

  get_internal_ctx(ctx)->amo_or(dest, val, pe);
}

template <typename T>
__host__ T rocshmem_atomic_fetch_xor(rocshmem_ctx_t ctx, T *dest, T val,
                                      int pe) {
  DPRINTF("Host function: rocshmem_atomic_fetch_xor\n");

  return get_internal_ctx(ctx)->amo_fetch_xor(dest, val, pe);
}

template <typename T>
__host__ void rocshmem_atomic_xor(rocshmem_ctx_t ctx, T *dest, T val,
                                   int pe) {
  DPRINTF("Host function: rocshmem_atomic_xor\n");

  get_internal_ctx(ctx)->amo_xor(dest, val, pe);
}

__host__ void rocshmem_ctx_fence(rocshmem_ctx_t ctx) {
  DPRINTF("Host function: rocshmem_ctx_fence\n");

  get_internal_ctx(ctx)->fence();
}

__host__ void rocshmem_ctx_quiet(rocshmem_ctx_t ctx) {
  DPRINTF("Host function: rocshmem_ctx_quiet\n");

  get_internal_ctx(ctx)->quiet();
}

__host__ void rocshmem_barrier_all() {
  DPRINTF("Host function: rocshmem_barrier_all\n");

  get_internal_ctx(ROCSHMEM_HOST_CTX_DEFAULT)->barrier_all();
}


__host__ void rocshmem_barrier_all_on_stream(hipStream_t stream) {
  DPRINTF("Host function: rocshmem_barrier_all_on_stream\n");

  get_internal_ctx(ROCSHMEM_HOST_CTX_DEFAULT)->barrier_all_on_stream(stream);
}

__host__ void rocshmem_alltoallmem_on_stream(rocshmem_team_t team, void *dest,
                                             const void *source, size_t size,
                                             hipStream_t stream) {
  DPRINTF("Host function: rocshmem_alltoallmem_on_stream\n");

  get_internal_ctx(ROCSHMEM_HOST_CTX_DEFAULT)
      ->alltoallmem_on_stream(team, dest, source, size, stream);
}

__host__ void rocshmem_broadcastmem_on_stream(rocshmem_team_t team, void *dest,
                                              const void *source, size_t nelems,
                                              int pe_root, hipStream_t stream) {
  DPRINTF("Host function: rocshmem_broadcastmem_on_stream\n");

  get_internal_ctx(ROCSHMEM_HOST_CTX_DEFAULT)
      ->broadcastmem_on_stream(team, dest, source, nelems, pe_root, stream);
}

__host__ void rocshmem_getmem_on_stream(void *dest, const void *source,
                                        size_t nelems, int pe,
                                        hipStream_t stream) {
  DPRINTF("Host function: rocshmem_getmem_on_stream\n");

  get_internal_ctx(ROCSHMEM_HOST_CTX_DEFAULT)
      ->getmem_on_stream(dest, source, nelems, pe, stream);
}

__host__ void rocshmem_putmem_on_stream(void *dest, const void *source,
                                        size_t nelems, int pe,
                                        hipStream_t stream) {
  DPRINTF("Host function: rocshmem_putmem_on_stream\n");

  get_internal_ctx(ROCSHMEM_HOST_CTX_DEFAULT)
      ->putmem_on_stream(dest, source, nelems, pe, stream);
}

__host__ void rocshmem_putmem_signal_on_stream(void *dest, const void *source,
                                               size_t nelems,
                                               uint64_t *sig_addr,
                                               uint64_t signal, int sig_op,
                                               int pe, hipStream_t stream) {
  DPRINTF("Host function: rocshmem_putmem_signal_on_stream\n");

  get_internal_ctx(ROCSHMEM_HOST_CTX_DEFAULT)
      ->putmem_signal_on_stream(dest, source, nelems, sig_addr, signal, sig_op,
                                pe, stream);
}

__host__ void rocshmem_signal_wait_until_on_stream(uint64_t *sig_addr, int cmp,
                                                   uint64_t cmp_value,
                                                   hipStream_t stream) {
  DPRINTF("Host function: rocshmem_signal_wait_until_on_stream\n");

  get_internal_ctx(ROCSHMEM_HOST_CTX_DEFAULT)
      ->signal_wait_until_on_stream(sig_addr, cmp, cmp_value, stream);
}

__host__ void rocshmem_sync_all() {
  DPRINTF("Host function: rocshmem_sync_all\n");

  get_internal_ctx(ROCSHMEM_HOST_CTX_DEFAULT)->sync_all();
}

template <typename T>
__host__ void rocshmem_broadcast([[maybe_unused]] rocshmem_ctx_t ctx, T *dest,
                                  const T *source, int nelem, int pe_root,
                                  int pe_start, int log_pe_stride, int pe_size,
                                  long *p_sync) {
  DPRINTF("Host function: rocshmem_broadcast\n");

  get_internal_ctx(ROCSHMEM_HOST_CTX_DEFAULT)
      ->broadcast<T>(dest, source, nelem, pe_root, pe_start, log_pe_stride,
                     pe_size, p_sync);
}

template <typename T>
__host__ void rocshmem_broadcast([[maybe_unused]] rocshmem_ctx_t ctx,
                                  rocshmem_team_t team, T *dest,
                                  const T *source, int nelem, int pe_root) {
  DPRINTF("Host function: Team-based rocshmem_broadcast\n");

  get_internal_ctx(ROCSHMEM_HOST_CTX_DEFAULT)
      ->broadcast<T>(team, dest, source, nelem, pe_root);
}

template <typename T, ROCSHMEM_OP Op>
__host__ void rocshmem_to_all([[maybe_unused]] rocshmem_ctx_t ctx, T *dest,
                               const T *source, int nreduce, int PE_start,
                               int logPE_stride, int PE_size, T *pWrk,
                               long *pSync) {
  DPRINTF("Host function: rocshmem_to_all\n");

  get_internal_ctx(ROCSHMEM_HOST_CTX_DEFAULT)
      ->to_all<T, Op>(dest, source, nreduce, PE_start, logPE_stride, PE_size,
                      pWrk, pSync);
}

template <typename T, ROCSHMEM_OP Op>
__host__ int rocshmem_reduce([[maybe_unused]] rocshmem_ctx_t ctx,
                               rocshmem_team_t team, T *dest, const T *source,
                               int nreduce) {
  DPRINTF("Host function: Team-based rocshmem_reduce\n");

  return get_internal_ctx(ROCSHMEM_HOST_CTX_DEFAULT)
              ->reduce<T, Op>(team, dest, source, nreduce);
}

template <typename T>
__host__ void rocshmem_wait_until(T *ivars, int cmp, T val) {
  DPRINTF("Host function: rocshmem_wait_until\n");

  get_internal_ctx(ROCSHMEM_HOST_CTX_DEFAULT)->wait_until(ivars, cmp, val);
}

template <typename T>
__host__ void rocshmem_wait_until_all(T *ivars, size_t nelems, const int* status,
                                       int cmp, T val) {
  DPRINTF("Host function: rocshmem_wait_until_all\n");

  get_internal_ctx(ROCSHMEM_HOST_CTX_DEFAULT)->wait_until_all(ivars,
      nelems, status, cmp, val);
}

template <typename T>
__host__ size_t rocshmem_wait_until_any(T *ivars, size_t nelems, const int* status,
                                       int cmp, T val) {
  DPRINTF("Host function: rocshmem_wait_until_any\n");

  return get_internal_ctx(ROCSHMEM_HOST_CTX_DEFAULT)->wait_until_any(ivars,
      nelems, status, cmp, val);
}

template <typename T>
__host__ size_t rocshmem_wait_until_some(T *ivars, size_t nelems, size_t* indices,
                                        const int* status, int cmp,
                                        T val) {
  DPRINTF("Host function: rocshmem_wait_until_some\n");

  return get_internal_ctx(ROCSHMEM_HOST_CTX_DEFAULT)->wait_until_some(ivars, nelems,
      indices, status, cmp, val);
}

template <typename T>
__host__ size_t rocshmem_wait_until_any_vector(T *ivars, size_t nelems, const int* status,
                                                int cmp, T* vals) {
  DPRINTF("Host function: rocshmem_wait_until_any_vector\n");

  return get_internal_ctx(ROCSHMEM_HOST_CTX_DEFAULT)->wait_until_any_vector(ivars,
      nelems, status, cmp, vals);
}

template <typename T>
__host__ void rocshmem_wait_until_all_vector(T *ivars, size_t nelems, const int* status,
                                              int cmp, T* vals) {
  DPRINTF("Host function: rocshmem_wait_until_all_vector\n");

  get_internal_ctx(ROCSHMEM_HOST_CTX_DEFAULT)->wait_until_all_vector(ivars,
      nelems, status, cmp, vals);
}

template <typename T>
__host__ size_t rocshmem_wait_until_some_vector(T *ivars, size_t nelems,
                                               size_t* indices,
                                               const int* status,
                                               int cmp, T* vals) {
  DPRINTF("Host function: rocshmem_wait_until_some_vector\n");

  return get_internal_ctx(ROCSHMEM_HOST_CTX_DEFAULT)->wait_until_some_vector(ivars,
      nelems, indices, status, cmp, vals);
}

template <typename T>
__host__ int rocshmem_test(T *ivars, int cmp, T val) {
  DPRINTF("Host function: rocshmem_testl\n");

  return get_internal_ctx(ROCSHMEM_HOST_CTX_DEFAULT)->test(ivars, cmp, val);
}

/**
 * Template generator for reductions
 **/
#define REDUCTION_GEN(T, Op)                                                  \
  template __host__ void rocshmem_to_all<T, Op>(                              \
      rocshmem_ctx_t ctx, T * dest, const T *source, int nreduce,             \
      int PE_start, int logPE_stride, int PE_size, T *pWrk, long *pSync);     \
  template __host__ int rocshmem_reduce<T, Op>(                               \
      rocshmem_ctx_t ctx, rocshmem_team_t team, T * dest, const T *source,    \
      int nreduce);

#define ARITH_REDUCTION_GEN(T)    \
  REDUCTION_GEN(T, ROCSHMEM_SUM) \
  REDUCTION_GEN(T, ROCSHMEM_MIN) \
  REDUCTION_GEN(T, ROCSHMEM_MAX) \
  REDUCTION_GEN(T, ROCSHMEM_PROD)

#define BITWISE_REDUCTION_GEN(T)  \
  REDUCTION_GEN(T, ROCSHMEM_OR)  \
  REDUCTION_GEN(T, ROCSHMEM_AND) \
  REDUCTION_GEN(T, ROCSHMEM_XOR)

#define INT_REDUCTION_GEN(T) \
  ARITH_REDUCTION_GEN(T)     \
  BITWISE_REDUCTION_GEN(T)

#define FLOAT_REDUCTION_GEN(T) ARITH_REDUCTION_GEN(T)

/**
 * Declare templates for the required datatypes (for the compiler)
 **/
#define RMA_GEN(T)                                                            \
  template __host__ void rocshmem_put<T>(                                     \
      rocshmem_ctx_t ctx, T * dest, const T *source, size_t nelems, int pe);  \
  template __host__ void rocshmem_put_nbi<T>(                                 \
      rocshmem_ctx_t ctx, T * dest, const T *source, size_t nelems, int pe);  \
  template __host__ void rocshmem_p<T>(rocshmem_ctx_t ctx, T * dest,          \
                                        T value, int pe);                     \
  template __host__ void rocshmem_get<T>(                                     \
      rocshmem_ctx_t ctx, T * dest, const T *source, size_t nelems, int pe);  \
  template __host__ void rocshmem_get_nbi<T>(                                 \
      rocshmem_ctx_t ctx, T * dest, const T *source, size_t nelems, int pe);  \
  template __host__ T rocshmem_g<T>(rocshmem_ctx_t ctx, const T *source,      \
                                     int pe);                                 \
  template __host__ void rocshmem_put<T>(T * dest, const T *source,           \
                                          size_t nelems, int pe);             \
  template __host__ void rocshmem_put_nbi<T>(T * dest, const T *source,       \
                                              size_t nelems, int pe);         \
  template __host__ void rocshmem_p<T>(T * dest, T value, int pe);            \
  template __host__ void rocshmem_get<T>(T * dest, const T *source,           \
                                          size_t nelems, int pe);             \
  template __host__ void rocshmem_get_nbi<T>(T * dest, const T *source,       \
                                              size_t nelems, int pe);         \
  template __host__ T rocshmem_g<T>(const T *source, int pe);                 \
  template __host__ void rocshmem_broadcast<T>(                               \
      rocshmem_ctx_t ctx, T * dest, const T *source, int nelem, int pe_root,  \
      int pe_start, int log_pe_stride, int pe_size, long *p_sync);            \
  template __host__ void rocshmem_broadcast<T>(                               \
      rocshmem_ctx_t ctx, rocshmem_team_t team, T * dest, const T *source,    \
      int nelem, int pe_root);

/**
 * Declare templates for the standard amo types
 */
#define AMO_STANDARD_GEN(T)                                                   \
  template __host__ T rocshmem_atomic_compare_swap<T>(                        \
      rocshmem_ctx_t ctx, T * dest, T cond, T value, int pe);                 \
  template __host__ T rocshmem_atomic_compare_swap<T>(T * dest, T cond,       \
                                                       T value, int pe);      \
  template __host__ T rocshmem_atomic_fetch_inc<T>(rocshmem_ctx_t ctx,        \
                                                    T * dest, int pe);        \
  template __host__ T rocshmem_atomic_fetch_inc<T>(T * dest, int pe);         \
  template __host__ void rocshmem_atomic_inc<T>(rocshmem_ctx_t ctx,           \
                                                 T * dest, int pe);           \
  template __host__ void rocshmem_atomic_inc<T>(T * dest, int pe);            \
  template __host__ T rocshmem_atomic_fetch_add<T>(                           \
      rocshmem_ctx_t ctx, T * dest, T value, int pe);                         \
  template __host__ T rocshmem_atomic_fetch_add<T>(T * dest, T value,         \
                                                    int pe);                  \
  template __host__ void rocshmem_atomic_add<T>(rocshmem_ctx_t ctx,           \
                                                 T * dest, T value, int pe);  \
  template __host__ void rocshmem_atomic_add<T>(T * dest, T value, int pe);

/**
 * Declare templates for the extended amo types
 */
#define AMO_EXTENDED_GEN(T)                                                   \
  template __host__ T rocshmem_atomic_fetch<T>(rocshmem_ctx_t ctx, T * dest,  \
                                                int pe);                      \
  template __host__ T rocshmem_atomic_fetch<T>(T * dest, int pe);             \
  template __host__ void rocshmem_atomic_set<T>(rocshmem_ctx_t ctx,           \
                                                 T * dest, T value, int pe);  \
  template __host__ void rocshmem_atomic_set<T>(T * dest, T value, int pe);   \
  template __host__ T rocshmem_atomic_swap<T>(rocshmem_ctx_t ctx, T * dest,   \
                                               T value, int pe);              \
  template __host__ T rocshmem_atomic_swap<T>(T * dest, T value, int pe);

/**
 * Declare templates for the bitwise amo types
 */
#define AMO_BITWISE_GEN(T)                                                    \
  template __host__ T rocshmem_atomic_fetch_and<T>(                           \
      rocshmem_ctx_t ctx, T * dest, T value, int pe);                         \
  template __host__ T rocshmem_atomic_fetch_and<T>(T * dest, T value,         \
                                                    int pe);                  \
  template __host__ void rocshmem_atomic_and<T>(rocshmem_ctx_t ctx,           \
                                                 T * dest, T value, int pe);  \
  template __host__ void rocshmem_atomic_and<T>(T * dest, T value, int pe);   \
  template __host__ T rocshmem_atomic_fetch_or<T>(rocshmem_ctx_t ctx,         \
                                                   T * dest, T value, int pe);\
  template __host__ T rocshmem_atomic_fetch_or<T>(T * dest, T value, int pe); \
  template __host__ void rocshmem_atomic_or<T>(rocshmem_ctx_t ctx, T * dest,  \
                                                T value, int pe);             \
  template __host__ void rocshmem_atomic_or<T>(T * dest, T value, int pe);    \
  template __host__ T rocshmem_atomic_fetch_xor<T>(                           \
      rocshmem_ctx_t ctx, T * dest, T value, int pe);                         \
  template __host__ T rocshmem_atomic_fetch_xor<T>(T * dest, T value,         \
                                                    int pe);                  \
  template __host__ void rocshmem_atomic_xor<T>(rocshmem_ctx_t ctx,           \
                                                 T * dest, T value, int pe);  \
  template __host__ void rocshmem_atomic_xor<T>(T * dest, T value, int pe);

/**
 * Declare templates for the wait types
 */
#define WAIT_GEN(T)                                                           \
  template __host__ void rocshmem_wait_until<T>(T *ivars, int cmp,            \
                                                 T val);                      \
  template __host__ int rocshmem_test<T>(T *ivars, int cmp, T val);           \
  template __host__ void Context::wait_until<T>(T *ivars, int cmp,            \
                                                T val);                       \
  template __host__ size_t rocshmem_wait_until_any<T>(T *ivars,               \
                                      size_t nelems, const int* status,       \
                                      int cmp, T val);                        \
  template __host__ void rocshmem_wait_until_all<T>(T *ivars,                 \
                                      size_t nelems, const int* status,       \
                                      int cmp, T val);                        \
  template __host__ size_t rocshmem_wait_until_some<T>(T *ivars, size_t nelems,\
                                      size_t* indices, const int* status,     \
                                      int cmp, T val);                        \
  template __host__ size_t rocshmem_wait_until_any_vector<T>(T *ivars,        \
                                      size_t nelems, const int* status,       \
                                      int cmp, T* vals);                      \
  template __host__ void rocshmem_wait_until_all_vector<T>(T *ivars,          \
                                      size_t nelems, const int* status,       \
                                      int cmp, T* vals);                      \
  template __host__ size_t rocshmem_wait_until_some_vector<T>(T *ivars,       \
                                      size_t nelems, size_t* indices,         \
                                      const int* status, int cmp,             \
                                      T* vals);                               \
  template __host__ int Context::test<T>(T *ivars, int cmp, T val);

/**
 * Define APIs to call the template functions
 **/

#define REDUCTION_DEF_GEN(T, TNAME, Op_API, Op)                               \
  __host__ void rocshmem_ctx_##TNAME##_##Op_API##_to_all(                     \
      rocshmem_ctx_t ctx, T *dest, const T *source, int nreduce,              \
      int PE_start, int logPE_stride, int PE_size, T *pWrk, long *pSync) {    \
    rocshmem_to_all<T, Op>(ctx, dest, source, nreduce, PE_start,              \
                            logPE_stride, PE_size, pWrk, pSync);              \
  }                                                                           \
  __host__ int rocshmem_ctx_##TNAME##_##Op_API##_reduce(                      \
      rocshmem_ctx_t ctx, rocshmem_team_t team, T *dest, const T *source,     \
      int nreduce) {                                                          \
    return rocshmem_reduce<T, Op>(ctx, team, dest, source, nreduce);          \
  }

#define ARITH_REDUCTION_DEF_GEN(T, TNAME)                                     \
  REDUCTION_DEF_GEN(T, TNAME, sum, ROCSHMEM_SUM)                              \
  REDUCTION_DEF_GEN(T, TNAME, min, ROCSHMEM_MIN)                              \
  REDUCTION_DEF_GEN(T, TNAME, max, ROCSHMEM_MAX)                              \
  REDUCTION_DEF_GEN(T, TNAME, prod, ROCSHMEM_PROD)

#define BITWISE_REDUCTION_DEF_GEN(T, TNAME)                                   \
  REDUCTION_DEF_GEN(T, TNAME, or, ROCSHMEM_OR)                                \
  REDUCTION_DEF_GEN(T, TNAME, and, ROCSHMEM_AND)                              \
  REDUCTION_DEF_GEN(T, TNAME, xor, ROCSHMEM_XOR)

#define INT_REDUCTION_DEF_GEN(T, TNAME)                                       \
  ARITH_REDUCTION_DEF_GEN(T, TNAME)                                           \
  BITWISE_REDUCTION_DEF_GEN(T, TNAME)

#define FLOAT_REDUCTION_DEF_GEN(T, TNAME) ARITH_REDUCTION_DEF_GEN(T, TNAME)

#define RMA_DEF_GEN(T, TNAME)                                                 \
  __host__ void rocshmem_ctx_##TNAME##_put(                                   \
      rocshmem_ctx_t ctx, T *dest, const T *source, size_t nelems, int pe) {  \
    rocshmem_put<T>(ctx, dest, source, nelems, pe);                           \
  }                                                                           \
  __host__ void rocshmem_ctx_##TNAME##_put_nbi(                               \
      rocshmem_ctx_t ctx, T *dest, const T *source, size_t nelems, int pe) {  \
    rocshmem_put_nbi<T>(ctx, dest, source, nelems, pe);                       \
  }                                                                           \
  __host__ void rocshmem_ctx_##TNAME##_p(rocshmem_ctx_t ctx, T *dest,         \
                                          T value, int pe) {                  \
    rocshmem_p<T>(ctx, dest, value, pe);                                      \
  }                                                                           \
  __host__ void rocshmem_ctx_##TNAME##_get(                                   \
      rocshmem_ctx_t ctx, T *dest, const T *source, size_t nelems, int pe) {  \
    rocshmem_get<T>(ctx, dest, source, nelems, pe);                           \
  }                                                                           \
  __host__ void rocshmem_ctx_##TNAME##_get_nbi(                               \
      rocshmem_ctx_t ctx, T *dest, const T *source, size_t nelems, int pe) {  \
    rocshmem_get_nbi<T>(ctx, dest, source, nelems, pe);                       \
  }                                                                           \
  __host__ T rocshmem_ctx_##TNAME##_g(rocshmem_ctx_t ctx, const T *source,    \
                                       int pe) {                              \
    return rocshmem_g<T>(ctx, source, pe);                                    \
  }                                                                           \
  __host__ void rocshmem_##TNAME##_put(T *dest, const T *source,              \
                                        size_t nelems, int pe) {              \
    rocshmem_put<T>(dest, source, nelems, pe);                                \
  }                                                                           \
  __host__ void rocshmem_##TNAME##_put_nbi(T *dest, const T *source,          \
                                            size_t nelems, int pe) {          \
    rocshmem_put_nbi<T>(dest, source, nelems, pe);                            \
  }                                                                           \
  __host__ void rocshmem_##TNAME##_p(T *dest, T value, int pe) {              \
    rocshmem_p<T>(dest, value, pe);                                           \
  }                                                                           \
  __host__ void rocshmem_##TNAME##_get(T *dest, const T *source,              \
                                        size_t nelems, int pe) {              \
    rocshmem_get<T>(dest, source, nelems, pe);                                \
  }                                                                           \
  __host__ void rocshmem_##TNAME##_get_nbi(T *dest, const T *source,          \
                                            size_t nelems, int pe) {          \
    rocshmem_get_nbi<T>(dest, source, nelems, pe);                            \
  }                                                                           \
  __host__ T rocshmem_##TNAME##_g(const T *source, int pe) {                  \
    return rocshmem_g<T>(source, pe);                                         \
  }                                                                           \
  __host__ void rocshmem_ctx_##TNAME##_broadcast(                             \
      rocshmem_ctx_t ctx, T *dest, const T *source, int nelem, int pe_root,   \
      int pe_start, int log_pe_stride, int pe_size, long *p_sync) {           \
    rocshmem_broadcast<T>(ctx, dest, source, nelem, pe_root, pe_start,        \
                           log_pe_stride, pe_size, p_sync);                   \
  }                                                                           \
  __host__ void rocshmem_ctx_##TNAME##_broadcast(                             \
      rocshmem_ctx_t ctx, rocshmem_team_t team, T *dest, const T *source,     \
      int nelem, int pe_root) {                                               \
    rocshmem_broadcast<T>(ctx, team, dest, source, nelem, pe_root);           \
  }

#define AMO_STANDARD_DEF_GEN(T, TNAME)                                        \
  __host__ T rocshmem_ctx_##TNAME##_atomic_compare_swap(                      \
      rocshmem_ctx_t ctx, T *dest, T cond, T value, int pe) {                 \
    return rocshmem_atomic_compare_swap<T>(ctx, dest, cond, value, pe);       \
  }                                                                           \
  __host__ T rocshmem_##TNAME##_atomic_compare_swap(T *dest, T cond, T value, \
                                                     int pe) {                \
    return rocshmem_atomic_compare_swap<T>(dest, cond, value, pe);            \
  }                                                                           \
  __host__ T rocshmem_ctx_##TNAME##_atomic_fetch_inc(rocshmem_ctx_t ctx,      \
                                                      T *dest, int pe) {      \
    return rocshmem_atomic_fetch_inc<T>(ctx, dest, pe);                       \
  }                                                                           \
  __host__ T rocshmem_##TNAME##_atomic_fetch_inc(T *dest, int pe) {           \
    return rocshmem_atomic_fetch_inc<T>(dest, pe);                            \
  }                                                                           \
  __host__ void rocshmem_ctx_##TNAME##_atomic_inc(rocshmem_ctx_t ctx,         \
                                                   T *dest, int pe) {         \
    rocshmem_atomic_inc<T>(ctx, dest, pe);                                    \
  }                                                                           \
  __host__ void rocshmem_##TNAME##_atomic_inc(T *dest, int pe) {              \
    rocshmem_atomic_inc<T>(dest, pe);                                         \
  }                                                                           \
  __host__ T rocshmem_ctx_##TNAME##_atomic_fetch_add(                         \
      rocshmem_ctx_t ctx, T *dest, T value, int pe) {                         \
    return rocshmem_atomic_fetch_add<T>(ctx, dest, value, pe);                \
  }                                                                           \
  __host__ T rocshmem_##TNAME##_atomic_fetch_add(T *dest, T value, int pe) {  \
    return rocshmem_atomic_fetch_add<T>(dest, value, pe);                     \
  }                                                                           \
  __host__ void rocshmem_ctx_##TNAME##_atomic_add(rocshmem_ctx_t ctx,         \
                                                   T *dest, T value, int pe) { \
    rocshmem_atomic_add<T>(ctx, dest, value, pe);                             \
  }                                                                           \
  __host__ void rocshmem_##TNAME##_atomic_add(T *dest, T value, int pe) {     \
    rocshmem_atomic_add<T>(dest, value, pe);                                  \
  }

#define AMO_EXTENDED_DEF_GEN(T, TNAME)                                        \
  __host__ T rocshmem_ctx_##TNAME##_atomic_fetch(rocshmem_ctx_t ctx,          \
                                                  T *source, int pe) {        \
    return rocshmem_atomic_fetch<T>(ctx, source, pe);                         \
  }                                                                           \
  __host__ T rocshmem_##TNAME##_atomic_fetch(T *source, int pe) {             \
    return rocshmem_atomic_fetch<T>(source, pe);                              \
  }                                                                           \
  __host__ void rocshmem_ctx_##TNAME##_atomic_set(rocshmem_ctx_t ctx,         \
                                                   T *dest, T value, int pe) {\
    rocshmem_atomic_set<T>(ctx, dest, value, pe);                             \
  }                                                                           \
  __host__ void rocshmem_##TNAME##_atomic_set(T *dest, T value, int pe) {     \
    rocshmem_atomic_set<T>(dest, value, pe);                                  \
  }                                                                           \
  __host__ T rocshmem_ctx_##TNAME##_atomic_swap(rocshmem_ctx_t ctx, T *dest,  \
                                                 T value, int pe) {           \
    return rocshmem_atomic_swap<T>(ctx, dest, value, pe);                     \
  }                                                                           \
  __host__ T rocshmem_##TNAME##_atomic_swap(T *dest, T value, int pe) {       \
    return rocshmem_atomic_swap<T>(dest, value, pe);                          \
  }

#define AMO_BITWISE_DEF_GEN(T, TNAME)                                         \
  __host__ T rocshmem_ctx_##TNAME##_atomic_fetch_and(                         \
      rocshmem_ctx_t ctx, T *dest, T value, int pe) {                         \
    return rocshmem_atomic_fetch_and<T>(ctx, dest, value, pe);                \
  }                                                                           \
  __host__ T rocshmem_##TNAME##_atomic_fetch_and(T *dest, T value, int pe) {  \
    return rocshmem_atomic_fetch_and<T>(dest, value, pe);                     \
  }                                                                           \
  __host__ void rocshmem_ctx_##TNAME##_atomic_and(rocshmem_ctx_t ctx,         \
                                                   T *dest, T value, int pe) {\
    rocshmem_atomic_and<T>(ctx, dest, value, pe);                             \
  }                                                                           \
  __host__ void rocshmem_##TNAME##_atomic_and(T *dest, T value, int pe) {     \
    rocshmem_atomic_and<T>(dest, value, pe);                                  \
  }                                                                           \
  __host__ T rocshmem_ctx_##TNAME##_atomic_fetch_or(                          \
      rocshmem_ctx_t ctx, T *dest, T value, int pe) {                         \
    return rocshmem_atomic_fetch_or<T>(ctx, dest, value, pe);                 \
  }                                                                           \
  __host__ T rocshmem_##TNAME##_atomic_fetch_or(T *dest, T value, int pe) {   \
    return rocshmem_atomic_fetch_or<T>(dest, value, pe);                      \
  }                                                                           \
  __host__ void rocshmem_ctx_##TNAME##_atomic_or(rocshmem_ctx_t ctx,          \
                                                  T *dest, T value, int pe) { \
    rocshmem_atomic_or<T>(ctx, dest, value, pe);                              \
  }                                                                           \
  __host__ void rocshmem_##TNAME##_atomic_or(T *dest, T value, int pe) {      \
    rocshmem_atomic_or<T>(dest, value, pe);                                   \
  }                                                                           \
  __host__ T rocshmem_ctx_##TNAME##_atomic_fetch_xor(                         \
      rocshmem_ctx_t ctx, T *dest, T value, int pe) {                         \
    return rocshmem_atomic_fetch_xor<T>(ctx, dest, value, pe);                \
  }                                                                           \
  __host__ T rocshmem_##TNAME##_atomic_fetch_xor(T *dest, T value, int pe) {  \
    return rocshmem_atomic_fetch_xor<T>(dest, value, pe);                     \
  }                                                                           \
  __host__ void rocshmem_ctx_##TNAME##_atomic_xor(rocshmem_ctx_t ctx,         \
                                                   T *dest, T value, int pe) {\
    rocshmem_atomic_xor<T>(ctx, dest, value, pe);                             \
  }                                                                           \
  __host__ void rocshmem_##TNAME##_atomic_xor(T *dest, T value, int pe) {     \
    rocshmem_atomic_xor<T>(dest, value, pe);                                  \
  }

#define WAIT_DEF_GEN(T, TNAME)                                                \
  __host__ void rocshmem_##TNAME##_wait_until(T *ivars, int cmp,              \
                                               T val) {                       \
    rocshmem_wait_until<T>(ivars, cmp, val);                                  \
  }                                                                           \
  __host__ size_t rocshmem_##TNAME##_wait_until_any(T *ivars, size_t nelems,  \
                                                     const int* status,       \
                                                     int cmp,                 \
                                                     T val) {                 \
    return rocshmem_wait_until_any<T>(ivars, nelems, status, cmp, val);       \
  }                                                                           \
  __host__ void rocshmem_##TNAME##_wait_until_all(T *ivars, size_t nelems,    \
                                                   const int* status,         \
                                                   int cmp,                   \
                                                   T val) {                   \
    rocshmem_wait_until_all<T>(ivars, nelems, status, cmp, val);              \
  }                                                                           \
  __host__ size_t rocshmem_##TNAME##_wait_until_some(T *ivars, size_t nelems, \
                                                    size_t* indices,          \
                                                    const int* status,        \
                                                    int cmp,                  \
                                                    T val) {                  \
    return rocshmem_wait_until_some<T>(ivars, nelems, indices, status, cmp, val); \
  }                                                                           \
  __host__ size_t rocshmem_##TNAME##_wait_until_any_vector(T *ivars,          \
                                                          size_t nelems,      \
                                                          const int* status,  \
                                                          int cmp,            \
                                                          T* vals) {          \
    return rocshmem_wait_until_any_vector<T>(ivars, nelems, status, cmp,      \
                                              vals);                          \
  }                                                                           \
  __host__ void rocshmem_##TNAME##_wait_until_all_vector(T *ivars,            \
                                                          size_t nelems,      \
                                                          const int* status,  \
                                                          int cmp,            \
                                                          T* vals) {          \
    rocshmem_wait_until_all_vector<T>(ivars, nelems, status, cmp, vals);      \
  }                                                                           \
  __host__ size_t rocshmem_##TNAME##_wait_until_some_vector(T *ivars,         \
                                                           size_t nelems,     \
                                                           size_t* indices,   \
                                                           const int* status, \
                                                           int cmp,           \
                                                           T* vals) {         \
    return rocshmem_wait_until_some_vector<T>(ivars, nelems, indices,         \
        status, cmp, vals);                                                   \
  }                                                                           \
  __host__ int rocshmem_##TNAME##_test(T *ivars, int cmp, T val) {            \
    return rocshmem_test<T>(ivars, cmp, val);                                 \
  }

/******************************************************************************
 ************************* Macro Invocation Per Type **************************
 *****************************************************************************/

// clang-format off
INT_REDUCTION_GEN(int)
INT_REDUCTION_GEN(short)
INT_REDUCTION_GEN(long)
INT_REDUCTION_GEN(long long)
FLOAT_REDUCTION_GEN(float)
FLOAT_REDUCTION_GEN(double)
// long double reduction fails. hipcc/device may not support long double.
// so disable it for now.
// FLOAT_REDUCTION_GEN(long double)

RMA_GEN(float)
RMA_GEN(double)
// RMA_GEN(long double)
RMA_GEN(char)
RMA_GEN(signed char)
RMA_GEN(short)
RMA_GEN(int)
RMA_GEN(long)
RMA_GEN(long long)
RMA_GEN(unsigned char)
RMA_GEN(unsigned short)
RMA_GEN(unsigned int)
RMA_GEN(unsigned long)
RMA_GEN(unsigned long long)

AMO_STANDARD_GEN(int)
AMO_STANDARD_GEN(long)
AMO_STANDARD_GEN(long long)
AMO_STANDARD_GEN(unsigned int)
AMO_STANDARD_GEN(unsigned long)
AMO_STANDARD_GEN(unsigned long long)

AMO_EXTENDED_GEN(float)
AMO_EXTENDED_GEN(double)
AMO_EXTENDED_GEN(int)
AMO_EXTENDED_GEN(long)
AMO_EXTENDED_GEN(long long)
AMO_EXTENDED_GEN(unsigned int)
AMO_EXTENDED_GEN(unsigned long)
AMO_EXTENDED_GEN(unsigned long long)

AMO_BITWISE_GEN(unsigned int)
AMO_BITWISE_GEN(unsigned long)
AMO_BITWISE_GEN(unsigned long long)

/* Supported synchronization types */
WAIT_GEN(float)
WAIT_GEN(double)
// WAIT_GEN(long double)
WAIT_GEN(char)
WAIT_GEN(unsigned char)
WAIT_GEN(unsigned short)
WAIT_GEN(signed char)
WAIT_GEN(short)
WAIT_GEN(int)
WAIT_GEN(long)
WAIT_GEN(long long)
WAIT_GEN(unsigned int)
WAIT_GEN(unsigned long)
WAIT_GEN(unsigned long long)

INT_REDUCTION_DEF_GEN(int, int)
INT_REDUCTION_DEF_GEN(short, short)
INT_REDUCTION_DEF_GEN(long, long)
INT_REDUCTION_DEF_GEN(long long, longlong)
FLOAT_REDUCTION_DEF_GEN(float, float)
FLOAT_REDUCTION_DEF_GEN(double, double)
// long double reduction fails. hipcc/device may not support long double.
// so disable it for now.
// FLOAT_REDUCTION_DEF_GEN(long double, longdouble)

RMA_DEF_GEN(float, float)
RMA_DEF_GEN(double, double)
RMA_DEF_GEN(char, char)
// RMA_DEF_GEN(long double, longdouble)
RMA_DEF_GEN(signed char, schar)
RMA_DEF_GEN(short, short)
RMA_DEF_GEN(int, int)
RMA_DEF_GEN(long, long)
RMA_DEF_GEN(long long, longlong)
RMA_DEF_GEN(unsigned char, uchar)
RMA_DEF_GEN(unsigned short, ushort)
RMA_DEF_GEN(unsigned int, uint)
RMA_DEF_GEN(unsigned long, ulong)
RMA_DEF_GEN(unsigned long long, ulonglong)
RMA_DEF_GEN(int8_t, int8)
RMA_DEF_GEN(int16_t, int16)
RMA_DEF_GEN(int32_t, int32)
RMA_DEF_GEN(int64_t, int64)
RMA_DEF_GEN(uint8_t, uint8)
RMA_DEF_GEN(uint16_t, uint16)
RMA_DEF_GEN(uint32_t, uint32)
RMA_DEF_GEN(uint64_t, uint64)
RMA_DEF_GEN(size_t, size)
RMA_DEF_GEN(ptrdiff_t, ptrdiff)

AMO_STANDARD_DEF_GEN(int, int)
AMO_STANDARD_DEF_GEN(long, long)
AMO_STANDARD_DEF_GEN(long long, longlong)
AMO_STANDARD_DEF_GEN(unsigned int, uint)
AMO_STANDARD_DEF_GEN(unsigned long, ulong)
AMO_STANDARD_DEF_GEN(unsigned long long, ulonglong)
AMO_STANDARD_DEF_GEN(int32_t, int32)
AMO_STANDARD_DEF_GEN(int64_t, int64)
AMO_STANDARD_DEF_GEN(uint32_t, uint32)
AMO_STANDARD_DEF_GEN(uint64_t, uint64)
AMO_STANDARD_DEF_GEN(size_t, size)
AMO_STANDARD_DEF_GEN(ptrdiff_t, ptrdiff)

AMO_EXTENDED_DEF_GEN(float, float)
AMO_EXTENDED_DEF_GEN(double, double)
AMO_EXTENDED_DEF_GEN(int, int)
AMO_EXTENDED_DEF_GEN(long, long)
AMO_EXTENDED_DEF_GEN(long long, longlong)
AMO_EXTENDED_DEF_GEN(unsigned int, uint)
AMO_EXTENDED_DEF_GEN(unsigned long, ulong)
AMO_EXTENDED_DEF_GEN(unsigned long long, ulonglong)
AMO_EXTENDED_DEF_GEN(int32_t, int32)
AMO_EXTENDED_DEF_GEN(int64_t, int64)
AMO_EXTENDED_DEF_GEN(uint32_t, uint32)
AMO_EXTENDED_DEF_GEN(uint64_t, uint64)
AMO_EXTENDED_DEF_GEN(size_t, size)
AMO_EXTENDED_DEF_GEN(ptrdiff_t, ptrdiff)

AMO_BITWISE_DEF_GEN(unsigned int, uint)
AMO_BITWISE_DEF_GEN(unsigned long, ulong)
AMO_BITWISE_DEF_GEN(unsigned long long, ulonglong)
AMO_BITWISE_DEF_GEN(int32_t, int32)
AMO_BITWISE_DEF_GEN(int64_t, int64)
AMO_BITWISE_DEF_GEN(uint32_t, uint32)
AMO_BITWISE_DEF_GEN(uint64_t, uint64)

WAIT_DEF_GEN(float, float)
WAIT_DEF_GEN(double, double)
// WAIT_DEF_GEN(long double, longdouble)
WAIT_DEF_GEN(char, char)
WAIT_DEF_GEN(signed char, schar)
WAIT_DEF_GEN(short, short)
WAIT_DEF_GEN(int, int)
WAIT_DEF_GEN(long, long)
WAIT_DEF_GEN(long long, longlong)
WAIT_DEF_GEN(unsigned char, uchar)
WAIT_DEF_GEN(unsigned short, ushort)
WAIT_DEF_GEN(unsigned int, uint)
WAIT_DEF_GEN(unsigned long, ulong)
WAIT_DEF_GEN(unsigned long long, ulonglong)
WAIT_DEF_GEN(uint64_t, uint64)
// clang-format on

}  // namespace rocshmem
