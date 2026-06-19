#define _GNU_SOURCE

#include <dlfcn.h>
#include <errno.h>
#include <fcntl.h>
#include <stdarg.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <time.h>
#include <unistd.h>

static uint64_t deploy_now_ns(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC_RAW, &ts);
  return (uint64_t)ts.tv_sec * 1000000000ULL + (uint64_t)ts.tv_nsec;
}

static int deploy_rank(void) {
  const char* names[] = {
      "SLURM_PROCID", "OMPI_COMM_WORLD_RANK", "PMIX_RANK", "PMI_RANK",
      "MV2_COMM_WORLD_RANK", "RANK", NULL};
  for (int i = 0; names[i] != NULL; ++i) {
    const char* v = getenv(names[i]);
    if (v && v[0]) return atoi(v);
  }
  return -1;
}

static void expand_path(char* dst, size_t dst_size, const char* tmpl, int rank, int pid) {
  size_t o = 0;
  for (size_t i = 0; tmpl[i] && o + 1 < dst_size; ++i) {
    if (tmpl[i] == '%' && tmpl[i + 1] == 'r') {
      o += (size_t)snprintf(dst + o, dst_size - o, "%d", rank);
      ++i;
    } else if (tmpl[i] == '%' && tmpl[i + 1] == 'p') {
      o += (size_t)snprintf(dst + o, dst_size - o, "%d", pid);
      ++i;
    } else {
      dst[o++] = tmpl[i];
    }
  }
  dst[o < dst_size ? o : dst_size - 1] = '\0';
}

static void deploy_log(const char* phase, uint64_t start_ns, uint64_t end_ns, int md,
                       size_t bytes, const char* detail) {
  int rank = deploy_rank();
  int pid = (int)getpid();
  uint64_t dur_us = end_ns > start_ns ? (end_ns - start_ns) / 1000ULL : 0ULL;
  char line[512];
  int n = snprintf(line, sizeof(line),
      "DEPLOY_TRACE rank=%d pid=%d phase=%s t_ns=%llu dur_us=%llu md=%d bytes=%lu detail=%s\n",
      rank, pid, phase, (unsigned long long)start_ns, (unsigned long long)dur_us,
      md, (unsigned long)bytes, detail ? detail : "none");
  if (n <= 0) return;

  const char* tmpl = getenv("RCCL_DEPLOY_TRACE_FILE");
  if (tmpl && tmpl[0]) {
    char path[512];
    expand_path(path, sizeof(path), tmpl, rank, pid);
    int fd = open(path, O_WRONLY | O_CREAT | O_APPEND, 0644);
    if (fd >= 0) {
      ssize_t unused = write(fd, line, (size_t)n);
      (void)unused;
      close(fd);
      return;
    }
  }
  ssize_t unused = write(STDERR_FILENO, line, (size_t)n);
  (void)unused;
}

__attribute__((constructor))
static void deploy_trace_constructor(void) {
  uint64_t now = deploy_now_ns();
  deploy_log("deploy.exec", now, now, 0, 0, "constructor");
}

typedef int (*mpi_init_fn_t)(int*, char***);
typedef int (*mpi_init_thread_fn_t)(int*, char***, int, int*);
typedef int (*pmix_init_fn_t)(void*, void*, size_t);

int MPI_Init(int* argc, char*** argv) {
  mpi_init_fn_t real_fn = (mpi_init_fn_t)dlsym(RTLD_NEXT, "MPI_Init");
  uint64_t start = deploy_now_ns();
  int ret = real_fn ? real_fn(argc, argv) : -1;
  deploy_log("mpi.init", start, deploy_now_ns(), ret, 0, "MPI_Init");
  return ret;
}

int MPI_Init_thread(int* argc, char*** argv, int required, int* provided) {
  mpi_init_thread_fn_t real_fn = (mpi_init_thread_fn_t)dlsym(RTLD_NEXT, "MPI_Init_thread");
  uint64_t start = deploy_now_ns();
  int ret = real_fn ? real_fn(argc, argv, required, provided) : -1;
  int md = provided ? *provided : required;
  deploy_log("mpi.init_thread", start, deploy_now_ns(), md, 0, "MPI_Init_thread");
  return ret;
}

int PMIx_Init(void* proc, void* info, size_t ninfo) {
  pmix_init_fn_t real_fn = (pmix_init_fn_t)dlsym(RTLD_NEXT, "PMIx_Init");
  uint64_t start = deploy_now_ns();
  int ret = real_fn ? real_fn(proc, info, ninfo) : -1;
  deploy_log("pmix.init", start, deploy_now_ns(), ret, ninfo, "PMIx_Init");
  return ret;
}
