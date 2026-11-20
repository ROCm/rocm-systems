// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/* Manual probe for the local checked-prefetch protocol. See the SVM report.
 * Run one named case per process with an external watchdog and health audit.
 * Requires the reviewed driver correction; it never loads a module itself.
 */
#define _GNU_SOURCE
#include <errno.h>
#include <fcntl.h>
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <sys/uio.h>
#include <unistd.h>
#include <time.h>
#include <hsa/hsa.h>
#include <hsa/hsa_ext_amd.h>

#define CHECK(c) do { if (!(c)) { fprintf(stderr, "FAIL line %d: %s errno=%d (%s)\n", __LINE__, #c, errno, strerror(errno)); exit(1); } } while (0)
struct version { uint32_t major, minor; uint64_t id, features; };
struct attr { uint32_t type, value; };
struct svm { uint64_t start, size; uint32_t op, nattr; struct attr attr; };
_Static_assert(sizeof(struct version) == 24, "version layout");
_Static_assert(sizeof(struct svm) == 32, "SVM layout");
static hsa_agent_t gpu, cpu;
static int fd;
static uint32_t gpuid;
static size_t page;
static hsa_status_t agent(hsa_agent_t a, void *unused) {
  hsa_device_type_t type;
  CHECK(hsa_agent_get_info(a, HSA_AGENT_INFO_DEVICE, &type) == HSA_STATUS_SUCCESS);
  if (type == HSA_DEVICE_TYPE_GPU && !gpu.handle) gpu = a;
  if (type == HSA_DEVICE_TYPE_CPU && !cpu.handle) cpu = a;
  return HSA_STATUS_SUCCESS;
}
static int prefetch(void *ptr, size_t size, uint32_t location) {
  struct svm args = {(uintptr_t)ptr, size, 0x525301, 1, {1, location}};
  errno = 0;
  int result = ioctl(fd, _IOWR('K', 0x20, struct svm), &args);
  int error = result ? errno : 0;
  printf("prefetch %zu pages -> %u: %d\n", size/page, location, error);
  fflush(stdout);
  return error;
}
static unsigned char pattern(size_t i) { return (unsigned char)((i * 13 + i / page + 7) % 251); }
static void verify(unsigned char *ptr, size_t size, int zero) {
  for (size_t i = 0; i < size; ++i) CHECK(ptr[i] == (zero ? 0 : pattern(i)));
}
static hsa_signal_value_t wait_completion(hsa_signal_t signal) {
  struct timespec start, now;
  CHECK(clock_gettime(CLOCK_MONOTONIC, &start) == 0);
  hsa_signal_value_t value;
  do {
    value = hsa_signal_load_scacquire(signal);
    if (value < 1) return value;
    usleep(1000);
    CHECK(clock_gettime(CLOCK_MONOTONIC, &now) == 0);
  } while (now.tv_sec - start.tv_sec < 5);
  CHECK(value < 1);
  return value;
}
int main(int argc, char **argv) {
  CHECK(argc == 2);
  CHECK(!strcmp(argv[1], "untouched") || !strcmp(argv[1], "cpu-untouched") ||
        !strcmp(argv[1], "data") || !strcmp(argv[1], "partial") ||
        !strcmp(argv[1], "split") || !strcmp(argv[1], "split-error") ||
        !strcmp(argv[1], "multi-vma") || !strcmp(argv[1], "frontend-error"));
  alarm(20);
  setbuf(stdout, NULL);
  CHECK(hsa_init() == HSA_STATUS_SUCCESS);
  CHECK(hsa_iterate_agents(agent, NULL) == HSA_STATUS_SUCCESS && gpu.handle);
  uint32_t node;
  CHECK(hsa_agent_get_info(gpu, (hsa_agent_info_t)HSA_AMD_AGENT_INFO_DRIVER_NODE_ID, &node) == HSA_STATUS_SUCCESS);
  char path[128];
  snprintf(path, sizeof(path), "/sys/class/kfd/kfd/topology/nodes/%u/gpu_id", node);
  FILE *info = fopen(path, "r"); CHECK(info); CHECK(fscanf(info, "%u", &gpuid) == 1); fclose(info);
  fd = open("/dev/kfd", O_RDWR | O_CLOEXEC); CHECK(fd >= 0);
  struct version version = {0};
  CHECK(ioctl(fd, _IOR('K', 0x01, struct version), &version) == 0);
  CHECK(version.major == 1 && version.id == UINT64_C(0x31564d5352444443) && (version.features & 1));
  uint32_t legacy[2] = {0};
  CHECK(ioctl(fd, _IOC(_IOC_READ, 'K', 0x01, 8), legacy) == 0);
  CHECK(legacy[0] == version.major && legacy[1] == version.minor);
  page = (size_t)sysconf(_SC_PAGESIZE);
  size_t size = 8 * page;
  unsigned char *ptr = mmap(NULL, size, PROT_READ | PROT_WRITE, MAP_ANONYMOUS | MAP_PRIVATE, -1, 0);
  CHECK(ptr != MAP_FAILED);
  int zero = !strcmp(argv[1], "untouched") || !strcmp(argv[1], "cpu-untouched");
  if (!zero) for (size_t i = 0; i < size; ++i) ptr[i] = pattern(i);
  if (!strcmp(argv[1], "cpu-untouched")) {
    CHECK(prefetch(ptr, size, 0) == 0); verify(ptr, size, 1);
  } else if (!strcmp(argv[1], "frontend-error")) {
    hsa_amd_svm_attribute_pair_t policy = {HSA_AMD_SVM_ATTRIB_PREFERRED_LOCATION, cpu.handle};
    CHECK(hsa_amd_svm_attributes_set(ptr, size, &policy, 1) == HSA_STATUS_SUCCESS);
    int pipefd[2]; CHECK(pipe2(pipefd, O_CLOEXEC) == 0);
    struct iovec iov = {ptr + 3 * page, page};
    CHECK(vmsplice(pipefd[1], &iov, 1, 0) == (ssize_t)page);
    hsa_signal_t dependency, completion;
    CHECK(hsa_signal_create(1, 0, NULL, &dependency) == HSA_STATUS_SUCCESS);
    CHECK(hsa_signal_create(1, 0, NULL, &completion) == HSA_STATUS_SUCCESS);
    CHECK(hsa_amd_svm_prefetch_async(ptr, size, gpu, 1, &dependency, completion) == HSA_STATUS_SUCCESS);
    usleep(20000);
    CHECK(hsa_signal_load_scacquire(completion) == 1);
    hsa_signal_store_screlease(dependency, 0);
    hsa_signal_value_t failure = wait_completion(completion);
    printf("HSA partial-migration completion: %" PRId64 "\n", (int64_t)failure);
    CHECK(failure < 0);
    CHECK(prefetch(ptr, size, 0) == 0); verify(ptr, size, 0);
    close(pipefd[0]); close(pipefd[1]);
    hsa_signal_store_screlease(completion, 1);
    CHECK(hsa_amd_svm_prefetch_async(ptr, size, gpu, 0, NULL, completion) == HSA_STATUS_SUCCESS);
    CHECK(wait_completion(completion) == 0);
    hsa_amd_svm_attribute_pair_t location = {HSA_AMD_SVM_ATTRIB_PREFETCH_LOCATION, 0};
    CHECK(hsa_amd_svm_attributes_get(ptr, size, &location, 1) == HSA_STATUS_SUCCESS);
    CHECK(location.value == gpu.handle);
    hsa_signal_store_screlease(completion, 1);
    CHECK(hsa_amd_svm_prefetch_async(ptr, size, cpu, 0, NULL, completion) == HSA_STATUS_SUCCESS);
    CHECK(wait_completion(completion) == 0); verify(ptr, size, 0);
    CHECK(hsa_signal_destroy(completion) == HSA_STATUS_SUCCESS);
    CHECK(hsa_signal_destroy(dependency) == HSA_STATUS_SUCCESS);
  } else if (!strcmp(argv[1], "partial")) {
    int pipefd[2]; CHECK(pipe2(pipefd, O_CLOEXEC) == 0);
    struct iovec iov = {ptr + 3 * page, page};
    CHECK(vmsplice(pipefd[1], &iov, 1, 0) == (ssize_t)page);
    CHECK(prefetch(ptr, size, gpuid) == EAGAIN);
    CHECK(prefetch(ptr, size, 0) == 0); verify(ptr, size, 0);
    close(pipefd[0]); close(pipefd[1]);
    CHECK(prefetch(ptr, size, gpuid) == 0);
    CHECK(prefetch(ptr, size, 0) == 0); verify(ptr, size, 0);
  } else if (!strcmp(argv[1], "split")) {
    CHECK(prefetch(ptr, size, gpuid) == 0);
    CHECK(prefetch(ptr + 2 * page, 2 * page, 0) == 0);
    CHECK(prefetch(ptr + 2 * page, 2 * page, gpuid) == 0);
    CHECK(prefetch(ptr, size, 0) == 0); verify(ptr, size, 0);
  } else if (!strcmp(argv[1], "split-error")) {
    CHECK(prefetch(ptr, size, gpuid) == 0);
    CHECK(mprotect(ptr, page, PROT_NONE) == 0);
    CHECK(prefetch(ptr + 2 * page, 2 * page, 0) != 0);
    CHECK(mprotect(ptr, page, PROT_READ | PROT_WRITE) == 0);
    CHECK(prefetch(ptr, size, 0) == 0); verify(ptr, size, 0);
  } else {
    if (!strcmp(argv[1], "multi-vma")) CHECK(mprotect(ptr + 4 * page, 4 * page, PROT_READ) == 0);
    for (int i = 0; i < 2; ++i) {
      CHECK(prefetch(ptr, size, gpuid) == 0);
      CHECK(prefetch(ptr, size, 0) == 0); verify(ptr, size, zero);
    }
  }
  CHECK(munmap(ptr, size) == 0); CHECK(close(fd) == 0);
  CHECK(hsa_shut_down() == HSA_STATUS_SUCCESS);
  puts("PASS");
  return 0;
}
