/*
 * Copyright © 2014 Advanced Micro Devices, Inc.
 *
 * Permission is hereby granted, free of charge, to any person
 * obtaining a copy of this software and associated documentation
 * files (the "Software"), to deal in the Software without
 * restriction, including without limitation the rights to use, copy,
 * modify, merge, publish, distribute, sublicense, and/or sell copies
 * of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice (including
 * the next paragraph) shall be included in all copies or substantial
 * portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 * NONINFRINGEMENT.  IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
 * HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
 * WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */
#include <stdlib.h>
#include <cstring>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <cstdio>
#include <strings.h>
#include <dlfcn.h>
#include <cassert>
#include <amdgpu.h>
#include "libhsakmt.h"
#include "inc/hsa/hsa.h"
#include "inc/hsa/hsa_ven_amd_loader.h"

int (*fn_amdgpu_device_get_fd)(HsaAMDGPUDeviceHandle device_handle);

hsa_signal_value_t (*fn_hsa_signal_load_relaxed)(hsa_signal_t signal);
hsa_signal_value_t (*fn_hsa_signal_wait_relaxed)(
    hsa_signal_t signal, hsa_signal_condition_t condition,
    hsa_signal_value_t compare_value, uint64_t timeout_hint,
    hsa_wait_state_t wait_state_hint);
void (*fn_hsa_signal_store_screlease)(hsa_signal_t hsa_signal,
                                      hsa_signal_value_t value);
hsa_status_t (*fn_hsa_ven_amd_loader_query_host_address)(
    const void *device_address, const void **host_address);

static const char dxg_device_name[] = "/dev/dxg";
static pid_t parent_pid = -1;
int hsakmt_debug_level;
bool hsakmt_forked;
static int dxg_fd = -1;

/* is_forked_child detects when the process has forked since the last
 * time this function was called. We cannot rely on pthread_atfork
 * because the process can fork without calling the fork function in
 * libc (using clone or calling the system call directly).
 */
bool is_forked_child(void) {
  pid_t cur_pid;

  if (hsakmt_forked)
    return true;

  cur_pid = getpid();

  if (parent_pid == -1) {
    parent_pid = cur_pid;
    return false;
  }

  if (parent_pid != cur_pid) {
    hsakmt_forked = true;
    return true;
  }

  return false;
}

/* Callbacks from pthread_atfork */
static void prepare_fork_handler(void) { pthread_mutex_lock(&hsakmt_mutex); }
static void parent_fork_handler(void) { pthread_mutex_unlock(&hsakmt_mutex); }
static void child_fork_handler(void) {
  pthread_mutex_init(&hsakmt_mutex, NULL);
  hsakmt_forked = true;
}

/* Call this from the child process after fork. This will clear all
 * data that is duplicated from the parent process, that is not valid
 * in the child.
 * The topology information is duplicated from the parent is valid
 * in the child process so it is not cleared
 */
static void clear_after_fork(void) {
  // TODO: fmm_clear_all_mem();
  if (dxg_fd) {
    close(dxg_fd);
    dxg_fd = -1;
  }
  dxg_open_count = 0;
  parent_pid = -1;
  hsakmt_forked = false;
}

static inline void init_page_size(void) {
#ifndef PAGE_SIZE
  PAGE_SIZE = sysconf(_SC_PAGESIZE);
#endif
  PAGE_SHIFT = ffs(PAGE_SIZE) - 1;
}

static HSAKMT_STATUS init_vars_from_env(void) {
  char *envvar;
  int debug_level;

  /* Normally libraries don't print messages. For debugging purpose, we'll
   * print messages if an environment variable, HSAKMT_DEBUG_LEVEL, is set.
   */
  hsakmt_debug_level = HSAKMT_DEBUG_LEVEL_DEFAULT;

  envvar = getenv("HSAKMT_DEBUG_LEVEL");
  if (envvar) {
    debug_level = atoi(envvar);
    if (debug_level >= HSAKMT_DEBUG_LEVEL_ERR &&
        debug_level <= HSAKMT_DEBUG_LEVEL_DEBUG)
      hsakmt_debug_level = debug_level;
  }

  /* Check whether to support Zero frame buffer */
  envvar = getenv("HSA_ZFB");
  if (envvar)
    zfb_support = atoi(envvar);

  /* Check whether to handle vendor specific aql packet */
  envvar = getenv("WSLKMT_VENDOR_PACKET");
  if (envvar)
    vendor_packet_support = atoi(envvar);

  return HSAKMT_STATUS_SUCCESS;
}

#define _HSAKMT_LOOKUP_SYMS(_sym)                                              \
  do {                                                                         \
    fn_##_sym =                                                                \
        reinterpret_cast<decltype(fn_##_sym)>(dlsym(RTLD_DEFAULT, #_sym));     \
    if (!fn_##_sym) {                                                          \
      pr_err("%s not found - %s\n", #_sym, dlerror());                         \
      return HSAKMT_STATUS_ERROR;                                              \
    }                                                                          \
  } while (0)

static HSAKMT_STATUS init_symbols(void) {
  _HSAKMT_LOOKUP_SYMS(hsa_signal_load_relaxed);
  _HSAKMT_LOOKUP_SYMS(hsa_signal_wait_relaxed);
  _HSAKMT_LOOKUP_SYMS(hsa_signal_store_screlease);

  hsa_status_t (*fn_hsa_system_get_extension_table)(
      uint16_t extension, uint16_t version_major, uint16_t version_minor,
      void *table);
  _HSAKMT_LOOKUP_SYMS(hsa_system_get_extension_table);
  hsa_ven_amd_loader_1_03_pfn_t table;
  fn_hsa_system_get_extension_table(HSA_EXTENSION_AMD_LOADER, 1, 3, &table);
  fn_hsa_ven_amd_loader_query_host_address =
      table.hsa_ven_amd_loader_query_host_address;

  return HSAKMT_STATUS_SUCCESS;
}

static void load_libdrm_amdgpu(void) {
  /* load libdrm_amdgpu */
  int fd;
  uint32_t major, minor;
  amdgpu_device_handle device_handle;
  amdgpu_device_initialize(fd, &major, &minor, &device_handle);
}

HSAKMT_STATUS HSAKMTAPI hsaKmtOpenKFD(void) {
  HSAKMT_STATUS result;
  int fd = -1;
  HsaSystemProperties sys_props;
  char *error;
  char *useSvmStr;

  pthread_mutex_lock(&hsakmt_mutex);

  /* If the process has forked, the child process must re-initialize
   * it's connection to DXG. Any references tracked by dxg_open_count
   * belong to the parent
   */
  if (is_forked_child())
    clear_after_fork();

  if (dxg_open_count == 0) {
    static bool atfork_installed = false;

    result = init_symbols();
    if (result != HSAKMT_STATUS_SUCCESS)
      goto open_failed;

    load_libdrm_amdgpu();

    result = init_vars_from_env();
    if (result != HSAKMT_STATUS_SUCCESS)
      goto open_failed;

    if (dxg_fd < 0) {
      fd = open(dxg_device_name, O_RDWR | O_CLOEXEC);

      if (fd == -1) {
        result = HSAKMT_STATUS_KERNEL_IO_CHANNEL_NOT_OPENED;
        goto open_failed;
      }

      dxg_fd = fd;
    }

    init_page_size();

    useSvmStr = getenv("HSA_USE_SVM");
    is_svm_api_supported = !(useSvmStr && !strcmp(useSvmStr, "0")) && false;

    // result = topology_sysfs_get_system_props(&sys_props);
    if (result != HSAKMT_STATUS_SUCCESS)
      goto topology_sysfs_failed;

    dxg_open_count = 1;

    if (!atfork_installed) {
      /* Atfork handlers cannot be uninstalled and
       * must be installed only once. Otherwise
       * prepare will deadlock when trying to take
       * the same lock multiple times.
       */
      pthread_atfork(prepare_fork_handler, parent_fork_handler,
                     child_fork_handler);
      atfork_installed = true;
    }
  } else {
    dxg_open_count++;
    result = HSAKMT_STATUS_KERNEL_ALREADY_OPENED;
  }

  pthread_mutex_unlock(&hsakmt_mutex);
  return result;
topology_sysfs_failed:
  close(fd);
open_failed:
  pthread_mutex_unlock(&hsakmt_mutex);

  return result;
}

HSAKMT_STATUS HSAKMTAPI hsaKmtCloseKFD(void) {
  HSAKMT_STATUS result;

  pthread_mutex_lock(&hsakmt_mutex);

  if (dxg_open_count > 0) {
    if (--dxg_open_count == 0) {
      close(dxg_fd);
    }

    result = HSAKMT_STATUS_SUCCESS;
  } else
    result = HSAKMT_STATUS_KERNEL_IO_CHANNEL_NOT_OPENED;

  pthread_mutex_unlock(&hsakmt_mutex);

  return result;
}
