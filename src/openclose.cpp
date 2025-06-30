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
#include <cassert>
#include "libhsakmt.h"

hsakmtRuntime *dxg_runtime = new hsakmtRuntime();
/* is_forked_child detects when the process has forked since the last
 * time this function was called. We cannot rely on pthread_atfork
 * because the process can fork without calling the fork function in
 * libc (using clone or calling the system call directly).
 */
bool is_forked_child(void) {
  if (dxg_runtime->is_forked)
    return true;

  pid_t cur_pid = getpid();
  if (dxg_runtime->parent_pid != cur_pid) {
    dxg_runtime->is_forked = true;
    dxg_runtime->parent_pid = cur_pid;
    return true;
  }

  return false;
}

/* Callbacks from pthread_atfork */
static void prepare_fork_handler(void) { pthread_mutex_lock(&dxg_runtime->hsakmt_mutex); }
static void parent_fork_handler(void) { pthread_mutex_unlock(&dxg_runtime->hsakmt_mutex); }
static void child_fork_handler(void) {
  pthread_mutex_init(&dxg_runtime->hsakmt_mutex, NULL);
  dxg_runtime->is_forked = true;
}

/* Call this from the child process after fork. This will clear all
 * data that is duplicated from the parent process, that is not valid
 * in the child.
 * The topology information is duplicated from the parent is valid
 * in the child process so it is not cleared
 */
static void clear_after_fork(void) {
  reset_suballocator();
  clear_allocation_map();

  if (dxg_runtime->dxg_fd >= 0) {
    close(dxg_runtime->dxg_fd);
    dxg_runtime->dxg_fd = -1;
  }
  delete dxg_runtime;
  dxg_runtime = new hsakmtRuntime();

}

static inline void init_page_size(void) {
  dxg_runtime->page_size = sysconf(_SC_PAGESIZE);
  dxg_runtime->page_shift = ffs(dxg_runtime->page_size) - 1;
}

static HSAKMT_STATUS init_vars_from_env(void) {
  char *envvar;
  int debug_level;

  /* Normally libraries don't print messages. For debugging purpose, we'll
   * print messages if an environment variable, HSAKMT_DEBUG_LEVEL, is set.
   */
  envvar = getenv("HSAKMT_DEBUG_LEVEL");
  if (envvar) {
    dxg_runtime->hsakmt_debug_level = atoi(envvar);
  }

  /* Check whether to support Zero frame buffer */
  envvar = getenv("HSA_ZFB");
  if (envvar)
    dxg_runtime->zfb_support = atoi(envvar);

  /* Check whether to handle vendor specific aql packet */
  envvar = getenv("WSLKMT_VENDOR_PACKET");
  if (envvar)
    dxg_runtime->vendor_packet_process = atoi(envvar);

  /* Decide whether to check available system memory before allocation */
  envvar = getenv("WSL_CHECK_AVAIL_SYSRAM");
  if (envvar)
    dxg_runtime->check_avail_sysram = !strcmp(envvar, "1");

  envvar = getenv("WSL_ENABLE_THUNK_SUB_ALLOCATOR");
  if (envvar)
    dxg_runtime->enable_thunk_sub_allocator = atoi(envvar);

  envvar = getenv("ROCR_VISIBLE_DEVICES");
  if (envvar) {
    std::string devices(envvar);
    size_t first_num_pos = devices.find_first_of("0123456789");
    if (first_num_pos != std::string::npos)
      dxg_runtime->default_node = std::stoi(devices.substr(first_num_pos)) + 1;
  }

  return HSAKMT_STATUS_SUCCESS;
}

HSAKMT_STATUS HSAKMTAPI hsaKmtOpenKFD(void) {
  HSAKMT_STATUS result;
  int fd = -1;
  HsaSystemProperties sys_props;
  char *error;

  pthread_mutex_lock(&dxg_runtime->hsakmt_mutex);

  /* If the process has forked, the child process must re-initialize
   * it's connection to DXG. Any references tracked by dxg_open_count
   * belong to the parent
   */
  if (is_forked_child())
    clear_after_fork();

  if (dxg_runtime->dxg_open_count == 0) {
    static bool atfork_installed = false;

    result = init_vars_from_env();
    if (result != HSAKMT_STATUS_SUCCESS)
      goto open_failed;

    if (dxg_runtime->dxg_fd < 0) {
      fd = open(dxg_runtime->dxg_device_name, O_RDWR | O_CLOEXEC);

      if (fd == -1) {
        result = HSAKMT_STATUS_KERNEL_IO_CHANNEL_NOT_OPENED;
        goto open_failed;
      }

      dxg_runtime->dxg_fd = fd;
    }

    init_page_size();

    char *useSvmStr = getenv("HSA_USE_SVM");
    dxg_runtime->is_svm_api_supported = !(useSvmStr && !strcmp(useSvmStr, "0")) && false;

    // result = topology_sysfs_get_system_props(&sys_props);
    if (result != HSAKMT_STATUS_SUCCESS)
      goto topology_sysfs_failed;

    dxg_runtime->dxg_open_count = 1;

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
    dxg_runtime->dxg_open_count++;
    result = HSAKMT_STATUS_KERNEL_ALREADY_OPENED;
  }

  reset_suballocator();
  pthread_mutex_unlock(&dxg_runtime->hsakmt_mutex);
  return result;
topology_sysfs_failed:
  close(fd);
open_failed:
  pthread_mutex_unlock(&dxg_runtime->hsakmt_mutex);

  return result;
}

HSAKMT_STATUS HSAKMTAPI hsaKmtCloseKFD(void) {
  HSAKMT_STATUS result;

  pthread_mutex_lock(&dxg_runtime->hsakmt_mutex);

  if (dxg_runtime->dxg_open_count > 0) {
    if (--dxg_runtime->dxg_open_count == 0) {
      close(dxg_runtime->dxg_fd);
      dxg_runtime->dxg_fd = -1;
    }

    result = HSAKMT_STATUS_SUCCESS;
  } else
    result = HSAKMT_STATUS_KERNEL_IO_CHANNEL_NOT_OPENED;

  pthread_mutex_unlock(&dxg_runtime->hsakmt_mutex);

  return result;
}
