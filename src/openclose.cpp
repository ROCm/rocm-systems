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
#include <sys/mman.h>
#include <sys/sysinfo.h>
#include <linux/mman.h>
#include <fcntl.h>
#include <unistd.h>
#include <cstdio>
#include <strings.h>
#include <cassert>

bool is_forked_child(void) {
  return dxg_runtime().IsForkedChild();
}

static void clear_after_fork(void) {
  dxg_runtime().ClearAfterFork();
}

static inline void init_page_size(void) {
  dxg_runtime().page_size = sysconf(_SC_PAGESIZE);
  dxg_runtime().page_shift = ffs(dxg_runtime().page_size) - 1;
}

static HSAKMT_STATUS init_vars_from_env(void) {
  char *envvar;
  int debug_level;
  auto &rt = dxg_runtime();

  /* Normally libraries don't print messages. For debugging purpose, we'll
   * print messages if an environment variable, HSAKMT_DEBUG_LEVEL, is set.
   */
  envvar = getenv("HSAKMT_DEBUG_LEVEL");
  if (envvar) {
    rt.hsakmt_debug_level = atoi(envvar);
  }

  /* Check whether to support Zero frame buffer */
  envvar = getenv("HSA_ZFB");
  if (envvar)
    rt.zfb_support = atoi(envvar);

  /* Check whether to handle vendor specific aql packet */
  envvar = getenv("WSLKMT_VENDOR_PACKET");
  if (envvar)
    rt.vendor_packet_process = atoi(envvar);

  /* Decide whether to check available system memory before allocation */
  envvar = getenv("WSL_CHECK_AVAIL_SYSRAM");
  if (envvar)
    rt.check_avail_sysram = !strcmp(envvar, "1");

  envvar = getenv("WSL_ENABLE_THUNK_SUB_ALLOCATOR");
  if (envvar)
    rt.enable_thunk_sub_allocator = atoi(envvar);

  envvar = getenv("ROCR_VISIBLE_DEVICES");
  if (envvar) {
    std::string devices(envvar);
    size_t first_num_pos = devices.find_first_of("0123456789");
    if (first_num_pos != std::string::npos)
      rt.default_node = std::stoi(devices.substr(first_num_pos)) + 1;
  }

  return HSAKMT_STATUS_SUCCESS;
}

HSAKMT_STATUS HSAKMTAPI hsaKmtOpenKFD(void) {
  HSAKMT_STATUS result;
  int fd = -1;
  HsaSystemProperties sys_props;
  char *error;
  auto &rt = dxg_runtime();

  pthread_mutex_lock(&rt.hsakmt_mutex);

  /* If the process has forked, the child process must re-initialize
   * it's connection to DXG. Any references tracked by dxg_open_count
   * belong to the parent
   */
  if (is_forked_child())
    clear_after_fork();

  if (rt.dxg_open_count == 0) {
    result = init_vars_from_env();
    if (result != HSAKMT_STATUS_SUCCESS)
      goto open_failed;

    if (rt.dxg_fd < 0) {
      fd = open(rt.dxg_device_name, O_RDWR | O_CLOEXEC);

      if (fd == -1) {
        result = HSAKMT_STATUS_KERNEL_IO_CHANNEL_NOT_OPENED;
        goto open_failed;
      }

      rt.dxg_fd = fd;
    }
    if (!wsl::thunk::dxcore::DxcoreLoader::Instance().Initialize()) {
        pr_err("Failed to load libdxcore.so\n");
        result = HSAKMT_STATUS_ERROR;
        goto dxcore_loader_failed;
    }

    hsakmt_hsa_loader_init();
    init_page_size();

    char *useSvmStr = getenv("HSA_USE_SVM");
    rt.is_svm_api_supported = !(useSvmStr && !strcmp(useSvmStr, "0")) && false;

    rt.dxg_open_count = 1;

    rt.InstallAtForkHandlers();
  } else {
    rt.dxg_open_count++;
    result = HSAKMT_STATUS_KERNEL_ALREADY_OPENED;
  }

  reset_suballocator();
  pthread_mutex_unlock(&rt.hsakmt_mutex);
  return result;
dxcore_loader_failed:
  close(fd);
open_failed:
  pthread_mutex_unlock(&rt.hsakmt_mutex);

  return result;
}

HSAKMT_STATUS HSAKMTAPI hsaKmtCloseKFD(void) {
  HSAKMT_STATUS result;
  auto &rt = dxg_runtime();

  pthread_mutex_lock(&rt.hsakmt_mutex);

  if (rt.dxg_open_count > 0) {
    if (--rt.dxg_open_count == 0) {
      wsl::thunk::dxcore::DxcoreLoader::Instance().Shutdown();
      rt.Reset();
    }

    result = HSAKMT_STATUS_SUCCESS;
  } else
    result = HSAKMT_STATUS_KERNEL_IO_CHANNEL_NOT_OPENED;

  pthread_mutex_unlock(&rt.hsakmt_mutex);

  return result;
}
