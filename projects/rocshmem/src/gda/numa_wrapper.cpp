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

#include "numa_wrapper.hpp"
#include "util.hpp"

#include "rocshmem/rocshmem.hpp"

#include <dlfcn.h>
#include <numa.h> // If not found, try installing libnuma-dev (e.g apt-get install libnuma-dev)
#include <numaif.h>

namespace rocshmem {

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wvla-cxx-extension"

NUMAWrapper numa;

NUMAWrapper::NUMAWrapper() {
  int err;

  numa_handle = dlopen("libnuma.so", RTLD_NOW);

  if (!numa_handle) {
    DPRINTF("Could not open libnuma. Returning\n");
    exit(1);
  }

  err = init_function_table();
  if (err != ROCSHMEM_SUCCESS) {
    DPRINTF("Could not construct libnuma function table \n");
    exit(1);
  }
}

NUMAWrapper::~NUMAWrapper() {
  if (numa_handle != nullptr) {
    dlclose(numa_handle);
  }
}

int NUMAWrapper::init_function_table() {
  DLSYM_HELPER(numa, numa_, numa_handle, bitmask_isbitset);
  DLSYM_HELPER(numa, numa_, numa_handle, get_mems_allowed);
  DLSYM_HELPER(numa, numa_, numa_handle, set_preferred);
  DLSYM_HELPER(numa, numa_, numa_handle, num_configured_nodes);
  DLSYM_HELPER(numa, numa_, numa_handle, num_configured_cpus);
  DLSYM_HELPER(numa, numa_, numa_handle, node_of_cpu);
  DLSYM_HELPER(numa, numa_, numa_handle, max_node);
  DLSYM_HELPER(numa, , numa_handle, move_pages);
  DLSYM_HELPER(numa, numa_, numa_handle, distance);
  return ROCSHMEM_SUCCESS;
}

int NUMAWrapper::bitmask_isbitset(const struct bitmask *bmp, unsigned int n) {
  return numa.bitmask_isbitset(bmp, n);
}

struct bitmask * NUMAWrapper::get_mems_allowed(void) {
  return numa.get_mems_allowed();
}

void NUMAWrapper::set_preferred(int node) {
  return numa.set_preferred(node);
}

int NUMAWrapper::num_configured_nodes() {
  return numa.num_configured_nodes();
}

int NUMAWrapper::num_configured_cpus(void) {
  return numa.num_configured_cpus();
}

int NUMAWrapper::node_of_cpu(int cpu) {
  return numa.node_of_cpu(cpu);
}

int NUMAWrapper::max_node(void) {
  return numa.max_node();
}

long NUMAWrapper::move_pages(int pid, unsigned long count, void *pages[count],
                             const int nodes[count], int status[count], int flags) {
  return numa.move_pages(pid, count, pages, nodes, status, flags);
}

int NUMAWrapper::distance(int node1, int node2) {
  return numa.distance(node1, node2);
}

#pragma clang diagnostic pop

}  // namespace rocshmem
