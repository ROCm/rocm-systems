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

#ifndef LIBRARY_SRC_GDA_NUMA_WRAPPER_HPP_
#define LIBRARY_SRC_GDA_NUMA_WRAPPER_HPP_

namespace rocshmem {

class NUMAWrapper;

extern NUMAWrapper numa;

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wvla-cxx-extension"

class NUMAWrapper {
  public:
    explicit NUMAWrapper();
    virtual ~NUMAWrapper();

    int bitmask_isbitset(const struct bitmask *bmp, unsigned int n);
    struct bitmask *get_mems_allowed(void);
    void set_preferred(int node);
    int num_configured_nodes();
    int num_configured_cpus(void);
    int node_of_cpu(int cpu);
    int max_node(void);
    long move_pages(int pid, unsigned long count, void *pages[count],
                    const int nodes[count], int status[count], int flags);
    int distance(int node1, int node2);

  private:
    struct numa_funcs_t {
      int (*bitmask_isbitset)(const struct bitmask *bmp, unsigned int n);
      struct bitmask* (*get_mems_allowed)(void);
      void (*set_preferred)(int node);
      int (*num_configured_nodes)();
      int (*num_configured_cpus)(void);
      int (*node_of_cpu)(int cpu);
      int (*max_node)(void);
      long (*move_pages)(int pid, unsigned long count, void *pages[count],
                         const int nodes[count], int status[count], int flags);
      int (*distance)(int node1, int node2);
    };

    /**
     * @brief External handle to the numa function table
     */
    struct numa_funcs_t numa;

    /**
     * @brief handle used for the dlopen of the InfiniBand Verbs library
     */
    void *numa_handle = nullptr;

    /**
     * @brief initialize function table
     */
    int init_function_table();
};

#pragma clang diagnostic pop

}  // namespace rocshmem

#endif  // LIBRARY_SRC_GDA_NUMA_WRAPPER_HPP_
