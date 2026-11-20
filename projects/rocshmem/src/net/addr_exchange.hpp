/******************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *****************************************************************************/

#ifndef ROCSHMEM_LIBRARY_SRC_NET_ADDR_EXCHANGE_HPP_
#define ROCSHMEM_LIBRARY_SRC_NET_ADDR_EXCHANGE_HPP_

#include <cstddef>
#include <functional>
#include <type_traits>
#include <vector>

namespace rocshmem {
namespace net {

/**
 * @brief Fixed-size per-PE all-gather primitive, injected by the caller.
 *
 * Signature matches Backend::symm_allgather(void* inout, size_t bytes_per_pe):
 * @p inout is a contiguous array of num_pes slots each @p bytes_per_pe long,
 * with the calling PE's slot pre-filled. Because Backend::symm_allgather is
 * protected and already spans both the MPI and TCP-socket bootstraps, the
 * conduit (an object the backend owns) supplies a lambda over it. Keeping this
 * a callback leaves net/ free of Backend / MPI coupling and gives every conduit
 * both bootstrap modes for free.
 */
using AllgatherFn = std::function<void(void *inout, size_t bytes_per_pe)>;

/**
 * @brief All-gather one trivially-copyable value per PE.
 *
 * @tparam T        Trivially-copyable per-PE payload (e.g. a {base,rkey} POD).
 * @param mine      This PE's contribution.
 * @param num_pes   Number of PEs.
 * @param my_pe     This PE's rank.
 * @param allgather The injected all-gather (see AllgatherFn).
 * @return Vector of size num_pes; element i is PE i's value.
 */
template <typename T>
std::vector<T> allgather_value(const T &mine, int num_pes, int my_pe,
                               const AllgatherFn &allgather) {
  static_assert(std::is_trivially_copyable<T>::value,
                "allgather_value payload must be trivially copyable");
  std::vector<T> all(static_cast<size_t>(num_pes));
  all[static_cast<size_t>(my_pe)] = mine;
  allgather(all.data(), sizeof(T));
  return all;
}

/**
 * @brief All-gather a fixed-length array of values per PE.
 *
 * Some registrations publish more than one value per PE (e.g. one rkey per NIC).
 * @p mine has @p per_pe elements; the result has num_pes * per_pe elements laid
 * out PE-major (result[pe * per_pe + k]).
 */
template <typename T>
std::vector<T> allgather_array(const std::vector<T> &mine, int num_pes,
                               int my_pe, int per_pe,
                               const AllgatherFn &allgather) {
  static_assert(std::is_trivially_copyable<T>::value,
                "allgather_array payload must be trivially copyable");
  std::vector<T> all(static_cast<size_t>(num_pes) * per_pe);
  for (int k = 0; k < per_pe; k++) {
    all[static_cast<size_t>(my_pe) * per_pe + k] = mine[static_cast<size_t>(k)];
  }
  allgather(all.data(), sizeof(T) * per_pe);
  return all;
}

}  // namespace net
}  // namespace rocshmem

#endif  // ROCSHMEM_LIBRARY_SRC_NET_ADDR_EXCHANGE_HPP_
