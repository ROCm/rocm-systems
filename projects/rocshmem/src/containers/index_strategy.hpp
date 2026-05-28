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

#ifndef LIBRARY_SRC_CONTAINERS_INDEX_STRATEGY_HPP_
#define LIBRARY_SRC_CONTAINERS_INDEX_STRATEGY_HPP_

/**
 * @file index_strategy.hpp
 *
 * @section
 * Q: What is an indexing strategy?
 *
 * A: The indexing strategy is a scheme used by __device__ code to access
 *    a container's raw memory region.
 *    The indexing strategy behaves like a simple forward iterator (in normal
 *    STL code).
 *
 * Q: What are the inputs and outputs of an indexing strategy?
 *
 * A: [INPUT]  an indexing strategy needs block and grid information
 *    [INPUT]  an indexing strategy needs to know the current index
 *    [INPUT]  an indexing strategy needs to know number of container elements
 *    [OUTPUT] an indexing strategy returns an index
 *
 * Q: What are the names of the strategies?
 *
 * A: Thread_Contiguous_Block_Agnostic
 *    Thread_Discontiguous_Block_Discontiguous
 *    Thread_Discontiguous_Block_Contiguous
 *
 * Q: What is the Thread_Contiguous_Block_Agnostic strategy?
 *
 * A: Thread_Contiguous_Block_Agnostic keeps a thread's memory accesses
 *    contiguous. Only one thread may access the container; the container
 *    is private to the thread (thread-private).
 *
 * Q: What is the Thread_Discontiguous_Block_Discontiguous strategy?
 *
 * A: Thread_Discontiguous_Block_Discontiguous does not keep a thread
 *    block's memory accesses contiguous. The container memory is accessed
 *    by multiple thread blocks and the accesses by different thread blocks
 *    are interleaved.
 *
 *    For example:
 *      assume grid_dim {8, 1, 1} block_dim {4, 1, 1}
 *      assume container._size = 72
 *
 *      In table below, '#' denotes thread block #'s accesses.
 *           0   1   2   3   4   5   6   7   8   9   10  11
 *         +---+---+---+---+---+---+---+---+---+---+---+---+
 *      0  | 0 | 0 | 0 | 0 | 1 | 1 | 1 | 1 | 2 | 2 | 2 | 2 |
 *         +---+---+---+---+---+---+---+---+---+---+---+---+
 *      1  | 3 | 3 | 3 | 3 | 4 | 4 | 4 | 4 | 5 | 5 | 5 | 5 |
 *         +---+---+---+---+---+---+---+---+---+---+---+---+
 *      2  | 6 | 6 | 6 | 6 | 7 | 7 | 7 | 7 | 0 | 0 | 0 | 0 |
 *         +---+---+---+---+---+---+---+---+---+---+---+---+
 *      3  | 1 | 1 | 1 | 1 | 2 | 2 | 2 | 2 | 3 | 3 | 3 | 3 |
 *         +---+---+---+---+---+---+---+---+---+---+---+---+
 *      4  | 4 | 4 | 4 | 4 | 5 | 5 | 5 | 5 | 6 | 6 | 6 | 6 |
 *         +---+---+---+---+---+---+---+---+---+---+---+---+
 *      5  | 7 | 7 | 7 | 7 | 0 | 0 | 0 | 0 | 1 | 1 | 1 | 1 |
 *         +---+---+---+---+---+---+---+---+---+---+---+---+
 *
 *      thread_id_00: grid{0, 0, 0} block{0, 0, 0}
 *      thread_id_00.2d_accesses := { {0,0},
 *                                    {2,8},
 *                                    {5,4} }
 *      thread_id_00.1d_accesses := { 0, 32, 64 }
 *
 *      thread_id_14: grid{3, 0, 0} block{2, 0, 0}
 *      thread_id_14.2d_accesses := { {1,2},
 *                                    {3,9} }
 *      thread_id_14.1d_accesses := { 14, 46 }
 *
 * Q: What is the Thread_Discontiguous_Block_Contiguous strategy?
 *
 * A: Thread_Discontiguous_Block_Contiguous does keep a thread block's memory
 *    accesses contiguous. The container memory is accessed by multiple
 *    thread blocks and the accesses by different thread blocks are
 *    restricted by artificial boundaries called 'tiles'.
 *
 *    For example:
 *      assume grid_dim {8, 1, 1} block_dim {4, 1, 1}
 *      assume container._size = 72
 *
 *      In table below, '#' denotes thread block #'s accesses.
 *           0   1   2   3   4   5   6   7   8   9   10  11
 *         +---+---+---+---+---+---+---+---+---+---+---+---+
 *      0  | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 1 | 1 |
 *         +---+---+---+---+---+---+---+---+---+---+---+---+
 *      1  | 1 | 1 | 1 | 1 | 1 | 1 | 2 | 2 | 2 | 2 | 2 | 2 |
 *         +---+---+---+---+---+---+---+---+---+---+---+---+
 *      2  | 2 | 2 | 2 | 3 | 3 | 3 | 3 | 3 | 3 | 3 | 3 | 3 |
 *         +---+---+---+---+---+---+---+---+---+---+---+---+
 *      3  | 4 | 4 | 4 | 4 | 4 | 4 | 4 | 4 | 4 | 5 | 5 | 5 |
 *         +---+---+---+---+---+---+---+---+---+---+---+---+
 *      4  | 5 | 5 | 5 | 5 | 5 | 5 | 6 | 6 | 6 | 6 | 6 | 6 |
 *         +---+---+---+---+---+---+---+---+---+---+---+---+
 *      5  | 6 | 6 | 6 | 7 | 7 | 7 | 7 | 7 | 7 | 7 | 7 | 7 |
 *         +---+---+---+---+---+---+---+---+---+---+---+---+
 *
 * Q: How do I use the strategies?
 *
 * A: You may use a strategy like this:
 *
 *      for (size_t i = _dev_idx.start();
 *           i < _dev_idx.end();
 *           i = _dev_idx.next(i)) {
 *          container[i] = ...
 *      }
 */

#include <hip/hip_runtime.h>
#include <cassert>

#include <algorithm>

namespace rocshmem {

class Identity {
 public:
  /**
   * @brief
   *
   * @return
   */
  __device__ size_t block_size() const {
    return hipBlockDim_x * hipBlockDim_y * hipBlockDim_z;
  }

  /**
   * @brief
   *
   * @return
   */
  __device__ size_t grid_size() const {
    return hipGridDim_x * hipGridDim_y * hipGridDim_z;
  }

  /**
   * @brief
   *
   * @return
   */
  __device__ size_t block_id() const {
    return hipBlockIdx_x + hipBlockIdx_y * hipGridDim_x +
           hipBlockIdx_z * hipGridDim_x * hipGridDim_y;
  }

  /**
   * @brief
   *
   * @return
   */
  __device__ size_t local_thread_id() const {
    return hipThreadIdx_x + hipThreadIdx_y * hipBlockDim_x +
           hipThreadIdx_z * hipBlockDim_x * hipBlockDim_y;
  }

  /**
   * @brief
   *
   * @return
   */
  __device__ size_t global_thread_id() const {
    return local_thread_id() + block_id() * block_size();
  }
};

class Thread_Contiguous_Block_Agnostic {
 public:
  /**
   * @brief
   */
  __host__ __device__ Thread_Contiguous_Block_Agnostic() = default;

  /**
   * @brief
   *
   * @param container_elems
   */
  __host__ __device__ Thread_Contiguous_Block_Agnostic(size_t container_elems)
      : _container_elems(container_elems) {}

  /**
   * @brief
   *
   * @return
   */
  __device__ size_t start() { return 0; }

  /**
   * @brief
   *
   * @return
   */
  __device__ size_t end() {
    assert(_container_elems);
    return _container_elems;
  }

  /**
   * @brief
   *
   * @param current_index
   *
   * @return
   */
  __device__ size_t next(size_t current_index) { return current_index + 1; }

 private:
  /**
   * @brief
   */
  size_t _container_elems{0};
};

class Thread_Discontiguous_Block_Discontiguous {
 public:
  /**
   * @brief
   */
  __host__ __device__ Thread_Discontiguous_Block_Discontiguous() = default;

  /**
   * @brief
   *
   * @param container_elems
   */
  __host__ __device__ explicit Thread_Discontiguous_Block_Discontiguous(
      size_t container_elems)
      : _container_elems(container_elems) {}

  /**
   * @brief
   *
   * @return
   */
  __device__ size_t start() { return _id.global_thread_id(); }

  /**
   * @brief
   *
   * @return
   */
  __device__ size_t end() {
    assert(_container_elems);
    return _container_elems;
  }

  /**
   * @brief
   *
   * @param current_index
   *
   * @return
   */
  __device__ size_t next(size_t current_index) {
    return current_index + _id.grid_size() * _id.block_size();
  }

 private:
  size_t _container_elems{0};

  Identity _id{};
};

class Thread_Discontiguous_Block_Contiguous {
 public:
  /**
   * @brief
   */
  __host__ __device__ Thread_Discontiguous_Block_Contiguous() = default;

  /**
   * @brief
   *
   * @param container_elems
   */
  __host__ __device__ explicit Thread_Discontiguous_Block_Contiguous(
      size_t container_elems)
      : _container_elems(container_elems) {
    size_t left_over = _container_elems % _id.grid_size();

    _tile_offset = _id.block_id() * (_container_elems / _id.grid_size()) +
                   min(_id.block_id(), left_over);

    _tile_num_elems = _container_elems / _id.grid_size();
    if (_id.block_id() < left_over) {
      _tile_num_elems++;
    }
  }

  /**
   * @brief
   *
   * @return
   */
  __device__ size_t start() {
    assert(_container_elems);
    return _tile_offset + _id.local_thread_id();
  }

  /**
   * @brief
   *
   * @return
   */
  __device__ size_t end() {
    assert(_container_elems);
    return _tile_offset + _tile_num_elems;
  }

  /**
   * @brief
   *
   * @param current_index
   *
   * @return
   */
  __device__ size_t next(size_t current_index) {
    assert(_container_elems);
    return current_index + _id.block_size();
  }

 private:
  size_t _container_elems{0};
  size_t _tile_offset{0};
  size_t _tile_num_elems{0};

  Identity _id{};
};

class Thread_Discontiguous_Block_Private {
 public:
  /**
   * @brief
   */
  __host__ __device__ Thread_Discontiguous_Block_Private() = default;

  /**
   * @brief
   *
   * @param container_elems
   */
  __host__ __device__ Thread_Discontiguous_Block_Private(size_t container_elems)
      : _container_elems(container_elems) {}

  /**
   * @brief
   *
   * @return
   */
  __device__ size_t start() { return _id.local_thread_id(); }

  /**
   * @brief
   *
   * @return
   */
  __device__ size_t end() {
    assert(_container_elems);
    return _container_elems;
  }

  /**
   * @brief
   *
   * @param current_index
   *
   * @return
   */
  __device__ size_t next(size_t current_index) {
    return current_index + _id.block_size();
  }

 private:
  size_t _container_elems{0};

  Identity _id{};
};

class Matrix_Private {
 public:
  /**
   * @brief
   */
  __host__ __device__ Matrix_Private() = default;

  /**
   * @brief
   *
   * @return
   */
  __device__ size_t start() { return _id.global_thread_id(); }

  /**
   * @brief
   *
   * @return
   */
  __device__ size_t end() {
    assert(false);
    return 0;
  }

  /**
   * @brief
   *
   * @param current_index
   *
   * @return
   */
  __device__ size_t next(size_t current_index) {
    assert(false);
    return 0;
  }

 private:
  Identity _id{};
};

class Matrix_Block {
 public:
  /**
   * @brief
   */
  __host__ __device__ Matrix_Block() = default;

  /**
   * @brief
   *
   * @return
   */
  __device__ size_t start() { return _id.block_id(); }

  /**
   * @brief
   *
   * @return
   */
  __device__ size_t end() {
    assert(false);
    return 0;
  }

  /**
   * @brief
   *
   * @param current_index
   *
   * @return
   */
  __device__ size_t next(size_t current_index) {
    assert(false);
    return 0;
  }

 private:
  Identity _id{};
};

class Matrix_Device {
 public:
  /**
   * @brief
   */
  __host__ __device__ Matrix_Device() = default;

  /**
   * @brief
   *
   * @return
   */
  __device__ size_t start() { return 0; }

  /**
   * @brief
   *
   * @return
   */
  __device__ size_t end() {
    assert(false);
    return 0;
  }

  /**
   * @brief
   *
   * @param current_index
   *
   * @return
   */
  __device__ size_t next(size_t current_index) {
    assert(false);
    return 0;
  }
};

enum class IndexStrategyEnum {
  TCBA = 0,
  TDBD = 1,
  TDBC = 2,
  TDBP = 3,
  MP = 4,
  MB = 5,
  MD = 6,
  UNSET = 7
};

class IndexStrategy {
 public:
  __host__ __device__ IndexStrategy() = default;

  __host__ __device__ IndexStrategy(IndexStrategyEnum ise) : _ise(ise) {}

  __host__ __device__ IndexStrategy(IndexStrategyEnum ise,
                                    size_t container_elems)
      : _ise(ise),
        _tcba(container_elems),
        _tdbd(container_elems),
        _tdbc(container_elems),
        _tdbp(container_elems) {}

  /**
   * @brief
   *
   * @return
   */
  __device__ size_t start() {
    switch (_ise) {
      case IndexStrategyEnum::TCBA:
        return _tcba.start();
      case IndexStrategyEnum::TDBD:
        return _tdbd.start();
      case IndexStrategyEnum::TDBC:
        return _tdbc.start();
      case IndexStrategyEnum::TDBP:
        return _tdbp.start();
      case IndexStrategyEnum::MP:
        return _mp.start();
      case IndexStrategyEnum::MB:
        return _mb.start();
      case IndexStrategyEnum::MD:
        return _md.start();
      case IndexStrategyEnum::UNSET:
        assert(false);
        return 0;
    }
  }

  /**
   * @brief
   *
   * @return
   */
  __device__ size_t end() {
    switch (_ise) {
      case IndexStrategyEnum::TCBA:
        return _tcba.end();
      case IndexStrategyEnum::TDBD:
        return _tdbd.end();
      case IndexStrategyEnum::TDBC:
        return _tdbc.end();
      case IndexStrategyEnum::TDBP:
        return _tdbp.end();
      case IndexStrategyEnum::MP:
        return _mp.end();
      case IndexStrategyEnum::MB:
        return _mb.end();
      case IndexStrategyEnum::MD:
        return _md.end();
      case IndexStrategyEnum::UNSET:
        assert(false);
        return 0;
    }
  }

  /**
   * @brief
   *
   * @param current_index
   *
   * @return
   */
  __device__ size_t next(size_t current_index) {
    switch (_ise) {
      case IndexStrategyEnum::TCBA:
        return _tcba.next(current_index);
      case IndexStrategyEnum::TDBD:
        return _tdbd.next(current_index);
      case IndexStrategyEnum::TDBC:
        return _tdbc.next(current_index);
      case IndexStrategyEnum::TDBP:
        return _tdbp.next(current_index);
      case IndexStrategyEnum::MP:
        return _mp.next(current_index);
      case IndexStrategyEnum::MB:
        return _mb.next(current_index);
      case IndexStrategyEnum::MD:
        return _md.next(current_index);
      case IndexStrategyEnum::UNSET:
        assert(false);
        return 0;
    }
  }

 private:
  IndexStrategyEnum _ise{IndexStrategyEnum::UNSET};
  Thread_Contiguous_Block_Agnostic _tcba{};
  Thread_Discontiguous_Block_Discontiguous _tdbd{};
  Thread_Discontiguous_Block_Contiguous _tdbc{};
  Thread_Discontiguous_Block_Private _tdbp{};
  Matrix_Private _mp{};
  Matrix_Block _mb{};
  Matrix_Device _md{};
};

}  // namespace rocshmem

#endif  // LIBRARY_SRC_CONTAINERS_INDEX_STRATEGY_HPP_
