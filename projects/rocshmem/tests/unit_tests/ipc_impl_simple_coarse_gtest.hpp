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

#ifndef ROCSHMEM_IPC_IMPL_SIMPLE_COARSE_GTEST_HPP
#define ROCSHMEM_IPC_IMPL_SIMPLE_COARSE_GTEST_HPP

#include "gtest/gtest.h"

#include <numeric>
#include <tuple>

#include <mpi.h>
#include "../src/memory/symmetric_heap.hpp"
#include "../src/ipc_policy.hpp"

#include <hip/hip_runtime.h>
#include <cassert>


namespace rocshmem {

__global__
void
kernel_simple_coarse_copy(IpcImpl *ipc_impl, int *src, int *dest, size_t bytes) {
    if (!threadIdx.x) {
      ipc_impl->ipcCopy(dest, src, bytes);
      ipc_impl->ipcFence();
    }
    __syncthreads();
}

__global__
void
kernel_simple_coarse_copy_block(IpcImpl *ipc_impl, int *src, int *dest, size_t bytes) {
    ipc_impl->ipcCopy_wg(dest, src, bytes);
    ipc_impl->ipcFence();
    __syncthreads();
}

__global__
void
kernel_simple_coarse_copy_warp(IpcImpl *ipc_impl, int *src, int *dest, size_t bytes) {
    ipc_impl->ipcCopy_wave(dest, src, bytes);
    ipc_impl->ipcFence();
    __syncthreads();
}

class IPCImplSimpleCoarse : public ::testing::TestWithParam<std::tuple<int, int, int>> {
    using HEAP_T = HeapMemory<HIPAllocator>;
    using MPI_T = RemoteHeapInfo<CommunicatorMPI>;
    using FN_T = void (*)(IpcImpl*, int*, int*, size_t);

  public:
    IPCImplSimpleCoarse() {
        MPIInstance::mpilib_dl_init();
        mpi_ = new MPI_T (heap_mem_.get_ptr(), heap_mem_.get_size(), MPI_COMM_WORLD);

        ipc_impl_.ipcHostInit(mpi_->my_pe(), mpi_->get_heap_bases(), MPI_COMM_WORLD);
        assert(ipc_impl_dptr_ == nullptr);
        hip_allocator_.allocate((void**)&ipc_impl_dptr_, sizeof(IpcImpl));
        CHECK_HIP(hipMemcpy(ipc_impl_dptr_, &ipc_impl_,
                            sizeof(IpcImpl), hipMemcpyHostToDevice));
    }

    virtual ~IPCImplSimpleCoarse() {
        if (ipc_impl_dptr_) {
            hip_allocator_.deallocate(ipc_impl_dptr_);
        }
        ipc_impl_.ipcHostStop();
        MPIInstance::mpilib_dl_close();
    }

    void launch(FN_T f, const dim3 grid, const dim3 block, int* src, int* dest, size_t bytes) {
        f<<<grid, block>>>(ipc_impl_dptr_, src, dest, bytes);
        CHECK_HIP(hipStreamSynchronize(nullptr));
    }

    enum TestType {
        READ = 0,
        WRITE = 1
    };

    virtual void copy(TestType test, dim3 grid, dim3 block) {
        FAIL();
    }

    void write(const dim3 grid, const dim3 block, size_t elems) {
        iota_golden(elems);
        initialize_src_buffer(WRITE);
        copy(WRITE, grid, block);
        validate_dest_buffer(WRITE);
    }

    void read(const dim3 grid, const dim3 block, size_t elems) {
        iota_golden(elems);
        initialize_src_buffer(READ);
        copy(READ, grid, block);
        validate_dest_buffer(READ);
    }

    void iota_golden(size_t elems) {
        golden_.resize(elems);
        std::iota(golden_.begin(), golden_.end(), 0);
    }

    void validate_golden(size_t elems) {
        ASSERT_EQ(golden_.size(), elems);
        for (int i = 0; i < static_cast<int>(golden_.size()); i++) {
            ASSERT_EQ(golden_[i], i);
        }
    }

    void initialize_src_buffer(TestType test) {
        if (!pe_initializes_src_buffer(test)) {
            return;
        }
        size_t bytes = golden_.size() * sizeof(int);
        auto dev_src = reinterpret_cast<int*>(ipc_impl_.ipc_bases[mpi_->my_pe()]);
        CHECK_HIP(hipMemcpy(dev_src, golden_.data(), bytes, hipMemcpyHostToDevice));
        CHECK_HIP(hipStreamSynchronize(nullptr));
    }

    bool pe_initializes_src_buffer(TestType test) {
        bool is_write_test = test;
        bool is_read_test = !test;
        return (is_write_test && mpi_->my_pe() == 0) ||
               (is_read_test && mpi_->my_pe() == 1);
    }

    void execute(TestType test, FN_T fn, const dim3 grid, const dim3 block) {
        if (mpi_->my_pe()) {
            mpi_->barrier();
            mpi_->barrier();
            return;
        }
        int *src{nullptr};
        int *dest{nullptr};
        if (test == WRITE) {
            src = reinterpret_cast<int*>(ipc_impl_.ipc_bases[0]);
            dest = reinterpret_cast<int*>(ipc_impl_.ipc_bases[1]);
        } else {
            src = reinterpret_cast<int*>(ipc_impl_.ipc_bases[1]);
            dest = reinterpret_cast<int*>(ipc_impl_.ipc_bases[0]);
        }
        size_t bytes = golden_.size() * sizeof(int);
        mpi_->barrier();
        launch(fn, grid, block, src, dest, bytes);
        mpi_->barrier();
    }

    void validate_dest_buffer(TestType test) {
        if (!pe_validates_dest_buffer(test)) {
            return;
        }

        auto dev_dest = reinterpret_cast<int*>(ipc_impl_.ipc_bases[mpi_->my_pe()]);
        for (int i = 0; i < static_cast<int>(golden_.size()); i++) {
            ASSERT_EQ(golden_[i], dev_dest[i]);
        }
    }

    bool pe_validates_dest_buffer(TestType test) {
        return !pe_initializes_src_buffer(test);
    }

  protected:
    std::vector<int> golden_;

    HEAP_T heap_mem_ {};
    MPI_T *mpi_{nullptr};

    IpcImpl ipc_impl_ {};
    IpcImpl *ipc_impl_dptr_ {nullptr};

    HIPAllocator hip_allocator_ {};
};

class DegenerateSimpleCoarse : public IPCImplSimpleCoarse {
  public:
    ~DegenerateSimpleCoarse() override {};
};

class ParameterizedBlockSimpleCoarse : public IPCImplSimpleCoarse {
  public:
    ~ParameterizedBlockSimpleCoarse() override {};

    void copy(IPCImplSimpleCoarse::TestType test, dim3 grid, dim3 block) override {
        execute(test, kernel_simple_coarse_copy_block, grid, block);
    }
};

class ParameterizedWarpSimpleCoarse : public IPCImplSimpleCoarse {
  public:
    ~ParameterizedWarpSimpleCoarse() override {};

    void copy(IPCImplSimpleCoarse::TestType test, dim3 grid, dim3 block) override {
        execute(test, kernel_simple_coarse_copy_warp, grid, block);
    }
};

class ParameterizedThreadSimpleCoarse : public IPCImplSimpleCoarse {
  public:
    ~ParameterizedThreadSimpleCoarse() override {};

    void copy(IPCImplSimpleCoarse::TestType test, dim3 grid, dim3 block) override {
        execute(test, kernel_simple_coarse_copy, grid, block);
    }
};

} // namespace rocshmem

#endif  // ROCSHMEM_IPC_IMPL_SIMPLE_COARSE_GTEST_HPP
