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

#ifndef ROCSHMEM_IPC_IMPL_SIMPLE_FINE_GTEST_HPP
#define ROCSHMEM_IPC_IMPL_SIMPLE_FINE_GTEST_HPP

#include "gtest/gtest.h"

#include <numeric>
#include <mpi.h>

#include "../src/atomic.hpp"
#include "../src/ipc_policy.hpp"
#include "../src/memory/notifier.hpp"
#include "../src/memory/symmetric_heap.hpp"
#include "../src/util.hpp"

#include <hip/hip_runtime.h>
#include <cassert>

namespace rocshmem {

const int WARP_SIZE = 64;

const int THREAD_TRANSFER_GRANULARITY = 8;  // DWORDX2

int warpsPerBlock(size_t block_size) {
    return ((block_size - 1) + WARP_SIZE) / WARP_SIZE;
}

// set signal pointer to ipc_impl.ipc_bases location at unused offset
// be careful with test size to not overrun this location
const uint32_t SIGNAL_OFFSET {67108864};

enum TestType {
    READ = 0,
    WRITE = 1
};

__device__
void
tiled_validator(bool *error, int *golden, int *dest, size_t bytes) {
    size_t elements {bytes / sizeof(int)};
    for (size_t i = get_flat_id(); i < elements; i += get_flat_grid_size()) {
        if (golden[i] != dest[i]) {
            printf("golden[%zu] %d != dest[%zu] %d\n", i, golden[i], i, dest[i]);
            *error = true;
        }
    }
}

template <typename NotifierT>
__global__
void
kernel_put_with_signal_tiled_validator(bool *error, int *golden, int *dest, size_t bytes, NotifierT *notifier) {
    detail::atomic::rocshmem_memory_orders orders{};
    if (!get_flat_id()) {
        while (detail::atomic::load<int, detail::atomic::memory_scope_system>(dest + SIGNAL_OFFSET, orders) != 0) {
            ;
        }
    }
    notifier->sync();
    tiled_validator(error, golden, dest, bytes);
}

template <typename NotifierT>
__global__
void
kernel_tiled_fine_copy(IpcImpl *ipc_impl, bool *error, int *golden, int *src, int *dest, size_t bytes, TestType test, NotifierT *notifier) {
    if (!get_flat_id()) {
        ipc_impl->ipcCopy(dest, src, bytes);
        ipc_impl->ipcFence();
        if (test == WRITE) {
            ipc_impl->ipcAMOFetchAdd(dest + SIGNAL_OFFSET, -1);
        }
    }
    if (test == READ) {
        notifier->sync();
        tiled_validator(error, golden, dest, bytes);
    }
}

template <typename NotifierT>
__global__
void
kernel_tiled_fine_copy_block(IpcImpl *ipc_impl, bool *error, int *golden, int *src, int *dest, size_t bytes, TestType test, NotifierT *notifier) {
    size_t block_bytes = blockDim.x * THREAD_TRANSFER_GRANULARITY;
    size_t block_byte_offset = blockIdx.x * block_bytes;
    for (size_t i = block_byte_offset; i < bytes; i += get_flat_grid_size() * THREAD_TRANSFER_GRANULARITY) {
	      int chunk = min(block_bytes, bytes - i);
        ipc_impl->ipcCopy_wg((char*)dest + i, (char*)src + i, chunk);
        ipc_impl->ipcFence();
        __syncthreads();
        if (test == WRITE) {
            if (!threadIdx.x) {
                ipc_impl->ipcAMOFetchAdd(dest + SIGNAL_OFFSET, -1);
            }
        }
    }
    if (test == READ) {
        notifier->sync();
        tiled_validator(error, golden, dest, bytes);
    }
}

template <typename NotifierT>
__global__
void
kernel_tiled_fine_copy_warp(IpcImpl *ipc_impl, bool *error, int *golden, int *src, int *dest, size_t bytes, TestType test, NotifierT *notifier) {
    size_t warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
    size_t warp_bytes = WARP_SIZE * THREAD_TRANSFER_GRANULARITY;
    size_t warp_byte_offset = warp_id * warp_bytes;
    for (size_t i = warp_byte_offset; i < bytes; i += get_flat_grid_size() * THREAD_TRANSFER_GRANULARITY) {
        int chunk = min(warp_bytes, bytes - i);
        ipc_impl->ipcCopy_wave(((char*)dest) + i, ((char*)src) + i, chunk);
        ipc_impl->ipcFence();
        if (test == WRITE) {
            if (!(threadIdx.x % WARP_SIZE)) {
                ipc_impl->ipcAMOFetchAdd(dest + SIGNAL_OFFSET, -1);
            }
        }
    }
    __syncthreads();
    if (test == READ) {
        notifier->sync();
        tiled_validator(error, golden, dest, bytes);
    }
}

class IPCImplTiledFine : public ::testing::TestWithParam<std::tuple<int, int, int>> {
    using HEAP_T = HeapMemory<HIPDefaultFinegrainedAllocator>;
    using MPI_T = RemoteHeapInfo<CommunicatorMPI>;
    using NotifierT = Notifier<detail::atomic::memory_scope_agent>;
    using NotifierProxyT = NotifierProxy<HIPAllocator, detail::atomic::memory_scope_agent>;
    using FN_T1 = void (*)(IpcImpl*, bool*, int*, int*, int*, size_t, TestType, NotifierT*);
    using FN_T2 = void (*)(bool*, int*, int*, size_t, NotifierT*);

  public:
    IPCImplTiledFine() {
        MPIInstance::mpilib_dl_init();
        mpi_ = new MPI_T (heap_mem_.get_ptr(), heap_mem_.get_size(), MPI_COMM_WORLD);

        ipc_impl_.ipcHostInit(mpi_->my_pe(), mpi_->get_heap_bases(), MPI_COMM_WORLD);

        assert(ipc_impl_dptr_ == nullptr);
        hip_allocator_.allocate((void**)&ipc_impl_dptr_, sizeof(IpcImpl));
        CHECK_HIP(hipMemcpy(ipc_impl_dptr_, &ipc_impl_, sizeof(IpcImpl), hipMemcpyHostToDevice));

        assert(error_dptr_ == nullptr);
        hip_allocator_.allocate((void**)&error_dptr_, sizeof(bool));
        *error_dptr_ = false;
    }

    ~IPCImplTiledFine() {
        if (ipc_impl_dptr_) {
            hip_allocator_.deallocate(ipc_impl_dptr_);
        }
        if (error_dptr_) {
            hip_allocator_.deallocate(error_dptr_);
        }
        if (golden_dptr_) {
            hip_allocator_.deallocate(golden_dptr_);
        }

        ipc_impl_.ipcHostStop();
        MPIInstance::mpilib_dl_close();
    }

    void launch(FN_T1 f, const dim3 grid, const dim3 block, int* src, int* dest, size_t bytes, TestType test) {
        f<<<grid, block>>>(ipc_impl_dptr_, error_dptr_, golden_dptr_, src, dest, bytes, test, notifier_.get());
        CHECK_HIP(hipStreamSynchronize(nullptr));
    }

    void launch(FN_T2 f, const dim3 grid, const dim3 block, int* dest, size_t bytes) {
        f<<<grid, block>>>(error_dptr_, golden_dptr_, dest, bytes, notifier_.get());
        CHECK_HIP(hipStreamSynchronize(nullptr));
    }

    virtual void copy(TestType test, dim3 grid, dim3 block) {
        FAIL();
    }

    void write(const dim3 grid, const dim3 block, size_t elems, int signal_value) {
        iota_golden(elems);
        initialize_signal(WRITE, signal_value);
        initialize_src_buffer(WRITE);
        copy(WRITE, grid, block);
        check_device_validation_errors(WRITE);
    }

    void read(const dim3 grid, const dim3 block, size_t elems) {
        iota_golden(elems);
        initialize_signal(READ);
        initialize_src_buffer(READ);
        copy(READ, grid, block);
        check_device_validation_errors(READ);
    }

    void iota_golden(size_t elems) {
        golden_.resize(elems);
        std::iota(golden_.begin(), golden_.end(), 0);

        assert(golden_dptr_ == nullptr);
        size_t golden_dptr_bytes {golden_.size() * sizeof(int)};
        hip_allocator_.allocate((void**)&golden_dptr_, golden_dptr_bytes);
        CHECK_HIP(hipMemcpy(golden_dptr_, golden_.data(), golden_dptr_bytes, hipMemcpyHostToDevice));
    }

    void validate_golden(size_t elems) {
        ASSERT_EQ(golden_.size(), elems);
        for (int i = 0; i < static_cast<int>(golden_.size()); i++) {
            ASSERT_EQ(golden_[i], i);
        }
    }

    void initialize_signal(TestType test, int signal_value = 0) {
        bool is_write_test = test;
        if (is_write_test && mpi_->my_pe() == 0) {
            int *dest = reinterpret_cast<int*>(ipc_impl_.ipc_bases[1]);
            *(dest + SIGNAL_OFFSET) = signal_value;
        }
    }

    void initialize_src_buffer(TestType test) {
        if (!pe_initializes_src_buffer(test)) {
            return;
        }
        size_t bytes = golden_.size() * sizeof(int);
        auto dev_src = reinterpret_cast<int*>(ipc_impl_.ipc_bases[mpi_->my_pe()]);
        CHECK_HIP(hipMemcpy(dev_src, golden_.data(), bytes, hipMemcpyHostToDevice));
    }

    bool pe_initializes_src_buffer(TestType test) {
        bool is_write_test = test;
        bool is_read_test = !test;
        return (is_write_test && mpi_->my_pe() == 0) ||
               (is_read_test && mpi_->my_pe() == 1);
    }

    void execute(TestType test, FN_T1 fn, const dim3 grid, const dim3 block) {
        size_t bytes = golden_.size() * sizeof(int);
        if (mpi_->my_pe()) {
            mpi_->barrier();
            if (test == WRITE) {
                int *dest = reinterpret_cast<int*>(ipc_impl_.ipc_bases[1]);
                FN_T2 val_fn = kernel_put_with_signal_tiled_validator;
                launch(val_fn, grid, block, dest, bytes);
                ASSERT_EQ(*(dest + SIGNAL_OFFSET), 0);
            }
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
        mpi_->barrier();
        launch(fn, grid, block, src, dest, bytes, test);
        mpi_->barrier();
    }

    void check_device_validation_errors(TestType test) {
        if (!pe_validates_dest_buffer(test)) {
            return;
        }
        ASSERT_EQ(*error_dptr_, false);
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
    HIPDefaultFinegrainedAllocator hip_allocator_ {};

    NotifierProxyT notifier_ {};

    HEAP_T heap_mem_ {};

    MPI_T *mpi_ {nullptr};

    std::vector<int> golden_;

    int *golden_dptr_ {nullptr};

    IpcImpl ipc_impl_ {};

    IpcImpl *ipc_impl_dptr_ {nullptr};

    bool *error_dptr_ {nullptr};
};

class DegenerateTiledFine : public IPCImplTiledFine {
  public:
    ~DegenerateTiledFine() override {};
};

class ParameterizedBlockTiledFine : public IPCImplTiledFine {
  public:
    ~ParameterizedBlockTiledFine() override {};

    void copy(TestType test, dim3 grid, dim3 block) override {
        execute(test, kernel_tiled_fine_copy_block, grid, block);
    }
};

class ParameterizedWarpTiledFine : public IPCImplTiledFine {
  public:
    ~ParameterizedWarpTiledFine() override {};

    void copy(TestType test, dim3 grid, dim3 block) override {
        execute(test, kernel_tiled_fine_copy_warp, grid, block);
    }
};

class ParameterizedThreadTiledFine : public IPCImplTiledFine {
  public:
    ~ParameterizedThreadTiledFine() override {};

    void copy(TestType test, dim3 grid, dim3 block) override {
        execute(test, kernel_tiled_fine_copy, grid, block);
    }
};

} // namespace rocshmem

#endif  // ROCSHMEM_IPC_IMPL_SIMPLE_FINE_GTEST_HPP
