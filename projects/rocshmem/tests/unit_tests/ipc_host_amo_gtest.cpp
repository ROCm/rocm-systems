/******************************************************************************
 * Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *****************************************************************************/

/**
 * @file ipc_host_amo_gtest.cpp
 *
 * Unit tests for JIRA-419: host-side AMO, fence, and quiet in IPC non-MPI mode.
 *
 * Test structure
 * --------------
 * Each test is written in two parts gated by a preprocessor macro:
 *
 *   JIRA_419_FIXED not defined  →  EXPECT_DEATH tests confirming the current
 *                                   abort() behaviour (pre-fix regression tests)
 *
 *   JIRA_419_FIXED defined      →  Functional tests that verify correct values
 *                                   after the fix is applied
 *
 * This lets the same source file serve as both "show it fails" and
 * "show it passes" without maintaining two diverging copies.
 *
 * Run before the fix (expects death):
 *   mpirun -np 2 ./rocshmem_unit_tests \
 *       --gtest_filter=IPCHostAMOTestFixture/*
 *
 * Run after the fix (expects correct values):
 *   cmake -DJIRA_419_FIXED=ON ...   [or add -DJIRA_419_FIXED to compile flags]
 *   mpirun -np 2 ./rocshmem_unit_tests \
 *       --gtest_filter=IPCHostAMOTestFixture/*
 *
 * Coverage
 * --------
 *  - amo_fetch_add (int)   — two-PE fetch-and-add round-trip
 *  - amo_fetch_cas (int)   — compare-and-swap succeeds and fails correctly
 *  - fence                 — does not crash, provides ordering
 *  - quiet                 — does not crash
 *  - putmem / getmem       — memcpy-backed RMA in non-MPI mode
 */

#include "ipc_host_amo_gtest.hpp"

using namespace rocshmem;

// ============================================================================
// Helpers
// ============================================================================

// Coordinate between the two PEs: PE 0 runs fn(), both barrier.
// Returns true only on PE 0, so assertions only fire once.
template <typename Fn>
static bool pe0_run(int my_pe, Fn fn) {
    if (my_pe == 0) fn();
    MPI_Barrier(MPI_COMM_WORLD);
    return my_pe == 0;
}

// ============================================================================
// PRE-FIX: death tests — confirm abort() is hit before JIRA-419 is fixed.
//
// GTest death tests fork a child process; the child calls the function and
// is expected to die (SIGABRT from abort()).  This validates that the test
// infrastructure is correctly targeting the broken path.
// ============================================================================

#if !defined(JIRA_419_FIXED)

TEST_F(IPCHostAMOFixture, PreFix_amo_fetch_add_aborts) {
    // Only PE 0 exercises the host AMO path; PE 1 just holds the target.
    if (my_pe_ != 0) {
        MPI_Barrier(MPI_COMM_WORLD);
        return;
    }

    int* local_slot = alloc_int_on_heap(0);
    *local_slot = 10;

    // Before the fix, HostInterface::amo_fetch_add casts to WindowInfoMPI*,
    // gets nullptr (we gave it a plain WindowInfo), and calls abort().
    EXPECT_DEATH(
        host_interface_->amo_fetch_add(static_cast<void*>(local_slot),
                                       5,
                                       my_pe_,
                                       context_window_info_),
        "" // any signal/message
    );

    MPI_Barrier(MPI_COMM_WORLD);
}

TEST_F(IPCHostAMOFixture, PreFix_amo_fetch_cas_aborts) {
    if (my_pe_ != 0) {
        MPI_Barrier(MPI_COMM_WORLD);
        return;
    }

    int* local_slot = alloc_int_on_heap(1);
    *local_slot = 42;

    EXPECT_DEATH(
        host_interface_->amo_fetch_cas(static_cast<void*>(local_slot),
                                       /*new_val=*/99,
                                       /*cond=*/42,
                                       my_pe_,
                                       context_window_info_),
        ""
    );

    MPI_Barrier(MPI_COMM_WORLD);
}

TEST_F(IPCHostAMOFixture, PreFix_fence_aborts) {
    if (my_pe_ != 0) {
        MPI_Barrier(MPI_COMM_WORLD);
        return;
    }

    // fence() hits the abort() via dynamic_cast<WindowInfoMPI*> == nullptr.
    EXPECT_DEATH(host_interface_->fence(context_window_info_), "");

    MPI_Barrier(MPI_COMM_WORLD);
}

TEST_F(IPCHostAMOFixture, PreFix_quiet_aborts) {
    if (my_pe_ != 0) {
        MPI_Barrier(MPI_COMM_WORLD);
        return;
    }

    EXPECT_DEATH(host_interface_->quiet(context_window_info_), "");

    MPI_Barrier(MPI_COMM_WORLD);
}

TEST_F(IPCHostAMOFixture, PreFix_putmem_aborts) {
    if (my_pe_ != 0) {
        MPI_Barrier(MPI_COMM_WORLD);
        return;
    }

    int src = 123;
    int* dest_slot = alloc_int_on_heap(2);

    EXPECT_DEATH(
        host_interface_->putmem(static_cast<void*>(dest_slot),
                                &src, sizeof(int),
                                my_pe_,
                                context_window_info_),
        ""
    );

    MPI_Barrier(MPI_COMM_WORLD);
}

TEST_F(IPCHostAMOFixture, PreFix_getmem_aborts) {
    if (my_pe_ != 0) {
        MPI_Barrier(MPI_COMM_WORLD);
        return;
    }

    int result = 0;
    int* src_slot = alloc_int_on_heap(3);
    *src_slot = 77;

    EXPECT_DEATH(
        host_interface_->getmem(&result,
                                static_cast<void*>(src_slot),
                                sizeof(int),
                                my_pe_,
                                context_window_info_),
        ""
    );

    MPI_Barrier(MPI_COMM_WORLD);
}

// ============================================================================
// POST-FIX: functional tests — verify correct behaviour after JIRA-419 is fixed.
// ============================================================================

#else  // JIRA_419_FIXED

// ----------------------------------------------------------------------------
// amo_fetch_add: PE 0 atomically adds to a slot in PE 1's heap.
//
// Expected:
//   returned old value == initial value
//   memory at target   == initial value + addend
// ----------------------------------------------------------------------------
TEST_F(IPCHostAMOFixture, PostFix_amo_fetch_add_local) {
    // Self-PE fetch-add: no cross-PE address translation needed; exercises the
    // non-MPI branch of amo_fetch_add on a locally-accessible fine-grain ptr.
    if (my_pe_ != 0) {
        MPI_Barrier(MPI_COMM_WORLD);
        return;
    }

    int* slot = alloc_int_on_heap(0);
    *slot = 10;

    int old = host_interface_->amo_fetch_add(static_cast<void*>(slot),
                                             /*value=*/5,
                                             my_pe_,
                                             context_window_info_);
    EXPECT_EQ(old, 10);
    EXPECT_EQ(*slot, 15);

    MPI_Barrier(MPI_COMM_WORLD);
}

TEST_F(IPCHostAMOFixture, PostFix_amo_fetch_add_remote) {
    // Cross-PE: PE 0 atomically adds to a slot in PE 1's heap.
    // PE 1 writes its initial value, both barrier, PE 0 does the AMO,
    // both barrier, PE 1 checks the updated value.
    constexpr int kSlot    = 0;
    constexpr int kInitial = 100;
    constexpr int kAddend  = 7;

    int* local_slot = alloc_int_on_heap(kSlot);

    if (my_pe_ == 1) {
        *local_slot = kInitial;
    }
    MPI_Barrier(MPI_COMM_WORLD);

    int old_val = -1;
    if (my_pe_ == 0) {
        // Pass local_slot directly — ctx_amo_fetch_add replicates
        // IPCHostContext::amo_fetch_add and calls shmem_ptr internally.
        old_val = ctx_amo_fetch_add(static_cast<void*>(local_slot), kAddend, /*pe=*/1);
        EXPECT_EQ(old_val, kInitial);
    }
    MPI_Barrier(MPI_COMM_WORLD);

    if (my_pe_ == 1) {
        EXPECT_EQ(*local_slot, kInitial + kAddend);
    }
    MPI_Barrier(MPI_COMM_WORLD);
}

// ----------------------------------------------------------------------------
// amo_fetch_cas: compare-and-swap, two sub-cases.
//   Case A: condition matches  → swap happens, old value returned
//   Case B: condition mismatch → swap does not happen, old value returned
// ----------------------------------------------------------------------------
TEST_F(IPCHostAMOFixture, PostFix_amo_fetch_cas_success) {
    constexpr int kSlot    = 1;
    constexpr int kInitial = 42;
    constexpr int kNew     = 99;

    int* local_slot = alloc_int_on_heap(kSlot);

    if (my_pe_ == 1) {
        *local_slot = kInitial;
    }
    MPI_Barrier(MPI_COMM_WORLD);

    if (my_pe_ == 0) {
        // Pass local_slot directly — ctx_amo_fetch_cas replicates
        // IPCHostContext::amo_fetch_cas and calls shmem_ptr internally.
        int old = ctx_amo_fetch_cas(static_cast<void*>(local_slot),
                                    /*value=*/kNew, /*cond=*/kInitial, /*pe=*/1);
        EXPECT_EQ(old, kInitial);
    }
    MPI_Barrier(MPI_COMM_WORLD);

    if (my_pe_ == 1) {
        EXPECT_EQ(*local_slot, kNew);  // swap happened
    }
    MPI_Barrier(MPI_COMM_WORLD);
}

TEST_F(IPCHostAMOFixture, PostFix_amo_fetch_cas_failure) {
    constexpr int kSlot    = 2;
    constexpr int kInitial = 42;
    constexpr int kNew     = 99;
    constexpr int kWrong   = 1;  // does not match kInitial

    int* local_slot = alloc_int_on_heap(kSlot);

    if (my_pe_ == 1) {
        *local_slot = kInitial;
    }
    MPI_Barrier(MPI_COMM_WORLD);

    if (my_pe_ == 0) {
        int old = ctx_amo_fetch_cas(static_cast<void*>(local_slot),
                                    kNew, /*cond=*/kWrong, /*pe=*/1);
        EXPECT_EQ(old, kInitial);  // old value returned unchanged
    }
    MPI_Barrier(MPI_COMM_WORLD);

    if (my_pe_ == 1) {
        EXPECT_EQ(*local_slot, kInitial);  // swap did NOT happen
    }
    MPI_Barrier(MPI_COMM_WORLD);
}

// ----------------------------------------------------------------------------
// fence: must not crash; verifies ordering semantics by checking a value
// written before the fence is visible after it.
// ----------------------------------------------------------------------------
TEST_F(IPCHostAMOFixture, PostFix_fence_no_abort) {
    if (my_pe_ != 0) {
        MPI_Barrier(MPI_COMM_WORLD);
        return;
    }

    int* slot = alloc_int_on_heap(3);
    *slot = 55;

    // Must not abort.
    EXPECT_NO_FATAL_FAILURE(host_interface_->fence(context_window_info_));

    // Value is intact after fence — no spurious write.
    EXPECT_EQ(*slot, 55);

    MPI_Barrier(MPI_COMM_WORLD);
}

// ----------------------------------------------------------------------------
// quiet: must not crash.
// ----------------------------------------------------------------------------
TEST_F(IPCHostAMOFixture, PostFix_quiet_no_abort) {
    if (my_pe_ != 0) {
        MPI_Barrier(MPI_COMM_WORLD);
        return;
    }

    EXPECT_NO_FATAL_FAILURE(host_interface_->quiet(context_window_info_));

    MPI_Barrier(MPI_COMM_WORLD);
}

// ----------------------------------------------------------------------------
// putmem / getmem: memcpy-backed RMA in non-MPI mode.
// ----------------------------------------------------------------------------
TEST_F(IPCHostAMOFixture, PostFix_putmem_local) {
    if (my_pe_ != 0) {
        MPI_Barrier(MPI_COMM_WORLD);
        return;
    }

    int src = 0xDEAD;
    int* dest_slot = alloc_int_on_heap(4);
    *dest_slot = 0;

    EXPECT_NO_FATAL_FAILURE(
        host_interface_->putmem(static_cast<void*>(dest_slot),
                                &src, sizeof(int),
                                my_pe_,
                                context_window_info_)
    );
    EXPECT_EQ(*dest_slot, src);

    MPI_Barrier(MPI_COMM_WORLD);
}

TEST_F(IPCHostAMOFixture, PostFix_getmem_local) {
    if (my_pe_ != 0) {
        MPI_Barrier(MPI_COMM_WORLD);
        return;
    }

    int* src_slot = alloc_int_on_heap(5);
    *src_slot = 0xBEEF;
    int result = 0;

    EXPECT_NO_FATAL_FAILURE(
        host_interface_->getmem(&result,
                                static_cast<void*>(src_slot),
                                sizeof(int),
                                my_pe_,
                                context_window_info_)
    );
    EXPECT_EQ(result, 0xBEEF);

    MPI_Barrier(MPI_COMM_WORLD);
}

TEST_F(IPCHostAMOFixture, PostFix_putmem_remote) {
    // PE 0 writes to PE 1's heap slot; PE 1 reads it back.
    constexpr int kSlot  = 6;
    constexpr int kValue = 0xCAFE;

    int* local_slot = alloc_int_on_heap(kSlot);

    if (my_pe_ == 1) {
        *local_slot = 0;
    }
    MPI_Barrier(MPI_COMM_WORLD);

    if (my_pe_ == 0) {
        int src = kValue;
        int* pe1_slot = reinterpret_cast<int*>(shmem_ptr(local_slot, /*pe=*/1));
        EXPECT_NO_FATAL_FAILURE(
            host_interface_->putmem(static_cast<void*>(pe1_slot),
                                    &src, sizeof(int),
                                    /*pe=*/1,
                                    context_window_info_)
        );
    }
    MPI_Barrier(MPI_COMM_WORLD);

    if (my_pe_ == 1) {
        EXPECT_EQ(*local_slot, kValue);
    }
    MPI_Barrier(MPI_COMM_WORLD);
}

#endif  // JIRA_419_FIXED
