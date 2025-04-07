/*
 * Copyright (C) 2016-2018 Advanced Micro Devices, Inc. All Rights Reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR
 * OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
 * ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
 * OTHER DEALINGS IN THE SOFTWARE.
 *
 */

#include "BaseDebug.hpp"
#include "KFDDBGTest.hpp"
#include <sys/ptrace.h>
#include <poll.h>
#include "hsakmt/linux/kfd_ioctl.h"
#include "KFDQMTest.hpp"
#include "PM4Queue.hpp"
#include "PM4Packet.hpp"
#include "Dispatch.hpp"
#include <string>

void KFDDBGTest::SetUp() {
    ROUTINE_START

    KFDBaseComponentTest::SetUp();

    ROUTINE_END
}

void KFDDBGTest::TearDown() {
    ROUTINE_START

    /* Reset the user trap handler */
    hsaKmtSetTrapHandler(m_NodeInfo.HsaDefaultGPUNode(), 0, 0, 0, 0);

    KFDBaseComponentTest::TearDown();

    ROUTINE_END
}

/*
 * To test debug attaching to a spawned process (i.e. attach prior to the tracee
 * opening a KFD device), have the child request the parent to PTRACE attach and
 * wait for the parent to debug attach then allow the child to runtime enable.
 *
 * The following will be exercised:
 * - The KFD shall create a KFD process on behalf of the tracee during debug
 *   attach since the tracee has not opened a KFD device.
 * - Runtime enable on the tracee shall raise an event to the debugging parent
 *   and block until parent has signalled that it has recieved the runtime
 *   enable event.
 * - Tracee should follow a similar hand shake for runtime disable and debug
 *   detach should follow.
 *
 * */
TEST_F(KFDDBGTest, AttachToSpawnedProcess) {
    TEST_START(TESTPROFILE_RUNALL)
    if (m_FamilyId >= FAMILY_AI) {

        if (hsaKmtCheckRuntimeDebugSupport()) {
            LOG() << "Skip test as debug API not supported";
            goto exit;
        }

        pid_t childPid = fork();

        if (childPid == 0) { /* Debugged process */
            uint32_t rDebug;
            int r;

            /* Let parent become the debugger and wait for attach. */
            ptrace(PTRACE_TRACEME);
            raise(SIGSTOP);

            r = hsaKmtOpenKFD();

            if (r != HSAKMT_STATUS_SUCCESS) {
                WARN() << "KFD open failed in debugged process" << std::endl;
                exit(1);
            }

            LOG() << std::dec << "--- Debugged PID " << getpid() << " runtime enable" << std::endl;

            r = hsaKmtRuntimeEnable(&rDebug, true);

            if (r != HSAKMT_STATUS_SUCCESS) {
                WARN() << "Runtime enabled failed" << std::endl;
                exit(1);
            }

            LOG() << std::dec << "--- Debugged PID " << getpid() << " runtime disable and exit" << std::endl;

            hsaKmtRuntimeDisable();

            exit(0);
        } else {
            BaseDebug *debug = new BaseDebug();
            struct kfd_runtime_info r_info;
            memset(&r_info, 0, sizeof(struct kfd_runtime_info));
            uint64_t runtimeMask = KFD_EC_MASK(EC_PROCESS_RUNTIME);
            int childStatus;

            waitpid(childPid, &childStatus, 0);
            while (!WIFSTOPPED(childStatus));

            /* Attach and let new debugged process continue with runtime enable */
            LOG() << std::dec << "Attaching to PID " << childPid  << std::endl;
            ASSERT_SUCCESS(debug->Attach(&r_info, sizeof(r_info), childPid, runtimeMask));
            ASSERT_EQ(r_info.runtime_state, DEBUG_RUNTIME_STATE_DISABLED);
            ASSERT_EQ(r_info.ttmp_setup, false);

            ptrace(PTRACE_CONT, childPid, NULL, NULL);

            /* Wait and unblock runtime enable */
            ASSERT_SUCCESS(debug->QueryDebugEvent(&runtimeMask, NULL, NULL, 5000));
            ASSERT_EQ(runtimeMask, KFD_EC_MASK(EC_PROCESS_RUNTIME));
            ASSERT_SUCCESS(debug->SendRuntimeEvent(runtimeMask, 0, 0));

            /* Wait and unblock runtime disable */
            ASSERT_SUCCESS(debug->QueryDebugEvent(&runtimeMask, NULL, NULL, 5000));
            ASSERT_EQ(runtimeMask, KFD_EC_MASK(EC_PROCESS_RUNTIME));
            ASSERT_SUCCESS(debug->SendRuntimeEvent(runtimeMask, 0, 0));

            LOG() << std::dec << "Detaching from PID " << childPid << std::endl;
            debug->Detach();

            ptrace(PTRACE_DETACH, childPid, NULL, NULL);

            LOG() << std::dec << "Waiting on PID " << childPid << " to exit" << std::endl;
            waitpid(childPid, &childStatus, 0);
            EXPECT_EQ(WIFEXITED(childStatus), true);
            EXPECT_EQ(WEXITSTATUS(childStatus), HSAKMT_STATUS_SUCCESS);
        }
    } else {
        LOG() << "Skipping test: Test not supported on family ID 0x"
              << m_FamilyId << "." << std::endl;
    }
exit:
    LOG() << std::endl;
    TEST_END
}

/*
 * Unlike AttachToSpawnedProcess, the debug parent will only attach after
 * a non-blocked runtime enable by the tracee.  The parent should expect
 * a status update that the tracee is runtime enabled on debug attach.
 * Cleanup with appropriate runtime disable and debug detach handshake.
 */
TEST_F(KFDDBGTest, AttachToRunningProcess) {
    TEST_START(TESTPROFILE_RUNALL)
    if (m_FamilyId >= FAMILY_AI) {

        if (hsaKmtCheckRuntimeDebugSupport()) {
            LOG() << "Skip test as debug API not supported";
            goto exit;
        }

    pid_t childPid = fork();

    if (childPid == 0) { /* Debugged process */
            uint32_t rDebug;
            int r;

            r = hsaKmtOpenKFD();

            if (r != HSAKMT_STATUS_SUCCESS) {
                WARN() << "KFD open failed in debugged process" << std::endl;
                exit(1);
             }

             LOG() << std::dec << "--- Debugged PID " << getpid() << " runtime enable" << std::endl;

             r = hsaKmtRuntimeEnable(&rDebug, true);
             if (r != HSAKMT_STATUS_SUCCESS) {
                 WARN() << "Runtime enabled failed" << std::endl;
                 exit(1);
             }

             /* Let parent become the debugger and wait for attach. */
             ptrace(PTRACE_TRACEME);
             raise(SIGSTOP);

             LOG() << std::dec << "--- Debugged PID " << getpid() << " runtime disable and exit" << std::endl;

             hsaKmtRuntimeDisable();

             exit(0);
        } else {
            BaseDebug *debug = new BaseDebug();
            struct kfd_runtime_info r_info;
            memset(&r_info, 0, sizeof(struct kfd_runtime_info));
            uint64_t runtimeMask = KFD_EC_MASK(EC_PROCESS_RUNTIME);
            int childStatus;

            waitpid(childPid, &childStatus, 0);
            while (!WIFSTOPPED(childStatus));

            /* Attach to running process and let it continue */
            LOG() << std::dec << "Attaching to PID " << childPid  << std::endl;
            ASSERT_SUCCESS(debug->Attach(&r_info, sizeof(r_info), childPid, runtimeMask));
            ASSERT_EQ(r_info.runtime_state, DEBUG_RUNTIME_STATE_ENABLED);
            ASSERT_EQ(r_info.ttmp_setup, true);

            ptrace(PTRACE_CONT, childPid, NULL, NULL);

            /* Wait and unblock runtime disable */
            ASSERT_SUCCESS(debug->QueryDebugEvent(&runtimeMask, NULL, NULL, 5000));
            ASSERT_EQ(runtimeMask, KFD_EC_MASK(EC_PROCESS_RUNTIME));
            ASSERT_SUCCESS(debug->SendRuntimeEvent(runtimeMask, 0, 0));

            LOG() << std::dec << "Detaching from PID " << childPid << std::endl;
            debug->Detach();

            ptrace(PTRACE_DETACH, childPid, NULL, NULL);

            LOG() << std::dec << "Waiting on PID " << childPid << " to exit" << std::endl;
            waitpid(childPid, &childStatus, 0);
            EXPECT_EQ(WIFEXITED(childStatus), true);
            EXPECT_EQ(WEXITSTATUS(childStatus), HSAKMT_STATUS_SUCCESS);
        }
    } else {
        LOG() << "Skipping test: Test not supported on family ID 0x"
              << m_FamilyId << "." << std::endl;
    }
exit:
    LOG() << std::endl;
    TEST_END
}

TEST_F(KFDDBGTest, HitTrapEvent) {
    TEST_START(TESTPROFILE_RUNALL)
    if (m_FamilyId >= FAMILY_AI) {
        int defaultGPUNode = m_NodeInfo.HsaDefaultGPUNode();

        if (hsaKmtCheckRuntimeDebugSupport()) {
            LOG() << "Skip test as debug API not supported";
            goto exit;
        }

        ASSERT_GE(defaultGPUNode, 0) << "failed to get default GPU Node";

        // create shader and trap bufs then enable 2nd level trap
        HsaMemoryBuffer isaBuf(PAGE_SIZE, defaultGPUNode, true, false, true);
        HsaMemoryBuffer trapStatusBuf(PAGE_SIZE, defaultGPUNode, true, false, false);

        HsaMemoryBuffer trap(PAGE_SIZE*2, defaultGPUNode, true, false, true);
        HsaMemoryBuffer tmaBuf(PAGE_SIZE, defaultGPUNode, false, false, false);

        ASSERT_SUCCESS(hsaKmtSetTrapHandler(defaultGPUNode,
                                            trap.As<void *>(),
                                            0x1000,
                                            tmaBuf.As<void*>(),
                                            0x1000));

        // compile and dispatch shader
        ASSERT_SUCCESS(m_pAsm->RunAssembleBuf(JumpToTrapIsa, isaBuf.As<char*>()));
        ASSERT_SUCCESS(m_pAsm->RunAssembleBuf(TrapHandlerIsa, trap.As<char*>()));

        uint32_t rDebug;
        ASSERT_SUCCESS(hsaKmtRuntimeEnable(&rDebug, true));

        BaseDebug *debug = new BaseDebug();
        struct kfd_runtime_info r_info;
        memset(&r_info, 0, sizeof(struct kfd_runtime_info));
        ASSERT_SUCCESS(debug->Attach(&r_info, sizeof(r_info), getpid(), 0));
        ASSERT_EQ(r_info.runtime_state, DEBUG_RUNTIME_STATE_ENABLED);

        PM4Queue queue;
        HsaQueueResource *qResources;
        ASSERT_SUCCESS(queue.Create(defaultGPUNode));

        unsigned int* trapStatus = trapStatusBuf.As<unsigned int*>();
        trapStatus[0] = 0;
        Dispatch *dispatch;
        dispatch = new Dispatch(isaBuf);
        dispatch->SetArgs(&trapStatus[0], NULL);
        dispatch->SetDim(1, 1, 1);

        /* Subscribe to trap events and submit the queue */
        uint64_t trapMask = KFD_EC_MASK(EC_QUEUE_WAVE_TRAP);
        debug->SetExceptionsEnabled(trapMask);
        dispatch->Submit(queue);

        /* Wait for trap event */
        uint32_t QueueId = -1;
        ASSERT_SUCCESS(debug->QueryDebugEvent(&trapMask, NULL, &QueueId, 5000));
        ASSERT_NE(QueueId, -1);
        ASSERT_EQ(trapMask, KFD_EC_MASK(EC_QUEUE_WAVE_TRAP) | KFD_EC_MASK(EC_QUEUE_NEW));

        dispatch->Sync();
        EXPECT_SUCCESS(queue.Destroy());

        ASSERT_NE(trapStatus[0], 0);

        debug->Detach();
        hsaKmtRuntimeDisable();

        delete dispatch;
    } else {
        LOG() << "Skipping test: Test not supported on family ID 0x"
              << m_FamilyId << "." << std::endl;
    }
exit:
    LOG() << std::endl;
    TEST_END
}

TEST_F(KFDDBGTest, HitTrapOnWaveStartEndEvent) {
    TEST_START(TESTPROFILE_RUNALL)
    if (m_FamilyId >= FAMILY_AI) {
        int defaultGPUNode = m_NodeInfo.HsaDefaultGPUNode();

        if (hsaKmtCheckRuntimeDebugSupport()) {
            LOG() << "Skip test as debug API not supported";
            goto exit;
        }

        ASSERT_GE(defaultGPUNode, 0) << "failed to get default GPU Node";

        // create shader and trap bufs then enable 2nd level trap
        HsaMemoryBuffer isaBuf(PAGE_SIZE, defaultGPUNode, true, false, true);
        HsaMemoryBuffer trap(PAGE_SIZE*2, defaultGPUNode, true, false, true);
        HsaMemoryBuffer tmaBuf(PAGE_SIZE, defaultGPUNode, false, false, false);

        ASSERT_SUCCESS(hsaKmtSetTrapHandler(defaultGPUNode,
                                            trap.As<void *>(),
                                            0x1000,
                                            tmaBuf.As<void*>(),
                                            0x1000));

        // compile and dispatch shader
        ASSERT_SUCCESS(m_pAsm->RunAssembleBuf(NoopIsa, isaBuf.As<char*>()));
        ASSERT_SUCCESS(m_pAsm->RunAssembleBuf(TrapHandlerIsa, trap.As<char*>()));

        uint32_t rDebug;
        ASSERT_SUCCESS(hsaKmtRuntimeEnable(&rDebug, true));

        BaseDebug *debug = new BaseDebug();
        struct kfd_runtime_info r_info;
        memset(&r_info, 0, sizeof(struct kfd_runtime_info));
        ASSERT_SUCCESS(debug->Attach(&r_info, sizeof(r_info), getpid(), 0));
        ASSERT_EQ(r_info.runtime_state, DEBUG_RUNTIME_STATE_ENABLED);

        PM4Queue queue;
        HsaQueueResource *qResources;
        ASSERT_SUCCESS(queue.Create(defaultGPUNode));

        for (int i = 0; i < 2; i++) {
            uint32_t enableMask = !!!(i % 2) ? KFD_DBG_TRAP_MASK_TRAP_ON_WAVE_START :
                                               KFD_DBG_TRAP_MASK_TRAP_ON_WAVE_END;
            uint32_t supportedMask = enableMask, reqMask = enableMask;
            debug->SetWaveLaunchOverride(KFD_DBG_TRAP_OVERRIDE_OR, &reqMask, &supportedMask);

            if (!!!(supportedMask & enableMask)) {
                EXPECT_SUCCESS(queue.Destroy());
                debug->Detach();
                hsaKmtRuntimeDisable();
                LOG() << "Skipping test: Trap on start/end override not supported." << std::endl;
                goto exit;
            }

	    // previous set mask
            ASSERT_EQ(reqMask, !!!(i % 2) ? 0 : KFD_DBG_TRAP_MASK_TRAP_ON_WAVE_START);

            Dispatch *dispatch;
            dispatch = new Dispatch(isaBuf);
            dispatch->SetArgs(NULL, NULL);
            dispatch->SetDim(1, 1, 1);

            /* Subscribe to trap events and submit the queue */
            uint64_t trapMask = KFD_EC_MASK(EC_QUEUE_WAVE_TRAP);
            debug->SetExceptionsEnabled(trapMask);
            dispatch->Submit(queue);

            /* Wait for trap event */
            uint32_t QueueId = -1;
            ASSERT_SUCCESS(debug->QueryDebugEvent(&trapMask, NULL, &QueueId, 5000));
            ASSERT_NE(QueueId, -1);
            ASSERT_EQ(trapMask, KFD_EC_MASK(EC_QUEUE_WAVE_TRAP) | KFD_EC_MASK(EC_QUEUE_NEW));

            dispatch->Sync();
            delete dispatch;
        }

        EXPECT_SUCCESS(queue.Destroy());

        debug->Detach();
        hsaKmtRuntimeDisable();
    } else {
        LOG() << "Skipping test: Test not supported on family ID 0x"
              << m_FamilyId << "." << std::endl;
    }
exit:
    LOG() << std::endl;
    TEST_END
}

TEST_F(KFDDBGTest, SuspendQueues) {
    TEST_START(TESTPROFILE_RUNALL)
    if (m_FamilyId >= FAMILY_AI) {
        int defaultGPUNode = m_NodeInfo.HsaDefaultGPUNode();

        if (hsaKmtCheckRuntimeDebugSupport()) {
            LOG() << "Skip test as debug API not supported";
            goto exit;
        }

        ASSERT_GE(defaultGPUNode, 0) << "failed to get default GPU Node";

        // create shader and trap bufs then enable 2nd level trap
        HsaMemoryBuffer isaBuf(PAGE_SIZE, defaultGPUNode, true, false, true);

        // compile and dispatch shader
        ASSERT_SUCCESS(m_pAsm->RunAssembleBuf(JumpToTrapIsa, isaBuf.As<char*>()));

        uint32_t rDebug;
        ASSERT_SUCCESS(hsaKmtRuntimeEnable(&rDebug, true));

        BaseDebug *debug = new BaseDebug();
        struct kfd_runtime_info r_info;
        memset(&r_info, 0, sizeof(struct kfd_runtime_info));
        ASSERT_SUCCESS(debug->Attach(&r_info, sizeof(r_info), getpid(), 0));
        ASSERT_EQ(r_info.runtime_state, DEBUG_RUNTIME_STATE_ENABLED);

        PM4Queue queue;
        HsaQueueResource *qResources;
        ASSERT_SUCCESS(queue.Create(defaultGPUNode));
        qResources = queue.GetResource();
        HSA_QUEUEID Queues[] = { qResources->QueueId };

        Dispatch *dispatch;
        dispatch = new Dispatch(isaBuf);
        dispatch->SetDim(1, 1, 1);
        dispatch->Submit(queue);

        uint32_t NumQueues = 1;
        uint32_t QueueIds[NumQueues];
        struct kfd_queue_snapshot_entry Snapshots[NumQueues];
        memset(Snapshots, 0, NumQueues * sizeof(struct kfd_queue_snapshot_entry));
        ASSERT_SUCCESS(debug->SuspendQueues(&NumQueues, Queues, &QueueIds[0], 0));

        // Suspend should fail as new queues cannot be suspended
        ASSERT_EQ(NumQueues, 0);
        ASSERT_NE(QueueIds[0] & KFD_DBG_QUEUE_INVALID_MASK, 0);

        // Snapshot queue, clear new queue status and suspend successfully.
        ASSERT_SUCCESS(debug->QueueSnapshot(0, (uint64_t)(&(Snapshots[0])), &NumQueues));
        ASSERT_EQ(NumQueues, 1);
        ASSERT_EQ(Snapshots[0].ctx_save_restore_area_size, 0);

        ASSERT_SUCCESS(debug->QueueSnapshot(KFD_EC_MASK(EC_QUEUE_NEW), (uint64_t)(&(Snapshots[0])),
                                            &NumQueues));
        ASSERT_EQ(NumQueues, 1);
        ASSERT_GT(Snapshots[0].ctx_save_restore_area_size, 0);

        ASSERT_SUCCESS(debug->SuspendQueues(&NumQueues, Queues, &QueueIds[0], 0));
        ASSERT_EQ(NumQueues, 1);
        ASSERT_EQ(QueueIds[0] & KFD_DBG_QUEUE_INVALID_MASK, 0);

        // Resume and destroy queue then clean up.
        ASSERT_SUCCESS(debug->ResumeQueues(&NumQueues, Queues, &QueueIds[0]));
        ASSERT_EQ(NumQueues, 1);
        ASSERT_EQ(QueueIds[0] & KFD_DBG_QUEUE_INVALID_MASK, 0);

        EXPECT_SUCCESS(queue.Destroy());

        debug->Detach();
        hsaKmtRuntimeDisable();

        delete dispatch;
    } else {
        LOG() << "Skipping test: Test not supported on family ID 0x"
              << m_FamilyId << "." << std::endl;
    }
exit:
    LOG() << std::endl;
    TEST_END
}

TEST_F(KFDDBGTest, HitMemoryViolation) {
    TEST_START(TESTPROFILE_RUNALL)
    if (m_FamilyId >= FAMILY_AI) {

        int defaultGPUNode = m_NodeInfo.HsaDefaultGPUNode();

        ASSERT_GE(defaultGPUNode, 0) << "failed to get default GPU Node";

        if (hsaKmtCheckRuntimeDebugSupport()) {
            LOG() << "Skip test as debug API not supported";
            goto exit;
        }

        pid_t childPid = fork();

        if (childPid == 0) { // Debugged process
            uint32_t rDebug;
            int r;

            // Refresh setup for HSA device and mem buffer use in child
            KFDBaseComponentTest::TearDown();
            KFDBaseComponentTest::SetUp();

            // Let parent become the debugger and wait for attach.
            ptrace(PTRACE_TRACEME);
            raise(SIGSTOP);

            r = hsaKmtRuntimeEnable(&rDebug, true);

            if (r != HSAKMT_STATUS_SUCCESS) {
                WARN() << "Runtime enabled failed" << std::endl;
                exit(1);
            }

            HsaMemoryBuffer isaBuf(PAGE_SIZE, defaultGPUNode, true, false, true);
            ASSERT_SUCCESS(m_pAsm->RunAssembleBuf(PersistentIterateIsa, isaBuf.As<char*>()));
            PM4Queue queue;
            HsaQueueResource *qResources;
            ASSERT_SUCCESS(queue.Create(defaultGPUNode));

            // Create memory violation event on dispatch
            HsaEvent *vmFaultEvent;
            HsaEventDescriptor eventDesc;
            eventDesc.EventType = HSA_EVENTTYPE_MEMORY;
            eventDesc.NodeId = defaultGPUNode;
            eventDesc.SyncVar.SyncVar.UserData = NULL;
            eventDesc.SyncVar.SyncVarSize = 0;
            r = hsaKmtCreateEvent(&eventDesc, true, false, &vmFaultEvent);

            if (r != HSAKMT_STATUS_SUCCESS) {
                WARN() << "Creating VM fault event failed" << std::endl;
                exit(1);
            }

            Dispatch dispatch(isaBuf);
            dispatch.SetDim(1, 1, 1);
            dispatch.SetPriv(false); //Override GFX11 CWSR WA
            dispatch.Submit(queue);

            // Queue immediately dies so halt process for tracer device inspection.
            raise(SIGSTOP);

            exit(0);
        } else {
            BaseDebug *debug = new BaseDebug();
            struct kfd_runtime_info r_info;
            memset(&r_info, 0, sizeof(struct kfd_runtime_info));
            uint64_t runtimeMask = KFD_EC_MASK(EC_PROCESS_RUNTIME);
            uint64_t memViolMask = KFD_EC_MASK(EC_DEVICE_MEMORY_VIOLATION);
            uint64_t subscribeMask = runtimeMask | memViolMask;
            uint64_t queryMask = 0;
            int childStatus;

            waitpid(childPid, &childStatus, 0);
            while (!WIFSTOPPED(childStatus));

            ASSERT_SUCCESS(debug->Attach(&r_info, sizeof(r_info), childPid, subscribeMask));
            ASSERT_EQ(r_info.runtime_state, DEBUG_RUNTIME_STATE_DISABLED);
            ASSERT_EQ(r_info.ttmp_setup, false);

            ptrace(PTRACE_CONT, childPid, NULL, NULL);

            // Wait and unblock runtime enable
            ASSERT_SUCCESS(debug->QueryDebugEvent(&runtimeMask, NULL, NULL, 5000));
            ASSERT_EQ(runtimeMask, KFD_EC_MASK(EC_PROCESS_RUNTIME));
            ASSERT_SUCCESS(debug->SendRuntimeEvent(runtimeMask, 0, 0));

            // Wait for memory violation
            uint32_t deviceId = -1;
            ASSERT_SUCCESS(debug->QueryDebugEvent(&queryMask, &deviceId, NULL, 5000));
            ASSERT_NE(deviceId, -1);
            ASSERT_EQ(queryMask, memViolMask);

            const std::vector<int> gpuNodes = m_NodeInfo.GetNodesWithGPU();
            uint32_t snapshotSize = gpuNodes.size();
            struct kfd_dbg_device_info_entry deviceInfo[snapshotSize];
            memset(deviceInfo, 0, snapshotSize * sizeof(struct kfd_dbg_device_info_entry));

            // Check device snapshot aligns with memory violation on target device.
            ASSERT_SUCCESS(debug->DeviceSnapshot(memViolMask, (uint64_t)(&deviceInfo[0]),
                                                 &snapshotSize));
            ASSERT_EQ(snapshotSize, gpuNodes.size());
            for (int i = 0; i < snapshotSize; i++) {
                if (deviceInfo[i].exception_status & memViolMask) {
                    ASSERT_EQ(deviceInfo[i].gpu_id, deviceId);
                    break;
                }
            }
            waitpid(childPid, &childStatus, 0);
            while (!WIFSTOPPED(childStatus));

            // Assume tracee queue has died and halted process
            ptrace(PTRACE_CONT, childPid, NULL, NULL);

            debug->Detach();

            ptrace(PTRACE_DETACH, childPid, NULL, NULL);

            waitpid(childPid, &childStatus, 0);
            EXPECT_EQ(WIFEXITED(childStatus), true);
            EXPECT_EQ(WEXITSTATUS(childStatus), HSAKMT_STATUS_SUCCESS);
        }
    } else {
        LOG() << "Skipping test: Test not supported on family ID 0x"
              << m_FamilyId << "." << std::endl;
    }
exit:
    LOG() << std::endl;
    TEST_END
}

TEST_F(KFDDBGTest, HitAddressWatch) {
    TEST_START(TESTPROFILE_RUNALL)
    if (m_FamilyId >= FAMILY_VI) {
        int defaultGPUNode = m_NodeInfo.HsaDefaultGPUNode();

        if (hsaKmtCheckRuntimeDebugSupport()) {
            LOG() << "Skip test as debug API not supported";
            goto exit;
        }

        ASSERT_GE(defaultGPUNode, 0) << "failed to get default GPU Node";
        HsaNodeProperties nodeProps;
        ASSERT_SUCCESS(hsaKmtGetNodeProperties(defaultGPUNode, &nodeProps));

        HsaMemoryBuffer readerBuf(PAGE_SIZE, defaultGPUNode, true, false, true);
        HsaMemoryBuffer writerBuf(PAGE_SIZE, defaultGPUNode, true, false, true);
        HsaMemoryBuffer trap(PAGE_SIZE*2, defaultGPUNode, true, false, true);
        HsaMemoryBuffer tmaBuf(PAGE_SIZE, defaultGPUNode, false, false, false);

        ASSERT_SUCCESS(m_pAsm->RunAssembleBuf(WatchReadIsa, readerBuf.As<char*>()));
        ASSERT_SUCCESS(m_pAsm->RunAssembleBuf(WatchWriteIsa, writerBuf.As<char*>()));
        ASSERT_SUCCESS(m_pAsm->RunAssembleBuf(TrapHandlerIsa, trap.As<char*>()));
        ASSERT_SUCCESS(hsaKmtSetTrapHandler(defaultGPUNode,
                                            trap.As<void *>(),
                                            0x1000,
                                            tmaBuf.As<void*>(),
                                            0x1000));

        uint32_t rDebug;
        ASSERT_SUCCESS(hsaKmtRuntimeEnable(&rDebug, true));

        struct kfd_runtime_info r_info;
        memset(&r_info, 0, sizeof(struct kfd_runtime_info));
        BaseDebug *debug = new BaseDebug();
        ASSERT_SUCCESS(debug->Attach(&r_info, sizeof(r_info), getpid(), 0));
        ASSERT_EQ(r_info.runtime_state, DEBUG_RUNTIME_STATE_ENABLED);

        const std::vector<int> gpuNodes = m_NodeInfo.GetNodesWithGPU();
        uint32_t numDevices = gpuNodes.size();
        struct kfd_dbg_device_info_entry deviceInfo[numDevices];
        memset(deviceInfo, 0, numDevices * sizeof(struct kfd_dbg_device_info_entry));
        ASSERT_SUCCESS(debug->DeviceSnapshot(0, (uint64_t)(&deviceInfo[0]), &numDevices));
        ASSERT_EQ(numDevices, gpuNodes.size());
        bool is_precise = nodeProps.Capability.ui32.PreciseMemoryOperationsSupported;

        if (is_precise) {
            uint32_t trapFlags = KFD_DBG_TRAP_FLAG_SINGLE_MEM_OP;
            ASSERT_SUCCESS(debug->SetFlags(&trapFlags));
        }

        uint32_t enableMask = KFD_DBG_TRAP_MASK_DBG_ADDRESS_WATCH;
        uint32_t supportedMask = enableMask;
        ASSERT_SUCCESS(debug->SetWaveLaunchOverride(KFD_DBG_TRAP_OVERRIDE_OR,
                                                    &enableMask,
                                                    &supportedMask));
        ASSERT_NE(supportedMask & KFD_DBG_TRAP_MASK_DBG_ADDRESS_WATCH, 0);
        ASSERT_EQ(enableMask & KFD_DBG_TRAP_MASK_DBG_ADDRESS_WATCH, 0); // previous set mask

        PM4Queue queue;
        ASSERT_SUCCESS(queue.Create(defaultGPUNode));
        const uint32_t watchMask = -1 & UINT_MAX;

        HsaMemoryBuffer targetBuf(PAGE_SIZE, defaultGPUNode, true, false, false);
        HsaMemoryBuffer resultBuf(PAGE_SIZE, defaultGPUNode, true, false, false);
        unsigned int *target = targetBuf.As<unsigned int*>();
        unsigned int *result = resultBuf.As<unsigned int*>();

        for (int mode = KFD_DBG_TRAP_ADDRESS_WATCH_MODE_READ;
                 mode < KFD_DBG_TRAP_ADDRESS_WATCH_MODE_ALL; mode++) {

            // atomics may not be supported on all devices so skip for now.
            if (mode != KFD_DBG_TRAP_ADDRESS_WATCH_MODE_READ &&
                mode != KFD_DBG_TRAP_ADDRESS_WATCH_MODE_NONREAD)
                continue;

            uint32_t watchId = -1;
            ASSERT_SUCCESS(debug->SetAddressWatch((uint64_t)(&target[0]), mode,
                                                  watchMask, deviceInfo[0].gpu_id, &watchId));
            ASSERT_EQ(watchId, 0);

            const HsaMemoryBuffer &shaderBuf =
                mode == KFD_DBG_TRAP_ADDRESS_WATCH_MODE_READ ? readerBuf : writerBuf;
            uint32_t preciseMask = 0x1;
            uint32_t watchStsMask = m_FamilyId >= FAMILY_GFX12 ? 0x1 : 0x80;
            result[0] = preciseMask;
            Dispatch dispatch(shaderBuf);
            dispatch.SetDim(1, 1, 1);
            dispatch.SetArgs(&target[0], &result[0]);
            dispatch.SetPriv(false); // Override GFX11 CWSR WA
            dispatch.Submit(queue);
            dispatch.Sync();

            /*
             * result[0] contains both the HW watch status result mask and the
             * precise memory operation value check (precise = 1, non-precise = 2)
             * For devices before GFX12, these masks did not bit wise overlap and
             * are added into result[0].
             * In GFX12 and above, they overlap and must be subtracted instead of masked
             * to assert the correct value.
             */
            if (m_FamilyId < FAMILY_GFX12) {
                ASSERT_EQ(result[0] & watchStsMask, watchStsMask);

                if (is_precise)
                    ASSERT_EQ(result[0] & preciseMask, preciseMask);
            } else {
                uint32_t maskCheck = result[0] - watchStsMask - preciseMask;
                ASSERT_EQ(maskCheck, is_precise ? 0 : 1);
            }

            ASSERT_SUCCESS(debug->ClearAddressWatch(deviceInfo[0].gpu_id, watchId));
            resultBuf.Fill(0);
            targetBuf.Fill(0);
        }

        ASSERT_SUCCESS(queue.Destroy());
        debug->Detach();
        hsaKmtRuntimeDisable();
    } else {
        LOG() << "Skipping test: Test not supported on family ID 0x"
              << m_FamilyId << "." << std::endl;
    }
exit:
    LOG() << std::endl;
    TEST_END
}
