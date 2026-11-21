/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Flow credit scaling (NCCL 2.32.6, "Flow: scale credits across FIFO depths"): protocols sharing one
// ncclFlowConnState count credits in a common ncclFlowFifoSlots-step window. A 4-slice Simple connection
// advances two steps per slice, so the 8-step grant of an LL receiver authorizes only four Simple sends,
// and a Simple connection resuming from an odd LL step aligns up to a slice boundary and republishes credit.
// The connections loop back to rank 0 through a fake window whose peer pointer is the local pointer.

#include "DeviceTestBase.hpp"

#include "nccl_device/impl/core__funcs.h"
#include "nccl_device/impl/lsa_flow__funcs.h"

namespace RcclUnitTesting
{
namespace
{

constexpr size_t kStepSize    = 64;
constexpr size_t kStateOffset = ncclFlowFifoSlots * kStepSize;

__global__ void kernelLlGrantSimpleSends(ncclDevComm comm, ncclSymPtr<char> fifo,
                                         ncclSymPtr<ncclFlowConnState> state, int* granted)
{
  ncclLsaFlowConn<ncclFlowProtocolLL> recv(comm, fifo, state, /*peer=*/0, /*nSlices=*/ncclFlowFifoSlots,
                                           kStepSize, ncclFlowTypeRecv, /*recvPost=*/true);
  recv.close();
  ncclLsaFlowConn<ncclFlowProtocolSimple> send(comm, fifo, state, /*peer=*/0, /*nSlices=*/4, kStepSize,
                                               ncclFlowTypeSend, /*recvPost=*/false);
  int n = 0;
  for (int i = 0; i < ncclFlowFifoSlots; i++) {
    send.waitSend();
    // Out of credit: waitSend gave up on the (pre-set) abort flag.
    if (send.cache < send.step) break;
    n++;
  }
  *granted = n;
}

__global__ void kernelSimpleResumeAfterLl(ncclDevComm comm, ncclSymPtr<char> fifo, ncclSymPtr<ncclFlowConnState> state)
{
  ncclLsaFlowConn<ncclFlowProtocolSimple> recv(comm, fifo, state, /*peer=*/0, /*nSlices=*/4, kStepSize,
                                               ncclFlowTypeRecv, /*recvPost=*/true);
  recv.close();
}

class FlowCreditTest : public DeviceTestBase
{
protected:
  DeviceBuffer<char> backing{kStateOffset + sizeof(ncclFlowConnState)};
  DeviceBuffer<int> peerDelta4G{1};
  DeviceBuffer<ncclWindow_vidmem> win{1};
  DeviceBuffer<uint32_t> abortFlag{1};
  ncclDevComm comm{};

  void SetUp() override
  {
    DeviceTestBase::SetUp();
    backing.zero();
    peerDelta4G.zero();
    abortFlag.upload(1);
    ncclWindow_vidmem hostWin{};
    hostWin.lsaFlatBase    = backing.ptr;
    hostWin.lsaPeerDelta4G = peerDelta4G.ptr;
    win.copyFrom(&hostWin, 1);
    comm.abortFlag = abortFlag.ptr;
  }

  ncclSymPtr<char> fifo() const { return ncclSymPtr<char>(win.ptr, 0); }
  ncclSymPtr<ncclFlowConnState> state() const { return ncclSymPtr<ncclFlowConnState>(win.ptr, kStateOffset); }

  ncclFlowConnState readState() const
  {
    ncclFlowConnState s;
    HIP_EXPECT(hipMemcpy(&s, backing.ptr + kStateOffset, sizeof(s), hipMemcpyDeviceToHost));
    return s;
  }
};

// An LL receiver grants ncclFlowFifoSlots steps; a 4-slice Simple sender on the same state may use only
// four of them, or it would overrun its four-slice FIFO.
TEST_F(FlowCreditTest, LlGrantAuthorizesFourSimpleSlices)
{
  DeviceBuffer<int> granted(1);
  kernelLlGrantSimpleSends<<<1, 1>>>(comm, fifo(), state(), granted.ptr);
  syncAndCheck();
  EXPECT_EQ(readState().signal[ncclFlowTypeSend], (uint64_t)ncclFlowFifoSlots);
  EXPECT_EQ(granted.download(), 4);
}

// LL left the receive step odd (3) after publishing credit 11; the Simple receiver must align to step 4 and
// republish credit 12 even though that is below its usual batch.
TEST_F(FlowCreditTest, SimpleResumeAlignsStepAndRefreshesCredit)
{
  ncclFlowConnState s{};
  s.step[ncclFlowTypeRecv]        = 3;
  s.signalValue[ncclFlowTypeRecv] = 3 + ncclFlowFifoSlots;
  s.signal[ncclFlowTypeSend]      = 3 + ncclFlowFifoSlots;
  HIP_CHECK(hipMemcpy(backing.ptr + kStateOffset, &s, sizeof(s), hipMemcpyHostToDevice));

  kernelSimpleResumeAfterLl<<<1, 1>>>(comm, fifo(), state());
  syncAndCheck();

  s = readState();
  EXPECT_EQ(s.step[ncclFlowTypeRecv], 4u);
  EXPECT_EQ(s.signal[ncclFlowTypeSend], 4u + ncclFlowFifoSlots);
  EXPECT_EQ(s.signalValue[ncclFlowTypeRecv], 4u + ncclFlowFifoSlots);
}

} // namespace
} // namespace RcclUnitTesting
