# IB-MSEG host tests, included from transport/host_tests.cmake. Each binary is the matching net_ib one against the
# fork's units and fakes: its test file includes the net_ib test file and adds the fork's tests. The fork declares
# its parameters extern, so each binary links the net_ib_mseg_params*.cc for the units it compiles for real.
include("${CMAKE_CURRENT_LIST_DIR}/../derived_transport_tests.cmake")

rccl_derived_transport_tests(
  DERIVED net_ib_multiseg
  HEADER_SUFFIX _mseg
  SELF_CHECK_SUITE NetIbMsegBaseTestExclusions
  BASE net_ib
    RENAME_MAP "${CMAKE_CURRENT_LIST_DIR}/../../../../tools/scripts/net_ib_multiseg/rename_map.txt"
    SEAM_DIR "${CMAKE_CURRENT_LIST_DIR}/shim"
    # connect-test.cc uses these outside its seam window; gin-test.cc #undefs them before its window.
    KEEP_ALIASES ncclIbAcceptImpl ncclIbConnectImpl ncclIbCreateFlushQp ncclIbRegMrDmaBufInternal ncclIbSetTrafficClass)

rccl_add_derived_transport_binary(rccl-UnitTestsMicroTransportNetIbMsegConnect net_ib_multiseg/connect-test.cc
  DERIVED net_ib_multiseg
  BASE net_ib
  EXTRA_DEFS RCCL_HOST_TEST_NET_IB_MSEG
    # NetIbMsegParamCopies compares the copies with the unhipified net_ib sources.
    NET_IB_MSEG_PARAM_COPIES_DIR="${CMAKE_CURRENT_SOURCE_DIR}/fakes"
    NET_IB_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}/../../src/transport/net_ib"
  SOURCES
    fakes/comm_fakes.cc
    fakes/env_fakes.cc
    fakes/gdr_peermem_fakes.cc
    fakes/gdr_probe_fakes.cc
    fakes/hip_fakes.cc
    fakes/hsa_fakes.cc
    fakes/ibvwrap_fakes.cc
    fakes/mlx5dvwrap_fakes.cc
    fakes/nccl_fakes.cc
    fakes/net_ib_mseg_params.cc
    fakes/net_ib_mseg_params_connect.cc
    fakes/net_ib_mseg_p2p_multiseg_fakes.cc
    fakes/os_fakes.cc
    fakes/rccl_mseg_fakes.cc
    fakes/rocmwrap_fakes.cc
    fakes/socket_fakes.cc
    fakes/transport_net_ib_p2p_fakes.cc
    fakes/transport_net_ib_recovery_fakes.cc
    fakes/transport_net_ib_resiliency_fakes.cc
    fakes/transport_net_ib_wqe_lat_fakes.cc
    ${RCCL_HOST_HIPIFY_DIR}/src/misc/utils.cc
    ${RCCL_HOST_HIPIFY_DIR}/src/transport/net_ib_multiseg/common_tmp.cc
    ${RCCL_HOST_HIPIFY_DIR}/src/transport/net_ib_multiseg/gdr_tmp.cc
    ${RCCL_HOST_HIPIFY_DIR}/src/transport/net_ib_multiseg/init_tmp.cc
    ${RCCL_HOST_HIPIFY_DIR}/src/transport/net_ib_multiseg/reg_tmp.cc
    ${RCCL_HOST_HIPIFY_DIR}/src/transport/net_ib_multiseg/seg_p2p.cc)

rccl_add_derived_transport_binary(rccl-UnitTestsMicroTransportNetIbMsegGin net_ib_multiseg/gin-test.cc
  DERIVED net_ib_multiseg
  BASE net_ib
  EXTRA_DEFS RCCL_HOST_TEST_NET_IB_MSEG
  SOURCES
    fakes/comm_fakes.cc
    fakes/env_fakes.cc
    fakes/gdr_peermem_fakes.cc
    fakes/gdr_probe_fakes.cc
    fakes/hip_fakes.cc
    fakes/hsa_fakes.cc
    fakes/ibvwrap_fakes.cc
    fakes/mlx5dvwrap_fakes.cc
    fakes/nccl_fakes.cc
    fakes/nccl_stubs.cc
    fakes/net_ib_mseg_params.cc
    fakes/net_ib_mseg_params_connect.cc
    fakes/net_ib_mseg_params_p2p.cc
    fakes/os_fakes.cc
    fakes/rccl_mseg_fakes.cc
    fakes/rocmwrap_fakes.cc
    fakes/socket_fakes.cc
    fakes/transport_net_ib_recovery_fakes.cc
    fakes/transport_net_ib_resiliency_fakes.cc
    fakes/transport_net_ib_wqe_lat_fakes.cc
    ${RCCL_HOST_HIPIFY_DIR}/src/misc/utils.cc
    ${RCCL_HOST_HIPIFY_DIR}/src/transport/net_ib_multiseg/common_tmp.cc
    ${RCCL_HOST_HIPIFY_DIR}/src/transport/net_ib_multiseg/connect_tmp.cc
    ${RCCL_HOST_HIPIFY_DIR}/src/transport/net_ib_multiseg/gdr_tmp.cc
    ${RCCL_HOST_HIPIFY_DIR}/src/transport/net_ib_multiseg/init_tmp.cc
    ${RCCL_HOST_HIPIFY_DIR}/src/transport/net_ib_multiseg/p2p_multiseg.cc  # p2p.cc calls its multi-segment hooks
    ${RCCL_HOST_HIPIFY_DIR}/src/transport/net_ib_multiseg/p2p_tmp.cc
    ${RCCL_HOST_HIPIFY_DIR}/src/transport/net_ib_multiseg/reg_tmp.cc
    ${RCCL_HOST_HIPIFY_DIR}/src/transport/net_ib_multiseg/seg_p2p.cc)

rccl_add_derived_transport_binary(rccl-UnitTestsMicroTransportNetIbMsegP2p net_ib_multiseg/net-ib-p2p-test.cc
  DERIVED net_ib_multiseg
  BASE net_ib
  EXTRA_DEFS RCCL_HOST_TEST_NET_IB_MSEG
  EXCLUDE_BASE_TESTS
    NetIbMsegP2pMicrotest.MultiSend_SmallPayloadUsesOnlyRotatedActiveQp replaced  # ctsFifo is a pointer
    NetIbMsegP2pMicrotest.MultiSend_WeightedPayloadAlignsChunksAndClampsFinalBytes replaced  # ctsFifo is a pointer
    NetIbMsegP2pMicrotest.MultiSend_PayloadAtSplitThresholdUsesWeightedPath replaced  # ctsFifo is a pointer
    NetIbMsegP2pMicrotest.MultiSend_RequestRotationSelectsOwningActiveQp replaced  # ctsFifo is a pointer
    NetIbMsegP2pMicrotest.Isend_MatchedReceivePublishesClampedRequestAndAdvancesFifo replaced  # ctsFifo is a pointer
    NetIbMsegP2pMicrotest.Isend_MultiReceiveDefersUntilEveryTagMatchesThenPostsBatch replaced  # ctsFifo is a pointer
    NetIbMsegP2pMicrotest.Isend_PostFailureWithEventsReturnsFailedRequestForDrain replaced  # ctsFifo is a pointer
    NetIbMsegP2pMicrotest.Isend_ReceiveNotPublishedOrTagMismatchReturnsWithoutRequest replaced  # ctsFifo is a pointer
    NetIbMsegP2pMicrotest.PostFifo_OwningDeviceSlotPostsSignaledInlineCts replaced  # remCtsFifo.elems is a pointer
    NetIbMsegP2pMicrotest.PostFifo_DifferentSlotIsUnsignaledUnlessResiliencyIsEnabled replaced  # remCtsFifo.elems is a pointer
    NetIbMsegP2pMicrotest.Irecv_ValidRequestPostsReceiveAndCtsMetadata replaced  # remCtsFifo.elems is a pointer
    NetIbMsegP2pMicrotest.Irecv_MultiReceivePublishesEveryDescriptorInOneNonInlineCts replaced  # remCtsFifo.elems is a pointer
    NetIbMsegP2pMicrotest.Irecv_PrepostedModeSkipsReceivePostButPublishesCts replaced  # remCtsFifo.elems is a pointer
    # The fork's GPU flush chains write and read in one post (Iflush_GpuScratchChainsWriteAndReadInOnePost), so
    # there is no separate read post to fail.
    NetIbMsegP2pMicrotest.DISABLED_Iflush_ReadPostFailurePropagatesError diverged
  SOURCES
    fakes/comm_fakes.cc
    fakes/env_fakes.cc
    fakes/gdr_peermem_fakes.cc
    fakes/gdr_probe_fakes.cc
    fakes/hip_fakes.cc
    fakes/hsa_fakes.cc
    fakes/ibvwrap_fakes.cc
    fakes/mlx5dvwrap_fakes.cc
    fakes/nccl_fakes.cc
    fakes/nccl_stubs.cc
    fakes/net_ib_mseg_params.cc
    fakes/net_ib_mseg_params_connect.cc
    fakes/net_ib_mseg_params_p2p.cc
    fakes/os_fakes.cc
    fakes/rccl_mseg_fakes.cc
    fakes/rocmwrap_fakes.cc
    fakes/socket_fakes.cc
    fakes/transport_net_ib_recovery_fakes.cc
    fakes/transport_net_ib_resiliency_fakes.cc
    fakes/transport_net_ib_wqe_lat_fakes.cc
    ${RCCL_HOST_HIPIFY_DIR}/src/misc/utils.cc
    ${RCCL_HOST_HIPIFY_DIR}/src/transport/net_ib_multiseg/common_tmp.cc
    ${RCCL_HOST_HIPIFY_DIR}/src/transport/net_ib_multiseg/connect_tmp.cc
    ${RCCL_HOST_HIPIFY_DIR}/src/transport/net_ib_multiseg/gdr_tmp.cc
    ${RCCL_HOST_HIPIFY_DIR}/src/transport/net_ib_multiseg/init_tmp.cc
    ${RCCL_HOST_HIPIFY_DIR}/src/transport/net_ib_multiseg/reg_tmp.cc
    ${RCCL_HOST_HIPIFY_DIR}/src/transport/net_ib_multiseg/seg_p2p.cc)

rccl_add_derived_transport_binary(rccl-UnitTestsMicroTransportNetIbMsegReg net_ib_multiseg/reg-test.cc
  DERIVED net_ib_multiseg
  BASE net_ib
  EXTRA_DEFS RCCL_HOST_TEST_NET_IB_MSEG
  ENABLE_BASE_TESTS
    # The fork unwinds the earlier devices' MRs (AICOMRCCL-3112).
    NetIbMsegRegMicrotest.DISABLED_RegMr_MultiDeviceFailureUnwindsEarlierRegistrations
  SOURCES
    fakes/env_fakes.cc
    fakes/gdr_peermem_fakes.cc
    fakes/gdr_probe_fakes.cc
    fakes/hip_fakes.cc
    fakes/ibvwrap_fakes.cc
    fakes/mlx5dvwrap_fakes.cc
    fakes/nccl_fakes.cc
    fakes/net_ib_mseg_params.cc
    fakes/net_ib_mseg_p2p_multiseg_fakes.cc
    fakes/os_fakes.cc
    fakes/rocmwrap_fakes.cc
    fakes/socket_fakes.cc
    fakes/transport_net_ib_fakes.cc
    fakes/transport_net_ib_p2p_fakes.cc
    fakes/transport_net_ib_recovery_fakes.cc
    fakes/transport_net_ib_resiliency_fakes.cc
    ${RCCL_HOST_HIPIFY_DIR}/src/misc/utils.cc
    ${RCCL_HOST_HIPIFY_DIR}/src/transport/net_ib_multiseg/common_tmp.cc
    ${RCCL_HOST_HIPIFY_DIR}/src/transport/net_ib_multiseg/gdr_tmp.cc
    ${RCCL_HOST_HIPIFY_DIR}/src/transport/net_ib_multiseg/init_tmp.cc
    ${RCCL_HOST_HIPIFY_DIR}/src/transport/net_ib_multiseg/seg_p2p.cc)
