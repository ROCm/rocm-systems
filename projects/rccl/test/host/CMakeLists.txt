cmake_minimum_required(VERSION 3.16)

include("${CMAKE_CURRENT_LIST_DIR}/cmake/RcclAssertSourceLine.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/cmake/RcclTargetHelpers.cmake")

# Checked against the original sources because hipify output does not exist yet
# at configure time. Paths are relative to this file, so the same checks cover
# the in-build and standalone contexts.
set(_rccl_src_root "${CMAKE_CURRENT_LIST_DIR}/../../src")
set(_rccl_gdr_map_pointer_default [=[
static ncclResult_t ncclGdrCudaMapPointer(void* data, size_t size, gdr_mh_t* mh, void** map, size_t* mapSize,
                                          gdr_info_t* info, uint32_t pinFlags = 0) {]=])
rccl_assert_source_text_count("${_rccl_src_root}/include/gdrwrap.h"
  "${_rccl_gdr_map_pointer_default}" 2
  "fakes/gdrwrap_fakes.h: micro_ncclGdrCudaMapPointer() default")
unset(_rccl_gdr_map_pointer_default)
rccl_assert_source_line("${_rccl_src_root}/graph/connect.cc"
  "NCCL_PARAM(MinNchannels, \"MIN_NCHANNELS\", -2)" "fakes/tuning_fakes.cc: ncclParamMinNchannels()")
rccl_assert_source_line("${_rccl_src_root}/graph/connect.cc"
  "NCCL_PARAM(MaxNchannels, \"MAX_NCHANNELS\", -2)" "fakes/tuning_fakes.cc: ncclParamMaxNchannels()")
rccl_assert_source_line("${_rccl_src_root}/enqueue/enqueue.cc"
  "RCCL_PARAM(ForceCe, \"FORCE_CE\", 1)" "fakes/wrap_fakes.cc: rcclParamForceCe()")
rccl_assert_source_line("${_rccl_src_root}/misc/rocmwrap.cc"
  "RCCL_PARAM(ForceEnableDMABUF, \"FORCE_ENABLE_DMABUF\", 0);"
  "fakes/rocmwrap_fakes.cc: rcclParamForceEnableDMABUF()")
rccl_assert_source_line("${CMAKE_CURRENT_LIST_DIR}/fakes/rocmwrap_fakes.cc"
  "int64_t rcclParamForceEnableDMABUF() { return g_loadParam(\"RCCL_FORCE_ENABLE_DMABUF\", 0); }"
  "the ForceEnableDMABUF fake default")
rccl_assert_source_line("${_rccl_src_root}/transport/net.cc"
  "NCCL_PARAM(GdrCopySyncEnable, \"GDRCOPY_SYNC_ENABLE\", 1);"
  "fakes/transport_stubs.cc: ncclParamGdrCopySyncEnable()")
rccl_assert_source_line("${_rccl_src_root}/transport/net.cc"
  "NCCL_PARAM(GdrCopyFlushEnable, \"GDRCOPY_FLUSH_ENABLE\", 0);"
  "fakes/transport_stubs.cc: ncclParamGdrCopyFlushEnable()")
# allocator-test.cc copies misc/utils.cc's pointer hash rather than linking utils.cc.
rccl_assert_source_line("${_rccl_src_root}/misc/utils.cc"
  "uintptr_t h = reinterpret_cast<uintptr_t>(key);" "allocator-test.cc: ncclHashPointer()")
rccl_assert_source_line("${_rccl_src_root}/misc/utils.cc"
  "h ^= h >> 32;" "allocator-test.cc: ncclHashPointer()")
rccl_assert_source_line("${_rccl_src_root}/misc/utils.cc"
  "h *= 0x9e3779b97f4a7c13;" "allocator-test.cc: ncclHashPointer()")
rccl_assert_source_line("${_rccl_src_root}/misc/utils.cc"
  "return (uint64_t)h >> (64 - hbits);" "allocator-test.cc: ncclHashPointer()")
# src/enqueue/task_prep/*.cc declare these five but define none, and param_redirect.h only rewrites
# NCCL_PARAM where it expands, so the hand-written bodies below are the only definitions on that link line.
rccl_assert_source_line("${_rccl_src_root}/init.cc"
  "NCCL_PARAM(MinCTAs, \"MIN_CTAS\", NCCL_CONFIG_UNDEF_INT)" "fakes/comm_fakes.cc: ncclParamMinCTAs()")
rccl_assert_source_line("${_rccl_src_root}/init.cc"
  "NCCL_PARAM(MaxCTAs, \"MAX_CTAS\", NCCL_CONFIG_UNDEF_INT)" "fakes/comm_fakes.cc: ncclParamMaxCTAs()")
rccl_assert_source_line("${_rccl_src_root}/init.cc"
  "NCCL_PARAM(NvlsChannels, \"NVLS_NCHANNELS\", NCCL_CONFIG_UNDEF_INT)"
  "fakes/comm_fakes.cc: ncclParamNvlsChannels()")
rccl_assert_source_line("${_rccl_src_root}/init.cc"
  "NCCL_PARAM(CGAClusterSize, \"CGA_CLUSTER_SIZE\", NCCL_CONFIG_UNDEF_INT)"
  "fakes/comm_fakes.cc: ncclParamCGAClusterSize()")
rccl_assert_source_line("${_rccl_src_root}/rccl_wrap.cc"
  "RCCL_PARAM(DdaEnable, \"DDA_ENABLE\", 1)" "fakes/rccl_wrap_fakes.cc: rcclParamDdaEnable()")
rccl_assert_source_line("${CMAKE_CURRENT_LIST_DIR}/fakes/rccl_wrap_fakes.cc"
  "int64_t g_rcclParamDdaEnable = 1" "the DdaEnable fake default")
# Both arms of the ROCM_VERSION split at enqueue.cc:437-441; enqueue_fakes.cc mirrors both, so pin both.
rccl_assert_source_line("${_rccl_src_root}/enqueue/enqueue.cc"
  "NCCL_PARAM(GraphRegister, \"GRAPH_REGISTER\", 1)"
  "fakes/enqueue_fakes.cc: ncclParamGraphRegister()")
rccl_assert_source_line("${_rccl_src_root}/enqueue/enqueue.cc"
  "NCCL_PARAM(GraphRegister, \"GRAPH_REGISTER\", 0)"
  "fakes/enqueue_fakes.cc: ncclParamGraphRegister()")
# tuning/cost-model-test.cc defines ncclFuncStr itself and parseList() indexes it as its prefix table,
# so a rename in production must not leave the test parsing NCCL_ALGO against stale names.
rccl_assert_source_line("${_rccl_src_root}/init.cc"
  "const char* ncclFuncStr[NCCL_NUM_FUNCTIONS + 4] = {\"Broadcast\", \"Reduce\", \"AllGather\", \"ReduceScatter\", \"AllReduce\","
  "tuning/cost-model-test.cc: ncclFuncStr")
# Real NCCL_PARAM lives in enqueue.cc; init.cc only has the extern. MicroInit
# does not compile enqueue.cc, so init_fakes.cc supplies the body.
rccl_assert_source_line("${_rccl_src_root}/enqueue/enqueue.cc"
  "NCCL_PARAM(P2pLL128Enable, \"P2P_LL128_ENABLE\", -1)" "fakes/init_fakes.cc: ncclParamP2pLL128Enable()")
rccl_assert_source_line("${CMAKE_CURRENT_LIST_DIR}/fakes/init_fakes.cc"
  "int64_t ncclParamP2pLL128Enable() { return g_loadParam(\"P2P_LL128_ENABLE\", -1); }"
  "the P2pLL128Enable fake default")
rccl_assert_source_line("${_rccl_src_root}/graph/connect.cc"
  "NCCL_PARAM(TreeCompact, \"TREE_COMPACT\", 0)" "fakes/init_fakes.cc: ncclParamTreeCompact()")
rccl_assert_source_line("${CMAKE_CURRENT_LIST_DIR}/fakes/init_fakes.cc"
  "int64_t ncclParamTreeCompact() { return g_loadParam(\"TREE_COMPACT\", 0); }"
  "the TreeCompact fake default")
rccl_assert_source_line("${_rccl_src_root}/transport/net_ib/connect.cc"
  "NCCL_PARAM(IbQpsPerConn, \"IB_QPS_PER_CONNECTION\", 1)" "fakes/transport_net_ib_fakes.cc: ncclParamIbQpsPerConn()")
rccl_assert_source_line("${CMAKE_CURRENT_LIST_DIR}/fakes/transport_net_ib_fakes.cc"
  "int64_t ncclParamIbQpsPerConn() { return g_loadParam(\"IB_QPS_PER_CONNECTION\", 1); }"
  "the IbQpsPerConn fake default")
# The net_ib_cast connect / qp_sharing fakes repeat these params; both copies must agree.
foreach(_param
    "NCCL_PARAM(IbCastTimeout, \"IB_TIMEOUT\", 20)"
    "NCCL_PARAM(IbCastRetryCnt, \"IB_RETRY_CNT\", 7)"
    "NCCL_PARAM(IbCastPkey, \"IB_PKEY\", 0)"
    "NCCL_PARAM(IbCastQpsPerConn, \"IB_QPS_PER_CONNECTION\", 2)"
    "NCCL_PARAM(IbCastUseInline, \"IB_USE_INLINE\", 0)"
    "NCCL_PARAM(IbCastGdrFlushDisable, \"GDR_FLUSH_DISABLE\", 1)"
    "NCCL_PARAM(IbCastSl, \"IB_SL\", -1)"
    "NCCL_PARAM(IbCastTc, \"IB_TC\", -1)"
    "RCCL_PARAM(IbCastUDMAPolicy, \"IB_UDMA_POLICY\", 0)")
  rccl_assert_source_line("${_rccl_src_root}/transport/net_ib_cast/connect.cc"
    "${_param}" "fakes/net_ib_cast_connect_fakes.cc")
  rccl_assert_source_line("${CMAKE_CURRENT_LIST_DIR}/fakes/net_ib_cast_connect_fakes.cc"
    "${_param}" "the connect.cc default")
endforeach()
foreach(_param
    "RCCL_PARAM(IbCastQpSharingEnable, \"IB_QP_SHARING_ENABLE\", 0)"
    "RCCL_PARAM(IbCastCommNGroups, \"IB_COMM_NGROUPS\", 4)"
    "RCCL_PARAM(IbCastQpSharingValidatePool, \"IB_QP_SHARING_VALIDATE_POOL\", 0)")
  rccl_assert_source_line("${_rccl_src_root}/transport/net_ib_cast/qp_sharing.cc"
    "${_param}" "fakes/net_ib_cast_qp_sharing_fakes.cc")
  rccl_assert_source_line("${CMAKE_CURRENT_LIST_DIR}/fakes/net_ib_cast_qp_sharing_fakes.cc"
    "${_param}" "the qp_sharing.cc default")
endforeach()
# The net_ib, net_ib_cast and socket fakes copy these lines from production; pin both ends ("<SEMI>" stands for a semicolon).
set(_rccl_cast_pins
  "transport/net_ib_cast/common.cc" "NCCL_PARAM(IbCastSplitDataOnQps, \"IB_SPLIT_DATA_ON_QPS\", 0)<SEMI>"
    "net_ib_cast_common_fakes.cc" "return g_loadParam(\"IB_SPLIT_DATA_ON_QPS\", 0)<SEMI> }"
  "transport/net_ib_cast/common.cc" "NCCL_PARAM(IbCastPrepostReceiveWorkRequests, \"IB_PREPOST_RECEIVE_WORK_REQUESTS\", -2)<SEMI>"
    "net_ib_cast_common_fakes.cc" "return g_loadParam(\"IB_PREPOST_RECEIVE_WORK_REQUESTS\", -2)<SEMI> }"
  "transport/net_ib_cast/common.cc" "NCCL_PARAM(IbCastAsyncEvents, \"IB_RETURN_ASYNC_EVENTS\", 1)<SEMI>"
    "net_ib_cast_common_fakes.cc" "return g_loadParam(\"IB_RETURN_ASYNC_EVENTS\", 1)<SEMI> }"
  "transport/net_ib_cast/common.cc" "RCCL_PARAM(IbCastOptionalRecvCompletion, \"IB_OPTIONAL_RECV_COMPLETION\", 0)<SEMI>"
    "net_ib_cast_common_fakes.cc" "return g_loadParam(\"RCCL_IB_OPTIONAL_RECV_COMPLETION\", 0)<SEMI> }"
  "transport/net_ib_cast/common.cc" "int IbCastNMergedDevs = -1<SEMI>"
    "net_ib_cast_common_fakes.cc" "int IbCastNMergedDevs = -1<SEMI>"
  "transport/net_ib_cast/common.cc" "int IbCastNDevs = -1<SEMI>"
    "net_ib_cast_common_fakes.cc" "int IbCastNDevs = -1<SEMI>"
  "transport/net_ib_cast/common.cc" "int IbCastRelaxedOrderingEnabled = 0<SEMI>"
    "net_ib_cast_common_fakes.cc" "int IbCastRelaxedOrderingEnabled = 0<SEMI>"
  "transport/net_ib_cast/common.cc" "if (ncclParamIbCastAsyncEvents() && COMPILER_ATOMIC_LOAD(&stat->fatalErrorCount, std::memory_order_relaxed)) {"
    "net_ib_cast_common_fakes.cc" "if (ncclParamIbCastAsyncEvents() && COMPILER_ATOMIC_LOAD(&stat->fatalErrorCount, std::memory_order_relaxed)) {"
  "transport/net_ib_cast/common.cc" "return &sComm->devs[devIndex].base<SEMI>"
    "net_ib_cast_common_fakes.cc" "if (base->isSend) return &((struct ncclIbSendComm*)base)->devs[devIndex].base<SEMI>"
  "transport/net_ib_cast/common.cc" "return &rComm->devs[devIndex].base<SEMI>"
    "net_ib_cast_common_fakes.cc" "return &((struct ncclIbRecvComm*)base)->devs[devIndex].base<SEMI>"
  "transport/net_ib_cast/init.cc" "RCCL_PARAM(IbCastQpsPerP2p, \"IB_QPS_PER_P2P\", 0)<SEMI>"
    "net_ib_cast_init_fakes.cc" "return g_loadParam(\"RCCL_IB_QPS_PER_P2P\", 0)<SEMI> }"
  "transport/net_ib_cast/init.cc" "RCCL_PARAM(IbCastP2pDisableCts, \"IB_P2P_DISABLE_CTS\", 1)<SEMI>"
    "net_ib_cast_init_fakes.cc" "return g_loadParam(\"RCCL_IB_P2P_DISABLE_CTS\", 1)<SEMI> }"
  "transport/net_ib_cast/init.cc" "NCCL_PARAM(IbCastOooRq, \"IB_OOO_RQ\", 0)"
    "net_ib_cast_init_fakes.cc" "return g_loadParam(\"IB_OOO_RQ\", 0)<SEMI> }"
  "transport/net_ib_cast/init.cc" "bool IbCastAinicRoce = 0<SEMI>"
    "net_ib_cast_init_fakes.cc" "bool IbCastAinicRoce = 0<SEMI>"
  "transport/net_ib_cast/init.cc" "bool IbCastMultiplaneEnable = false<SEMI>"
    "net_ib_cast_init_fakes.cc" "bool IbCastMultiplaneEnable = false<SEMI>"
  "transport/net_ib_cast/init.cc" "bool IbCastOffloadEnabled = 0<SEMI>"
    "net_ib_cast_init_fakes.cc" "bool IbCastOffloadEnabled = 0<SEMI>"
  "transport/net_ib_cast/init.cc" "bool IbCastUseInline = 0<SEMI>"
    "net_ib_cast_init_fakes.cc" "bool IbCastUseInline = 0<SEMI>"
  "transport/net_ib_cast/init.cc" "bool IbCastAinicCtsInlineData = 0<SEMI>"
    "net_ib_cast_init_fakes.cc" "bool IbCastAinicCtsInlineData = 0<SEMI>"
  "transport/net_ib_cast/init.cc" "int IbCastGdrFlushDisable = 0<SEMI>"
    "net_ib_cast_init_fakes.cc" "int IbCastGdrFlushDisable = 0<SEMI>"
  "transport/net_ib/p2p.cc" "NCCL_PARAM(IbArThreshold, \"IB_AR_THRESHOLD\", -2)<SEMI>"
    "transport_net_ib_p2p_fakes.cc" "return g_loadParam(\"IB_AR_THRESHOLD\", -2)<SEMI> }"
  "transport/net_ib/p2p.cc" "int64_t ncclIbArThreshold = 8192<SEMI>"
    "transport_net_ib_p2p_fakes.cc" "int64_t ncclIbArThreshold = 8192<SEMI>"
  "transport/net_ib/p2p.cc" "NCCL_PARAM(IbReceiverSideMatchingScheme, \"IB_RECEIVER_SIDE_MATCHING_SCHEME\", -2)<SEMI>"
    "transport_net_ib_p2p_fakes.cc" "return g_loadParam(\"IB_RECEIVER_SIDE_MATCHING_SCHEME\", -2)<SEMI>"
  "transport/net_ib/p2p.cc" "RCCL_PARAM(IbGdrFlushGpuMemNoRelaxedOrdering, \"GDR_FLUSH_GPU_MEM_NO_RELAXED_ORDERING\", 1)<SEMI>"
    "transport_net_ib_p2p_fakes.cc" "return g_loadParam(\"RCCL_GDR_FLUSH_GPU_MEM_NO_RELAXED_ORDERING\", 1)<SEMI>"
  "transport/net_ib/p2p_resiliency.cc" "NCCL_PARAM(IbResiliencyPortFailover, \"IB_RESILIENCY_PORT_FAILOVER\", 0)<SEMI>"
    "transport_net_ib_resiliency_fakes.cc" "return g_loadParam(\"IB_RESILIENCY_PORT_FAILOVER\", 0)<SEMI>"
  "misc/rocmwrap.cc" "NCCL_PARAM(DmaBufEnable, \"DMABUF_ENABLE\", 1)<SEMI>"
    "rocmwrap_fakes.cc" "int64_t ncclParamDmaBufEnable() { return g_loadParam(\"DMABUF_ENABLE\", 1)<SEMI> }"
  "transport/net_ib_cast/p2p.cc" "NCCL_PARAM(IbCastArThreshold, \"IB_AR_THRESHOLD\", -2)<SEMI>"
    "net_ib_cast_p2p_fakes.cc" "return g_loadParam(\"IB_AR_THRESHOLD\", -2)<SEMI> }"
  "transport/net_ib_cast/p2p.cc" "int64_t IbCastArThreshold = 8192<SEMI>"
    "net_ib_cast_p2p_fakes.cc" "int64_t IbCastArThreshold = 8192<SEMI>"
  "transport/net_ib_cast/p2p.cc" "NCCL_PARAM(IbCastReceiverSideMatchingScheme, \"IB_RECEIVER_SIDE_MATCHING_SCHEME\", -2)<SEMI>"
    "net_ib_cast_p2p_fakes.cc" "return g_loadParam(\"IB_RECEIVER_SIDE_MATCHING_SCHEME\", -2)<SEMI> }"
  "transport/net_ib_cast/p2p.cc" "RCCL_PARAM(IbCastGdrFlushGpuMemNoRelaxedOrdering, \"GDR_FLUSH_GPU_MEM_NO_RELAXED_ORDERING\", 1)<SEMI>"
    "net_ib_cast_p2p_fakes.cc" "return g_loadParam(\"RCCL_GDR_FLUSH_GPU_MEM_NO_RELAXED_ORDERING\", 1)<SEMI>"
  "transport/net_ib_cast/p2p.cc" "const char* IbCastReqTypeStr[] = {\"Unused\", \"Send\", \"Recv\", \"Flush\", \"IPut\", \"IGet\", \"Failed\"}<SEMI>"
    "net_ib_cast_p2p_fakes.cc" "const char* IbCastReqTypeStr[] = {\"Unused\", \"Send\", \"Recv\", \"Flush\", \"IPut\", \"IGet\", \"Failed\"}<SEMI>"
  "transport/net_ib_cast/p2p_resiliency.cc" "NCCL_PARAM(IbCastResiliencyPortFailover, \"IB_RESILIENCY_PORT_FAILOVER\", 0)<SEMI>"
    "net_ib_cast_p2p_resiliency_fakes.cc" "return g_loadParam(\"IB_RESILIENCY_PORT_FAILOVER\", 0)<SEMI> }"
  "transport/net_ib_cast/p2p_resiliency.cc" "NCCL_PARAM(IbCastResiliencyPortFailoverMaxAttempts, \"IB_RESILIENCY_PORT_FAILOVER_MAX_ATTEMPTS\", 1)<SEMI>"
    "net_ib_cast_p2p_resiliency_fakes.cc" "return g_loadParam(\"IB_RESILIENCY_PORT_FAILOVER_MAX_ATTEMPTS\", 1)<SEMI>"
  "transport/net_ib_cast/p2p_resiliency.cc" "NCCL_PARAM(IbCastResiliencyPortFailoverProbeDelay, \"IB_RESILIENCY_PORT_FAILOVER_PROBE_DELAY\", 10)<SEMI>"
    "net_ib_cast_p2p_resiliency_fakes.cc" "return g_loadParam(\"IB_RESILIENCY_PORT_FAILOVER_PROBE_DELAY\", 10)<SEMI>"
  "misc/socket.cc" "static uint64_t cached = NCCL_SOCKET_MAGIC<SEMI>"
    "socket_fakes.cc" "uint64_t ncclSocketDefaultMagic(void) { return NCCL_SOCKET_MAGIC<SEMI> }")
while(_rccl_cast_pins)
  list(POP_FRONT _rccl_cast_pins _src _prod _fake _copy)
  string(REPLACE "<SEMI>" ";" _prod "${_prod}")
  string(REPLACE "<SEMI>" ";" _copy "${_copy}")
  rccl_assert_source_line("${_rccl_src_root}/${_src}" "${_prod}" "fakes/${_fake}")
  rccl_assert_source_line("${CMAKE_CURRENT_LIST_DIR}/fakes/${_fake}" "${_copy}" "the ${_src} copy")
endwhile()
rccl_assert_source_line("${CMAKE_CURRENT_LIST_DIR}/fakes/tuning_fakes.cc"
  "int64_t g_paramMinNchannels = -2;" "the MinNchannels fake default")
rccl_assert_source_line("${CMAKE_CURRENT_LIST_DIR}/fakes/tuning_fakes.cc"
  "int64_t g_paramMaxNchannels = -2;" "the MaxNchannels fake default")
rccl_assert_source_line("${CMAKE_CURRENT_LIST_DIR}/fakes/wrap_fakes.cc"
  "static int64_t DefaultParamForceCe() { return 1; }" "the ForceCe fake default")
rccl_assert_source_line("${_rccl_src_root}/init.cc"
  "NCCL_PARAM(SetCpuStackSize, \"SET_CPU_STACK_SIZE\", 1)" "fakes/linux_fakes.cc: ncclParamSetCpuStackSize()")
rccl_assert_source_line("${CMAKE_CURRENT_LIST_DIR}/fakes/linux_fakes.cc"
  "int64_t ncclParamSetCpuStackSize() { return g_loadParam(\"SET_CPU_STACK_SIZE\", 1); }"
  "the SetCpuStackSize fake default")
# RasDiagnosticsTests.cpp (rccl-UnitTests) classifies report lines by kRasFailureMarkers and kRasHealthMarkers;
# each marker must stay both in those lists and in every source file that emits it (comma-separated).
set(_rccl_ras_tests "${CMAKE_CURRENT_LIST_DIR}/../RasDiagnosticsTests.cpp")
set(_rccl_ras_markers
  "ras/diagnostics_checks_common.cc|diagnostics incomplete"
  "ras/diagnostics_env.cc,ras/diagnostics_gpu.cc|mismatch across"
  "ras/diagnostics_env.cc|differ across ranks"
  "ras/diagnostics_env.cc|comparison may be"
  "ras/diagnostics_gpu.cc|uncorrected volatile errors on rank(s)"
  "ras/diagnostics_gpu.cc|at or above threshold"
  "ras/diagnostics_gpu.cc|inactive link(s)")
foreach(_entry IN LISTS _rccl_ras_markers)
  string(REPLACE "|" ";" _ras_pair "${_entry}")
  list(GET _ras_pair 0 _ras_sources)
  list(GET _ras_pair 1 _marker)
  string(REPLACE "," ";" _ras_files "${_ras_sources}")
  foreach(_ras_cc IN LISTS _ras_files)
    rccl_assert_source_line("${_rccl_src_root}/${_ras_cc}" "${_marker}" "the RAS markers in test/RasDiagnosticsTests.cpp")
  endforeach()
  rccl_assert_source_line("${_rccl_ras_tests}" "\"${_marker}\"," "the report lines it marks in src/${_ras_sources}")
endforeach()
# The check labels, the report header and the skip line it parses the report by. src emits each as the start of a
# literal; a check label may be passed on without its ": ".
set(_rccl_ras_labels
  "ras/diagnostics_gpu.cc|GPU inventory: "
  "ras/diagnostics_gpu.cc|ECC: "
  "ras/diagnostics_env.cc|NCCL environment: "
  "ras/ras.cc|NCCL DIAG === RAS Diagnostics ==="
  "ras/ras.cc|NCCL DIAG INFO RAS diagnostics skipped: NCCL_RAS_ENABLE=0")
foreach(_entry IN LISTS _rccl_ras_labels)
  string(REPLACE "|" ";" _ras_pair "${_entry}")
  list(GET _ras_pair 0 _ras_cc)
  list(GET _ras_pair 1 _label)
  string(REGEX REPLACE ": $" "" _src_label "${_label}")
  rccl_assert_source_line("${_rccl_src_root}/${_ras_cc}" "\"${_src_label}"
    "the RAS labels in test/RasDiagnosticsTests.cpp")
  rccl_assert_source_line("${_rccl_ras_tests}" "= \"${_label}\"" "the report lines it labels in src/${_ras_cc}")
endforeach()
# The driver labels are macros on the AMD SMI arm of diagnostics_gpu.cc.
rccl_assert_source_line("${_rccl_src_root}/ras/diagnostics_gpu.cc"
  "#define RAS_DIAG_DRIVER_LABEL \"HIP driver version\"" "kDriver in test/RasDiagnosticsTests.cpp")
rccl_assert_source_line("${_rccl_ras_tests}" "= \"HIP driver version: \""
  "RAS_DIAG_DRIVER_LABEL in src/ras/diagnostics_gpu.cc")
rccl_assert_source_line("${_rccl_src_root}/ras/diagnostics_gpu.cc"
  "#define RAS_DIAG_GPU_DRIVER_LABEL \"AMD GPU driver version\"" "kGpuDriver in test/RasDiagnosticsTests.cpp")
rccl_assert_source_line("${_rccl_ras_tests}" "= \"AMD GPU driver version: \""
  "RAS_DIAG_GPU_DRIVER_LABEL in src/ras/diagnostics_gpu.cc")
unset(_rccl_ras_tests)
unset(_rccl_ras_markers)
unset(_rccl_ras_labels)
unset(_src_label)
unset(_ras_pair)
unset(_ras_sources)
unset(_ras_files)
unset(_ras_cc)
unset(_marker)
# misc/gdr-probe-test.cc pins the GDR probe's registration flags to production's; these
# keep its kRegAccessFlags and both reg.cc copies spelling the same set.
set(_rccl_reg_access_flags
  "IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ | IBV_ACCESS_REMOTE_ATOMIC;")
foreach(_reg_cc transport/net_ib/reg.cc transport/net_ib_cast/reg.cc)
  rccl_assert_source_line("${_rccl_src_root}/${_reg_cc}"
    "${_rccl_reg_access_flags}" "misc/gdr-probe-test.cc: kRegAccessFlags")
  rccl_assert_source_line("${_rccl_src_root}/${_reg_cc}"
    "if (relaxedOrdering) flags |= IBV_ACCESS_RELAXED_ORDERING;" "misc/gdr-probe-test.cc: the relaxed-ordering flags")
endforeach()
rccl_assert_source_line("${CMAKE_CURRENT_LIST_DIR}/misc/gdr-probe-test.cc"
  "${_rccl_reg_access_flags}" "the production registration flags")
unset(_rccl_reg_access_flags)
# graph_fakes.cc hand-writes these params, whose NCCL_PARAMs are in files the graph/ binaries don't compile.
# Each entry is <src file>|<production line>|<fake body, up to its ';'>.
foreach(_pin
    "init.cc|NCCL_PARAM(P2pDisable, \"P2P_DISABLE\", 0)|ncclParamP2pDisable() { return g_loadParam(\"P2P_DISABLE\", 0)"
    "init.cc|NCCL_PARAM(MultiRankGpuEnable, \"MULTI_RANK_GPU_ENABLE\", 0)|ncclParamMultiRankGpuEnable() { return g_loadParam(\"MULTI_RANK_GPU_ENABLE\", 0)"
    "init.cc|NCCL_PARAM(WorkArgsBytes, \"WORK_ARGS_BYTES\", INT64_MAX)|ncclParamWorkArgsBytes() { return g_loadParam(\"WORK_ARGS_BYTES\", INT64_MAX)"
    "graph/rccl_graph_gen.cc|RCCL_PARAM(IntraGraphGen, \"INTRA_GRAPH_GEN\", 0)|rcclParamIntraGraphGen() { return g_loadParam(\"RCCL_INTRA_GRAPH_GEN\", 0)"
    "graph/rccl_graph_gen.cc|RCCL_PARAM(InterGraphGen, \"INTER_GRAPH_GEN\", 0)|rcclParamInterGraphGen() { return g_loadParam(\"RCCL_INTER_GRAPH_GEN\", 0)")
  string(REPLACE "|" ";" _pin "${_pin}")
  list(GET _pin 0 _src)
  list(GET _pin 1 _prod)
  list(GET _pin 2 _fake)
  rccl_assert_source_line("${_rccl_src_root}/${_src}" "${_prod}" "fakes/graph_fakes.cc")
  rccl_assert_source_line("${CMAKE_CURRENT_LIST_DIR}/fakes/graph_fakes.cc" "${_fake}" "the graph_fakes.cc default")
endforeach()
# Pin the copied function-name table and its order to init.cc.
foreach(_func_str_line
    "const char* ncclFuncStr[NCCL_NUM_FUNCTIONS + 4] = {\"Broadcast\", \"Reduce\", \"AllGather\", \"ReduceScatter\", \"AllReduce\","
    "\"AlltoAllPivot\", \"AlltoAllGda\", \"AlltoAllvGda\","
    "\"SendRecv\"};")
  rccl_assert_source_line("${_rccl_src_root}/init.cc" "${_func_str_line}"
    "ras/client-support-test.cc: ncclFuncStr")
  rccl_assert_source_line("${CMAKE_CURRENT_LIST_DIR}/ras/client-support-test.cc" "${_func_str_line}"
    "the production ncclFuncStr table")
endforeach()
# dda-coll-seams.cc copies these rccl_wrap.cc DDA param defaults; pin both copies.
foreach(_dda_param
    "RCCL_PARAM(DdaLL, \"DDA_LL\", 1)"
    "RCCL_PARAM(DdaLLOneShotThreshold, \"DDA_LL_ONESHOT_THRESHOLD\", (size_t)(1) * 1024 * 1024)"
    "RCCL_PARAM(DdaLL128OneShotThreshold, \"DDA_LL128_ONESHOT_THRESHOLD\", (size_t)(4) * 1024 * 1024)"
    "RCCL_PARAM(DdaLL128AsyncWire, \"DDA_LL128_ASYNC_WIRE\", 0)")
  rccl_assert_source_line("${_rccl_src_root}/rccl_wrap.cc" "${_dda_param}" "algorithms/dda/dda-coll-seams.cc")
  rccl_assert_source_line("${CMAKE_CURRENT_LIST_DIR}/algorithms/dda/dda-coll-seams.cc" "${_dda_param}"
    "the rccl_wrap.cc DDA param")
endforeach()
rccl_assert_source_line("${_rccl_src_root}/rccl_wrap.cc"
  "RCCL_PARAM(DdaLLTwoShotThreshold, \"DDA_LL_TWOSHOT_THRESHOLD\", (size_t)(2) * 1024 * 1024)"
  "dda-coll-seams.cc: DefaultDdaLLTwoShotThreshold()")
rccl_assert_source_line("${CMAKE_CURRENT_LIST_DIR}/algorithms/dda/dda-coll-seams.cc"
  "DefaultDdaLLTwoShotThreshold(const ncclComm*) { return (size_t)(2) * 1024 * 1024; }"
  "the rccl_wrap.cc DdaLLTwoShotThreshold default")
rccl_assert_source_line("${_rccl_src_root}/rccl_wrap.cc"
  "RCCL_PARAM(DdaLL128TwoShotThreshold, \"DDA_LL128_TWOSHOT_THRESHOLD\", (size_t)(64) * 1024 * 1024)"
  "dda-coll-seams.cc: DefaultDdaLL128TwoShotThreshold()")
rccl_assert_source_line("${CMAKE_CURRENT_LIST_DIR}/algorithms/dda/dda-coll-seams.cc"
  "DefaultDdaLL128TwoShotThreshold(const ncclComm*) { return (size_t)(64) * 1024 * 1024; }"
  "the rccl_wrap.cc DdaLL128TwoShotThreshold default")
unset(_func_str_line)
# fakes/topo_stubs.cc copies the AMD arm of topoPathTypeStr; pin both copies to one spelling.
foreach(_path_cc "${_rccl_src_root}/graph/topo.cc" "${CMAKE_CURRENT_LIST_DIR}/fakes/topo_stubs.cc")
  rccl_assert_source_line("${_path_cc}" "const char* topoPathTypeStr[] = {\"LOC\", \"XGMI\", \"NVB\", \"C2C\", \"PIX\", \
\"PXB\", \"P2C\", \"PXN\", \"PHB\", \"SYS\", \"NET\", \"DIS\"};" "fakes/topo_stubs.cc: topoPathTypeStr")
endforeach()
rccl_assert_source_line("${_rccl_src_root}/transport/net_ib_cast/p2p.cc"
  "int cqMaxPollEvent = 4;" "net-ib-cast-p2p-test.cc: kPollBatch")
# fakes/net_ib_cast_common_fakes.cc copies IbCastStatsCheckFatalCount's gate from common.cc.
foreach(_common_cc "${_rccl_src_root}/transport/net_ib_cast/common.cc"
    "${CMAKE_CURRENT_LIST_DIR}/fakes/net_ib_cast_common_fakes.cc")
  rccl_assert_source_line("${_common_cc}"
    "if (ncclParamIbCastAsyncEvents() && COMPILER_ATOMIC_LOAD(&stat->fatalErrorCount, std::memory_order_relaxed)) {"
    "fakes/net_ib_cast_common_fakes.cc: DefaultIbCastStatsCheckFatalCount")
endforeach()
foreach(_path_cc "${_rccl_src_root}/transport/net_ib_cast/p2p_resiliency_recovery.cc"
                 "${CMAKE_CURRENT_LIST_DIR}/fakes/net_ib_cast_p2p_resiliency_recovery_fakes.cc")
  rccl_assert_source_line("${_path_cc}" "NCCL_PARAM(IbCastResiliencyPortRecovery, \"IB_RESILIENCY_PORT_RECOVERY\", 0);"
    "fakes/net_ib_cast_p2p_resiliency_recovery_fakes.cc: ncclParamIbCastResiliencyPortRecovery()")
endforeach()
rccl_assert_source_line("${_rccl_src_root}/transport/net.cc"
  "NCCL_PARAM(NetOptionalRecvCompletion, \"NET_OPTIONAL_RECV_COMPLETION\", 1);"
  "fakes/transport_stubs.cc: ncclParamNetOptionalRecvCompletion()")
rccl_assert_source_line("${CMAKE_CURRENT_LIST_DIR}/fakes/transport_stubs.cc"
  "int64_t ncclParamNetOptionalRecvCompletion() { return g_loadParam(\"NET_OPTIONAL_RECV_COMPLETION\", 1); }"
  "the NetOptionalRecvCompletion fake default")
# DiagnosticsTests.cpp (rccl-UnitTests) finds failed checks by kDiagFailureMarkers; each marker must stay
# both in that list and in the report line it marks.
set(_rccl_diag_tests "${CMAKE_CURRENT_LIST_DIR}/../DiagnosticsTests.cpp")
set(_rccl_diag_core_markers "transport detect returned" "p2p: check returned")
set(_rccl_diag_p2p_markers
  "p2p: setup failed" "p2p: destination buffer unavailable" "p2p: peer-memory import failed"
  "p2p: write mismatch" "p2p: read mismatch" "p2p: topology check failed" "p2p: launch/check failed"
  "p2p: resource cleanup failed")
foreach(_marker IN LISTS _rccl_diag_core_markers)
  rccl_assert_source_line("${_rccl_src_root}/diagnostics.cc"
    "NCCL DIAG [INFO] ${_marker}" "kDiagFailureMarkers in test/DiagnosticsTests.cpp")
endforeach()
foreach(_marker IN LISTS _rccl_diag_p2p_markers)
  rccl_assert_source_line("${_rccl_src_root}/diagnostics/p2p.cc"
    "NCCL DIAG [INFO] ${_marker}" "kDiagFailureMarkers in test/DiagnosticsTests.cpp")
endforeach()
# The local setup line is assembled from NCCL_DIAG_P2P_RUNTIME; Report_EachReasonHasItsOwnLine in
# diag-p2p-test.cc checks its printed form.
rccl_assert_source_line("${_rccl_src_root}/diagnostics/p2p.cc"
  "NCCL DIAG [INFO] p2p: local \" NCCL_DIAG_P2P_RUNTIME" "kDiagFailureMarkers in test/DiagnosticsTests.cpp")
foreach(_marker IN LISTS _rccl_diag_core_markers _rccl_diag_p2p_markers ITEMS "p2p: local HIP setup failed")
  rccl_assert_source_line("${_rccl_diag_tests}" "\"${_marker}\"," "the report line it marks in src/diagnostics")
endforeach()
unset(_rccl_diag_tests)
unset(_rccl_diag_core_markers)
unset(_rccl_diag_p2p_markers)
unset(_rccl_src_root)

# Define the per-target micro-test source lists shared by BOTH build paths.
function(rccl_define_micro_source_lists)
  # rccl-UnitTestsMicro -- the broad-coverage micro binary.
  set(TEST_MICRO_SOURCE_FILES
    param-utils-test.cc                  # src/include/param/utils.h, src/include/param/parser_common.h
    param-parsers-test.cc                # src/include/param/parser_{default,enum,bitset,list}.h
    group-test.cc
    comm-dump-test.cc                    # src/commDump.cc
    timer-test.cc                        # src/include/timer.h
    wrap-test.cc
    dev-runtime-test.cc
    nccl-utility-test.cc                 # src/include/nccl_device/utility.h (header-only, no fakes)
    # sym-kernels-index-test.cc: added conditionally below, only when GENERATE_SYM_KERNELS is ON.
    process-isolated-runner-test.cc
    # real IsArchMatch, for wrap-test.cc (path is build-directory-relative).
    ${RCCL_HOST_HIPIFY_DIR}/src/misc/archinfo.cc
    # real ncclP2pImportShareableBuffer, for dev-runtime-test.cc's cuMem window import.
    ${RCCL_HOST_HIPIFY_DIR}/src/transport/p2p_tmp.cc
    # real devcomm compat tables, for dev_runtime.cc's getNcclVersionCompat.
    ${RCCL_HOST_HIPIFY_DIR}/src/devcomm/devcomm_v22902.cc
    ${RCCL_HOST_HIPIFY_DIR}/src/devcomm/devcomm_v22907.cc
    # real ncclRmaProxyEnabled / ncclLaunchRma / ncclRmaCeInit, for dev_runtime.cc and group.cc
    # (rma_tmp.cc: see RMA_CC_PATH in rma/host_tests.cmake).
    ${RCCL_HOST_HIPIFY_DIR}/src/rma/rma_tmp.cc
    ${RCCL_HOST_HIPIFY_DIR}/src/rma/rma_ce.cc
    fakes/amdsmi_fakes.cc
    fakes/channel_fakes.cc
    fakes/argcheck_fakes.cc
    fakes/dev_runtime_micro_fakes.cc
    fakes/mgmt_task_enq_fakes.cc         # src/enqueue/mgmt_task_enq.cc
    fakes/nccl_device_core_fakes.cc      # src/nccl_device/core.cc ncclTeam* accessors
    fakes/nccl_fakes.cc
    fakes/search_fakes.cc
    fakes/libc_fakes.cc
    fakes/ras_fakes.cc
    fakes/ras_net_fakes.cc
    fakes/ras_client_support_fakes.cc
    fakes/ras_message_fakes.cc
    fakes/ras_param_fakes.cc
    fakes/rma_fakes.cc
    fakes/comm_fakes.cc
    fakes/recorder_fakes.cc
    fakes/profiler_fakes.cc
    fakes/roctx_fakes.cc
    fakes/hsa_fakes.cc                   # hsa_status_string (rocmwrap.h's HSACHECK)
    fakes/rocmwrap_fakes.cc
    fakes/topo_stubs.cc
    fakes/utils_fakes.cc
    fakes/collective_stubs.cc
    fakes/os_fakes.cc
    fakes/socket_fakes.cc
    fakes/devcomm_fakes.cc
    fakes/hip_fakes.cc
    fakes/env_fakes.cc
    fakes/ce_fakes.cc
    fakes/collectives_fakes.cc
    fakes/dda_fakes.cc
    fakes/rccl_arch_thresholds_fakes.cc
    fakes/strongstream_stubs.cc
    fakes/sym_kernels_fakes.cc
    fakes/transport_stubs.cc
    fakes/enqueue_fakes.cc               # ncclParamGraphRegister
    fakes/transport_p2p_reg_fakes.cc     # p2p.cc registration, renamed in by transport_p2p_reg_seam.h
    fakes/tuning_fakes.cc
    fakes/wrap_fakes.cc
    ../common/main_altrsmi.cpp
    ../common/GpuCountProbe.cpp
    ../common/ProcessIsolatedTestRunner.cpp
    PARENT_SCOPE)

  # rccl-UnitTestsMicroBootstrap -- src/bootstrap.cc. Own binary: bootstrap_micro_fakes.cc replaces misc/socket.cc
  # wholesale, and other binaries link bootstrap_stubs.cc for this unit's symbols.
  set(TEST_MICRO_BOOTSTRAP_SOURCE_FILES
    bootstrap-test.cc
    fakes/bootstrap_micro_fakes.cc
    fakes/env_fakes.cc
    fakes/hip_fakes.cc
    fakes/nccl_fakes.cc
    fakes/os_fakes.cc
    ${RCCL_HOST_HIPIFY_DIR}/src/misc/utils.cc
    ../common/main_altrsmi.cpp
    PARENT_SCOPE)

  # rccl-UnitTestsMicroCollectives -- src/collectives.cc. Own binary: fakes/collectives_fakes.cc, which
  # most targets link, defines this file's string helpers.
  set(TEST_MICRO_COLLECTIVES_SOURCE_FILES
    collectives-test.cc
    fakes/bootstrap_stubs.cc
    fakes/ce_fakes.cc
    fakes/collconfig_fakes.cc
    fakes/comm_fakes.cc
    fakes/dda_fakes.cc
    fakes/dev_runtime_fakes.cc
    fakes/enqueue_fakes.cc
    fakes/group_fakes.cc
    fakes/hip_fakes.cc
    fakes/nccl_fakes.cc
    fakes/nccl_stubs.cc
    fakes/mgmt_task_enq_fakes.cc
    fakes/rccl_wrap_fakes.cc
    fakes/recorder_fakes.cc
    fakes/strongstream_stubs.cc
    fakes/sym_kernels_fakes.cc
    ${RCCL_HOST_HIPIFY_DIR}/src/misc/archinfo.cc
    ../common/main_altrsmi.cpp
    PARENT_SCOPE)

  # rccl-UnitTestsMicroInit(+-uncached,-faultinj) -- src/init.cc.
  set(TEST_MICRO_INIT_SOURCE_FILES
    init-test.cc
    fakes/init_fakes.cc
    fakes/transport_net_ib_common_fakes.cc  # src/transport/net_ib/common.cc ncclNetIb
    fakes/transport_net_ib_fakes.cc      # src/transport/net_ib/connect.cc GDR flush flag seams
    fakes/bootstrap_stubs.cc
    fakes/topo_stubs.cc
    fakes/search_fakes.cc
    fakes/transport_stubs.cc
    fakes/nccl_stubs.cc
    fakes/channel_fakes.cc
    fakes/nccl_fakes.cc
    fakes/hip_fakes.cc
    # Shared per-production-TU fakes (see MICROTEST_README.md for the map).
    fakes/amdsmi_fakes.cc
    fakes/api_trace_fakes.cc
    fakes/dev_runtime_fakes.cc
    fakes/enqueue_fakes.cc               # src/enqueue/enqueue.cc (rcclEffectiveP2pBatchEnable)
    fakes/env_fakes.cc
    fakes/env_plugin_fakes.cc
    fakes/gin_fakes.cc
    fakes/group_fakes.cc
    fakes/init_nvtx_fakes.cc
    fakes/kernel_config_fakes.cc
    fakes/libc_interposers.cc
    fakes/mem_manager_fakes.cc
    fakes/mgmt_task_enq_fakes.cc
    fakes/os_fakes.cc
    fakes/profiler_fakes.cc
    fakes/rccl_wrap_fakes.cc
    fakes/recorder_fakes.cc
    fakes/rocmwrap_fakes.cc
    fakes/strongstream_stubs.cc
    fakes/sym_kernels_fakes.cc
    fakes/tuning_fakes.cc
    fakes/rccl_arch_thresholds_fakes.cc
    ${RCCL_HOST_HIPIFY_DIR}/src/misc/argcheck.cc
    ${RCCL_HOST_HIPIFY_DIR}/src/misc/archinfo.cc
    ${RCCL_HOST_HIPIFY_DIR}/src/misc/utils.cc
    ../common/main_altrsmi.cpp
    ../common/GpuCountProbe.cpp
    ../common/ProcessIsolatedTestRunner.cpp
    PARENT_SCOPE)

  # rccl-UnitTestsMicroHostDispatch -- host_test_dispatch.cc, the rccl-UnitTestsHostAll front end.
  # Own binary like every other unit under test; it also makes the dispatcher run its own tests.
  set(TEST_MICRO_HOST_DISPATCH_SOURCE_FILES
    host-test-dispatch-test.cc
    ../common/main_altrsmi.cpp
    PARENT_SCOPE)

  # rccl-UnitTestsMicroCpuset -- src/include/cpuset.h. Own binary: the test defines os/linux.cc's
  # ncclOsCpu* one-liners, which would otherwise sit in the shared binary as test-TU definitions.
  set(TEST_MICRO_CPUSET_SOURCE_FILES
    cpuset-test.cc
    ../common/main_altrsmi.cpp
    PARENT_SCOPE)

  # rccl-UnitTestsMicroDebug -- src/debug.cc. Own binary: it is the real ncclDebugLog, which
  # every shared fake set also defines.
  set(TEST_MICRO_DEBUG_SOURCE_FILES
    debug-test.cc
    ../common/main_altrsmi.cpp
    PARENT_SCOPE)

  # rccl-UnitTestsMicroCe -- src/ce_coll.cc. Its own binary: every other target links ce_fakes.cc.
  set(TEST_MICRO_CE_SOURCE_FILES
    ce-coll-test.cc
    fakes/allocator_fakes.cc             # src/allocator.cc
    fakes/bootstrap_stubs.cc
    fakes/topo_stubs.cc
    fakes/search_fakes.cc
    fakes/transport_stubs.cc
    fakes/nccl_stubs.cc
    fakes/channel_fakes.cc
    fakes/nccl_fakes.cc
    fakes/hip_fakes.cc
    fakes/ce_reduce_fakes.cc             # src/device/ce_reduce.cc
    fakes/collectives_fakes.cc
    fakes/comm_fakes.cc
    fakes/dev_runtime_fakes.cc
    fakes/devcomm_fakes.cc               # ncclTeamLsa
    fakes/env_fakes.cc
    fakes/mgmt_task_enq_fakes.cc
    fakes/os_fakes.cc
    fakes/profiler_fakes.cc
    fakes/rccl_wrap_fakes.cc             # rcclParamCeAllReduce, rcclParamCeReduceScatter, rcclParamCeArStagingBytes
    fakes/recorder_fakes.cc
    fakes/rma_fakes.cc
    fakes/roctx_fakes.cc
    fakes/strongstream_stubs.cc
    fakes/tuning_fakes.cc
    ${RCCL_HOST_HIPIFY_DIR}/src/misc/argcheck.cc
    ${RCCL_HOST_HIPIFY_DIR}/src/misc/archinfo.cc
    ${RCCL_HOST_HIPIFY_DIR}/src/misc/utils.cc
    ../common/main_altrsmi.cpp
    PARENT_SCOPE)

  # rccl-UnitTestsMicroMemManager -- src/mem_manager.cc. Own binary: mem_manager.cc defines ncclMemTrack and
  # ncclMemTrackImportFromPeer (also in nccl_fakes.cc) and ncclCommMemSuspend (also in collective_stubs.cc).
  set(TEST_MICRO_MEM_MANAGER_SOURCE_FILES
    mem-manager-test.cc
    fakes/mem_manager_micro_fakes.cc
    fakes/bootstrap_stubs.cc
    fakes/hip_fakes.cc
    ../common/main_altrsmi.cpp
    PARENT_SCOPE)

  # rccl-UnitTestsMicroAllocator -- src/allocator.cc. Own binary: every other binary that needs
  # ncclShadowPoolToHost links allocator_fakes.cc instead.
  set(TEST_MICRO_ALLOCATOR_SOURCE_FILES
    allocator-test.cc
    fakes/hip_fakes.cc
    fakes/nccl_fakes.cc
    fakes/rocmwrap_fakes.cc
    ../common/main_altrsmi.cpp
    PARENT_SCOPE)

  # rccl-UnitTestsMicroMnnvl -- src/mnnvl.cc. Own binary: nccl_stubs.cc, which most targets link, defines
  # ncclMnnvlCheck, and mnnvl.cc defines it too.
  set(TEST_MICRO_MNNVL_SOURCE_FILES
    mnnvl-test.cc
    fakes/hip_fakes.cc
    fakes/nccl_fakes.cc
    fakes/utils_fakes.cc
    ../common/main_altrsmi.cpp
    PARENT_SCOPE)

  # rccl-UnitTestsMicroAlloc -- src/include/alloc.h. Own binary: alloc.h's inline side-stream map and
  # shutdown and skip-free state would be shared with every other unit in rccl-UnitTestsMicro.
  set(TEST_MICRO_ALLOC_SOURCE_FILES
    alloc-test.cc
    fakes/hip_fakes.cc
    fakes/nccl_fakes.cc
    fakes/os_fakes.cc
    fakes/utils_fakes.cc
    ../common/main_altrsmi.cpp
    PARENT_SCOPE)

  # rccl-UnitTestsMicroSymKernels -- REAL src/sym_kernels.cc + generated host TU.
  set(TEST_MICRO_SYM_KERNELS_SOURCE_FILES
    sym-kernels-test.cc
    fakes/bootstrap_stubs.cc
    fakes/topo_stubs.cc
    fakes/search_fakes.cc
    fakes/transport_stubs.cc
    fakes/nccl_stubs.cc
    fakes/channel_fakes.cc
    fakes/nccl_fakes.cc
    fakes/hip_fakes.cc
    fakes/env_fakes.cc
    fakes/devcomm_fakes.cc
    fakes/gin_fakes.cc                   # ncclGetRailedGinType, read by sym_kernels.cc
    fakes/nccl_device_core_fakes.cc      # ncclTeamRail, read by sym_kernels.cc
    fakes/mgmt_task_enq_fakes.cc
    fakes/os_fakes.cc
    fakes/profiler_fakes.cc
    fakes/strongstream_stubs.cc
    fakes/recorder_fakes.cc
    ${RCCL_HOST_HIPIFY_DIR}/src/misc/argcheck.cc
    ${RCCL_HOST_HIPIFY_DIR}/src/misc/archinfo.cc
    ${RCCL_HOST_HIPIFY_DIR}/src/misc/utils.cc
    ../common/main_altrsmi.cpp
    PARENT_SCOPE)

  # rccl-UnitTestsMicroCftDevRuntime -- src/cft_dev_runtime.cc + src/nccl_device/core.cc. Own
  # binary: rccl-UnitTestsMicro fakes most of core.cc, which the LE-info queries call.
  set(TEST_MICRO_CFT_DEV_RUNTIME_SOURCE_FILES
    cft-dev-runtime-test.cc
    fakes/bootstrap_stubs.cc
    fakes/topo_stubs.cc
    fakes/transport_stubs.cc
    fakes/nccl_stubs.cc
    fakes/channel_fakes.cc
    fakes/nccl_fakes.cc
    fakes/hip_fakes.cc
    fakes/collectives_fakes.cc
    fakes/comm_fakes.cc
    fakes/dev_runtime_fakes.cc
    fakes/env_fakes.cc
    fakes/mgmt_task_enq_fakes.cc
    fakes/os_fakes.cc
    fakes/recorder_fakes.cc
    fakes/strongstream_stubs.cc
    fakes/tuning_fakes.cc
    ${RCCL_HOST_HIPIFY_DIR}/src/misc/argcheck.cc
    ${RCCL_HOST_HIPIFY_DIR}/src/misc/archinfo.cc
    ${RCCL_HOST_HIPIFY_DIR}/src/misc/utils.cc
    ../common/main_altrsmi.cpp
    PARENT_SCOPE)

  # rccl-UnitTestsMicroProxy -- src/proxy.cc. Own binary: it compiles the real proxy.cc, which
  # proxy_fakes.cc fakes for the other binaries.
  set(TEST_MICRO_PROXY_SOURCE_FILES
    proxy-test.cc
    fakes/proxy_micro_fakes.cc
    fakes/env_fakes.cc
    fakes/hip_fakes.cc
    fakes/os_fakes.cc
    ../common/main_altrsmi.cpp
    PARENT_SCOPE)
endfunction()

# Create one host-only micro-test binary and apply all the boilerplate that
# differs only by build mode. The per-target specifics are passed in:
#   SOURCES         the target's source list
#   EXTRA_INCLUDES  include dirs appended after the mode's include-dir var
#   EXTRA_DEFS      compile definitions appended after the mode's define baseline
#   CATEGORIES_YAML CTest category yaml (in-build only; ignored standalone)
#   REMOVE_DEFS     defines to drop from the in-build baseline before EXTRA_DEFS are
#                   added; ignored standalone, whose baseline is fixed
function(rccl_add_micro_binary _name)
  cmake_parse_arguments(_ramb "OBJECT" "CATEGORIES_YAML"
    "SOURCES;EXTRA_INCLUDES;EXTRA_DEFS;REMOVE_DEFS" ${ARGN})
  # OBJECT builds an object library with the same compile settings, for sources several binaries share.
  if(_ramb_OBJECT)
    add_library(${_name} OBJECT ${_ramb_SOURCES})
  else()
    add_executable(${_name} ${_ramb_SOURCES})
  endif()

  if(BUILD_TESTS)
    set(_ramb_base_defs ${RCCL_COMMON_COMPILE_DEFS})
    if(_ramb_REMOVE_DEFS)
      list(REMOVE_ITEM _ramb_base_defs ${_ramb_REMOVE_DEFS})
    endif()
    target_include_directories(${_name} PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR} ${RCCL_COMMON_INCLUDE_DIRS} ${_ramb_EXTRA_INCLUDES})
    target_compile_definitions(${_name} PRIVATE
      ${_ramb_base_defs} ${_ramb_EXTRA_DEFS})
    rccl_apply_rocshmem_gin_optout(${_name})
    rccl_apply_hip_compile_only(${_name})
    target_link_libraries(${_name} PRIVATE ${MICRO_TEST_LINK_LIBS})
    add_dependencies(${_name} hipify_all copy_nccl_device_headers)
    # csv-tuner-test.cc includes csv_tuner.cc, which includes the generated
    # embedded-tuner header. These targets do not link rccl, so without this the
    # header may not exist yet in a clean parallel build.
    if(TARGET rccl_embedded_tuner_configs)
      add_dependencies(${_name} rccl_embedded_tuner_configs)
    endif()
    rccl_apply_host_only_flags(${_name})
    if(NOT _ramb_OBJECT)
      rocm_install(TARGETS ${_name} COMPONENT tests)
      if(_ramb_CATEGORIES_YAML)
        rccl_register_test_categories(${_name} "${_ramb_CATEGORIES_YAML}")
      endif()
    endif()
  else()
    # Mirror add_test_executable(): record for the standalone coverage loop.
    if(_ramb_OBJECT)
      list(APPEND HOST_TEST_OBJECT_LIBS ${_name})
      set(HOST_TEST_OBJECT_LIBS "${HOST_TEST_OBJECT_LIBS}" PARENT_SCOPE)
    else()
      list(APPEND HOST_TEST_BINARIES ${_name})
      set(HOST_TEST_BINARIES "${HOST_TEST_BINARIES}" PARENT_SCOPE)
    endif()
    target_compile_options(${_name} PRIVATE
      -Wno-unused-parameter -Wno-unused-variable -ffunction-sections)
    target_include_directories(${_name} PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR} ${RCCL_INCLUDE_DIRS} ${_ramb_EXTRA_INCLUDES})
    target_compile_definitions(${_name} PRIVATE
      __HIP_PLATFORM_AMD__ NCCL_OS_LINUX ROCM_VERSION=70000 ${_ramb_EXTRA_DEFS})
    # -no-pie only when the non-PIC RCCL-vendored libgtest.a is linked; forcing
    # it against PIC (system/FetchContent) GTest breaks PIE-only toolchains.
    set(_ramb_link_opts -no-hip-rt -Wl,--gc-sections)
    if(_MICRO_LINK_NO_PIE)
      list(APPEND _ramb_link_opts -no-pie)
    endif()
    target_link_options(${_name} PRIVATE ${_ramb_link_opts})
    target_link_libraries(${_name} PRIVATE
      GTest::GTest Threads::Threads fmt::fmt-header-only)
  endif()
  target_link_libraries(${_name} PRIVATE nlohmann_json::nlohmann_json)
endfunction()

# One binary per test file. SOURCES (relative to test/host/) compile once into object libs named from OBJECT_PREFIX,
# unless EXTRA_DEFS or in-tree REMOVE_DEFS is set. Wrap it in a macro so HOST_TEST_BINARIES reaches the top-level scope.
function(rccl_add_unit_micro_binary _name _test)
  cmake_parse_arguments(_raumb "PROCESS_ISOLATED" "OBJECT_PREFIX;CATEGORIES_YAML"
    "SOURCES;COMMON_INCLUDES;COMMON_DEFS;EXTRA_DEFS;REMOVE_DEFS" ${ARGN})
  if(NOT _raumb_OBJECT_PREFIX)
    message(FATAL_ERROR "rccl_add_unit_micro_binary(${_name}): needs OBJECT_PREFIX")
  endif()
  if(_raumb_UNPARSED_ARGUMENTS)
    message(FATAL_ERROR "rccl_add_unit_micro_binary(${_name}): unknown arguments: ${_raumb_UNPARSED_ARGUMENTS}")
  endif()
  set(_raumb_shared ${_raumb_SOURCES} ../common/main_altrsmi.cpp)
  if(_raumb_PROCESS_ISOLATED)
    # GpuCountProbe.cpp resolves the runner's gpuCountViaFreshImage(); the two always link together.
    list(APPEND _raumb_shared ../common/GpuCountProbe.cpp ../common/ProcessIsolatedTestRunner.cpp)
  endif()
  set(_raumb_args
    EXTRA_INCLUDES ${_raumb_COMMON_INCLUDES}
    EXTRA_DEFS ${_MICRO_MODE_DEFS} ${_raumb_COMMON_DEFS} ${_raumb_EXTRA_DEFS})
  set(_raumb_sources ${_test})
  if(_raumb_EXTRA_DEFS OR (BUILD_TESTS AND _raumb_REMOVE_DEFS))
    list(APPEND _raumb_sources ${_raumb_shared})
    list(APPEND _raumb_args REMOVE_DEFS ${_raumb_REMOVE_DEFS})
  else()
    foreach(_src ${_raumb_shared})
      string(REPLACE "${RCCL_HOST_HIPIFY_DIR}/" "" _obj "${_src}")
      string(MAKE_C_IDENTIFIER "${_raumb_OBJECT_PREFIX}-${_obj}" _obj)
      if(NOT TARGET ${_obj})
        rccl_add_micro_binary(${_obj} OBJECT SOURCES ${_src} ${_raumb_args})
      endif()
      list(APPEND _raumb_sources $<TARGET_OBJECTS:${_obj}>)
    endforeach()
  endif()
  rccl_add_micro_binary(${_name}
    SOURCES ${_raumb_sources}
    ${_raumb_args}
    CATEGORIES_YAML "${_raumb_CATEGORIES_YAML}")
  set(HOST_TEST_BINARIES "${HOST_TEST_BINARIES}" PARENT_SCOPE)
  set(HOST_TEST_OBJECT_LIBS "${HOST_TEST_OBJECT_LIBS}" PARENT_SCOPE)
endfunction()

# =====================================================================
# Standalone project setup (only used when NOT in an in-RCCL-build).
# =====================================================================
function(rccl_standalone_setup)
  find_package(Threads REQUIRED)
  include(FetchContent)

  # --- Resolve GTest: system -> RCCL-vendored -> FetchContent fallback ---
  # Track whether we linked the RCCL-vendored non-PIC libgtest.a: only that path
  # needs -no-pie on the micro binaries. System/FetchContent GTest is PIC, and
  # forcing -no-pie there is unnecessary and breaks PIE-only toolchains.
  set(_micro_link_no_pie FALSE)
  find_package(GTest QUIET)
  if(NOT GTest_FOUND)
    set(_rccl_gtest_root "${CMAKE_CURRENT_SOURCE_DIR}/../../build/gtest")
    if(EXISTS "${_rccl_gtest_root}/lib/libgtest.a" OR EXISTS "${_rccl_gtest_root}/lib64/libgtest.a")
      message(STATUS "Using RCCL-vendored GTest: ${_rccl_gtest_root}")
      find_package(GTest REQUIRED PATHS "${_rccl_gtest_root}" NO_DEFAULT_PATH)
      set(_micro_link_no_pie TRUE)
    else()
      message(STATUS "Downloading GTest via FetchContent")
      FetchContent_Declare(googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG        v1.14.0
        GIT_SHALLOW    TRUE
      )
      FetchContent_MakeAvailable(googletest)
      add_library(GTest::GTest ALIAS gtest)
    endif()
  endif()

  # Both find_package(GTest) code paths above (system and RCCL-vendored) export
  # only the modern GTest::gtest target; GTest::GTest is a legacy alias CMake's
  # own FindGTest module stopped providing. Add it once here instead of after
  # every branch so target_link_libraries(... GTest::GTest) below works
  # regardless of which resolution path was taken.
  if(TARGET GTest::gtest AND NOT TARGET GTest::GTest)
    # Promote before aliasing: GTest's package config creates GTest::gtest as a
    # non-GLOBAL imported target, and ALIAS of one of those needs CMake 3.18,
    # above this project's declared 3.16 minimum. Promotion works from 3.11 and
    # only from the directory that created the target, which is this file.
    set_target_properties(GTest::gtest PROPERTIES IMPORTED_GLOBAL TRUE)
    add_library(GTest::GTest ALIAS GTest::gtest)
  endif()

  # --- fmt: the RCCL-vendored header-only copy (already defined for in-tree builds) ---
  if(NOT TARGET fmt::fmt-header-only)
    add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/../../external/fmt"
                     "${CMAKE_BINARY_DIR}/external/fmt")
  endif()

  # --- Resolve RCCL build directory ---
  set(RCCL_BUILD_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../build" CACHE PATH
      "RCCL build directory (auto-detects release/debug subdirs)")

  if(NOT EXISTS "${RCCL_BUILD_DIR}/hipify")
    foreach(_subdir release debug Release Debug RelWithDebInfo)
      if(EXISTS "${RCCL_BUILD_DIR}/${_subdir}/hipify")
        set(RCCL_BUILD_DIR "${RCCL_BUILD_DIR}/${_subdir}")
        message(STATUS "Detected build subdir layout: ${RCCL_BUILD_DIR}")
        break()
      endif()
    endforeach()
  endif()

  set(RCCL_HIPIFY_DIR "${RCCL_BUILD_DIR}/hipify")
  set(RCCL_TEST_DIR "${CMAKE_CURRENT_SOURCE_DIR}/..")

  if(NOT EXISTS "${RCCL_HIPIFY_DIR}/src/graph/search.cc")
    message(FATAL_ERROR "Hipified sources not found at ${RCCL_HIPIFY_DIR}. Run the RCCL build first.")
  endif()

  find_package(Python3 COMPONENTS Interpreter REQUIRED)

  # csv_tuner.cc needs the embedded-tuner header. The RCCL build normally emits
  # it; generate it here so a standalone test build works against an older tree.
  if(NOT EXISTS "${RCCL_BUILD_DIR}/include/rccl_tuner_embedded_configs.h")
    execute_process(
      COMMAND ${Python3_EXECUTABLE}
              ${CMAKE_CURRENT_SOURCE_DIR}/../../tuner/embed_tuner_configs.py
              ${CMAKE_CURRENT_SOURCE_DIR}/../../tuner
              ${RCCL_BUILD_DIR}/include/rccl_tuner_embedded_configs.h
      RESULT_VARIABLE _gen_tuner_result
      ERROR_VARIABLE _gen_tuner_error)
    if(_gen_tuner_result)
      message(FATAL_ERROR "embed_tuner_configs.py failed: ${_gen_tuner_error}")
    endif()
  endif()
  message(STATUS "CXX compiler: ${CMAKE_CXX_COMPILER}")
  message(STATUS "Hipified sources: ${RCCL_HIPIFY_DIR}")

  # --- Detect whether the system amdsmi.h exposes the UALoE fabric struct ---
  # Compile-only (STATIC_LIBRARY) avoids link failures from amdsmi symbols.
  include(CheckCXXSourceCompiles)
  unset(AMDSMI_FABRIC_API CACHE)
  unset(AMDSMI_HEADER_HAS_FABRIC_TELEMETRY CACHE)
  set(CMAKE_TRY_COMPILE_TARGET_TYPE STATIC_LIBRARY)
  check_cxx_source_compiles("
    #include <amd_smi/amdsmi.h>
    int main() {
      amdsmi_fabric_telemetry_category_t cat = AMDSMI_FABRIC_TELEMETRY_CATEGORY_UNKNOWN;
      (void)cat;
      return 0;
    }
  " AMDSMI_HEADER_HAS_FABRIC_TELEMETRY)
  check_cxx_source_compiles("
    #include <amd_smi/amdsmi.h>
    #if !defined(AMDSMI_FABRIC_MAX_LOCAL_GPUS)
    #error amdsmi.h has no UALoE fabric types
    #endif
    int main() {
      amdsmi_fabric_info_t info{};
      amdsmi_fabric_info_v1_t v1{};
      (void)info;
      (void)v1;
      return 0;
    }
  " AMDSMI_FABRIC_API)
  set(CMAKE_TRY_COMPILE_TARGET_TYPE EXECUTABLE)
  if(AMDSMI_HEADER_HAS_FABRIC_TELEMETRY AND NOT AMDSMI_FABRIC_API)
    set(AMDSMI_FABRIC_API TRUE)
    message(STATUS "amdsmi fabric telemetry types found - enabling AMDSMI_FABRIC_DIRECT")
  elseif(AMDSMI_FABRIC_API)
    message(STATUS "amdsmi fabric API found - enabling AMDSMI_FABRIC_DIRECT")
  else()
    message(STATUS "amdsmi fabric API not found (AMDSMI_FABRIC_DIRECT disabled)")
  endif()

  # --- Detect HIP fabric API support (mirrors top-level CMakeLists.txt) ---
  # mem_manager.h only defines its own hipMemFabricHandle_st fallback when
  # HIP_FABRIC_API is undefined; on HIP runtimes that already provide the type
  # (e.g. ROCm 7.14+), omitting this define causes a redefinition conflict
  # with hip_runtime_api.h.
  # Compile-only (STATIC_LIBRARY), like the amdsmi probe above: with the default
  # EXECUTABLE target type these checks link, which drags in the HIP runtime at
  # configure time even though the binary itself is linked with -no-hip-rt.
  include(CheckSymbolExists)
  set(CMAKE_TRY_COMPILE_TARGET_TYPE STATIC_LIBRARY)
  check_symbol_exists("hipMemImportFromShareableHandle" "hip/hip_runtime_api.h" HIP_FABRIC_API_FUNC)
  check_cxx_source_compiles("
    #include <hip/hip_runtime_api.h>
    int main() {
      hipMemFabricHandle_t handle;
      (void)handle;
      return 0;
    }
  " HIP_FABRIC_HANDLE_TYPE)
  set(CMAKE_TRY_COMPILE_TARGET_TYPE EXECUTABLE)
  if(HIP_FABRIC_API_FUNC AND HIP_FABRIC_HANDLE_TYPE)
    set(HIP_FABRIC_API ON)
    message(STATUS "HIP Fabric API enabled (hipMemImportFromShareableHandle and hipMemFabricHandle_t found)")
  else()
    set(HIP_FABRIC_API OFF)
  endif()

  # --- Common include paths for hipified RCCL headers ---
  set(RCCL_INCLUDE_DIRS
    ${RCCL_HIPIFY_DIR}/src/include
    ${RCCL_HIPIFY_DIR}/src/include/plugin
    ${RCCL_HIPIFY_DIR}/src/include/plugin/tuner
    ${RCCL_BUILD_DIR}/include
    ${RCCL_HIPIFY_DIR}/src
    ${RCCL_HIPIFY_DIR}/src/graph
    ${RCCL_TEST_DIR}
    ${RCCL_TEST_DIR}/common
  )

  # --- accl-profiler plugin directory ---
  set(ACCL_PROFILER_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../plugins/profiler/accl")
  set(RCCL_SRC_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../..")

  # --- Wrapper library for production .cc files that need link stubs ---
  add_library(rccl-source-wrappers STATIC
    mem_manager_src_wrapper.cc
    mem_manager_wrapper_test.cc
    fakes/roctx_fakes.cc
    bootstrap_wrapper.cc
    init_stubs.cpp
    wrapper_link_stubs.cpp
    hip_prop_r0600_stub.cpp
    accl_profiler_wrapper.cc
  )
  target_include_directories(rccl-source-wrappers PRIVATE
    ${ACCL_PROFILER_DIR}
    ${RCCL_INCLUDE_DIRS}
  )
  # Standalone HostUnitTests never find_package(hip); in-tree already has hip::host.
  if(NOT TARGET hip::host)
    find_package(hip QUIET)
  endif()
  foreach(_hip_tgt hip::host hip::device)
    if(TARGET ${_hip_tgt})
      target_include_directories(rccl-source-wrappers PRIVATE
        $<TARGET_PROPERTY:${_hip_tgt},INTERFACE_INCLUDE_DIRECTORIES>)
      target_compile_definitions(rccl-source-wrappers PRIVATE
        $<TARGET_PROPERTY:${_hip_tgt},INTERFACE_COMPILE_DEFINITIONS>)
    endif()
  endforeach()
  # accl_profiler_wrapper.cc needs the non-hipified profiler headers;
  # scope the extra include to just that file to avoid os.h/rocmwrap.h
  # conflicts with the hipified sources used by other wrappers.
  set_source_files_properties(accl_profiler_wrapper.cc PROPERTIES
    COMPILE_FLAGS "-I${RCCL_SRC_ROOT}/src/include/plugin -I${RCCL_SRC_ROOT}/src/include"
  )
  target_compile_definitions(rccl-source-wrappers PRIVATE
    __HIP_PLATFORM_AMD__
    NCCL_OS_LINUX
    ROCM_VERSION=70000
    # init.cc selects amdsmi_wrap.h (amd_smi/amdsmi.h, ships with ROCm) over the
    # legacy rocm_smi_wrap.h; the amd_smi_* wrapper API it declares is stubbed by
    # init_fakes.cc, so no SMI runtime is linked. The in-RCCL-build path inherits
    # this from RCCL_COMMON_COMPILE_DEFS instead.
    USE_AMDSMI
    $<$<BOOL:${AMDSMI_FABRIC_API}>:AMDSMI_FABRIC_DIRECT>
    $<$<BOOL:${AMDSMI_HEADER_HAS_FABRIC_TELEMETRY}>:AMDSMI_HEADER_HAS_FABRIC_TELEMETRY>
    $<$<BOOL:${HIP_FABRIC_API}>:HIP_FABRIC_API>
  )
  target_link_libraries(rccl-source-wrappers PRIVATE
    GTest::GTest
    fmt::fmt-header-only
  )

  # --- Main test executable ---
  add_test_executable(rccl-HostUnitTests
    main.cpp
    ${RCCL_TEST_DIR}/BitOpsTests.cpp
    ${RCCL_TEST_DIR}/IommuPassthrough_test.cpp
    ${RCCL_TEST_DIR}/VersionInfoTests.cpp
    ${RCCL_TEST_DIR}/DdaCollCommonTests.cpp
    ${RCCL_TEST_DIR}/RomeTopoConsensusTests.cpp
    RomeTopoConsensusLogTests.cpp
    process-isolated-runner-pool-test.cc
    ${RCCL_TEST_DIR}/MiscTests.cpp
    ${RCCL_TEST_DIR}/VersionGateTests.cpp
    ${RCCL_TEST_DIR}/TimeoutTests.cpp
    ${RCCL_TEST_DIR}/EnqueueCountTests.cpp
    ${RCCL_TEST_DIR}/BootstrapBidirTests.cpp
    ${RCCL_TEST_DIR}/AltRsmiTests.cpp
    ${RCCL_TEST_DIR}/AcclProfilerTests.cpp
    ${RCCL_TEST_DIR}/common/GpuCountProbe.cpp
    ${RCCL_TEST_DIR}/common/ProcessIsolatedTestRunner.cpp
    ${RCCL_HIPIFY_DIR}/src/misc/kernel_config.cc
    ${RCCL_HIPIFY_DIR}/src/misc/alt_rsmi.cc
    ${RCCL_HIPIFY_DIR}/src/graph/rome_topo_consensus.cc
  )

  target_include_directories(rccl-HostUnitTests PRIVATE
    ${ACCL_PROFILER_DIR}
    ${RCCL_INCLUDE_DIRS}
  )
  set_source_files_properties(${RCCL_TEST_DIR}/AcclProfilerTests.cpp PROPERTIES
    COMPILE_FLAGS "-I${RCCL_SRC_ROOT}/src/include/plugin -I${RCCL_SRC_ROOT}/src/include"
  )

  target_compile_definitions(rccl-HostUnitTests PRIVATE
    __HIP_PLATFORM_AMD__
    NCCL_OS_LINUX
    ROCM_VERSION=70000
    RCCL_EXPOSE_STATIC
    ARSMI_TEST_BUILD
    $<$<BOOL:${AMDSMI_FABRIC_API}>:AMDSMI_FABRIC_DIRECT>
    $<$<BOOL:${AMDSMI_HEADER_HAS_FABRIC_TELEMETRY}>:AMDSMI_HEADER_HAS_FABRIC_TELEMETRY>
    $<$<BOOL:${HIP_FABRIC_API}>:HIP_FABRIC_API>
  )

  # -no-pie only for the non-PIC RCCL-vendored libgtest.a (see _micro_link_no_pie).
  set(_host_link_opts -no-hip-rt)
  if(_micro_link_no_pie)
    list(APPEND _host_link_opts -no-pie)
  endif()
  target_link_options(rccl-HostUnitTests PRIVATE ${_host_link_opts})

  target_link_libraries(rccl-HostUnitTests PRIVATE
    GTest::GTest
    Threads::Threads
    fmt::fmt-header-only
    # RCCL_TEST_CODE_COVERAGE compiles a dlsym() call in
    # ProcessIsolatedTestRunner.cpp; the in-tree test build links dl for the same
    # runner. Empty on platforms where libdl is folded into libc (glibc >= 2.34),
    # required on older ones.
    ${CMAKE_DL_LIBS}
    -Wl,--whole-archive
    rccl-source-wrappers
    -Wl,--no-whole-archive
  )

  # --- Per-mode variables consumed by the shared micro-binary block ---------
  set(RCCL_HOST_HIPIFY_DIR "${RCCL_HIPIFY_DIR}" PARENT_SCOPE)
  set(RCCL_INCLUDE_DIRS "${RCCL_INCLUDE_DIRS}" PARENT_SCOPE)
  set(_MICRO_MODE_DEFS
    $<$<BOOL:${AMDSMI_FABRIC_API}>:AMDSMI_FABRIC_DIRECT>
    $<$<BOOL:${AMDSMI_HEADER_HAS_FABRIC_TELEMETRY}>:AMDSMI_HEADER_HAS_FABRIC_TELEMETRY>
    $<$<BOOL:${HIP_FABRIC_API}>:HIP_FABRIC_API>
    PARENT_SCOPE)
  set(_MICRO_AMDSMI_DEF USE_AMDSMI PARENT_SCOPE)
  set(_MICRO_LINK_NO_PIE "${_micro_link_no_pie}" PARENT_SCOPE)
  # ENABLE_DEVICE_LINKER defaults ON, so the enqueue arm that SHIPS is the
  # -devlinker one. The in-RCCL-build path builds only the single inherited-arm
  # binary; run_host_tests.sh (the only in-repo workflow that runs these) uses
  # this standalone path, so build the shipping arm explicitly too.
  set(_ENQ_BUILD_DEVLINKER TRUE PARENT_SCOPE)
  # Likewise roctx.cc's ROCTX_ENABLE arm (-roctx).
  set(_MICRO_BUILD_ROCTX_VARIANT TRUE PARENT_SCOPE)
  # No GENERATE_SYM_KERNELS variable exists in this standalone project(); check
  # for the generated host TU on disk instead.
  if(EXISTS "${RCCL_HIPIFY_DIR}/gensrc/symmetric/sym_kernels_host.cc")
    set(_MICRO_BUILD_SYM_KERNELS TRUE PARENT_SCOPE)
  else()
    set(_MICRO_BUILD_SYM_KERNELS FALSE PARENT_SCOPE)
  endif()
  set(HOST_TEST_BINARIES "${HOST_TEST_BINARIES}" PARENT_SCOPE)
endfunction()

# =====================================================================
# Standalone Coverage
# =====================================================================
function(rccl_standalone_coverage)
  option(ENABLE_CODE_COVERAGE "Enable code coverage" OFF)
  if(ENABLE_CODE_COVERAGE)
    # HOST_TEST_BINARIES is populated by add_test_executable() and by
    # rccl_add_micro_binary(), so new test binaries are instrumented automatically.
    foreach(_cov_tgt ${HOST_TEST_BINARIES})
      target_compile_options(${_cov_tgt} PRIVATE -fprofile-instr-generate -fcoverage-mapping)
      target_link_options(${_cov_tgt} PRIVATE -fprofile-instr-generate -fcoverage-mapping)
      target_compile_definitions(${_cov_tgt} PRIVATE RCCL_TEST_CODE_COVERAGE=1)
      target_link_libraries(${_cov_tgt} PRIVATE ${CMAKE_DL_LIBS})
    endforeach()
    foreach(_cov_tgt ${HOST_TEST_OBJECT_LIBS})
      target_compile_options(${_cov_tgt} PRIVATE -fprofile-instr-generate -fcoverage-mapping)
      target_compile_definitions(${_cov_tgt} PRIVATE RCCL_TEST_CODE_COVERAGE=1)
    endforeach()
    target_compile_options(rccl-source-wrappers PRIVATE -fprofile-instr-generate -fcoverage-mapping)
  endif()
endfunction()

# --- nlohmann/json: the RCCL-vendored single header ------------------
add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/../../external/nlohmann_json"
                 "${CMAKE_BINARY_DIR}/external/nlohmann_json")

# =====================================================================
# Driver: pick the mode, then define the micro-test binaries ONCE.
# =====================================================================
if(BUILD_TESTS)
  set(RCCL_HOST_HIPIFY_DIR "${PROJECT_BINARY_DIR}/hipify")
  set(_MICRO_MODE_DEFS "")
  set(_MICRO_AMDSMI_DEF "")
  set(_ENQ_BUILD_DEVLINKER FALSE)
  set(_MICRO_BUILD_ROCTX_VARIANT FALSE)
  set(_MICRO_BUILD_SYM_KERNELS ${GENERATE_SYM_KERNELS})
else()
  # --- hipcc is mandatory (must be resolved before project() at top level) ---
  # Auto-detect hipcc via ROCM_PATH, or pass -DCMAKE_CXX_COMPILER=hipcc.
  if(NOT DEFINED CMAKE_CXX_COMPILER OR CMAKE_CXX_COMPILER STREQUAL "")
    if(NOT ROCM_PATH)
      if(DEFINED ENV{ROCM_PATH})
        set(ROCM_PATH "$ENV{ROCM_PATH}")
      else()
        set(ROCM_PATH "/opt/rocm")
      endif()
    endif()
    find_program(_hipcc hipcc HINTS ${ROCM_PATH} PATH_SUFFIXES bin NO_CMAKE_FIND_ROOT_PATH)
    if(NOT _hipcc)
      find_program(_hipcc hipcc)
    endif()
    if(_hipcc)
      set(CMAKE_CXX_COMPILER "${_hipcc}")
    else()
      message(FATAL_ERROR
        "hipcc not found. Either:\n"
        "  cmake -B build -DCMAKE_CXX_COMPILER=/path/to/hipcc\n"
        "  cmake -B build -DROCM_PATH=/opt/rocm-6.x")
    endif()
  endif()

  # hipcc needs --gcc-install-dir to find C++ standard headers.
  find_program(_system_gxx g++ REQUIRED)
  execute_process(
    COMMAND ${_system_gxx} -print-file-name=libstdc++.a
    OUTPUT_VARIABLE _gcc_libstdcpp
    OUTPUT_STRIP_TRAILING_WHITESPACE
  )
  get_filename_component(_gcc_install_dir "${_gcc_libstdcpp}" DIRECTORY)
  set(CMAKE_CXX_FLAGS_INIT "--gcc-install-dir=${_gcc_install_dir} --offload-host-only")

  project(rccl-HostUnitTests LANGUAGES CXX)

  # hipcc prepends -O3 and CMake's Debug flags carry no -O, so Debug built at -O3.
  # Appended, not prepended: the last -O on the command line wins.
  string(APPEND CMAKE_CXX_FLAGS_DEBUG " -O0")

  option(HOST_TEST_CCACHE "Route host-test compiles through ccache when available" ON)
  if(HOST_TEST_CCACHE AND NOT CMAKE_CXX_COMPILER_LAUNCHER)
    find_program(CCACHE_PROGRAM ccache)
    if(CCACHE_PROGRAM)
      message(STATUS "Using ccache: ${CCACHE_PROGRAM}")
      set(CMAKE_CXX_COMPILER_LAUNCHER "${CCACHE_PROGRAM}")
    endif()
  endif()

  set(CMAKE_CXX_STANDARD 17)
  set(CMAKE_CXX_STANDARD_REQUIRED ON)

  add_compile_options(-Wno-tautological-constant-out-of-range-compare)

  # Wrapper around add_executable() that also records the target in
  # HOST_TEST_BINARIES, so the coverage function instruments every host-only test
  # binary without a separately-maintained list. A macro (not a function) so
  # list(APPEND) mutates the caller's variable. Use this instead of
  # add_executable() for every test binary in the standalone build.
  macro(add_test_executable _name)
    list(APPEND HOST_TEST_BINARIES ${_name})
    add_executable(${_name} ${ARGN})
  endmacro()

  rccl_standalone_setup()
endif()

# =====================================================================
# Shared micro-test binary definitions (identical for BOTH build paths).
# =====================================================================
rccl_define_micro_source_lists()

# Real src/ siblings that several binaries compile. They read their params through g_loadParam (nccl_fakes.h).
set_source_files_properties(
  ${RCCL_HOST_HIPIFY_DIR}/src/dev_runtime.cc
  ${RCCL_HOST_HIPIFY_DIR}/src/rccl_dev_runtime.cc
  ${RCCL_HOST_HIPIFY_DIR}/src/rma/rma_tmp.cc
  PROPERTIES
  GENERATED TRUE
  COMPILE_OPTIONS "-include;fakes/nccl_fakes.h;-include;fakes/param_redirect.h")
# rocshmem.hpp is only on the include path under ENABLE_ROCSHMEM.
if(ENABLE_ROCSHMEM)
  list(APPEND TEST_MICRO_INIT_SOURCE_FILES fakes/rocshmem_fakes.cc)
endif()

# Mark every hipified misc/ oracle TU any target below compiles as GENERATED.
# The property is directory-scoped and idempotent, so one call up front covers
# all targets -- no need to repeat it per binary.
rccl_mark_oracle_tus_generated(${RCCL_HOST_HIPIFY_DIR} argcheck archinfo utils kernel_config)

# ---------------------------------------------------------------------------
# rccl-UnitTestsMicroBootstrap
# ---------------------------------------------------------------------------
rccl_add_micro_binary(rccl-UnitTestsMicroBootstrap
  SOURCES ${TEST_MICRO_BOOTSTRAP_SOURCE_FILES}
  EXTRA_DEFS
    ${_MICRO_MODE_DEFS}
    BOOTSTRAP_CC_PATH="${RCCL_HOST_HIPIFY_DIR}/src/bootstrap.cc"
    RCCL_OS_FAKES_OMIT_ncclSocketClose   # fakes/bootstrap_micro_fakes.cc defines it
  CATEGORIES_YAML "${CMAKE_CURRENT_SOURCE_DIR}/../test_categories_micro_bootstrap.yaml")

# ---------------------------------------------------------------------------
# rccl-UnitTestsMicro
# ---------------------------------------------------------------------------
set_source_files_properties(
  ${RCCL_HOST_HIPIFY_DIR}/src/transport/p2p_tmp.cc
  ${RCCL_HOST_HIPIFY_DIR}/src/devcomm/devcomm_v22902.cc
  ${RCCL_HOST_HIPIFY_DIR}/src/devcomm/devcomm_v22907.cc
  PROPERTIES GENERATED TRUE)
# fakes/rma_fakes.cc defines these two as seams; rename the real ones.
set_source_files_properties(${RCCL_HOST_HIPIFY_DIR}/src/rma/rma_ce.cc PROPERTIES
  GENERATED TRUE
  COMPILE_DEFINITIONS "ncclRmaCePutLaunch=ncclRmaCePutLaunchUut;ncclRmaCeWaitLaunch=ncclRmaCeWaitLaunchUut")
rccl_add_micro_binary(rccl-UnitTestsMicro
  SOURCES ${TEST_MICRO_SOURCE_FILES}
  EXTRA_INCLUDES
    ${RCCL_HOST_HIPIFY_DIR}/src/devcomm              # devcomm_v22902.h, included unqualified
    ${RCCL_HOST_HIPIFY_DIR}/src/include/nccl_device  # device.h includes rccl_ptr.h unqualified
  EXTRA_DEFS
    ${_MICRO_MODE_DEFS}
    COMM_DUMP_CC_PATH="${RCCL_HOST_HIPIFY_DIR}/src/commDump.cc"
    GROUP_CC_PATH="${RCCL_HOST_HIPIFY_DIR}/src/group.cc"
    WRAP_CC_PATH="${RCCL_HOST_HIPIFY_DIR}/src/rccl_wrap.cc"
    RCCL_RMA_FAKES_OMIT_ncclRmaInitialized  # rma_tmp.cc defines it
    RCCL_TRANSPORT_STUBS_OMIT_COLLECTIVE_FLOOR
    DEV_RUNTIME_CC_PATH="${RCCL_HOST_HIPIFY_DIR}/src/dev_runtime.cc"
    DEV_RUNTIME_SEGMENTS_CC_PATH="${RCCL_HOST_HIPIFY_DIR}/src/dev_runtime_segments.cc"
    RCCL_DEV_RUNTIME_CC_PATH="${RCCL_HOST_HIPIFY_DIR}/src/rccl_dev_runtime.cc"
  CATEGORIES_YAML "${CMAKE_CURRENT_SOURCE_DIR}/../test_categories_micro.yaml")
# wrap-test.cc includes rccl_wrap.cc. Production librccl always compiles the LL128
# arm; without this the broadcast protocol tests exercise the compiled-out fallback.
set_source_files_properties(wrap-test.cc PROPERTIES COMPILE_DEFINITIONS ENABLE_LL128)

# ---------------------------------------------------------------------------
# rccl-UnitTestsMicroCollectives
# ---------------------------------------------------------------------------
rccl_add_micro_binary(rccl-UnitTestsMicroCollectives
  SOURCES ${TEST_MICRO_COLLECTIVES_SOURCE_FILES}
  EXTRA_DEFS
    ${_MICRO_MODE_DEFS}
    COLLECTIVES_CC_PATH="${RCCL_HOST_HIPIFY_DIR}/src/collectives.cc"
  CATEGORIES_YAML "${CMAKE_CURRENT_SOURCE_DIR}/../test_categories_micro_collectives.yaml")
# collectives.cc defines a kernel. An empty embedded device binary stands in for the device pass, so the
# host-only object registers it without referencing a fatbin symbol nothing provides.
file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/empty_gpubinary" "")
target_compile_options(rccl-UnitTestsMicroCollectives PRIVATE
  "SHELL:-Xclang -fcuda-include-gpubinary -Xclang ${CMAKE_CURRENT_BINARY_DIR}/empty_gpubinary")

include("${CMAKE_CURRENT_LIST_DIR}/device/host_tests.cmake")

# ---------------------------------------------------------------------------
# rccl-UnitTestsMicroCpuset
# ---------------------------------------------------------------------------
rccl_add_micro_binary(rccl-UnitTestsMicroCpuset
  SOURCES ${TEST_MICRO_CPUSET_SOURCE_FILES}
  EXTRA_DEFS
    ${_MICRO_MODE_DEFS}
  CATEGORIES_YAML "${CMAKE_CURRENT_SOURCE_DIR}/../test_categories_micro_cpuset.yaml")

# sym_kernels_host.cc only exists when sym kernels are generated (GENERATE_SYM_KERNELS
# in-build / present on disk standalone).
if(_MICRO_BUILD_SYM_KERNELS)
  target_sources(rccl-UnitTestsMicro PRIVATE sym-kernels-index-test.cc)
  target_compile_definitions(rccl-UnitTestsMicro PRIVATE
    SYM_KERNELS_HOST_CC_PATH="${RCCL_HOST_HIPIFY_DIR}/gensrc/symmetric/sym_kernels_host.cc")
elseif(NOT BUILD_TESTS)
  message(WARNING "Skipping sym-kernels-index-test.cc for rccl-UnitTestsMicro: missing ${RCCL_HOST_HIPIFY_DIR}/gensrc/symmetric/sym_kernels_host.cc")
endif()

include("${CMAKE_CURRENT_LIST_DIR}/algorithms/dda/host_tests.cmake")

include("${CMAKE_CURRENT_LIST_DIR}/rma/host_tests.cmake")

# ---------------------------------------------------------------------------
# rccl-UnitTestsMicroDebug
# ---------------------------------------------------------------------------
rccl_add_micro_binary(rccl-UnitTestsMicroDebug
  SOURCES ${TEST_MICRO_DEBUG_SOURCE_FILES}
  EXTRA_DEFS
    ${_MICRO_MODE_DEFS}
    DEBUG_CC_PATH="${RCCL_HOST_HIPIFY_DIR}/src/debug.cc"
  CATEGORIES_YAML "${CMAKE_CURRENT_SOURCE_DIR}/../test_categories_micro_debug.yaml")

include("${CMAKE_CURRENT_LIST_DIR}/scheduler/host_tests.cmake")

# In-build, RCCL_CUMEM_DMABUF_EXPORT_SUPPORTED arrives with rccl's own compile definitions; standalone
# repeats the top-level probe, so rocmwrap.h's NCCL_CUMEM_DMABUF_EXPORT_GATE matches what rccl compiles.
set(_MICRO_DMABUF_EXPORT_DEF "")
if(NOT BUILD_TESTS)
  include(CheckCXXSymbolExists)
  set(CMAKE_TRY_COMPILE_TARGET_TYPE STATIC_LIBRARY)
  set(CMAKE_REQUIRED_DEFINITIONS -D__HIP_PLATFORM_AMD__)
  check_cxx_symbol_exists(hipMemGetHandleForAddressRange "hip/hip_runtime_api.h"
    _MICRO_HIP_MEM_GET_HANDLE_FOR_ADDR_RANGE)
  unset(CMAKE_REQUIRED_DEFINITIONS)
  set(CMAKE_TRY_COMPILE_TARGET_TYPE EXECUTABLE)
  if(_MICRO_HIP_MEM_GET_HANDLE_FOR_ADDR_RANGE)
    set(_MICRO_DMABUF_EXPORT_DEF RCCL_CUMEM_DMABUF_EXPORT_SUPPORTED)
  endif()
endif()

include("${CMAKE_CURRENT_LIST_DIR}/gin/host_tests.cmake")

# ---------------------------------------------------------------------------
# rccl-UnitTestsMicroInit and variants
# ---------------------------------------------------------------------------
foreach(_variant "" "-uncached" "-faultinj")
  set(_init_tgt "rccl-UnitTestsMicroInit${_variant}")
  rccl_add_micro_binary(${_init_tgt}
    SOURCES ${TEST_MICRO_INIT_SOURCE_FILES}
    REMOVE_DEFS ENABLE_FAULT_INJECTION HIP_HOST_UNCACHED_MEMORY HIP_UNCACHED_MEMORY
    EXTRA_DEFS ${_MICRO_MODE_DEFS} ${_MICRO_AMDSMI_DEF} INIT_CC_PATH="${RCCL_HOST_HIPIFY_DIR}/src/init.cc"
    CATEGORIES_YAML "${CMAKE_CURRENT_SOURCE_DIR}/../test_categories_micro_init.yaml")
  if(_variant STREQUAL "-uncached")
    target_compile_definitions(${_init_tgt} PRIVATE HIP_HOST_UNCACHED_MEMORY HIP_UNCACHED_MEMORY)
  elseif(_variant STREQUAL "-faultinj")
    target_compile_definitions(${_init_tgt} PRIVATE ENABLE_FAULT_INJECTION)
  endif()
endforeach()

# ---------------------------------------------------------------------------
# rccl-UnitTestsMicroMemManager
# ---------------------------------------------------------------------------
rccl_add_micro_binary(rccl-UnitTestsMicroMemManager
  SOURCES ${TEST_MICRO_MEM_MANAGER_SOURCE_FILES}
  EXTRA_DEFS
    ${_MICRO_MODE_DEFS}
    MEM_MANAGER_CC_PATH="${RCCL_HOST_HIPIFY_DIR}/src/mem_manager.cc"
  CATEGORIES_YAML "${CMAKE_CURRENT_SOURCE_DIR}/../test_categories_micro_mem_manager.yaml")

# ---------------------------------------------------------------------------
# rccl-UnitTestsMicroAllocator
# ---------------------------------------------------------------------------
rccl_add_micro_binary(rccl-UnitTestsMicroAllocator
  SOURCES ${TEST_MICRO_ALLOCATOR_SOURCE_FILES}
  EXTRA_DEFS
    ${_MICRO_MODE_DEFS}
    ALLOCATOR_CC_PATH="${RCCL_HOST_HIPIFY_DIR}/src/allocator.cc"
  CATEGORIES_YAML "${CMAKE_CURRENT_SOURCE_DIR}/../test_categories_micro_allocator.yaml")

# ---------------------------------------------------------------------------
# rccl-UnitTestsMicroMnnvl
# ---------------------------------------------------------------------------
rccl_add_micro_binary(rccl-UnitTestsMicroMnnvl
  SOURCES ${TEST_MICRO_MNNVL_SOURCE_FILES}
  EXTRA_DEFS
    ${_MICRO_MODE_DEFS}
    MNNVL_CC_PATH="${RCCL_HOST_HIPIFY_DIR}/src/mnnvl.cc"
  CATEGORIES_YAML "${CMAKE_CURRENT_SOURCE_DIR}/../test_categories_micro_mnnvl.yaml")

# ---------------------------------------------------------------------------
# rccl-UnitTestsMicroAlloc
# ---------------------------------------------------------------------------
rccl_add_micro_binary(rccl-UnitTestsMicroAlloc
  SOURCES ${TEST_MICRO_ALLOC_SOURCE_FILES}
  EXTRA_DEFS
    ${_MICRO_MODE_DEFS}
  CATEGORIES_YAML "${CMAKE_CURRENT_SOURCE_DIR}/../test_categories_micro_alloc.yaml")

include("${CMAKE_CURRENT_LIST_DIR}/enqueue/host_tests.cmake")

include("${CMAKE_CURRENT_LIST_DIR}/nccl_device/host_tests.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/devcomm/host_tests.cmake")

# ---------------------------------------------------------------------------
# rccl-UnitTestsMicroCftDevRuntime
# ---------------------------------------------------------------------------
rccl_add_micro_binary(rccl-UnitTestsMicroCftDevRuntime
  SOURCES ${TEST_MICRO_CFT_DEV_RUNTIME_SOURCE_FILES}
  EXTRA_DEFS
    ${_MICRO_MODE_DEFS}
    CORE_CC_PATH="${RCCL_HOST_HIPIFY_DIR}/src/nccl_device/core.cc"
    CFT_DEV_RUNTIME_CC_PATH="${RCCL_HOST_HIPIFY_DIR}/src/cft_dev_runtime.cc"
  CATEGORIES_YAML "${CMAKE_CURRENT_SOURCE_DIR}/../test_categories_micro_cft_dev_runtime.yaml")

# ---------------------------------------------------------------------------
# rccl-UnitTestsMicroCe
# ---------------------------------------------------------------------------
# In-build, CE_BATCH_ASYNC_SUPPORTED arrives with rccl's own compile definitions when CE_ENABLED.
set(_MICRO_CE_BATCH_DEF "")
if(NOT BUILD_TESTS)
  include(CheckCXXSymbolExists)
  set(CMAKE_TRY_COMPILE_TARGET_TYPE STATIC_LIBRARY)
  set(CMAKE_REQUIRED_DEFINITIONS -D__HIP_PLATFORM_AMD__)
  check_cxx_symbol_exists(hipMemcpyBatchAsync "hip/hip_runtime_api.h" _MICRO_HIP_MEMCPY_BATCH_ASYNC)
  unset(CMAKE_REQUIRED_DEFINITIONS)
  set(CMAKE_TRY_COMPILE_TARGET_TYPE EXECUTABLE)
  if(_MICRO_HIP_MEMCPY_BATCH_ASYNC)
    set(_MICRO_CE_BATCH_DEF CE_BATCH_ASYNC_SUPPORTED)
  endif()
endif()
rccl_add_micro_binary(rccl-UnitTestsMicroCe
  SOURCES ${TEST_MICRO_CE_SOURCE_FILES}
  EXTRA_DEFS
    ${_MICRO_MODE_DEFS}
    ${_MICRO_AMDSMI_DEF}
    ${_MICRO_CE_BATCH_DEF}
    RCCL_STUBS_OMIT_ncclCeFinalize       # ce_coll.cc defines it
    CE_COLL_CC_PATH="${RCCL_HOST_HIPIFY_DIR}/src/ce_coll.cc"
  CATEGORIES_YAML "${CMAKE_CURRENT_SOURCE_DIR}/../test_categories_micro.yaml")

include("${CMAKE_CURRENT_LIST_DIR}/param/host_tests.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/register/host_tests.cmake")

include("${CMAKE_CURRENT_LIST_DIR}/ras/host_tests.cmake")

include("${CMAKE_CURRENT_LIST_DIR}/plugin/host_tests.cmake")

# ---------------------------------------------------------------------------
# rccl-UnitTestsMicroSymKernels
# ---------------------------------------------------------------------------
if(_MICRO_BUILD_SYM_KERNELS)
  rccl_add_micro_binary(rccl-UnitTestsMicroSymKernels
    SOURCES ${TEST_MICRO_SYM_KERNELS_SOURCE_FILES}
    EXTRA_INCLUDES
      ${RCCL_HOST_HIPIFY_DIR}/src/device               # sym_kernels.cc includes "tdm/tdmCopy.h"
    EXTRA_DEFS
      ${_MICRO_MODE_DEFS}
      ${_MICRO_AMDSMI_DEF}
      GENERATE_SYM_KERNELS
      SYM_KERNELS_HOST_CC_PATH="${RCCL_HOST_HIPIFY_DIR}/gensrc/symmetric/sym_kernels_host.cc"
      SYM_KERNELS_CC_PATH="${RCCL_HOST_HIPIFY_DIR}/src/sym_kernels.cc"
    CATEGORIES_YAML "${CMAKE_CURRENT_SOURCE_DIR}/../test_categories_micro.yaml")
endif()

include("${CMAKE_CURRENT_LIST_DIR}/tuning/host_tests.cmake")

# ---------------------------------------------------------------------------
# rccl-UnitTestsMicroProxy
# ---------------------------------------------------------------------------
rccl_add_micro_binary(rccl-UnitTestsMicroProxy
  SOURCES ${TEST_MICRO_PROXY_SOURCE_FILES}
  EXTRA_DEFS
    ${_MICRO_MODE_DEFS}
    PROXY_CC_PATH="${RCCL_HOST_HIPIFY_DIR}/src/proxy.cc"
    RCCL_OS_FAKES_OMIT_ncclSocketClose   # fakes/proxy_micro_fakes.cc defines it
  CATEGORIES_YAML "${CMAKE_CURRENT_SOURCE_DIR}/../test_categories_micro_proxy.yaml")

include("${CMAKE_CURRENT_LIST_DIR}/graph/host_tests.cmake")

# ---------------------------------------------------------------------------
# rccl-UnitTestsMicroDiagnostics -- src/diagnostics.cc
# ---------------------------------------------------------------------------
# Root binaries share rccl-root-obj objects, so pass no COMMON_DEFS/COMMON_INCLUDES with that prefix. A binary with
# EXTRA_DEFS compiles its own sources and takes its *_CC_PATH there; the rest set it with target_compile_definitions().
rccl_add_unit_micro_binary(rccl-UnitTestsMicroDiagnostics diagnostics-test.cc
  OBJECT_PREFIX rccl-root-obj
  SOURCES
    fakes/dev_runtime_micro_fakes.cc
    fakes/env_fakes.cc
    fakes/hip_fakes.cc
    fakes/mgmt_task_enq_fakes.cc
    fakes/nccl_device_core_fakes.cc
    fakes/nccl_fakes.cc
  CATEGORIES_YAML "${CMAKE_CURRENT_SOURCE_DIR}/../test_categories_micro_diagnostics.yaml")
target_compile_definitions(rccl-UnitTestsMicroDiagnostics PRIVATE
  DIAGNOSTICS_CC_PATH="${RCCL_HOST_HIPIFY_DIR}/src/diagnostics.cc")

include("${CMAKE_CURRENT_LIST_DIR}/diagnostics/host_tests.cmake")

include("${CMAKE_CURRENT_LIST_DIR}/transport/host_tests.cmake")

include("${CMAKE_CURRENT_LIST_DIR}/misc/host_tests.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/os/host_tests.cmake")

# ---------------------------------------------------------------------------
# rccl-UnitTestsMicroHostDispatch
# ---------------------------------------------------------------------------
rccl_add_micro_binary(rccl-UnitTestsMicroHostDispatch
  SOURCES ${TEST_MICRO_HOST_DISPATCH_SOURCE_FILES}
  EXTRA_DEFS ${_MICRO_MODE_DEFS}
  CATEGORIES_YAML "${CMAKE_CURRENT_SOURCE_DIR}/../test_categories_micro.yaml")

# ---------------------------------------------------------------------------
# rccl-UnitTestsHostAll
# ---------------------------------------------------------------------------
# The single gtest-shaped front door for every rccl-UnitTestsMicro* binary above: it discovers
# them at run time, so a test harness integrates with this one target and never needs a copy of
# the binary list. Deliberately NOT named rccl-UnitTestsMicro*, so it cannot front itself.
#
# Defined through the same helper as the binaries it runs purely so the two build modes, the
# install rule and the host-only flags stay in one place; it compiles no production source and
# links no fakes.
rccl_add_micro_binary(rccl-UnitTestsHostAll
  SOURCES host_test_dispatch.cc
  EXTRA_DEFS ${_MICRO_MODE_DEFS})

# =====================================================================
# Standalone-only tail: coverage, the runner's binary list, no CTest registration.
# =====================================================================
if(BUILD_TESTS)
  # In-RCCL-build path registers these binaries via CTest categories
  # (rccl_register_test_categories, called inside rccl_add_micro_binary).
  return()
endif()

rccl_standalone_coverage()

# run_host_tests.sh runs every binary listed here, so a new binary needs no runner edit.
# rccl-UnitTestsHostAll is left out: it reruns every rccl-UnitTestsMicro* binary itself.
set(_host_test_binaries ${HOST_TEST_BINARIES})
list(REMOVE_ITEM _host_test_binaries rccl-UnitTestsHostAll)
string(REPLACE ";" "\n" _host_test_binaries "${_host_test_binaries}")
file(WRITE "${CMAKE_BINARY_DIR}/host_test_binaries.txt" "${_host_test_binaries}\n")
