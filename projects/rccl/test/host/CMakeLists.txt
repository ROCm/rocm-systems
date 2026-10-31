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
  "transport/net_ib/p2p.cc" "NCCL_PARAM(IbArThreshold, \"IB_AR_THRESHOLD\", -2)<SEMI>"
    "transport_net_ib_p2p_fakes.cc" "return g_loadParam(\"IB_AR_THRESHOLD\", -2)<SEMI> }"
  "transport/net_ib/p2p.cc" "int64_t ncclIbArThreshold = 8192<SEMI>"
    "transport_net_ib_p2p_fakes.cc" "int64_t ncclIbArThreshold = 8192<SEMI>"
  "transport/net_ib/p2p.cc" "NCCL_PARAM(IbReceiverSideMatchingScheme, \"IB_RECEIVER_SIDE_MATCHING_SCHEME\", -2)<SEMI>"
    "transport_net_ib_p2p_fakes.cc" "return g_loadParam(\"IB_RECEIVER_SIDE_MATCHING_SCHEME\", -2)<SEMI>"
  "transport/net_ib/p2p_resiliency.cc" "NCCL_PARAM(IbResiliencyPortFailover, \"IB_RESILIENCY_PORT_FAILOVER\", 0)<SEMI>"
    "transport_net_ib_resiliency_fakes.cc" "return g_loadParam(\"IB_RESILIENCY_PORT_FAILOVER\", 0)<SEMI>"
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
# gdr-probe-test.cc pins the GDR probe's registration flags to production's; these
# keep its kRegAccessFlags and both reg.cc copies spelling the same set.
set(_rccl_reg_access_flags
  "IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ | IBV_ACCESS_REMOTE_ATOMIC;")
foreach(_reg_cc transport/net_ib/reg.cc transport/net_ib_cast/reg.cc)
  rccl_assert_source_line("${_rccl_src_root}/${_reg_cc}"
    "${_rccl_reg_access_flags}" "gdr-probe-test.cc: kRegAccessFlags")
  rccl_assert_source_line("${_rccl_src_root}/${_reg_cc}"
    "if (relaxedOrdering) flags |= IBV_ACCESS_RELAXED_ORDERING;" "gdr-probe-test.cc: the relaxed-ordering flags")
endforeach()
rccl_assert_source_line("${CMAKE_CURRENT_LIST_DIR}/gdr-probe-test.cc"
  "${_rccl_reg_access_flags}" "the production registration flags")
unset(_rccl_reg_access_flags)
# graph_fakes.cc hand-writes these params, whose NCCL_PARAMs are in files rccl-UnitTestsMicroTopo doesn't compile.
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
    "client-support-test.cc: ncclFuncStr")
  rccl_assert_source_line("${CMAKE_CURRENT_LIST_DIR}/client-support-test.cc" "${_func_str_line}"
    "the production ncclFuncStr table")
endforeach()
# dda-coll-seams.cc copies these rccl_wrap.cc DDA param defaults; pin both copies.
foreach(_dda_param
    "RCCL_PARAM(DdaLL, \"DDA_LL\", 1)"
    "RCCL_PARAM(DdaLLOneShotThreshold, \"DDA_LL_ONESHOT_THRESHOLD\", (size_t)(1) * 1024 * 1024)"
    "RCCL_PARAM(DdaLL128OneShotThreshold, \"DDA_LL128_ONESHOT_THRESHOLD\", (size_t)(4) * 1024 * 1024)")
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
unset(_rccl_src_root)

# Define the per-target micro-test source lists shared by BOTH build paths.
function(rccl_define_micro_source_lists)
  # rccl-UnitTestsMicro -- the broad-coverage micro binary.
  set(TEST_MICRO_SOURCE_FILES
    p2p-test.cc
    coll-net-test.cc                     # src/transport/coll_net.cc
    rma-proxy-launch-test.cc            # RMA proxy queue + stream-memop coverage
    rma-proxy-progress-test.cc
    gin-plugin-init-test.cc
    rma-plugin-init-test.cc              # src/plugin/rma.cc ncclRmaPluginInit leak fix (AICOMRCCL-3322)
    rma-test.cc
    rma-ce-test.cc
    ras-test.cc
    ras-diagnostics-common-test.cc       # src/ras/diagnostics_checks_common.cc
    ras-collectives-test.cc              # src/ras/collectives.cc
    ras-diagnostics-test.cc              # src/ras/diagnostics.cc
    rasnet-test.cc                       # src/ras/rasnet.cc
    ras-diagnostics-gpu-test.cc              # src/ras/diagnostics_gpu.cc
    ras-peers-test.cc                    # src/ras/peers.cc
    ras-diagnostics-env-test.cc          # src/ras/diagnostics_env.cc
    ras-param-test.cc                    # src/ras/ras_param.cc
    net-telemetry-test.cc                # src/transport/net_telemetry.cc, src/include/net_telemetry.h
    net-socket-test.cc                   # src/transport/net_socket.cc, src/transport/net_socket/common.cc
    tuning-test.cc                       # src/tuning/tuning.cc
    tuning-general-test.cc
    csv-tuner-test.cc                    # src/plugin/tuner/csv_tuner.cc
    param-utils-test.cc                  # src/include/param/utils.h, src/include/param/parser_common.h
    param-parsers-test.cc                # src/include/param/parser_{default,enum,bitset,list}.h
    param-registry-test.cc               # src/param/param_registry.cc
    ras-client-test.cc
    group-test.cc
    comm-dump-test.cc                    # src/commDump.cc
    timer-test.cc                        # src/include/timer.h
    devcomm-test.cc
    wrap-test.cc
    dev-runtime-test.cc
    nccl-utility-test.cc                 # src/include/nccl_device/utility.h (header-only, no fakes)
    # sym-kernels-index-test.cc: added conditionally below, only when GENERATE_SYM_KERNELS is ON.
    gdr-probe-test.cc                    # src/misc/gdr_probe.cc runtime GDR registration probe
    dv_symbols_test.cc                   # src/misc/ionicdvsymbols.cc, src/misc/mlx5dvsymbols.cc
    register-test.cc                     # src/register/register.cc
    coll-reg-test.cc                     # src/register/coll_reg.cc
    sendrecv-reg-test.cc                 # src/register/sendrecv_reg.cc
    process-isolated-runner-test.cc
    rccl-graph-gen-test.cc               # src/graph/rccl_graph_gen.cc ring construction + cut balancing
    # real IsArchMatch, for wrap-test.cc (path is build-directory-relative).
    ${RCCL_HOST_HIPIFY_DIR}/src/misc/archinfo.cc
    fakes/channel_fakes.cc
    fakes/argcheck_fakes.cc
    fakes/dev_runtime_micro_fakes.cc
    fakes/nccl_device_core_fakes.cc      # src/nccl_device/core.cc ncclTeam* accessors
    fakes/ibvwrap_fakes.cc               # src/misc/ibvwrap.cc PD / MR registration seams
    fakes/nccl_fakes.cc
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

  # rccl-UnitTestsMicroWarpSpeed -- scheduler UUTs under ENABLE_WARP_SPEED.
  set(TEST_MICRO_SCHEDULER_SOURCE_FILES
    scheduler-test.cc
    fakes/enqueue_fakes.cc
    fakes/bootstrap_stubs.cc
    fakes/topo_stubs.cc
    fakes/transport_stubs.cc
    fakes/nccl_stubs.cc
    fakes/channel_fakes.cc
    fakes/nccl_fakes.cc
    fakes/hip_fakes.cc
    # Shared per-production-TU fakes
    fakes/ce_fakes.cc
    fakes/collectives_fakes.cc
    fakes/comm_fakes.cc
    fakes/dev_runtime_fakes.cc
    fakes/env_fakes.cc
    fakes/mgmt_task_enq_fakes.cc
    fakes/os_fakes.cc
    fakes/proxy_fakes.cc
    fakes/profiler_fakes.cc
    fakes/rccl_wrap_fakes.cc
    fakes/recorder_fakes.cc
    fakes/register_stubs.cc
    fakes/roctx_fakes.cc
    fakes/strongstream_stubs.cc
    fakes/sym_kernels_fakes.cc
    fakes/sym_kernels_index_fakes.cc
    fakes/tuning_fakes.cc
    ${RCCL_HOST_HIPIFY_DIR}/src/misc/argcheck.cc
    ${RCCL_HOST_HIPIFY_DIR}/src/misc/archinfo.cc
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

  # rccl-UnitTestsMicroOnerank -- src/device/onerank.cu. Own binary: fakes/sched_stubs.cc fakes
  # ncclLaunchOneRank.
  set(TEST_MICRO_ONERANK_SOURCE_FILES
    onerank-test.cc
    fakes/hip_fakes.cc
    fakes/nccl_fakes.cc
    ../common/main_altrsmi.cpp
    PARENT_SCOPE)

  # rccl-UnitTestsMicroMgmtTaskEnq -- src/enqueue/mgmt_task_enq.cc. Own binary: it defines
  # ncclMgmtTaskEnqueue, which fakes/mgmt_task_enq_fakes.cc fakes for every other target.
  set(TEST_MICRO_MGMT_TASK_ENQ_SOURCE_FILES
    mgmt-task-enq-test.cc
    fakes/group_fakes.cc
    fakes/nccl_fakes.cc
    fakes/nccl_stubs.cc
    fakes/utils_fakes.cc
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
    ../common/ProcessIsolatedTestRunner.cpp
    PARENT_SCOPE)

  # rccl-UnitTestsMicroCrypt -- src/misc/crypt.cc. Own binary: keeps crypt.cc's file-scope state isolated.
  set(TEST_MICRO_CRYPT_SOURCE_FILES
    crypt-test.cc
    ../common/main_altrsmi.cpp
    PARENT_SCOPE)

  # rccl-UnitTestsMicroStrongstream -- src/misc/strongstream.cc. Own binary: the test defines the HIP
  # calls it records, and strongstream_stubs.cc / nccl_fakes.cc already define this unit's symbols.
  set(TEST_MICRO_STRONGSTREAM_SOURCE_FILES
    strongstream-test.cc
    ../common/main_altrsmi.cpp
    PARENT_SCOPE)

  # rccl-UnitTestsMicroUtils -- src/misc/utils.cc. Own binary: the test seams HIP, env and libc
  # around utils.cc, which other micro binaries link whole as an oracle TU.
  set(TEST_MICRO_UTILS_SOURCE_FILES
    utils-test.cc
    ../common/main_altrsmi.cpp
    PARENT_SCOPE)

  # rccl-UnitTestsMicroHostDispatch -- host_test_dispatch.cc, the rccl-UnitTestsHostAll front end.
  # Own binary like every other unit under test; it also makes the dispatcher run its own tests.
  set(TEST_MICRO_HOST_DISPATCH_SOURCE_FILES
    host-test-dispatch-test.cc
    ../common/main_altrsmi.cpp
    PARENT_SCOPE)

  # rccl-UnitTestsMicroArgcheck -- src/misc/argcheck.cc. Own binary: the test seams the HIP, window
  # and bootstrap calls around argcheck.cc, which other micro binaries link whole as an oracle TU.
  set(TEST_MICRO_ARGCHECK_SOURCE_FILES
    argcheck-test.cc
    ../common/main_altrsmi.cpp
    PARENT_SCOPE)

  # rccl-UnitTestsMicroCpuset -- src/include/cpuset.h. Own binary: the test defines os/linux.cc's
  # ncclOsCpu* one-liners, which would otherwise sit in the shared binary as test-TU definitions.
  set(TEST_MICRO_CPUSET_SOURCE_FILES
    cpuset-test.cc
    ../common/main_altrsmi.cpp
    PARENT_SCOPE)

  # rccl-UnitTestsMicroConnect -- REAL src/graph/connect.cc (#included via CONNECT_CC_PATH) with the real
  # ring/tree helpers it calls; connect-test.cc fakes the few remaining symbols itself.
  set(TEST_MICRO_CONNECT_SOURCE_FILES
    connect-test.cc
    fakes/hip_fakes.cc
    ${RCCL_HOST_HIPIFY_DIR}/src/graph/rings.cc
    ${RCCL_HOST_HIPIFY_DIR}/src/graph/trees.cc
    ${RCCL_HOST_HIPIFY_DIR}/src/graph/rccl_graph_gen.cc
    ${RCCL_HOST_HIPIFY_DIR}/src/misc/archinfo.cc
    ../common/main_altrsmi.cpp
    PARENT_SCOPE)

  # src/algorithms/dda/{all_gather,all_reduce,alltoall,reduce_scatter}/dda_*.cu launchers. Own binary:
  # rccl-UnitTestsMicro links fakes/dda_fakes.cc, which defines the same DDA entry points.
  set(TEST_MICRO_DDA_COLL_SOURCE_FILES
    algorithms/dda/all_gather/dda-all-gather-fabric-test.cc
    algorithms/dda/all_gather/dda-all-gather-fabric-ll-test.cc
    algorithms/dda/all_gather/dda-all-gather-fabric-ll128-test.cc
    algorithms/dda/all_gather/dda-all-gather-ipc-test.cc
    algorithms/dda/all_reduce/dda-all-reduce-fabric-test.cc
    algorithms/dda/all_reduce/dda-all-reduce-fabric-ll-test.cc
    algorithms/dda/all_reduce/dda-all-reduce-fabric-ll128-test.cc
    algorithms/dda/alltoall/dda-alltoall-fabric-test.cc
    algorithms/dda/alltoall/dda-alltoall-fabric-ll-test.cc
    algorithms/dda/alltoall/dda-alltoall-fabric-ll128-test.cc
    algorithms/dda/alltoall/dda-alltoall-ipc-test.cc
    algorithms/dda/reduce_scatter/dda-reduce-scatter-fabric-test.cc
    algorithms/dda/reduce_scatter/dda-reduce-scatter-fabric-ll-test.cc
    algorithms/dda/reduce_scatter/dda-reduce-scatter-fabric-ll128-test.cc
    algorithms/dda/reduce_scatter/dda-reduce-scatter-ipc-test.cc
    algorithms/dda/dda-coll-seams.cc
    fakes/dda_ipc_nranks_fakes.cc
    fakes/hip_fakes.cc
    fakes/nccl_fakes.cc
    ../common/main_altrsmi.cpp
    PARENT_SCOPE)

  # rccl-UnitTestsMicroEnqueue(+-devlinker) -- src/enqueue.cc.
  set(TEST_MICRO_ENQUEUE_SOURCE_FILES
    enqueue-test.cc
    fakes/enqueue_test_deps.cc
    fakes/bootstrap_stubs.cc
    fakes/topo_stubs.cc
    fakes/transport_stubs.cc
    fakes/nccl_stubs.cc
    fakes/channel_fakes.cc
    fakes/nccl_fakes.cc
    fakes/hip_fakes.cc
    # Shared per-production-TU fakes (see MICROTEST_README.md for the map).
    fakes/allocator_fakes.cc
    fakes/ce_fakes.cc
    fakes/collconfig_fakes.cc
    fakes/collectives_fakes.cc
    fakes/comm_fakes.cc
    fakes/dev_runtime_fakes.cc
    fakes/env_fakes.cc
    fakes/group_fakes.cc
    fakes/mgmt_task_enq_fakes.cc
    fakes/os_fakes.cc
    fakes/proxy_fakes.cc
    fakes/profiler_fakes.cc
    fakes/rccl_wrap_fakes.cc
    fakes/recorder_fakes.cc
    fakes/register_stubs.cc
    fakes/rma_fakes.cc
    fakes/roctx_fakes.cc
    fakes/sched_stubs.cc
    fakes/strongstream_stubs.cc
    fakes/sym_kernels_fakes.cc
    fakes/tuning_fakes.cc
    fakes/rccl_arch_thresholds_fakes.cc
    ${RCCL_HOST_HIPIFY_DIR}/src/misc/argcheck.cc
    ${RCCL_HOST_HIPIFY_DIR}/src/misc/archinfo.cc
    ${RCCL_HOST_HIPIFY_DIR}/src/misc/utils.cc
    ../common/main_altrsmi.cpp
    PARENT_SCOPE)

  # rccl-UnitTestsMicroDebug -- src/debug.cc. Own binary: it is the real ncclDebugLog, which
  # every shared fake set also defines.
  set(TEST_MICRO_DEBUG_SOURCE_FILES
    debug-test.cc
    ../common/main_altrsmi.cpp
    PARENT_SCOPE)

  # rccl-UnitTestsMicroApiTrace -- src/misc/api_trace.cc. Own binary: it defines the public
  # NCCL_API entry points that fakes/api_trace_fakes.cc supplies elsewhere.
  set(TEST_MICRO_API_TRACE_SOURCE_FILES
    api-trace-test.cc
    fakes/debug_fakes.cc
    fakes/recorder_fakes.cc
    fakes/roctx_fakes.cc
    ../common/main_altrsmi.cpp
    PARENT_SCOPE)

  # rccl-UnitTestsMicroRecorder -- src/misc/recorder.cc. Own binary: it is the real
  # rccl::Recorder, which fakes/recorder_fakes.cc supplies elsewhere.
  set(TEST_MICRO_RECORDER_SOURCE_FILES
    recorder-test.cc
    fakes/debug_fakes.cc
    ../common/main_altrsmi.cpp
    PARENT_SCOPE)

  # rccl-UnitTestsMicroCollTrace -- src/misc/latency_profiler/*.cc. Own binary: its ncclDebugLog,
  # HIP event and bootstrap seams would collide with the shared fakes.
  set(TEST_MICRO_COLL_TRACE_SOURCE_FILES
    coll-trace-test.cc
    fakes/debug_fakes.cc
    fakes/collectives_fakes.cc
    fakes/env_fakes.cc
    fakes/strongstream_stubs.cc
    ../common/main_altrsmi.cpp
    PARENT_SCOPE)

  # rccl-UnitTestsMicroCe -- src/ce_coll.cc. Its own binary: every other target links ce_fakes.cc.
  set(TEST_MICRO_CE_SOURCE_FILES
    ce-coll-test.cc
    fakes/allocator_fakes.cc             # src/allocator.cc
    fakes/bootstrap_stubs.cc
    fakes/topo_stubs.cc
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

  # rccl-UnitTestsMicroGinHostProxy -- src/gin/gin_host_proxy.cc + src/gin/proxy_gpucontext/. Own
  # binary: gin-plugin-init-test.cc in rccl-UnitTestsMicro defines ncclGinProxy and ncclGinProxyVersion.
  set(TEST_MICRO_GIN_HOST_PROXY_SOURCE_FILES
    gin-host-proxy-test.cc
    fakes/hip_fakes.cc
    fakes/nccl_fakes.cc
    ../common/main_altrsmi.cpp
    PARENT_SCOPE)

  # rccl-UnitTestsMicroClientSupport -- src/ras/client_support.cc. Its own binary: this test's
  # inline stand-ins for ncclSocketsSameNode/rasPeerInfoToString/rasPeerToString (client_support.cc's
  # callees into src/ras/peers.cc) collide with the REAL peers.cc compiled into rccl-UnitTestsMicro
  # for ras-peers-test.cc, which in turn needs those 3 real and client_support.cc's own
  # ncclSocketToHost/rasGpuDevsToString faked -- the opposite split. The two can't share a binary.
  set(TEST_MICRO_CLIENT_SUPPORT_SOURCE_FILES
    client-support-test.cc               # src/ras/client_support.cc
    fakes/hip_fakes.cc
    fakes/env_fakes.cc
    fakes/libc_fakes.cc
    fakes/ras_fakes.cc
    fakes/ras_param_fakes.cc
    fakes/nccl_fakes.cc                  # ncclDebugLog/ncclDebugMask (checks.h logging macros)
    fakes/collectives_fakes.cc           # ncclFuncToString
    ../common/main_altrsmi.cpp
    PARENT_SCOPE)

  # rccl-UnitTestsMicroRmaProxy -- REAL src/rma/rma_proxy.cc.
  set(TEST_MICRO_RMA_PROXY_SOURCE_FILES
    rma-proxy-test.cc
    # Shared per-production-TU fakes (see MICROTEST_README.md for the map).
    fakes/hip_fakes.cc
    fakes/nccl_fakes.cc
    fakes/os_fakes.cc
    fakes/rma_fakes.cc
    fakes/rocmwrap_fakes.cc
    ${RCCL_HOST_HIPIFY_DIR}/src/misc/utils.cc
    ../common/main_altrsmi.cpp
    PARENT_SCOPE)

  # rccl-UnitTestsMicroSymKernels -- REAL src/sym_kernels.cc + generated host TU,
  # plus the REAL symmetric cost model (src/tuning/sym_model.cc, src/tuning/sym_model/*.cc).
  set(TEST_MICRO_SYM_KERNELS_SOURCE_FILES
    sym-kernels-test.cc
    sym-model-test.cc
    fakes/bootstrap_stubs.cc
    fakes/topo_stubs.cc
    fakes/transport_stubs.cc
    fakes/nccl_stubs.cc
    fakes/channel_fakes.cc
    fakes/nccl_fakes.cc
    fakes/hip_fakes.cc
    fakes/env_fakes.cc
    fakes/devcomm_fakes.cc
    fakes/nccl_device_core_fakes.cc      # ncclTeamRail, read by the GIN cost model
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

  # rccl-UnitTestsMicroNvls -- REAL src/transport/nvls.cc, which RCCL always compiles
  # as its NVLS stubs (the multicast code needs CUDART_VERSION >= 12010). Its own
  # binary: fakes/transport_stubs.cc and fakes/collective_stubs.cc fake the ncclNvls*
  # entry points nvls.cc defines for real.
  set(TEST_MICRO_NVLS_SOURCE_FILES
    nvls-test.cc
    fakes/nccl_fakes.cc
    ../common/main_altrsmi.cpp
    PARENT_SCOPE)

  # rccl-UnitTestsMicroNcclDevice -- src/nccl_device/ requirement builders. Own binary:
  # fakes/dev_runtime_micro_fakes.cc fakes three of them in rccl-UnitTestsMicro.
  set(TEST_MICRO_NCCL_DEVICE_SOURCE_FILES
    nccl-device-requirements-test.cc
    ../common/main_altrsmi.cpp
    PARENT_SCOPE)

  # rccl-UnitTestsMicroCore -- src/nccl_device/core.cc. Own binary:
  # dev_runtime_micro_fakes.cc and devcomm_fakes.cc fake five of its seven functions.
  set(TEST_MICRO_CORE_SOURCE_FILES
    core-test.cc
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

  # rccl-UnitTestsMicroTaskPrep -- host-only microtests for src/enqueue/task_prep/
  # and src/enqueue/task_sched/.
  # Its own binary, not rccl-UnitTestsMicro: that target compiles group.cc, rccl_wrap.cc and
  # dev_runtime.cc as real units under test, and this suite fakes all three to drive them.
  # The four task_prep/ files are separate TUs so task_prep.cc's three sibling externs resolve to
  # real code. Each leaves this source list for its own test TU, which #includes its *_CC_PATH, as
  # that unit comes under test; listing both would be a duplicate symbol. The seven task_sched/
  # files instead share one test TU (task-sched-test.cc): none references another, and each defines
  # exactly one distinct extern-linkage function, so there is no cross-file resolution to preserve.
  set(TEST_MICRO_TASKPREP_SOURCE_FILES
    task-prep-test.cc
    task-classify-test.cc
    task-pretuning-test.cc
    task-posttuning-test.cc
    task-sched-test.cc
    fakes/bootstrap_stubs.cc
    fakes/topo_stubs.cc
    fakes/transport_stubs.cc
    fakes/nccl_stubs.cc
    fakes/channel_fakes.cc
    fakes/nccl_fakes.cc
    fakes/hip_fakes.cc
    # Shared per-production-TU fakes (see MICROTEST_README.md for the map).
    fakes/allocator_fakes.cc             # src/allocator.cc
    fakes/ce_fakes.cc                    # src/ce_coll.cc
    fakes/collectives_fakes.cc           # src/collectives.cc name tables/helpers
    fakes/comm_fakes.cc                  # src/init.cc comm lifecycle + CTA/channel params
    fakes/dev_runtime_fakes.cc           # src/dev_runtime.cc
    fakes/enqueue_fakes.cc               # src/enqueue/enqueue.cc
    fakes/env_fakes.cc                   # src/misc/param.cc + getenv interposition
    fakes/group_fakes.cc                 # src/group.cc
    fakes/mgmt_task_enq_fakes.cc         # src/enqueue/mgmt_task_enq.cc
    fakes/os_fakes.cc                    # src/os/linux.cc aligned alloc
    fakes/profiler_fakes.cc             # src/plugin/profiler.cc proxy-trace dump
    fakes/proxy_fakes.cc                 # src/proxy.cc
    fakes/rccl_wrap_fakes.cc             # src/rccl_wrap.cc
    fakes/recorder_fakes.cc              # src/recorder.cc
    fakes/register_stubs.cc              # src/register/*.cc
    fakes/rma_fakes.cc                   # src/rma/*.cc
    fakes/roctx_fakes.cc                # src/misc/roctx.cc
    fakes/sched_stubs.cc                 # src/scheduler/*.cc + deep launch paths
    fakes/strongstream_stubs.cc          # src/misc/strongstream.cc
    fakes/sym_kernels_fakes.cc           # src/sym_kernels.cc
    fakes/tuning_fakes.cc                # src/graph/tuning.cc
    ${RCCL_HOST_HIPIFY_DIR}/src/config/algorithm_parser.cc   # real alg-selector parsing
    ${RCCL_HOST_HIPIFY_DIR}/src/config/algorithm_registry.cc  # real per-func alg validity mask
    ${RCCL_HOST_HIPIFY_DIR}/src/config/collconfig.cc  # real CTA-policy/alg-mask resolution
    ${RCCL_HOST_HIPIFY_DIR}/src/misc/argcheck.cc  # real PtrCheck/CommCheck
    ${RCCL_HOST_HIPIFY_DIR}/src/misc/archinfo.cc  # real IsArchMatch
    ${RCCL_HOST_HIPIFY_DIR}/src/misc/utils.cc     # real busId/hash helpers
    ../common/main_altrsmi.cpp
    PARENT_SCOPE)

  # rccl-UnitTestsMicroParam -- src/param/{param,c_api}.cc, src/misc/param.cc. Own binary: misc/param.cc
  # defines ncclGetEnv, initEnv and ncclLoadParam, which the shared fakes in rccl-UnitTestsMicro also define.
  set(TEST_MICRO_PARAM_SOURCE_FILES
    param-test.cc
    ../common/ProcessIsolatedTestRunner.cpp
    ../common/main_altrsmi.cpp
    PARENT_SCOPE)

  # rccl-UnitTestsMicroGraph: the real src/graph topology TUs, which every other micro binary fakes.
  set(TEST_MICRO_GRAPH_SOURCE_FILES
    graph-dev-test.cc
    fakes/amdsmi_fakes.cc
    fakes/env_fakes.cc
    fakes/hip_fakes.cc
    fakes/nccl_fakes.cc
    fakes/os_fakes.cc
    fakes/rccl_wrap_fakes.cc
    fakes/transport_stubs.cc
    fakes/tuning_fakes.cc
    ${RCCL_HOST_HIPIFY_DIR}/src/graph/paths.cc
    ${RCCL_HOST_HIPIFY_DIR}/src/graph/rome_models.cc
    ${RCCL_HOST_HIPIFY_DIR}/src/graph/search.cc
    ${RCCL_HOST_HIPIFY_DIR}/src/graph/topo.cc
    ${RCCL_HOST_HIPIFY_DIR}/src/graph/xml.cc
    ${RCCL_HOST_HIPIFY_DIR}/src/misc/archinfo.cc
    ${RCCL_HOST_HIPIFY_DIR}/src/misc/nvmlwrap_stub.cc
    ${RCCL_HOST_HIPIFY_DIR}/src/misc/utils.cc
    ../common/main_altrsmi.cpp
    PARENT_SCOPE)

  # rccl-UnitTestsMicroPlugin -- src/plugin/net.cc, net/net_v*.cc, plugin_open.cc. Own binary: gin-plugin-init-test.cc
  # defines ncclPluginLibPaths, ncclClosePluginLib and ncclGetNetPluginLib in rccl-UnitTestsMicro.
  set(TEST_MICRO_PLUGIN_SOURCE_FILES
    plugin-net-test.cc
    plugin-open-test.cc
    net-v6-test.cc
    net-v7-test.cc
    net-v8-test.cc
    net-v9-test.cc
    net-v10-test.cc
    net-v11-test.cc
    net-v12-test.cc
    fakes/env_fakes.cc
    fakes/nccl_fakes.cc
    fakes/transport_stubs.cc
    ../common/ProcessIsolatedTestRunner.cpp
    ../common/main_altrsmi.cpp
    PARENT_SCOPE)

  # rccl-UnitTestsMicroGinHost -- src/gin/gin_host.cc (GIN_PROXY_NTHREADS, proxy affinity). Own
  # binary: gin-plugin-init-test.cc already defines ncclParamGinEnable in
  # rccl-UnitTestsMicro, and gin_fakes.cc supplies ncclGinQueryLastError to Init.
  set(TEST_MICRO_GIN_HOST_SOURCE_FILES
    gin-host-test.cc
    fakes/nccl_fakes.cc
    fakes/hip_fakes.cc
    fakes/env_fakes.cc
    fakes/bootstrap_stubs.cc             # src/bootstrap.cc AllGather seam
    fakes/nccl_device_core_fakes.cc      # src/nccl_device/core.cc ncclTeam* accessors
    fakes/os_fakes.cc                    # src/os/linux.cc proxy-thread affinity seams
    fakes/topo_stubs.cc                  # src/graph/topo.cc ncclTopoGetLocalGinDevs seam
    ../common/main_altrsmi.cpp
    PARENT_SCOPE)

  # rccl-UnitTestsMicroProfiler -- src/plugin/profiler.cc and its profiler_v1..v7.cc adapters. Own binary: the
  # tests define the logging, env, plugin-open and dlsym seams the shared fakes also define.
  set(TEST_MICRO_PROFILER_SOURCE_FILES
    profiler-test.cc
    profiler-v1-test.cc
    profiler-v2-test.cc
    profiler-v3-test.cc
    profiler-v4-test.cc
    profiler-v5-test.cc
    profiler-v6-test.cc
    profiler-v7-test.cc
    fakes/utils_fakes.cc
    ../common/main_altrsmi.cpp
    PARENT_SCOPE)
    
  # rccl-UnitTestsMicroDdaIpc -- src/algorithms/dda/all_reduce/dda_all_reduce_ipc.cu. Own binary:
  # rccl-UnitTestsMicro links fakes/dda_fakes.cc, which defines the same ncclAllReduceDdaIpc* symbols.
  set(TEST_MICRO_DDA_IPC_SOURCE_FILES
    dda-allreduce-ipc-test.cc
    fakes/dda_ipc_nranks_fakes.cc
    fakes/hip_fakes.cc
    fakes/nccl_fakes.cc
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

  # rccl-UnitTestsMicroRomeModels
  set(TEST_MICRO_ROME_MODELS_SOURCE_FILES
    rome-models-test.cc
    fakes/nccl_fakes.cc
    fakes/hip_fakes.cc
    fakes/env_fakes.cc
    fakes/topo_stubs.cc
    ${RCCL_HOST_HIPIFY_DIR}/src/misc/archinfo.cc
    ../common/main_altrsmi.cpp
    PARENT_SCOPE)

  # rccl-UnitTestsMicroRoctxNvml -- src/misc/roctx.cc + src/misc/nvmlwrap_stub.cc. Own binary:
  # fakes/comm_fakes.cc stubs the roctx_scoped_range_in this target compiles for real.
  set(TEST_MICRO_ROCTX_NVML_SOURCE_FILES
    roctx-nvml-stub-test.cc
    ../common/main_altrsmi.cpp
    PARENT_SCOPE)

  # rccl-UnitTestsMicroAmdsmiWrap -- src/misc/amdsmi_wrap.cc. Own binary: fakes/amdsmi_fakes.cc
  # defines the amd_smi_* wrappers this target compiles for real.
  set(TEST_MICRO_AMDSMI_WRAP_SOURCE_FILES
    amdsmi-wrap-test.cc
    fakes/hip_fakes.cc
    ${RCCL_HOST_HIPIFY_DIR}/src/misc/utils.cc           # real busIdToInt64
    ../common/main_altrsmi.cpp
    PARENT_SCOPE)

  # rccl-UnitTestsMicroRocmwrap -- src/misc/rocmwrap.cc. Own binary: fakes/rocmwrap_fakes.cc
  # and fakes/nccl_fakes.cc define rocmLibraryInit, ncclCuMemEnable and ncclCuMemHandleType.
  set(TEST_MICRO_ROCMWRAP_SOURCE_FILES
    rocmwrap-test.cc
    fakes/hip_fakes.cc
    ${RCCL_HOST_HIPIFY_DIR}/src/misc/archinfo.cc        # real GetGcnArchName/IsArchMatch
    ${RCCL_HOST_HIPIFY_DIR}/src/misc/kernel_config.cc   # real option matching
    ../common/main_altrsmi.cpp
    PARENT_SCOPE)

  # rccl-UnitTestsMicroNet -- src/transport/net.cc. Own binary: transport_stubs.cc, which most
  # targets link, defines rcclUseAinic and netTransport, and net.cc defines both.
  set(TEST_MICRO_NET_SOURCE_FILES
    net-test.cc
    fakes/collectives_fakes.cc           # func/algo/proto name tables
    fakes/env_fakes.cc
    fakes/hip_fakes.cc
    fakes/nccl_fakes.cc
    fakes/nccl_stubs.cc                  # core floor, incl. proxy profiler hooks
    fakes/channel_fakes.cc
    fakes/net_ib_cast_common_fakes.cc    # netIbCast
    fakes/net_ib_cast_connect_fakes.cc   # rcclCastNetP2pPolicy
    fakes/mgmt_task_enq_fakes.cc
    fakes/os_fakes.cc
    fakes/proxy_fakes.cc
    fakes/rccl_wrap_fakes.cc             # NET registration segment count / commit
    fakes/rocmwrap_fakes.cc
    fakes/topo_stubs.cc
    fakes/transport_net_ib_common_fakes.cc  # src/transport/net_ib/common.cc ncclNetIb
    fakes/transport_net_ib_connect_net_fakes.cc  # rcclNetP2pPolicy, GDR flush-flag accessors
    fakes/transport_net_ib_fakes.cc
    fakes/transport_p2p_fakes.cc         # src/transport/p2p.cc shareable buffers
    ${RCCL_HOST_HIPIFY_DIR}/src/misc/archinfo.cc  # real IsArchMatch
    ${RCCL_HOST_HIPIFY_DIR}/src/misc/utils.cc     # real parseStringList/matchIfList for NCCL_IB_HCA
    ../common/main_altrsmi.cpp
    PARENT_SCOPE)

  # src/diagnostics/p2p.cc; fakes transport/p2p.cc, which rccl-UnitTestsMicro compiles.
  set(TEST_MICRO_DIAGNOSTICS_SOURCE_FILES
    diag-p2p-test.cc
    diag-ib-write-bw-test.cc
    diagnostics-test.cc
    diag-p2p-device-test.cc
    fakes/argcheck_fakes.cc
    fakes/dev_runtime_micro_fakes.cc     # bootstrap collectives
    fakes/env_fakes.cc
    fakes/diagnostics_fakes.cc           # src/diagnostics.cc external-tool runners
    fakes/diagnostics_p2p_device_fakes.cc  # src/diagnostics/device/p2p.cu
    fakes/hip_fakes.cc
    fakes/libc_fakes.cc
    fakes/nccl_device_core_fakes.cc      # src/nccl_device/core.cc ncclTeam* accessors
    fakes/nccl_fakes.cc
    fakes/topo_stubs.cc
    fakes/transport_net_ib_fakes.cc      # src/transport/net_ib/connect.cc params
    fakes/transport_p2p_fakes.cc         # src/transport/p2p.cc shareable buffers
    ../common/main_altrsmi.cpp
    PARENT_SCOPE)

  # rccl-UnitTestsMicroGinHostFinalize -- host finalize / connect generation.
  # Not rccl-UnitTestsMicroGinHost: gin-host-test.cc defines the same ConnectOnce
  # seams (ncclParamGinType, ncclOsCpuCount, bootstrapAllGather). Not
  # rccl-UnitTestsMicro: that target defines the GIN host entry points in
  # fakes/dev_runtime_micro_fakes.cc for dev-runtime-test.cc.
  # os_fakes.cc for ncclOsCpuCount / ncclOsSetAffinity: no test starts a progress
  # thread, but -fprofile-instr-generate / -fcoverage-mapping keep ncclGinProgress
  # live against --gc-sections. The test TU defines the ConnectOnce / DevCommSetup
  # link seams (topology, bootstrap, team and thread-name helpers, ncclParamGinType).
  set(TEST_MICRO_GIN_HOST_FINALIZE_SOURCE_FILES
    gin-host-finalize-test.cc
    fakes/os_fakes.cc
    ../common/main_altrsmi.cpp
    PARENT_SCOPE)

  # rccl-UnitTestsMicroTopo -- src/graph/{xml,topo,paths,search,rome_models,connect,rings,trees}.cc, one
  # test TU each. Own binary: rccl-UnitTestsMicroGraph links tuning_fakes.cc, which defines connect.cc's params,
  # and os_fakes.cc, whose CPU and sysfs seams these tests cannot drive.
  set(TEST_MICRO_TOPO_SOURCE_FILES
    xml-test.cc
    topo-test.cc
    paths-test.cc
    search-test.cc
    graph-rome-models-test.cc
    graph-connect-test.cc
    rings-test.cc
    trees-test.cc
    fakes/graph_fakes.cc
    fakes/amdsmi_fakes.cc
    fakes/hip_fakes.cc
    ${RCCL_HOST_HIPIFY_DIR}/src/misc/archinfo.cc
    ${RCCL_HOST_HIPIFY_DIR}/src/misc/utils.cc
    ../common/main_altrsmi.cpp
    PARENT_SCOPE)

  # src/algorithms/dda/
  set(TEST_MICRO_DDA_SOURCE_FILES
    algorithms/dda/fabric/fabric-gpu-barrier-test.cc
    algorithms/dda/fabric/fabric-init-test.cc
    algorithms/dda/fabric/fabric-mem-handler-test.cc
    algorithms/dda/ipc/ipc-mem-handler-test.cc
    ${RCCL_HOST_HIPIFY_DIR}/src/algorithms/dda/device/device_buffer.cc  # real DeviceBuffer
    ${RCCL_HOST_HIPIFY_DIR}/src/misc/archinfo.cc  # real IsArchMatch
    fakes/bootstrap_stubs.cc             # bootstrapAllGather
    fakes/comm_fakes.cc                  # allocTracker
    fakes/env_fakes.cc                   # NCCL_CUMEM_SKIP_FREE
    fakes/hip_fakes.cc
    fakes/nccl_fakes.cc
    fakes/rccl_wrap_fakes.cc             # DDA params and thresholds
    fakes/utils_fakes.cc                 # getBusId (alloc.h side stream)
    ../common/main_altrsmi.cpp
    PARENT_SCOPE)

  # rccl-UnitTestsMicroTransport -- src/transport/. Tests mirror that tree
  # under test/host/transport; one binary owns the subsystem's shared socket,
  # network and GDRCopy seams.
  set(TEST_MICRO_TRANSPORT_SOURCE_FILES
    transport/net_ib/common-test.cc
    transport/net_ib/gdr-test.cc
    transport/net_ib/init-test.cc
    transport/net_ib/net-ib-gid-inspect-test.cc
    transport/net_ib/reg-test.cc
    transport/rma-socket-test.cc
    transport/rma-socket-mem-test.cc
    transport/rma-socket-progress-recver-test.cc
    transport/rma-socket-setup-test.cc
    fakes/env_fakes.cc
    fakes/gdr_peermem_fakes.cc
    fakes/gdr_probe_fakes.cc
    fakes/gdrwrap_fakes.cc
    fakes/hip_fakes.cc
    fakes/ibvwrap_fakes.cc
    fakes/mlx5dvwrap_fakes.cc
    fakes/nccl_fakes.cc
    fakes/os_fakes.cc
    fakes/rocmwrap_fakes.cc
    fakes/socket_fakes.cc
    fakes/transport_net_ib_fakes.cc
    fakes/transport_net_ib_p2p_fakes.cc
    fakes/transport_net_ib_recovery_fakes.cc
    fakes/transport_net_ib_resiliency_fakes.cc
    fakes/transport_net_socket_common_fakes.cc
    fakes/transport_net_socket_fakes.cc
    ${RCCL_HOST_HIPIFY_DIR}/src/misc/utils.cc
    ../common/main_altrsmi.cpp
    PARENT_SCOPE)

  # rccl-UnitTestsMicroNetIbCastDataPath -- src/transport/net_ib_cast/{p2p,scheduler,net_ib_ops_fault}.cc. Own
  # binary: it compiles the real ibvwrap.cc, which rccl-UnitTestsMicro replaces with fakes/ibvwrap_fakes.cc.
  set(TEST_MICRO_NET_IB_CAST_DATA_PATH_SOURCE_FILES
    net-ib-cast-p2p-test.cc
    net-ib-cast-scheduler-test.cc
    net-ib-ops-fault-test.cc
    fakes/env_fakes.cc
    fakes/hip_fakes.cc
    fakes/libc_fakes.cc
    fakes/nccl_fakes.cc
    fakes/net_ib_cast_common_fakes.cc
    fakes/net_ib_cast_connect_fakes.cc
    fakes/net_ib_cast_init_fakes.cc
    fakes/net_ib_cast_p2p_resiliency_fakes.cc
    fakes/socket_fakes.cc
    fakes/transport_stubs.cc
    ${RCCL_HOST_HIPIFY_DIR}/src/misc/ibvsymbols.cc   # buildIbvSymbols, never called
    ${RCCL_HOST_HIPIFY_DIR}/src/misc/ibvwrap.cc      # real verbs wrappers; tests set the ibvSymbols slots
    ${RCCL_HOST_HIPIFY_DIR}/src/transport/net_ib_cast/qp_sharing.cc  # real commId routing table
    ${RCCL_HOST_HIPIFY_DIR}/src/transport/net_telemetry.cc         # real telemetry counters
    ../common/main_altrsmi.cpp
    PARENT_SCOPE)

  # rccl-UnitTestsMicroP2pResiliency -- src/transport/net_ib_cast/p2p_resiliency*.cc. Own binary: the
  # net_ib_cast fakes stand in for connect.cc, p2p.cc and common.cc, which other targets do not fake.
  set(TEST_MICRO_P2P_RESILIENCY_SOURCE_FILES
    p2p-resiliency-test.cc
    p2p-resiliency-recovery-test.cc
    p2p-resiliency-recovery-ainic-test.cc
    fakes/ibvwrap_fakes.cc
    fakes/libc_fakes.cc                  # g_malloc / g_free for the recovery context
    fakes/nccl_fakes.cc
    fakes/net_ib_cast_capability_fakes.cc
    fakes/net_ib_cast_common_fakes.cc
    fakes/net_ib_cast_connect_fakes.cc
    fakes/net_ib_cast_init_fakes.cc
    fakes/net_ib_cast_p2p_fakes.cc
    fakes/net_ib_cast_qp_sharing_fakes.cc
    fakes/os_fakes.cc                    # ncclOsGetPageSize for ncclIbMalloc
    ../common/main_altrsmi.cpp
    PARENT_SCOPE)

  # rccl-UnitTestsMicroNetIbCastInit -- src/transport/net_ib_cast/{init,common,capability,reg,gdr,multiplane,gin,
  # net_ib_cast_gid_inspect}.cc. Own binary: it fakes gdr_probe.cc, which rccl-UnitTestsMicro compiles.
  set(TEST_MICRO_NET_IB_CAST_INIT_SOURCE_FILES
    net-ib-cast-init-test.cc
    net-ib-cast-capability-test.cc
    net-ib-cast-common-test.cc
    net-ib-cast-gdr-test.cc
    net-ib-cast-gid-inspect-test.cc
    net-ib-cast-gin-test.cc
    net-ib-cast-multiplane-test.cc
    net-ib-cast-reg-test.cc
    fakes/env_fakes.cc
    fakes/gdr_peermem_fakes.cc
    fakes/gdr_probe_fakes.cc
    fakes/hip_fakes.cc
    fakes/ibvwrap_fakes.cc
    fakes/ionicdvwrap_fakes.cc
    fakes/libc_fakes.cc
    fakes/mlx5dvwrap_fakes.cc
    fakes/nccl_fakes.cc
    fakes/net_ib_cast_connect_fakes.cc
    fakes/net_ib_cast_p2p_fakes.cc
    fakes/net_ib_cast_p2p_resiliency_fakes.cc
    fakes/net_ib_cast_p2p_resiliency_recovery_fakes.cc
    fakes/net_ib_cast_qp_sharing_fakes.cc
    fakes/net_ib_cast_scheduler_fakes.cc
    fakes/os_fakes.cc
    fakes/rocmwrap_fakes.cc              # rocmLibraryInit (gdr.cc DMA-BUF probe)
    fakes/socket_fakes.cc
    fakes/transport_stubs.cc             # rcclUseAinic, NET_OPTIONAL_RECV_COMPLETION
    ${RCCL_HOST_HIPIFY_DIR}/src/graph/xml.cc             # real parser for the multiplane map
    ${RCCL_HOST_HIPIFY_DIR}/src/misc/utils.cc            # real NCCL_IB_HCA parsing, pciPathToInt64
    ${RCCL_HOST_HIPIFY_DIR}/src/transport/net_telemetry.cc  # real, off unless RCCL_TELEMETRY_ENABLE=1
    ../common/ProcessIsolatedTestRunner.cpp
    ../common/main_altrsmi.cpp
    PARENT_SCOPE)

  # src/misc/ IB verbs and direct-verbs loaders
  set(TEST_MICRO_MISC_SOURCE_FILES
    misc/gdr-peermem-test.cc
    misc/ibvsymbols-test.cc
    misc/dvsymbols-test.cc               # mlx5dvsymbols.cc + ionicdvsymbols.cc
    misc/ibvwrap-test.cc
    misc/ionicdvwrap-test.cc
    misc/mlx5dvwrap-test.cc
    fakes/dl_fakes.cc                    # dlopen/dlvsym
    fakes/env_fakes.cc                   # NCCL_IBVERBS_LIB
    fakes/libc_fakes.cc                  # gdr_peermem.cc opendir
    fakes/nccl_fakes.cc
    fakes/net_ib_ops_fault_fakes.cc      # ibvwrap.cc fault shims
    fakes/transport_stubs.cc             # rcclUseAinic
    ../common/ProcessIsolatedTestRunner.cpp
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
  cmake_parse_arguments(_ramb "" "CATEGORIES_YAML"
    "SOURCES;EXTRA_INCLUDES;EXTRA_DEFS;REMOVE_DEFS" ${ARGN})

  if(BUILD_TESTS)
    set(_ramb_base_defs ${RCCL_COMMON_COMPILE_DEFS})
    if(_ramb_REMOVE_DEFS)
      list(REMOVE_ITEM _ramb_base_defs ${_ramb_REMOVE_DEFS})
    endif()
    add_executable(${_name} ${_ramb_SOURCES})
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
    rocm_install(TARGETS ${_name} COMPONENT tests)
    if(_ramb_CATEGORIES_YAML)
      rccl_register_test_categories(${_name} "${_ramb_CATEGORIES_YAML}")
    endif()
  else()
    add_executable(${_name} ${_ramb_SOURCES})
    # Mirror add_test_executable(): record for the HOST_TEST_COVERAGE loop.
    list(APPEND HOST_TEST_BINARIES ${_name})
    set(HOST_TEST_BINARIES "${HOST_TEST_BINARIES}" PARENT_SCOPE)
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

  # csv_tuner.cc needs the embedded-tuner header. The RCCL build normally emits
  # it; generate it here so a standalone test build works against an older tree.
  if(NOT EXISTS "${RCCL_BUILD_DIR}/include/rccl_tuner_embedded_configs.h")
    # No COMMAND_ERROR_IS_FATAL here: that needs CMake 3.19 and this project
    # declares 3.16, so check the result explicitly or a FATAL_ERROR from the
    # generator is swallowed and the build fails later on a missing include.
    execute_process(COMMAND ${CMAKE_COMMAND}
      -DTUNER_DIR=${CMAKE_CURRENT_SOURCE_DIR}/../../tuner
      -DOUT_FILE=${RCCL_BUILD_DIR}/include/rccl_tuner_embedded_configs.h
      -P ${CMAKE_CURRENT_SOURCE_DIR}/../../cmake/GenerateEmbeddedTunerConfigs.cmake
      RESULT_VARIABLE _gen_tuner_result
      ERROR_VARIABLE _gen_tuner_error)
    if(_gen_tuner_result)
      message(FATAL_ERROR "GenerateEmbeddedTunerConfigs.cmake failed: ${_gen_tuner_error}")
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
    ${RCCL_TEST_DIR}/DdaNranksRelaxConsensusTests.cpp
    RomeTopoConsensusLogTests.cpp
    ${RCCL_TEST_DIR}/MiscTests.cpp
    ${RCCL_TEST_DIR}/VersionGateTests.cpp
    ${RCCL_TEST_DIR}/TimeoutTests.cpp
    ${RCCL_TEST_DIR}/EnqueueCountTests.cpp
    ${RCCL_TEST_DIR}/BootstrapBidirTests.cpp
    ${RCCL_TEST_DIR}/AltRsmiTests.cpp
    ${RCCL_TEST_DIR}/AcclProfilerTests.cpp
    ${RCCL_TEST_DIR}/common/ProcessIsolatedTestRunner.cpp
    ${RCCL_HIPIFY_DIR}/src/misc/kernel_config.cc
    ${RCCL_HIPIFY_DIR}/src/misc/alt_rsmi.cc
    ${RCCL_HIPIFY_DIR}/src/graph/rome_topo_consensus.cc
    ${RCCL_HIPIFY_DIR}/src/algorithms/dda/ipc/dda_nranks_relax_consensus.cc
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
  option(HOST_TEST_COVERAGE "Build the host-only test binaries with llvm source-based coverage" ON)
  if(HOST_TEST_COVERAGE)
    # HOST_TEST_BINARIES is populated by add_test_executable() and by
    # rccl_add_micro_binary(), so new test binaries are instrumented automatically.
    foreach(_cov_tgt ${HOST_TEST_BINARIES})
      target_compile_options(${_cov_tgt} PRIVATE -fprofile-instr-generate -fcoverage-mapping)
      target_link_options(${_cov_tgt} PRIVATE -fprofile-instr-generate -fcoverage-mapping)
      target_compile_definitions(${_cov_tgt} PRIVATE RCCL_TEST_CODE_COVERAGE=1)
      target_link_libraries(${_cov_tgt} PRIVATE ${CMAKE_DL_LIBS})
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
rccl_add_micro_binary(rccl-UnitTestsMicro
  SOURCES ${TEST_MICRO_SOURCE_FILES}
  EXTRA_INCLUDES
    ${RCCL_HOST_HIPIFY_DIR}/src/devcomm              # devcomm_v22902.h, included unqualified
    ${RCCL_HOST_HIPIFY_DIR}/src/include/nccl_device  # device.h includes rccl_ptr.h unqualified
  EXTRA_DEFS
    ${_MICRO_MODE_DEFS}
    P2P_CC_PATH="${RCCL_HOST_HIPIFY_DIR}/src/transport/p2p_tmp.cc"
    COLL_NET_CC_PATH="${RCCL_HOST_HIPIFY_DIR}/src/transport/coll_net.cc"
    RCCL_TRANSPORT_STUBS_OMIT_COLL_NET
    RCCL_COLLECTIVE_STUBS_OMIT_COLL_NET
    RAS_CC_PATH="${RCCL_HOST_HIPIFY_DIR}/src/ras/ras.cc"
    RAS_DIAGNOSTICS_COMMON_CC_PATH="${RCCL_HOST_HIPIFY_DIR}/src/ras/diagnostics_checks_common.cc"
    # collectives_tmp.cc, not collectives.cc: src/CMakeLists.txt lists src/collectives.cc first,
    # so add_file_unique() renames this one. Re-check after any source-list reorder.
    COLLECTIVES_CC_PATH="${RCCL_HOST_HIPIFY_DIR}/src/ras/collectives_tmp.cc"
    RAS_DIAGNOSTICS_CC_PATH="${RCCL_HOST_HIPIFY_DIR}/src/ras/diagnostics_tmp.cc"
    RASNET_CC_PATH="${RCCL_HOST_HIPIFY_DIR}/src/ras/rasnet.cc"
    RAS_DIAGNOSTICS_GPU_CC_PATH="${RCCL_HOST_HIPIFY_DIR}/src/ras/diagnostics_gpu.cc"
    RAS_PEERS_CC_PATH="${RCCL_HOST_HIPIFY_DIR}/src/ras/peers.cc"
    RAS_DIAGNOSTICS_ENV_CC_PATH="${RCCL_HOST_HIPIFY_DIR}/src/ras/diagnostics_env.cc"
    RAS_PARAM_CC_PATH="${RCCL_HOST_HIPIFY_DIR}/src/ras/ras_param.cc"
    COMM_DUMP_CC_PATH="${RCCL_HOST_HIPIFY_DIR}/src/commDump.cc"
    NET_TELEMETRY_CC_PATH="${RCCL_HOST_HIPIFY_DIR}/src/transport/net_telemetry.cc"
    NET_SOCKET_CC_PATH="${RCCL_HOST_HIPIFY_DIR}/src/transport/net_socket.cc"
    # common_tmp.cc: src/CMakeLists.txt lists transport/net_ib/common.cc first.
    NET_SOCKET_COMMON_CC_PATH="${RCCL_HOST_HIPIFY_DIR}/src/transport/net_socket/common_tmp.cc"
    # graph/tuning.cc keeps the basename, so add_file_unique() renames this one.
    TUNING_CC_PATH="${RCCL_HOST_HIPIFY_DIR}/src/tuning/tuning_tmp.cc"
    TUNING_GENERAL_CC_PATH="${RCCL_HOST_HIPIFY_DIR}/src/tuning/tuning_general.cc"
    CSV_TUNER_CC_PATH="${RCCL_HOST_HIPIFY_DIR}/src/plugin/tuner/csv_tuner.cc"
    # Source tuner/ dir, so a test can check the generator embedded the CSVs verbatim.
    RCCL_TUNER_CSV_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}/../../tuner"
    RAS_CLIENT_CC_PATH="${RCCL_HOST_HIPIFY_DIR}/src/ras/client.cc"
    RMA_PROXY_LAUNCH_CC_PATH="${RCCL_HOST_HIPIFY_DIR}/src/rma/rma_proxy_launch.cc"
    RMA_PROXY_PROGRESS_CC_PATH="${RCCL_HOST_HIPIFY_DIR}/src/rma/rma_proxy_progress.cc"
    # rma_tmp.cc, not rma.cc: src/CMakeLists.txt lists plugin/rma.cc first, so
    # add_file_unique() renames this one. Re-check after any source-list reorder.
    RMA_CC_PATH="${RCCL_HOST_HIPIFY_DIR}/src/rma/rma_tmp.cc"
    # Per the note above, plugin/rma.cc comes first and so keeps its own name.
    RMA_PLUGIN_CC_PATH="${RCCL_HOST_HIPIFY_DIR}/src/plugin/rma.cc"
    # rma_ce.cc has no basename collision, so it keeps its own name.
    RMA_CE_CC_PATH="${RCCL_HOST_HIPIFY_DIR}/src/rma/rma_ce.cc"
    GROUP_CC_PATH="${RCCL_HOST_HIPIFY_DIR}/src/group.cc"
    DEVCOMM_V22902_CC_PATH="${RCCL_HOST_HIPIFY_DIR}/src/devcomm/devcomm_v22902.cc"
    DEVCOMM_V22907_CC_PATH="${RCCL_HOST_HIPIFY_DIR}/src/devcomm/devcomm_v22907.cc"
    WRAP_CC_PATH="${RCCL_HOST_HIPIFY_DIR}/src/rccl_wrap.cc"
    REGISTER_CC_PATH="${RCCL_HOST_HIPIFY_DIR}/src/register/register.cc"
    COLL_REG_CC_PATH="${RCCL_HOST_HIPIFY_DIR}/src/register/coll_reg.cc"
    SENDRECV_REG_CC_PATH="${RCCL_HOST_HIPIFY_DIR}/src/register/sendrecv_reg.cc"
    RCCL_RMA_FAKES_OMIT_ncclRmaInitialized
    RCCL_RMA_FAKES_OMIT_PROXY_LAUNCH_FLOOR
    RCCL_TRANSPORT_STUBS_OMIT_COLLECTIVE_FLOOR
    DEV_RUNTIME_CC_PATH="${RCCL_HOST_HIPIFY_DIR}/src/dev_runtime.cc"
    DEV_RUNTIME_SEGMENTS_CC_PATH="${RCCL_HOST_HIPIFY_DIR}/src/dev_runtime_segments.cc"
    RCCL_DEV_RUNTIME_CC_PATH="${RCCL_HOST_HIPIFY_DIR}/src/rccl_dev_runtime.cc"
    # plugin/gin.cc is the first gin.cc in SRC_FILES, so hipify keeps the name.
    GIN_CC_PATH="${RCCL_HOST_HIPIFY_DIR}/src/plugin/gin.cc"
    GDR_PROBE_CC_PATH="${RCCL_HOST_HIPIFY_DIR}/src/misc/gdr_probe.cc"
    IONICDV_SYMBOLS_CC_PATH="${RCCL_HOST_HIPIFY_DIR}/src/misc/ionicdvsymbols.cc"
    MLX5DV_SYMBOLS_CC_PATH="${RCCL_HOST_HIPIFY_DIR}/src/misc/mlx5dvsymbols.cc"
    DV_PROVIDER_FAKE_IONIC_NAME="$<TARGET_FILE_NAME:rccl-micro-dv-provider-fake-ionic>"
    DV_PROVIDER_FAKE_IONIC_NO_GDA_NAME="$<TARGET_FILE_NAME:rccl-micro-dv-provider-fake-ionic-no-gda>"
    DV_PROVIDER_FAKE_MLX5_NAME="$<TARGET_FILE_NAME:rccl-micro-dv-provider-fake-mlx5>"
    RCCL_GRAPH_GEN_CC_PATH="${RCCL_HOST_HIPIFY_DIR}/src/graph/rccl_graph_gen.cc"
    PARAM_REGISTRY_CC_PATH="${RCCL_HOST_HIPIFY_DIR}/src/param/param_registry.cc"
  CATEGORIES_YAML "${CMAKE_CURRENT_SOURCE_DIR}/../test_categories_micro.yaml")

# Fake ibverbs provider dv libraries for dv_symbols_test.cc; the SONAME is what the builders' dlopen matches.
# The test finds them beside its own executable, in the build tree and in an install.
foreach(_fake ionic ionic-no-gda mlx5)
  string(REGEX REPLACE "-.*" "" _dv "${_fake}")
  add_library(rccl-micro-dv-provider-fake-${_fake} SHARED fakes/dv_provider_fake.cc)
  set_target_properties(rccl-micro-dv-provider-fake-${_fake} PROPERTIES NO_SONAME ON)
  target_compile_options(rccl-micro-dv-provider-fake-${_fake} PRIVATE --offload-host-only)
  target_link_options(rccl-micro-dv-provider-fake-${_fake} PRIVATE -no-hip-rt -Wl,-soname,lib${_dv}.so)
  add_dependencies(rccl-UnitTestsMicro rccl-micro-dv-provider-fake-${_fake})
  if(BUILD_TESTS)
    install(TARGETS rccl-micro-dv-provider-fake-${_fake} LIBRARY DESTINATION ${CMAKE_INSTALL_BINDIR} COMPONENT tests)
  endif()
endforeach()
target_compile_definitions(rccl-micro-dv-provider-fake-ionic-no-gda PRIVATE DV_PROVIDER_FAKE_IONIC_NO_GDA)
target_link_options(rccl-micro-dv-provider-fake-ionic-no-gda PRIVATE
  -Wl,--version-script=${CMAKE_CURRENT_SOURCE_DIR}/fakes/dv_provider_fake_ionic_no_gda.map)
set_property(TARGET rccl-micro-dv-provider-fake-ionic-no-gda APPEND PROPERTY
  LINK_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/fakes/dv_provider_fake_ionic_no_gda.map")

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

# ---------------------------------------------------------------------------
# rccl-UnitTestsMicroOnerank
# ---------------------------------------------------------------------------
rccl_add_micro_binary(rccl-UnitTestsMicroOnerank
  SOURCES ${TEST_MICRO_ONERANK_SOURCE_FILES}
  EXTRA_INCLUDES
    ${RCCL_HOST_HIPIFY_DIR}/src/include/nccl_device  # device/op128.h includes rccl_ptr.h unqualified
    ${RCCL_HOST_HIPIFY_DIR}/src/device               # common_kernel.h includes "tdm/tdmCopy.h"
  EXTRA_DEFS
    ${_MICRO_MODE_DEFS}
    ONERANK_CC_PATH="${RCCL_HOST_HIPIFY_DIR}/src/device/onerank.cu.cpp"
  CATEGORIES_YAML "${CMAKE_CURRENT_SOURCE_DIR}/../test_categories_micro_onerank.yaml")
# onerank.cu defines kernels too; see rccl-UnitTestsMicroCollectives.
target_compile_options(rccl-UnitTestsMicroOnerank PRIVATE
  "SHELL:-Xclang -fcuda-include-gpubinary -Xclang ${CMAKE_CURRENT_BINARY_DIR}/empty_gpubinary")

# ---------------------------------------------------------------------------
# rccl-UnitTestsMicroMgmtTaskEnq
# ---------------------------------------------------------------------------
rccl_add_micro_binary(rccl-UnitTestsMicroMgmtTaskEnq
  SOURCES ${TEST_MICRO_MGMT_TASK_ENQ_SOURCE_FILES}
  EXTRA_DEFS
    ${_MICRO_MODE_DEFS}
    MGMT_TASK_ENQ_CC_PATH="${RCCL_HOST_HIPIFY_DIR}/src/enqueue/mgmt_task_enq.cc"
  CATEGORIES_YAML "${CMAKE_CURRENT_SOURCE_DIR}/../test_categories_micro_mgmt_task_enq.yaml")

# ---------------------------------------------------------------------------
# rccl-UnitTestsMicroGinHost -- src/gin/gin_host.cc (GIN_PROXY_NTHREADS, proxy affinity)
# ---------------------------------------------------------------------------
rccl_add_micro_binary(rccl-UnitTestsMicroGinHost
  SOURCES ${TEST_MICRO_GIN_HOST_SOURCE_FILES}
  EXTRA_INCLUDES
    ${RCCL_HOST_HIPIFY_DIR}/src/include/plugin
    ${RCCL_HOST_HIPIFY_DIR}/src/include/nccl_device
  EXTRA_DEFS
    ${_MICRO_MODE_DEFS}
    GIN_HOST_CC_PATH="${RCCL_HOST_HIPIFY_DIR}/src/gin/gin_host.cc"
  CATEGORIES_YAML "${CMAKE_CURRENT_SOURCE_DIR}/../test_categories_micro_gin_host.yaml")

# ---------------------------------------------------------------------------
# rccl-UnitTestsMicroCrypt
# ---------------------------------------------------------------------------
rccl_add_micro_binary(rccl-UnitTestsMicroCrypt
  SOURCES ${TEST_MICRO_CRYPT_SOURCE_FILES}
  EXTRA_DEFS
    ${_MICRO_MODE_DEFS}
    CRYPT_CC_PATH="${RCCL_HOST_HIPIFY_DIR}/src/misc/crypt.cc"
  CATEGORIES_YAML "${CMAKE_CURRENT_SOURCE_DIR}/../test_categories_micro_crypt.yaml")

# ---------------------------------------------------------------------------
# rccl-UnitTestsMicroStrongstream
# ---------------------------------------------------------------------------
rccl_add_micro_binary(rccl-UnitTestsMicroStrongstream
  SOURCES ${TEST_MICRO_STRONGSTREAM_SOURCE_FILES}
  EXTRA_DEFS
    ${_MICRO_MODE_DEFS}
    STRONGSTREAM_CC_PATH="${RCCL_HOST_HIPIFY_DIR}/src/misc/strongstream.cc"
  CATEGORIES_YAML "${CMAKE_CURRENT_SOURCE_DIR}/../test_categories_micro_strongstream.yaml")

# ---------------------------------------------------------------------------
# rccl-UnitTestsMicroUtils
# ---------------------------------------------------------------------------
rccl_add_micro_binary(rccl-UnitTestsMicroUtils
  SOURCES ${TEST_MICRO_UTILS_SOURCE_FILES}
  EXTRA_DEFS
    ${_MICRO_MODE_DEFS}
    UTILS_CC_PATH="${RCCL_HOST_HIPIFY_DIR}/src/misc/utils.cc"
  CATEGORIES_YAML "${CMAKE_CURRENT_SOURCE_DIR}/../test_categories_micro_utils.yaml")

# ---------------------------------------------------------------------------
# rccl-UnitTestsMicroArgcheck
# ---------------------------------------------------------------------------
rccl_add_micro_binary(rccl-UnitTestsMicroArgcheck
  SOURCES ${TEST_MICRO_ARGCHECK_SOURCE_FILES}
  EXTRA_DEFS
    ${_MICRO_MODE_DEFS}
    ARGCHECK_CC_PATH="${RCCL_HOST_HIPIFY_DIR}/src/misc/argcheck.cc"
  CATEGORIES_YAML "${CMAKE_CURRENT_SOURCE_DIR}/../test_categories_micro_argcheck.yaml")

# ---------------------------------------------------------------------------
# rccl-UnitTestsMicroCpuset
# ---------------------------------------------------------------------------
rccl_add_micro_binary(rccl-UnitTestsMicroCpuset
  SOURCES ${TEST_MICRO_CPUSET_SOURCE_FILES}
  EXTRA_DEFS
    ${_MICRO_MODE_DEFS}
  CATEGORIES_YAML "${CMAKE_CURRENT_SOURCE_DIR}/../test_categories_micro_cpuset.yaml")

# ---------------------------------------------------------------------------
# rccl-UnitTestsMicroDdaIpc -- src/algorithms/dda/all_reduce/dda_all_reduce_ipc.cu
# ---------------------------------------------------------------------------
rccl_add_micro_binary(rccl-UnitTestsMicroDdaIpc
  SOURCES ${TEST_MICRO_DDA_IPC_SOURCE_FILES}
  EXTRA_DEFS
    ${_MICRO_MODE_DEFS}
    DDA_ALL_REDUCE_IPC_CC_PATH="${RCCL_HOST_HIPIFY_DIR}/src/algorithms/dda/all_reduce/dda_all_reduce_ipc.cu.cpp"
  CATEGORIES_YAML "${CMAKE_CURRENT_SOURCE_DIR}/../test_categories_micro_dda_ipc.yaml")
# The TU defines __global__ kernels; an embedded (empty) GPU binary keeps clang from referencing an
# external __hip_fatbin_<cuid>. Registration itself is faked in fakes/hip_fakes.cc.
file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/empty_gpubinary" "")
target_compile_options(rccl-UnitTestsMicroDdaIpc PRIVATE
  "SHELL:-Xclang -fcuda-include-gpubinary -Xclang ${CMAKE_CURRENT_BINARY_DIR}/empty_gpubinary")

# sym_kernels_host.cc only exists when sym kernels are generated (GENERATE_SYM_KERNELS
# in-build / present on disk standalone).
if(_MICRO_BUILD_SYM_KERNELS)
  target_sources(rccl-UnitTestsMicro PRIVATE sym-kernels-index-test.cc)
  target_compile_definitions(rccl-UnitTestsMicro PRIVATE
    SYM_KERNELS_HOST_CC_PATH="${RCCL_HOST_HIPIFY_DIR}/gensrc/symmetric/sym_kernels_host.cc")
elseif(NOT BUILD_TESTS)
  message(WARNING "Skipping sym-kernels-index-test.cc for rccl-UnitTestsMicro: missing ${RCCL_HOST_HIPIFY_DIR}/gensrc/symmetric/sym_kernels_host.cc")
endif()

# ---------------------------------------------------------------------------
# rccl-UnitTestsMicroDdaColl
# ---------------------------------------------------------------------------
rccl_add_micro_binary(rccl-UnitTestsMicroDdaColl
  SOURCES ${TEST_MICRO_DDA_COLL_SOURCE_FILES}
  EXTRA_DEFS
    ${_MICRO_MODE_DEFS}
    DDA_ALL_GATHER_FABRIC_CC_PATH="${RCCL_HOST_HIPIFY_DIR}/src/algorithms/dda/all_gather/dda_all_gather_fabric.cu.cpp"
    DDA_ALL_GATHER_FABRIC_LL_CC_PATH="${RCCL_HOST_HIPIFY_DIR}/src/algorithms/dda/all_gather/dda_all_gather_fabric_ll.cu.cpp"
    DDA_ALL_GATHER_FABRIC_LL128_CC_PATH="${RCCL_HOST_HIPIFY_DIR}/src/algorithms/dda/all_gather/dda_all_gather_fabric_ll128.cu.cpp"
    DDA_ALL_GATHER_IPC_CC_PATH="${RCCL_HOST_HIPIFY_DIR}/src/algorithms/dda/all_gather/dda_all_gather_ipc.cu.cpp"
    DDA_ALL_REDUCE_FABRIC_CC_PATH="${RCCL_HOST_HIPIFY_DIR}/src/algorithms/dda/all_reduce/dda_all_reduce_fabric.cu.cpp"
    DDA_ALL_REDUCE_FABRIC_LL_CC_PATH="${RCCL_HOST_HIPIFY_DIR}/src/algorithms/dda/all_reduce/dda_all_reduce_fabric_ll.cu.cpp"
    DDA_ALL_REDUCE_FABRIC_LL128_CC_PATH="${RCCL_HOST_HIPIFY_DIR}/src/algorithms/dda/all_reduce/dda_all_reduce_fabric_ll128.cu.cpp"
    DDA_ALLTOALL_FABRIC_CC_PATH="${RCCL_HOST_HIPIFY_DIR}/src/algorithms/dda/alltoall/dda_alltoall_fabric.cu.cpp"
    DDA_ALLTOALL_FABRIC_LL_CC_PATH="${RCCL_HOST_HIPIFY_DIR}/src/algorithms/dda/alltoall/dda_alltoall_fabric_ll.cu.cpp"
    DDA_ALLTOALL_FABRIC_LL128_CC_PATH="${RCCL_HOST_HIPIFY_DIR}/src/algorithms/dda/alltoall/dda_alltoall_fabric_ll128.cu.cpp"
    DDA_ALLTOALL_IPC_CC_PATH="${RCCL_HOST_HIPIFY_DIR}/src/algorithms/dda/alltoall/dda_alltoall_ipc.cu.cpp"
    DDA_REDUCE_SCATTER_FABRIC_CC_PATH="${RCCL_HOST_HIPIFY_DIR}/src/algorithms/dda/reduce_scatter/dda_reduce_scatter_fabric.cu.cpp"
    DDA_REDUCE_SCATTER_FABRIC_LL_CC_PATH="${RCCL_HOST_HIPIFY_DIR}/src/algorithms/dda/reduce_scatter/dda_reduce_scatter_fabric_ll.cu.cpp"
    DDA_REDUCE_SCATTER_FABRIC_LL128_CC_PATH="${RCCL_HOST_HIPIFY_DIR}/src/algorithms/dda/reduce_scatter/dda_reduce_scatter_fabric_ll128.cu.cpp"
    DDA_REDUCE_SCATTER_IPC_CC_PATH="${RCCL_HOST_HIPIFY_DIR}/src/algorithms/dda/reduce_scatter/dda_reduce_scatter_ipc.cu.cpp"
  CATEGORIES_YAML "${CMAKE_CURRENT_SOURCE_DIR}/../test_categories_micro_dda_coll.yaml")
# The launchers define __global__ kernels; an embedded empty GPU binary keeps clang from referencing an
# undefined __hip_fatbin_<cuid>. Registration is faked in dda-coll-seams.cc.
file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/empty_gpubinary" "")
target_compile_options(rccl-UnitTestsMicroDdaColl PRIVATE
  "SHELL:-Xclang -fcuda-include-gpubinary -Xclang ${CMAKE_CURRENT_BINARY_DIR}/empty_gpubinary")

# ---------------------------------------------------------------------------
# rccl-UnitTestsMicroRmaProxy
# ---------------------------------------------------------------------------
rccl_add_micro_binary(rccl-UnitTestsMicroRmaProxy
  SOURCES ${TEST_MICRO_RMA_PROXY_SOURCE_FILES}
  EXTRA_DEFS
    ${_MICRO_MODE_DEFS}
    RMA_PROXY_CC_PATH="${RCCL_HOST_HIPIFY_DIR}/src/rma/rma_proxy.cc"
  CATEGORIES_YAML "${CMAKE_CURRENT_SOURCE_DIR}/../test_categories_micro_rma_proxy.yaml")

# ---------------------------------------------------------------------------
# rccl-UnitTestsMicroDebug
# ---------------------------------------------------------------------------
rccl_add_micro_binary(rccl-UnitTestsMicroDebug
  SOURCES ${TEST_MICRO_DEBUG_SOURCE_FILES}
  EXTRA_DEFS
    ${_MICRO_MODE_DEFS}
    DEBUG_CC_PATH="${RCCL_HOST_HIPIFY_DIR}/src/debug.cc"
  CATEGORIES_YAML "${CMAKE_CURRENT_SOURCE_DIR}/../test_categories_micro_debug.yaml")

# ---------------------------------------------------------------------------
# rccl-UnitTestsMicroApiTrace
# ---------------------------------------------------------------------------
rccl_add_micro_binary(rccl-UnitTestsMicroApiTrace
  SOURCES ${TEST_MICRO_API_TRACE_SOURCE_FILES}
  # The rocprofiler-register branch would need librocprofiler-register, so api_trace.cc's
  # RCCL_ROCPROFILER_REGISTER block is not built here and not in this binary's coverage.
  REMOVE_DEFS RCCL_ROCPROFILER_REGISTER=1
  EXTRA_DEFS
    ${_MICRO_MODE_DEFS}
    API_TRACE_CC_PATH="${RCCL_HOST_HIPIFY_DIR}/src/misc/api_trace.cc"
  CATEGORIES_YAML "${CMAKE_CURRENT_SOURCE_DIR}/../test_categories_micro_api_trace.yaml")

# ---------------------------------------------------------------------------
# rccl-UnitTestsMicroRecorder
# ---------------------------------------------------------------------------
rccl_add_micro_binary(rccl-UnitTestsMicroRecorder
  SOURCES ${TEST_MICRO_RECORDER_SOURCE_FILES}
  EXTRA_DEFS
    ${_MICRO_MODE_DEFS}
    RECORDER_CC_PATH="${RCCL_HOST_HIPIFY_DIR}/src/misc/recorder.cc"
  CATEGORIES_YAML "${CMAKE_CURRENT_SOURCE_DIR}/../test_categories_micro_recorder.yaml")

# ---------------------------------------------------------------------------
# rccl-UnitTestsMicroCollTrace
# ---------------------------------------------------------------------------
rccl_add_micro_binary(rccl-UnitTestsMicroCollTrace
  SOURCES ${TEST_MICRO_COLL_TRACE_SOURCE_FILES}
  EXTRA_DEFS
    ${_MICRO_MODE_DEFS}
    COLL_TRACE_CC_PATH="${RCCL_HOST_HIPIFY_DIR}/src/misc/latency_profiler/CollTrace.cc"
    COLL_TRACE_EVENT_CC_PATH="${RCCL_HOST_HIPIFY_DIR}/src/misc/latency_profiler/CollTraceEvent.cc"
    COLL_TRACE_FUNC_CC_PATH="${RCCL_HOST_HIPIFY_DIR}/src/misc/latency_profiler/CollTraceFunc.cc"
    COLL_TRACE_UTILS_CC_PATH="${RCCL_HOST_HIPIFY_DIR}/src/misc/latency_profiler/CollTraceUtils.cc"
  CATEGORIES_YAML "${CMAKE_CURRENT_SOURCE_DIR}/../test_categories_micro_coll_trace.yaml")

# ---------------------------------------------------------------------------
# rccl-UnitTestsMicroWarpSpeed -- tests for ENABLE_WARP_SPEED
# ---------------------------------------------------------------------------
rccl_add_micro_binary(rccl-UnitTestsMicroWarpSpeed
  SOURCES ${TEST_MICRO_SCHEDULER_SOURCE_FILES}
  EXTRA_DEFS
    ${_MICRO_MODE_DEFS}
    ENABLE_WARP_SPEED
    # Not inherited via RCCL_COMMON_COMPILE_DEFS (unlike ENABLE_FAULT_INJECTION); the default-ON, shipping config.
    GENERATE_SYM_KERNELS
    # Hipify renames this to *_tmp.cc: src/enqueue/task_sched/allgatherv_sched.cc keeps the basename instead.
    ALLGATHERV_SCHED_CC_PATH="${RCCL_HOST_HIPIFY_DIR}/src/scheduler/allgatherv_sched_tmp.cc"
    SYMMETRIC_SCHED_CC_PATH="${RCCL_HOST_HIPIFY_DIR}/src/scheduler/symmetric_sched.cc"
  CATEGORIES_YAML "${CMAKE_CURRENT_SOURCE_DIR}/../test_categories_micro_scheduler.yaml")

# ---------------------------------------------------------------------------
# rccl-UnitTestsMicroGinHostProxy
# ---------------------------------------------------------------------------
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
# The test redirects rocmwrap.cc's DmaBufEnable param; keep the name and default in step.
foreach(_f "${CMAKE_CURRENT_LIST_DIR}/../../src/misc/rocmwrap.cc" "${CMAKE_CURRENT_LIST_DIR}/gin-host-proxy-test.cc")
  rccl_assert_source_line("${_f}" "NCCL_PARAM(DmaBufEnable, \"DMABUF_ENABLE\", 1)" "gin-host-proxy-test.cc")
endforeach()
rccl_add_micro_binary(rccl-UnitTestsMicroGinHostProxy
  SOURCES ${TEST_MICRO_GIN_HOST_PROXY_SOURCE_FILES}
  EXTRA_INCLUDES
    ${RCCL_HOST_HIPIFY_DIR}/src/include/plugin
    ${RCCL_HOST_HIPIFY_DIR}/src/include/nccl_device
  EXTRA_DEFS
    ${_MICRO_MODE_DEFS}
    ${_MICRO_DMABUF_EXPORT_DEF}
    GIN_HOST_PROXY_CC_PATH="${RCCL_HOST_HIPIFY_DIR}/src/gin/gin_host_proxy.cc"
    GPUCONTEXT_CC_PATH="${RCCL_HOST_HIPIFY_DIR}/src/gin/proxy_gpucontext/gpucontext.cc"
    GPUCONTEXT_V1_CC_PATH="${RCCL_HOST_HIPIFY_DIR}/src/gin/proxy_gpucontext/gpucontext_v1.cc"
    GPUCONTEXT_V2_CC_PATH="${RCCL_HOST_HIPIFY_DIR}/src/gin/proxy_gpucontext/gpucontext_v2.cc"
  CATEGORIES_YAML "${CMAKE_CURRENT_SOURCE_DIR}/../test_categories_micro_gin_host_proxy.yaml")

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
# rccl-UnitTestsMicroConnect -- ncclTopoPreset / ncclTopoPostset unstubbed.
# ---------------------------------------------------------------------------
set_source_files_properties(
  ${RCCL_HOST_HIPIFY_DIR}/src/graph/rings.cc
  ${RCCL_HOST_HIPIFY_DIR}/src/graph/trees.cc
  ${RCCL_HOST_HIPIFY_DIR}/src/graph/rccl_graph_gen.cc
  PROPERTIES GENERATED TRUE)
rccl_add_micro_binary(rccl-UnitTestsMicroConnect
  SOURCES ${TEST_MICRO_CONNECT_SOURCE_FILES}
  EXTRA_DEFS
    ${_MICRO_MODE_DEFS}
    ${_MICRO_AMDSMI_DEF}
    CONNECT_CC_PATH="${RCCL_HOST_HIPIFY_DIR}/src/graph/connect.cc"
  CATEGORIES_YAML "${CMAKE_CURRENT_SOURCE_DIR}/../test_categories_micro_connect.yaml")

# ---------------------------------------------------------------------------
# rccl-UnitTestsMicroEnqueue (+ shipping -devlinker arm standalone)
# ---------------------------------------------------------------------------
set(_enq_common_args
  SOURCES ${TEST_MICRO_ENQUEUE_SOURCE_FILES}
  EXTRA_INCLUDES
    ${RCCL_HOST_HIPIFY_DIR}/src/device               # device/common.h (neutered)
    ${RCCL_HOST_HIPIFY_DIR}/src/include/nccl_device
  EXTRA_DEFS
    ${_MICRO_MODE_DEFS}
    ${_MICRO_AMDSMI_DEF}
    # Per-symbol omissions from the shared nccl_stubs.cc; enqueue.cc itself
    # supplies both, so neither needs a replacement definition.
    RCCL_STUBS_OMIT_ncclInitKernelsForDevice
    RCCL_STUBS_OMIT_ncclParamGraphStreamOrdering
    # Production librccl always builds this arm (CMakeLists.txt LL128_ENABLED=ON).
    # Standalone host CI otherwise compiles enqueue.cc without it, so addP2pToPlan's
    # gfx1250 ENABLE=1 window is unreachable there.
    ENABLE_LL128
    ENQUEUE_CC_PATH="${RCCL_HOST_HIPIFY_DIR}/src/enqueue/enqueue.cc"
  CATEGORIES_YAML "${CMAKE_CURRENT_SOURCE_DIR}/../test_categories_micro_enqueue.yaml")

rccl_add_micro_binary(rccl-UnitTestsMicroEnqueue ${_enq_common_args})
if(_ENQ_BUILD_DEVLINKER)
  rccl_add_micro_binary(rccl-UnitTestsMicroEnqueue-devlinker ${_enq_common_args})
  target_compile_definitions(rccl-UnitTestsMicroEnqueue-devlinker PRIVATE RCCL_DEVICE_LINKER)
endif()

# ---------------------------------------------------------------------------
# rccl-UnitTestsMicroNvls
# ---------------------------------------------------------------------------
rccl_add_micro_binary(rccl-UnitTestsMicroNvls
  SOURCES ${TEST_MICRO_NVLS_SOURCE_FILES}
  EXTRA_DEFS
    ${_MICRO_MODE_DEFS}
    NVLS_CC_PATH="${RCCL_HOST_HIPIFY_DIR}/src/transport/nvls.cc"
  CATEGORIES_YAML "${CMAKE_CURRENT_SOURCE_DIR}/../test_categories_micro.yaml")

# ---------------------------------------------------------------------------
# rccl-UnitTestsMicroNcclDevice
# ---------------------------------------------------------------------------
rccl_add_micro_binary(rccl-UnitTestsMicroNcclDevice
  SOURCES ${TEST_MICRO_NCCL_DEVICE_SOURCE_FILES}
  EXTRA_DEFS
    ${_MICRO_MODE_DEFS}
    CFT_BARRIER_CC_PATH="${RCCL_HOST_HIPIFY_DIR}/src/nccl_device/cft_barrier.cc"
    GIN_BARRIER_CC_PATH="${RCCL_HOST_HIPIFY_DIR}/src/nccl_device/gin_barrier.cc"
    GIN_SCRATCH_CC_PATH="${RCCL_HOST_HIPIFY_DIR}/src/nccl_device/gin_scratch.cc"
    LL_A2A_CC_PATH="${RCCL_HOST_HIPIFY_DIR}/src/nccl_device/ll_a2a.cc"
    LSA_BARRIER_CC_PATH="${RCCL_HOST_HIPIFY_DIR}/src/nccl_device/lsa_barrier.cc"
  CATEGORIES_YAML "${CMAKE_CURRENT_SOURCE_DIR}/../test_categories_micro_nccl_device.yaml")

# ---------------------------------------------------------------------------
# rccl-UnitTestsMicroCore
# ---------------------------------------------------------------------------
rccl_add_micro_binary(rccl-UnitTestsMicroCore
  SOURCES ${TEST_MICRO_CORE_SOURCE_FILES}
  EXTRA_DEFS
    ${_MICRO_MODE_DEFS}
    CORE_CC_PATH="${RCCL_HOST_HIPIFY_DIR}/src/nccl_device/core.cc"
  CATEGORIES_YAML "${CMAKE_CURRENT_SOURCE_DIR}/../test_categories_micro_core.yaml")

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

# ---------------------------------------------------------------------------
# rccl-UnitTestsMicroParam
# ---------------------------------------------------------------------------
rccl_add_micro_binary(rccl-UnitTestsMicroParam
  SOURCES ${TEST_MICRO_PARAM_SOURCE_FILES}
  EXTRA_DEFS
    ${_MICRO_MODE_DEFS}
    # param_tmp.cc, not param.cc: src/CMakeLists.txt lists misc/param.cc first, so
    # add_file_unique() renames this one.
    PARAM_CC_PATH="${RCCL_HOST_HIPIFY_DIR}/src/param/param_tmp.cc"
    C_API_CC_PATH="${RCCL_HOST_HIPIFY_DIR}/src/param/c_api.cc"
    MISC_PARAM_CC_PATH="${RCCL_HOST_HIPIFY_DIR}/src/misc/param.cc"
    PARAM_REGISTRY_CC_PATH="${RCCL_HOST_HIPIFY_DIR}/src/param/param_registry.cc"
  CATEGORIES_YAML "${CMAKE_CURRENT_SOURCE_DIR}/../test_categories_micro_param.yaml")

# ---------------------------------------------------------------------------
# rccl-UnitTestsMicroClientSupport
# ---------------------------------------------------------------------------
rccl_add_micro_binary(rccl-UnitTestsMicroClientSupport
  SOURCES ${TEST_MICRO_CLIENT_SUPPORT_SOURCE_FILES}
  EXTRA_DEFS
    ${_MICRO_MODE_DEFS}
    CLIENT_SUPPORT_CC_PATH="${RCCL_HOST_HIPIFY_DIR}/src/ras/client_support.cc"
  CATEGORIES_YAML "${CMAKE_CURRENT_SOURCE_DIR}/../test_categories_micro.yaml")

# ---------------------------------------------------------------------------
# rccl-UnitTestsMicroPlugin
# ---------------------------------------------------------------------------
rccl_add_micro_binary(rccl-UnitTestsMicroPlugin
  SOURCES ${TEST_MICRO_PLUGIN_SOURCE_FILES}
  EXTRA_DEFS
    ${_MICRO_MODE_DEFS}
    PLUGIN_NET_CC_PATH="${RCCL_HOST_HIPIFY_DIR}/src/plugin/net.cc"
    PLUGIN_OPEN_CC_PATH="${RCCL_HOST_HIPIFY_DIR}/src/plugin/plugin_open.cc"
    NET_V6_CC_PATH="${RCCL_HOST_HIPIFY_DIR}/src/plugin/net/net_v6.cc"
    NET_V7_CC_PATH="${RCCL_HOST_HIPIFY_DIR}/src/plugin/net/net_v7.cc"
    NET_V8_CC_PATH="${RCCL_HOST_HIPIFY_DIR}/src/plugin/net/net_v8.cc"
    NET_V9_CC_PATH="${RCCL_HOST_HIPIFY_DIR}/src/plugin/net/net_v9.cc"
    NET_V10_CC_PATH="${RCCL_HOST_HIPIFY_DIR}/src/plugin/net/net_v10.cc"
    NET_V11_CC_PATH="${RCCL_HOST_HIPIFY_DIR}/src/plugin/net/net_v11.cc"
    NET_V12_CC_PATH="${RCCL_HOST_HIPIFY_DIR}/src/plugin/net/net_v12.cc"
    RCCL_TRANSPORT_STUBS_OMIT_ncclGpuGdrSupport  # net.cc defines it
  CATEGORIES_YAML "${CMAKE_CURRENT_SOURCE_DIR}/../test_categories_micro_plugin.yaml")

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
      SYM_MODEL_CC_PATH="${RCCL_HOST_HIPIFY_DIR}/src/tuning/sym_model.cc"
      # gin_tmp.cc, not gin.cc: src/CMakeLists.txt lists plugin/gin.cc first, so hipify renames this one.
      SYM_MODEL_GIN_CC_PATH="${RCCL_HOST_HIPIFY_DIR}/src/tuning/sym_model/gin_tmp.cc"
      SYM_MODEL_LSA_CC_PATH="${RCCL_HOST_HIPIFY_DIR}/src/tuning/sym_model/lsa.cc"
      SYM_MODEL_LSA_BASE_CC_PATH="${RCCL_HOST_HIPIFY_DIR}/src/tuning/sym_model/lsa_base.cc"
      SYM_MODEL_LSA_A2A_CC_PATH="${RCCL_HOST_HIPIFY_DIR}/src/tuning/sym_model/lsa_a2a.cc"
    CATEGORIES_YAML "${CMAKE_CURRENT_SOURCE_DIR}/../test_categories_micro.yaml")
endif()

# ---------------------------------------------------------------------------
# rccl-UnitTestsMicroProfiler
# ---------------------------------------------------------------------------
rccl_add_micro_binary(rccl-UnitTestsMicroProfiler
  SOURCES ${TEST_MICRO_PROFILER_SOURCE_FILES}
  EXTRA_DEFS
    ${_MICRO_MODE_DEFS}
    PROFILER_CC_PATH="${RCCL_HOST_HIPIFY_DIR}/src/plugin/profiler.cc"
    PROFILER_V1_CC_PATH="${RCCL_HOST_HIPIFY_DIR}/src/plugin/profiler/profiler_v1.cc"
    PROFILER_V2_CC_PATH="${RCCL_HOST_HIPIFY_DIR}/src/plugin/profiler/profiler_v2.cc"
    PROFILER_V3_CC_PATH="${RCCL_HOST_HIPIFY_DIR}/src/plugin/profiler/profiler_v3.cc"
    PROFILER_V4_CC_PATH="${RCCL_HOST_HIPIFY_DIR}/src/plugin/profiler/profiler_v4.cc"
    PROFILER_V5_CC_PATH="${RCCL_HOST_HIPIFY_DIR}/src/plugin/profiler/profiler_v5.cc"
    PROFILER_V6_CC_PATH="${RCCL_HOST_HIPIFY_DIR}/src/plugin/profiler/profiler_v6.cc"
    PROFILER_V7_CC_PATH="${RCCL_HOST_HIPIFY_DIR}/src/plugin/profiler/profiler_v7.cc"
  CATEGORIES_YAML "${CMAKE_CURRENT_SOURCE_DIR}/../test_categories_micro_profiler.yaml")

# ---------------------------------------------------------------------------
# rccl-UnitTestsMicroTaskPrep -- host-only microtests for src/enqueue/task_prep/.
# Its own binary (not rccl-UnitTestsMicro): the four task_prep/config production
# files are compiled as real TUs so task_prep.cc's sibling externs resolve to
# real code; the surrounding subsystems are faked. See the source-list comment
# in rccl_define_micro_source_lists() for the rationale.
# ---------------------------------------------------------------------------
# task_prep/ and config/ oracle TUs live outside src/misc/, so mark them GENERATED
# here (rccl_mark_oracle_tus_generated only covers the misc/ argcheck/archinfo/utils/kernel_config set).
set_source_files_properties(
  ${RCCL_HOST_HIPIFY_DIR}/src/config/algorithm_parser.cc
  ${RCCL_HOST_HIPIFY_DIR}/src/config/algorithm_registry.cc
  ${RCCL_HOST_HIPIFY_DIR}/src/config/collconfig.cc
  PROPERTIES GENERATED TRUE)
rccl_add_micro_binary(rccl-UnitTestsMicroTaskPrep
  SOURCES ${TEST_MICRO_TASKPREP_SOURCE_FILES}
  EXTRA_INCLUDES
    ${RCCL_HOST_HIPIFY_DIR}/src/device               # device/common.h (neutered)
    ${RCCL_HOST_HIPIFY_DIR}/src/include/nccl_device
  EXTRA_DEFS
    ${_MICRO_MODE_DEFS}
    ${_MICRO_AMDSMI_DEF}
    # Hipify preserves src/enqueue/task_prep/ and renames none of the four (no basename collisions).
    # Same for src/enqueue/task_sched/: the _tmp.cc rename lands on src/scheduler/allgatherv_sched.cc,
    # the only basename collision (see the ALLGATHERV_SCHED_CC_PATH note above).
    TASK_PREP_CC_PATH="${RCCL_HOST_HIPIFY_DIR}/src/enqueue/task_prep/task_prep.cc"
    TASK_CLASSIFY_CC_PATH="${RCCL_HOST_HIPIFY_DIR}/src/enqueue/task_prep/task_classify.cc"
    TASK_PRETUNING_CC_PATH="${RCCL_HOST_HIPIFY_DIR}/src/enqueue/task_prep/task_pretuning.cc"
    TASK_POSTTUNING_CC_PATH="${RCCL_HOST_HIPIFY_DIR}/src/enqueue/task_prep/task_posttuning.cc"
    TASK_SCHED_DISPATCH_CC_PATH="${RCCL_HOST_HIPIFY_DIR}/src/enqueue/task_sched/task_sched.cc"
    TASK_SCHED_SYM_CC_PATH="${RCCL_HOST_HIPIFY_DIR}/src/enqueue/task_sched/sym_sched.cc"
    TASK_SCHED_LEGACY_CC_PATH="${RCCL_HOST_HIPIFY_DIR}/src/enqueue/task_sched/legacy_sched.cc"
    TASK_SCHED_ALLGATHERV_CC_PATH="${RCCL_HOST_HIPIFY_DIR}/src/enqueue/task_sched/allgatherv_sched.cc"
    TASK_SCHED_P2P_CC_PATH="${RCCL_HOST_HIPIFY_DIR}/src/enqueue/task_sched/p2p_sched.cc"
    TASK_SCHED_RMA_CC_PATH="${RCCL_HOST_HIPIFY_DIR}/src/enqueue/task_sched/rma_sched.cc"
    TASK_SCHED_CE_CC_PATH="${RCCL_HOST_HIPIFY_DIR}/src/enqueue/task_sched/ce_sched.cc"
  CATEGORIES_YAML "${CMAKE_CURRENT_SOURCE_DIR}/../test_categories_micro_taskprep.yaml")

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

# ---------------------------------------------------------------------------
# rccl-UnitTestsMicroRomeModels
# ---------------------------------------------------------------------------
rccl_add_micro_binary(rccl-UnitTestsMicroRomeModels
  SOURCES ${TEST_MICRO_ROME_MODELS_SOURCE_FILES}
  EXTRA_DEFS
    ${_MICRO_MODE_DEFS}
    ROME_MODELS_CC_PATH="${RCCL_HOST_HIPIFY_DIR}/src/graph/rome_models.cc"
  CATEGORIES_YAML "${CMAKE_CURRENT_SOURCE_DIR}/../test_categories_micro_rome_models.yaml")

# ---------------------------------------------------------------------------
# rccl-UnitTestsMicroRoctxNvml
# ---------------------------------------------------------------------------
set(_roctx_nvml_common_args
  SOURCES ${TEST_MICRO_ROCTX_NVML_SOURCE_FILES}
  EXTRA_DEFS
    ${_MICRO_MODE_DEFS}
    ROCTX_CC_PATH="${RCCL_HOST_HIPIFY_DIR}/src/misc/roctx.cc"
    NVMLWRAP_STUB_CC_PATH="${RCCL_HOST_HIPIFY_DIR}/src/misc/nvmlwrap_stub.cc"
  CATEGORIES_YAML "${CMAKE_CURRENT_SOURCE_DIR}/../test_categories_micro_roctx_nvml.yaml")

rccl_add_micro_binary(rccl-UnitTestsMicroRoctxNvml ${_roctx_nvml_common_args})
# -roctx builds roctx.cc's ROCTX_ENABLE arm against the stand-in fakes/roctx_include.
if(_MICRO_BUILD_ROCTX_VARIANT)
  rccl_add_micro_binary(rccl-UnitTestsMicroRoctxNvml-roctx ${_roctx_nvml_common_args})
  target_compile_definitions(rccl-UnitTestsMicroRoctxNvml-roctx PRIVATE ROCTX_ENABLE)
  target_include_directories(rccl-UnitTestsMicroRoctxNvml-roctx BEFORE PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/fakes/roctx_include)
endif()

# ---------------------------------------------------------------------------
# rccl-UnitTestsMicroRocmwrap
# ---------------------------------------------------------------------------
rccl_add_micro_binary(rccl-UnitTestsMicroRocmwrap
  SOURCES ${TEST_MICRO_ROCMWRAP_SOURCE_FILES}
  EXTRA_DEFS
    ${_MICRO_MODE_DEFS}
    ${_MICRO_DMABUF_EXPORT_DEF}
    ROCMWRAP_CC_PATH="${RCCL_HOST_HIPIFY_DIR}/src/misc/rocmwrap.cc"
  CATEGORIES_YAML "${CMAKE_CURRENT_SOURCE_DIR}/../test_categories_micro_rocmwrap.yaml")

# ---------------------------------------------------------------------------
# rccl-UnitTestsMicroAmdsmiWrap
# ---------------------------------------------------------------------------
rccl_add_micro_binary(rccl-UnitTestsMicroAmdsmiWrap
  SOURCES ${TEST_MICRO_AMDSMI_WRAP_SOURCE_FILES}
  EXTRA_DEFS
    ${_MICRO_MODE_DEFS}
    AMDSMI_WRAP_CC_PATH="${RCCL_HOST_HIPIFY_DIR}/src/misc/amdsmi_wrap.cc"
  CATEGORIES_YAML "${CMAKE_CURRENT_SOURCE_DIR}/../test_categories_micro_amdsmi_wrap.yaml")

# ---------------------------------------------------------------------------
# rccl-UnitTestsMicroGraph is its own binary because the others link fakes for the graph symbols these TUs define.
# ---------------------------------------------------------------------------
set_source_files_properties(
  ${RCCL_HOST_HIPIFY_DIR}/src/graph/paths.cc
  ${RCCL_HOST_HIPIFY_DIR}/src/graph/rome_models.cc
  ${RCCL_HOST_HIPIFY_DIR}/src/graph/search.cc
  ${RCCL_HOST_HIPIFY_DIR}/src/graph/topo.cc
  ${RCCL_HOST_HIPIFY_DIR}/src/graph/xml.cc
  ${RCCL_HOST_HIPIFY_DIR}/src/misc/nvmlwrap_stub.cc
  PROPERTIES GENERATED TRUE)
rccl_add_micro_binary(rccl-UnitTestsMicroGraph
  SOURCES ${TEST_MICRO_GRAPH_SOURCE_FILES}
  EXTRA_DEFS
    ${_MICRO_MODE_DEFS}
    ${_MICRO_AMDSMI_DEF}
    RCCL_NCCL_FAKES_OMIT_GRAPH
    RCCL_TRANSPORT_STUBS_OMIT_ncclPxnDisable
  CATEGORIES_YAML "${CMAKE_CURRENT_SOURCE_DIR}/../test_categories_micro_graph.yaml")

# ---------------------------------------------------------------------------
# rccl-UnitTestsMicroTopo
# ---------------------------------------------------------------------------
rccl_add_micro_binary(rccl-UnitTestsMicroTopo
  SOURCES ${TEST_MICRO_TOPO_SOURCE_FILES}
  # The warp-speed arms of connect.cc call rcclGetMaxWarpsPerBlock/rcclWarpSpeedComputeNChannels (rccl_wrap.cc).
  REMOVE_DEFS ENABLE_WARP_SPEED
  EXTRA_DEFS
    ${_MICRO_MODE_DEFS}
    ${_MICRO_AMDSMI_DEF}
    XML_CC_PATH="${RCCL_HOST_HIPIFY_DIR}/src/graph/xml.cc"
    TOPO_CC_PATH="${RCCL_HOST_HIPIFY_DIR}/src/graph/topo.cc"
    PATHS_CC_PATH="${RCCL_HOST_HIPIFY_DIR}/src/graph/paths.cc"
    SEARCH_CC_PATH="${RCCL_HOST_HIPIFY_DIR}/src/graph/search.cc"
    ROME_MODELS_CC_PATH="${RCCL_HOST_HIPIFY_DIR}/src/graph/rome_models.cc"
    CONNECT_CC_PATH="${RCCL_HOST_HIPIFY_DIR}/src/graph/connect.cc"
    RINGS_CC_PATH="${RCCL_HOST_HIPIFY_DIR}/src/graph/rings.cc"
    TREES_CC_PATH="${RCCL_HOST_HIPIFY_DIR}/src/graph/trees.cc"
  CATEGORIES_YAML "${CMAKE_CURRENT_SOURCE_DIR}/../test_categories_micro_topo.yaml")

# ---------------------------------------------------------------------------
# rccl-UnitTestsMicroDiagnostics
# ---------------------------------------------------------------------------
# p2p.cu's <<<>>> must not reach hip_fakes.cc's aborting hipLaunchKernel; see diag-p2p-device-test.cc.
set_source_files_properties(diag-p2p-device-test.cc PROPERTIES
  COMPILE_OPTIONS "-fno-hip-new-launch-api;-fuse-cuid=none")
rccl_add_micro_binary(rccl-UnitTestsMicroDiagnostics
  SOURCES ${TEST_MICRO_DIAGNOSTICS_SOURCE_FILES}
  EXTRA_DEFS
    ${_MICRO_MODE_DEFS}
    DIAG_P2P_CC_PATH="${RCCL_HOST_HIPIFY_DIR}/src/diagnostics/p2p.cc"
    DIAG_IB_WRITE_BW_CC_PATH="${RCCL_HOST_HIPIFY_DIR}/src/diagnostics/ib_write_bw.cc"
    DIAGNOSTICS_CC_PATH="${RCCL_HOST_HIPIFY_DIR}/src/diagnostics.cc"
    DIAG_P2P_DEVICE_CU_PATH="${RCCL_HOST_HIPIFY_DIR}/src/diagnostics/device/p2p.cu.cpp"
  CATEGORIES_YAML "${CMAKE_CURRENT_SOURCE_DIR}/../test_categories_micro_diagnostics.yaml")

# ---------------------------------------------------------------------------
# rccl-UnitTestsMicroGinHostFinalize -- host finalize for src/gin/gin_host.cc.
# Separate from rccl-UnitTestsMicroGinHost: both compile gin_host.cc, and their
# test TUs define the same link seams.
# ---------------------------------------------------------------------------
rccl_add_micro_binary(rccl-UnitTestsMicroGinHostFinalize
  SOURCES ${TEST_MICRO_GIN_HOST_FINALIZE_SOURCE_FILES}
  EXTRA_DEFS
    ${_MICRO_MODE_DEFS}
    # Hipify keeps gin/gin_host.cc (no basename collision with another gin_host.cc).
    GIN_HOST_CC_PATH="${RCCL_HOST_HIPIFY_DIR}/src/gin/gin_host.cc"
  CATEGORIES_YAML "${CMAKE_CURRENT_SOURCE_DIR}/../test_categories_micro_gin_host_finalize.yaml")

# ---------------------------------------------------------------------------
# rccl-UnitTestsMicroDda
# ---------------------------------------------------------------------------
set_source_files_properties(
  ${RCCL_HOST_HIPIFY_DIR}/src/algorithms/dda/device/device_buffer.cc
  PROPERTIES GENERATED TRUE)
rccl_add_micro_binary(rccl-UnitTestsMicroDda
  SOURCES ${TEST_MICRO_DDA_SOURCE_FILES}
  EXTRA_DEFS
    ${_MICRO_MODE_DEFS}
    FABRIC_MEM_HANDLER_CC_PATH="${RCCL_HOST_HIPIFY_DIR}/src/algorithms/dda/fabric/fabric_mem_handler.cc"
    # hipify renames .cu to .cu.cpp
    FABRIC_GPU_BARRIER_CC_PATH="${RCCL_HOST_HIPIFY_DIR}/src/algorithms/dda/fabric/fabric_gpu_barrier.cu.cpp"
    FABRIC_INIT_CC_PATH="${RCCL_HOST_HIPIFY_DIR}/src/algorithms/dda/fabric/fabric_init.cu.cpp"
    IPC_MEM_HANDLER_CC_PATH="${RCCL_HOST_HIPIFY_DIR}/src/algorithms/dda/ipc/ipc_mem_handler.cc"
  CATEGORIES_YAML "${CMAKE_CURRENT_SOURCE_DIR}/../test_categories_micro_dda.yaml")

# ---------------------------------------------------------------------------
# rccl-UnitTestsMicroTransport
# ---------------------------------------------------------------------------
rccl_add_micro_binary(rccl-UnitTestsMicroTransport
  SOURCES ${TEST_MICRO_TRANSPORT_SOURCE_FILES}
  EXTRA_INCLUDES
    ${RCCL_HOST_HIPIFY_DIR}/src/transport
  EXTRA_DEFS
    ${_MICRO_MODE_DEFS}
    ENABLE_FAULT_INJECTION
    NET_IB_COMMON_CC_PATH="${RCCL_HOST_HIPIFY_DIR}/src/transport/net_ib/common.cc"
    NET_IB_GDR_CC_PATH="${RCCL_HOST_HIPIFY_DIR}/src/transport/net_ib/gdr.cc"
    # Hipify renames this to *_tmp.cc: src/init.cc keeps the basename.
    NET_IB_INIT_CC_PATH="${RCCL_HOST_HIPIFY_DIR}/src/transport/net_ib/init_tmp.cc"
    NET_IB_GID_INSPECT_CC_PATH="${RCCL_HOST_HIPIFY_DIR}/src/transport/net_ib/net_ib_gid_inspect.cc"
    NET_IB_REG_CC_PATH="${RCCL_HOST_HIPIFY_DIR}/src/transport/net_ib/reg.cc"
    RMA_SOCKET_CC_PATH="${RCCL_HOST_HIPIFY_DIR}/src/transport/rma_socket.cc"
    RMA_SOCKET_MEM_CC_PATH="${RCCL_HOST_HIPIFY_DIR}/src/transport/rma_socket_mem.cc"
    RMA_SOCKET_PROGRESS_RECVER_CC_PATH="${RCCL_HOST_HIPIFY_DIR}/src/transport/rma_socket_progress_recver.cc"
    RMA_SOCKET_SETUP_CC_PATH="${RCCL_HOST_HIPIFY_DIR}/src/transport/rma_socket_setup.cc"
  CATEGORIES_YAML "${CMAKE_CURRENT_SOURCE_DIR}/../test_categories_micro_transport.yaml")

# ---------------------------------------------------------------------------
# rccl-UnitTestsMicroNetIbCastDataPath
# ---------------------------------------------------------------------------
set_source_files_properties(
  ${RCCL_HOST_HIPIFY_DIR}/src/misc/ibvsymbols.cc
  ${RCCL_HOST_HIPIFY_DIR}/src/misc/ibvwrap.cc
  ${RCCL_HOST_HIPIFY_DIR}/src/transport/net_ib_cast/qp_sharing.cc
  ${RCCL_HOST_HIPIFY_DIR}/src/transport/net_telemetry.cc
  PROPERTIES GENERATED TRUE)
rccl_add_micro_binary(rccl-UnitTestsMicroNetIbCastDataPath
  SOURCES ${TEST_MICRO_NET_IB_CAST_DATA_PATH_SOURCE_FILES}
  EXTRA_INCLUDES
    ${RCCL_HOST_HIPIFY_DIR}/src/transport/net_ib_cast  # common_cast.h, included unqualified
    ${RCCL_HOST_HIPIFY_DIR}/src/transport              # net_ib_limits.h
  EXTRA_DEFS
    ${_MICRO_MODE_DEFS}
    # Shipping librccl builds with FAULT_INJECTION=ON, and net_ib_ops_fault.cc is all under it.
    ENABLE_FAULT_INJECTION
    # p2p_tmp.cc, not p2p.cc: src/CMakeLists.txt lists transport/net_ib/p2p.cc first, so
    # add_file_unique() renames this one. Re-check after any source-list reorder.
    NET_IB_CAST_P2P_CC_PATH="${RCCL_HOST_HIPIFY_DIR}/src/transport/net_ib_cast/p2p_tmp.cc"
    NET_IB_CAST_SCHEDULER_CC_PATH="${RCCL_HOST_HIPIFY_DIR}/src/transport/net_ib_cast/scheduler.cc"
    NET_IB_OPS_FAULT_CC_PATH="${RCCL_HOST_HIPIFY_DIR}/src/transport/net_ib_cast/net_ib_ops_fault.cc"
  CATEGORIES_YAML "${CMAKE_CURRENT_SOURCE_DIR}/../test_categories_micro_net_ib_cast_data_path.yaml")

# ---------------------------------------------------------------------------
# rccl-UnitTestsMicroP2pResiliency
# ---------------------------------------------------------------------------
rccl_add_micro_binary(rccl-UnitTestsMicroP2pResiliency
  SOURCES ${TEST_MICRO_P2P_RESILIENCY_SOURCE_FILES}
  EXTRA_INCLUDES
    ${RCCL_HOST_HIPIFY_DIR}/src/transport/net_ib_cast
    ${RCCL_HOST_HIPIFY_DIR}/src/transport
  EXTRA_DEFS
    ${_MICRO_MODE_DEFS}
    # The _tmp names: src/CMakeLists.txt lists net_ib/p2p_resiliency*.cc first, so
    # add_file_unique() renames these. Re-check after any source-list reorder.
    P2P_RESILIENCY_CC_PATH="${RCCL_HOST_HIPIFY_DIR}/src/transport/net_ib_cast/p2p_resiliency_tmp.cc"
    P2P_RESILIENCY_RECOVERY_CC_PATH="${RCCL_HOST_HIPIFY_DIR}/src/transport/net_ib_cast/p2p_resiliency_recovery_tmp.cc"
    P2P_RESILIENCY_RECOVERY_AINIC_CC_PATH="${RCCL_HOST_HIPIFY_DIR}/src/transport/net_ib_cast/p2p_resiliency_recovery_ainic.cc"
  CATEGORIES_YAML "${CMAKE_CURRENT_SOURCE_DIR}/../test_categories_micro_p2p_resiliency.yaml")

# ---------------------------------------------------------------------------
# rccl-UnitTestsMicroNetIbCastInit
# ---------------------------------------------------------------------------
set_source_files_properties(
  ${RCCL_HOST_HIPIFY_DIR}/src/graph/xml.cc
  ${RCCL_HOST_HIPIFY_DIR}/src/misc/utils.cc
  ${RCCL_HOST_HIPIFY_DIR}/src/transport/net_telemetry.cc
  PROPERTIES GENERATED TRUE)
rccl_add_micro_binary(rccl-UnitTestsMicroNetIbCastInit
  SOURCES ${TEST_MICRO_NET_IB_CAST_INIT_SOURCE_FILES}
  EXTRA_INCLUDES
    ${RCCL_HOST_HIPIFY_DIR}/src/transport/net_ib_cast  # common_cast.h and the other cast headers
    ${RCCL_HOST_HIPIFY_DIR}/src/transport              # net_ib_limits.h, net_ib_gid_inspect.h
  EXTRA_DEFS
    ${_MICRO_MODE_DEFS}
    ${_MICRO_AMDSMI_DEF}
    ENABLE_FAULT_INJECTION  # net_ib_cast_gid_inspect.cc and the GID-change hook exist only with it
    # init, common, reg, gdr and gin share basenames with net_ib/ (and init.cc with src/), which
    # src/CMakeLists.txt lists first, so add_file_unique() renames these copies.
    NET_IB_CAST_INIT_CC_PATH="${RCCL_HOST_HIPIFY_DIR}/src/transport/net_ib_cast/init_tmp.cc"
    NET_IB_CAST_COMMON_CC_PATH="${RCCL_HOST_HIPIFY_DIR}/src/transport/net_ib_cast/common_tmp.cc"
    NET_IB_CAST_REG_CC_PATH="${RCCL_HOST_HIPIFY_DIR}/src/transport/net_ib_cast/reg_tmp.cc"
    NET_IB_CAST_GDR_CC_PATH="${RCCL_HOST_HIPIFY_DIR}/src/transport/net_ib_cast/gdr_tmp.cc"
    NET_IB_CAST_GIN_CC_PATH="${RCCL_HOST_HIPIFY_DIR}/src/transport/net_ib_cast/gin_tmp.cc"
    NET_IB_CAST_CAPABILITY_CC_PATH="${RCCL_HOST_HIPIFY_DIR}/src/transport/net_ib_cast/capability.cc"
    NET_IB_CAST_MULTIPLANE_CC_PATH="${RCCL_HOST_HIPIFY_DIR}/src/transport/net_ib_cast/multiplane.cc"
    NET_IB_CAST_GID_INSPECT_CC_PATH="${RCCL_HOST_HIPIFY_DIR}/src/transport/net_ib_cast/net_ib_cast_gid_inspect.cc"
  CATEGORIES_YAML "${CMAKE_CURRENT_SOURCE_DIR}/../test_categories_micro_net_ib_cast_init.yaml")

# ---------------------------------------------------------------------------
# rccl-UnitTestsMicroNet
# ---------------------------------------------------------------------------
rccl_add_micro_binary(rccl-UnitTestsMicroNet
  SOURCES ${TEST_MICRO_NET_SOURCE_FILES}
  EXTRA_INCLUDES
    ${RCCL_HOST_HIPIFY_DIR}/src/transport/net_ib_cast  # the net_ib_cast fakes' headers
    ${RCCL_HOST_HIPIFY_DIR}/src/transport              # net_ib_limits.h
  EXTRA_DEFS
    ${_MICRO_MODE_DEFS}
    ${_MICRO_DMABUF_EXPORT_DEF}  # standalone: the cuMem DMA-BUF arms in-build gets from rccl
    # net_tmp.cc, not net.cc: src/CMakeLists.txt lists plugin/net.cc first, so
    # add_file_unique() renames this one. Re-check after any source-list reorder.
    NET_CC_PATH="${RCCL_HOST_HIPIFY_DIR}/src/transport/net_tmp.cc"
  CATEGORIES_YAML "${CMAKE_CURRENT_SOURCE_DIR}/../test_categories_micro_net.yaml")

# ---------------------------------------------------------------------------
# rccl-UnitTestsMicroMisc
# ---------------------------------------------------------------------------
rccl_add_micro_binary(rccl-UnitTestsMicroMisc
  SOURCES ${TEST_MICRO_MISC_SOURCE_FILES}
  REMOVE_DEFS ENABLE_QP_TRACKING  # its create/destroy QP prologue dereferences the fake QP and PD
  EXTRA_INCLUDES
    ${RCCL_HOST_HIPIFY_DIR}/src/transport/net_ib_cast  # net_ib_ops_fault.h, included unqualified
  EXTRA_DEFS
    ${_MICRO_MODE_DEFS}
    ENABLE_FAULT_INJECTION  # FAULT_INJECTION defaults ON, so this is the ibvwrap.cc arm that ships
    GDR_PEERMEM_CC_PATH="${RCCL_HOST_HIPIFY_DIR}/src/misc/gdr_peermem.cc"
    IBVSYMBOLS_CC_PATH="${RCCL_HOST_HIPIFY_DIR}/src/misc/ibvsymbols.cc"
    IBVWRAP_CC_PATH="${RCCL_HOST_HIPIFY_DIR}/src/misc/ibvwrap.cc"
    IONICDVSYMBOLS_CC_PATH="${RCCL_HOST_HIPIFY_DIR}/src/misc/ionicdvsymbols.cc"
    IONICDVWRAP_CC_PATH="${RCCL_HOST_HIPIFY_DIR}/src/misc/ionicdvwrap.cc"
    MLX5DVSYMBOLS_CC_PATH="${RCCL_HOST_HIPIFY_DIR}/src/misc/mlx5dvsymbols.cc"
    MLX5DVWRAP_CC_PATH="${RCCL_HOST_HIPIFY_DIR}/src/misc/mlx5dvwrap.cc"
  CATEGORIES_YAML "${CMAKE_CURRENT_SOURCE_DIR}/../test_categories_micro_misc.yaml")

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
# Standalone-only tail: coverage + no CTest registration.
# =====================================================================
if(BUILD_TESTS)
  # In-RCCL-build path registers these binaries via CTest categories
  # (rccl_register_test_categories, called inside rccl_add_micro_binary).
  return()
endif()

rccl_standalone_coverage()
