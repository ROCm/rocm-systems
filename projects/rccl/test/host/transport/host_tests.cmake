# ---------------------------------------------------------------------------
# src/transport/: one binary per test file, each linking only the fakes its unit needs.
# ---------------------------------------------------------------------------
set_source_files_properties(
  ${RCCL_HOST_HIPIFY_DIR}/src/graph/xml.cc
  ${RCCL_HOST_HIPIFY_DIR}/src/misc/archinfo.cc
  ${RCCL_HOST_HIPIFY_DIR}/src/misc/utils.cc
  ${RCCL_HOST_HIPIFY_DIR}/src/transport/net_telemetry.cc
  PROPERTIES GENERATED TRUE)
# Real sibling units a test file reaches. They read their params through g_loadParam (declared in nccl_fakes.h), as
# the unit under test does.
set(_MICRO_TRANSPORT_SIBLINGS
  ${RCCL_HOST_HIPIFY_DIR}/src/transport/net_ib/common.cc
  ${RCCL_HOST_HIPIFY_DIR}/src/transport/net_ib/connect_tmp.cc
  ${RCCL_HOST_HIPIFY_DIR}/src/transport/net_ib/gdr.cc
  ${RCCL_HOST_HIPIFY_DIR}/src/transport/net_ib/init_tmp.cc
  ${RCCL_HOST_HIPIFY_DIR}/src/transport/net_ib/p2p_tmp.cc
  ${RCCL_HOST_HIPIFY_DIR}/src/transport/net_ib/reg.cc
  ${RCCL_HOST_HIPIFY_DIR}/src/transport/net_ib_cast/capability.cc
  ${RCCL_HOST_HIPIFY_DIR}/src/transport/net_ib_cast/common_tmp.cc
  ${RCCL_HOST_HIPIFY_DIR}/src/transport/net_ib_cast/gdr_tmp.cc
  ${RCCL_HOST_HIPIFY_DIR}/src/transport/net_ib_cast/init_tmp.cc
  ${RCCL_HOST_HIPIFY_DIR}/src/transport/net_ib_cast/net_ib_ops_fault.cc
  ${RCCL_HOST_HIPIFY_DIR}/src/transport/net_ib_cast/reg_tmp.cc
  ${RCCL_HOST_HIPIFY_DIR}/src/transport/net_ib_cast/scheduler.cc
  ${RCCL_HOST_HIPIFY_DIR}/src/transport/rma_socket.cc
  ${RCCL_HOST_HIPIFY_DIR}/src/transport/rma_socket_mem.cc
  ${RCCL_HOST_HIPIFY_DIR}/src/transport/rma_socket_progress_recver.cc
  ${RCCL_HOST_HIPIFY_DIR}/src/transport/rma_socket_setup.cc)
set_source_files_properties(${_MICRO_TRANSPORT_SIBLINGS} PROPERTIES
  GENERATED TRUE
  COMPILE_OPTIONS "-include;fakes/nccl_fakes.h;-include;fakes/param_redirect.h")
# Their own tests compile these two behind gdrwrap_seam.h, which must follow gdrwrap.h.
set_property(SOURCE
  ${RCCL_HOST_HIPIFY_DIR}/src/transport/rma_socket_mem.cc
  ${RCCL_HOST_HIPIFY_DIR}/src/transport/rma_socket_progress_recver.cc
  APPEND PROPERTY COMPILE_OPTIONS "-include;gdrwrap.h;-include;fakes/gdrwrap_seam.h")
set(_MICRO_TRANSPORT_DEFS
  ${_MICRO_DMABUF_EXPORT_DEF}  # standalone: the cuMem DMA-BUF arms in-build gets from rccl
  ${_MICRO_AMDSMI_DEF}         # graph/xml.cc
  # Shipping builds set it; it gates whole TUs and net_ib_cast's ncclIbNetCommBase fields.
  ENABLE_FAULT_INJECTION
  COLL_NET_CC_PATH="${RCCL_HOST_HIPIFY_DIR}/src/transport/coll_net.cc"
  GENERIC_CC_PATH="${RCCL_HOST_HIPIFY_DIR}/src/transport/generic.cc"
  NET_IB_COMMON_CC_PATH="${RCCL_HOST_HIPIFY_DIR}/src/transport/net_ib/common.cc"
  # graph/connect.cc and diagnostics/p2p.cc keep the original basenames.
  NET_IB_CONNECT_CC_PATH="${RCCL_HOST_HIPIFY_DIR}/src/transport/net_ib/connect_tmp.cc"
  NET_IB_GDR_CC_PATH="${RCCL_HOST_HIPIFY_DIR}/src/transport/net_ib/gdr.cc"
  NET_IB_GIN_CC_PATH="${RCCL_HOST_HIPIFY_DIR}/src/transport/net_ib/gin_tmp.cc"
  # Hipify renames this to *_tmp.cc: src/init.cc keeps the basename.
  NET_IB_INIT_CC_PATH="${RCCL_HOST_HIPIFY_DIR}/src/transport/net_ib/init_tmp.cc"
  NET_IB_GID_INSPECT_CC_PATH="${RCCL_HOST_HIPIFY_DIR}/src/transport/net_ib/net_ib_gid_inspect.cc"
  NET_IB_P2P_CC_PATH="${RCCL_HOST_HIPIFY_DIR}/src/transport/net_ib/p2p_tmp.cc"
  NET_IB_P2P_RESILIENCY_CC_PATH="${RCCL_HOST_HIPIFY_DIR}/src/transport/net_ib/p2p_resiliency.cc"
  NET_IB_P2P_RESILIENCY_RECOVERY_CC_PATH="${RCCL_HOST_HIPIFY_DIR}/src/transport/net_ib/p2p_resiliency_recovery.cc"
  NET_IB_REG_CC_PATH="${RCCL_HOST_HIPIFY_DIR}/src/transport/net_ib/reg.cc"
  NET_IB_WQE_LAT_MON_CC_PATH="${RCCL_HOST_HIPIFY_DIR}/src/transport/net_ib/wqe_lat_mon.cc"
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
  # p2p_tmp.cc, not p2p.cc: src/CMakeLists.txt lists transport/net_ib/p2p.cc first, so
  # add_file_unique() renames this one. Re-check after any source-list reorder.
  NET_IB_CAST_P2P_CC_PATH="${RCCL_HOST_HIPIFY_DIR}/src/transport/net_ib_cast/p2p_tmp.cc"
  NET_IB_CAST_QP_SHARING_CC_PATH="${RCCL_HOST_HIPIFY_DIR}/src/transport/net_ib_cast/qp_sharing.cc"
  NET_IB_CAST_SCHEDULER_CC_PATH="${RCCL_HOST_HIPIFY_DIR}/src/transport/net_ib_cast/scheduler.cc"
  NET_IB_OPS_FAULT_CC_PATH="${RCCL_HOST_HIPIFY_DIR}/src/transport/net_ib_cast/net_ib_ops_fault.cc"
  NET_SOCKET_CC_PATH="${RCCL_HOST_HIPIFY_DIR}/src/transport/net_socket.cc"
  # common_tmp.cc: src/CMakeLists.txt lists transport/net_ib/common.cc first.
  NET_SOCKET_COMMON_CC_PATH="${RCCL_HOST_HIPIFY_DIR}/src/transport/net_socket/common_tmp.cc"
  NET_TELEMETRY_CC_PATH="${RCCL_HOST_HIPIFY_DIR}/src/transport/net_telemetry.cc"
  # net_tmp.cc, not net.cc: src/CMakeLists.txt lists plugin/net.cc first, so
  # add_file_unique() renames this one. Re-check after any source-list reorder.
  NET_CC_PATH="${RCCL_HOST_HIPIFY_DIR}/src/transport/net_tmp.cc"
  NVLS_CC_PATH="${RCCL_HOST_HIPIFY_DIR}/src/transport/nvls.cc"
  P2P_CC_PATH="${RCCL_HOST_HIPIFY_DIR}/src/transport/p2p_tmp.cc"
  RMA_SOCKET_CC_PATH="${RCCL_HOST_HIPIFY_DIR}/src/transport/rma_socket.cc"
  RMA_SOCKET_MEM_CC_PATH="${RCCL_HOST_HIPIFY_DIR}/src/transport/rma_socket_mem.cc"
  RMA_SOCKET_PROGRESS_RECVER_CC_PATH="${RCCL_HOST_HIPIFY_DIR}/src/transport/rma_socket_progress_recver.cc"
  RMA_SOCKET_SETUP_CC_PATH="${RCCL_HOST_HIPIFY_DIR}/src/transport/rma_socket_setup.cc")

# One src/transport/ test file per binary; see rccl_add_unit_micro_binary().
macro(rccl_add_transport_micro_binary _name _test)
  rccl_add_unit_micro_binary(${_name} transport/${_test} ${ARGN}
    OBJECT_PREFIX rccl-transport-obj
    COMMON_INCLUDES
      ${RCCL_HOST_HIPIFY_DIR}/src/transport
      ${RCCL_HOST_HIPIFY_DIR}/src/transport/net_ib_cast  # common_cast.h and the other cast headers
    COMMON_DEFS ${_MICRO_TRANSPORT_DEFS}
    CATEGORIES_YAML "${CMAKE_CURRENT_SOURCE_DIR}/../test_categories_micro_transport.yaml")
endmacro()

rccl_add_transport_micro_binary(rccl-UnitTestsMicroTransportCollNet coll-net-test.cc
  SOURCES
    fakes/bootstrap_stubs.cc
    fakes/channel_fakes.cc
    fakes/comm_fakes.cc
    fakes/env_fakes.cc
    fakes/hip_fakes.cc
    fakes/hsa_fakes.cc
    fakes/nccl_fakes.cc
    fakes/os_fakes.cc
    fakes/rocmwrap_fakes.cc
    fakes/topo_stubs.cc
    fakes/transport_stubs.cc
    ${RCCL_HOST_HIPIFY_DIR}/src/misc/utils.cc
  EXTRA_DEFS
    # transport_stubs.cc would define coll_net.cc a second time
    RCCL_TRANSPORT_STUBS_OMIT_COLL_NET)

# No shared fakes: transport_stubs.cc defines generic.cc's connect functions, so the test TU
# fakes the transport calls generic.cc makes.
rccl_add_transport_micro_binary(rccl-UnitTestsMicroTransportGeneric generic-test.cc)

rccl_add_transport_micro_binary(rccl-UnitTestsMicroTransportNetSocket net-socket-test.cc
  SOURCES
    fakes/env_fakes.cc
    fakes/hip_fakes.cc
    fakes/libc_fakes.cc
    fakes/nccl_fakes.cc
    fakes/os_fakes.cc
    fakes/socket_fakes.cc)

rccl_add_transport_micro_binary(rccl-UnitTestsMicroTransportNetTelemetry net-telemetry-test.cc
  PROCESS_ISOLATED
  SOURCES
    fakes/env_fakes.cc
    fakes/hip_fakes.cc                   # hipGetDeviceCount, for the runner's GPU-pool probe
    fakes/libc_fakes.cc)

rccl_add_transport_micro_binary(rccl-UnitTestsMicroTransportNet net-test.cc
  SOURCES
    fakes/collectives_fakes.cc
    fakes/env_fakes.cc
    fakes/hip_fakes.cc
    fakes/hsa_fakes.cc
    fakes/nccl_fakes.cc
    fakes/nccl_stubs.cc
    fakes/net_ib_cast_common_fakes.cc
    fakes/net_ib_cast_connect_fakes.cc
    fakes/os_fakes.cc
    fakes/proxy_fakes.cc
    fakes/rccl_wrap_fakes.cc
    fakes/rocmwrap_fakes.cc
    fakes/topo_stubs.cc
    fakes/transport_net_ib_common_fakes.cc
    fakes/transport_net_ib_connect_net_fakes.cc
    fakes/transport_net_ib_fakes.cc
    fakes/transport_p2p_fakes.cc
    ${RCCL_HOST_HIPIFY_DIR}/src/misc/archinfo.cc
    ${RCCL_HOST_HIPIFY_DIR}/src/misc/utils.cc)

rccl_add_transport_micro_binary(rccl-UnitTestsMicroTransportNetIbCommon net_ib/common-test.cc
  SOURCES
    fakes/env_fakes.cc
    fakes/gdr_peermem_fakes.cc
    fakes/gdr_probe_fakes.cc
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
    ${RCCL_HOST_HIPIFY_DIR}/src/misc/utils.cc
    ${RCCL_HOST_HIPIFY_DIR}/src/transport/net_ib/gdr.cc
    ${RCCL_HOST_HIPIFY_DIR}/src/transport/net_ib/init_tmp.cc
    ${RCCL_HOST_HIPIFY_DIR}/src/transport/net_ib/reg.cc)

rccl_add_transport_micro_binary(rccl-UnitTestsMicroTransportNetIbConnect net_ib/connect-test.cc
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
    fakes/os_fakes.cc
    fakes/rocmwrap_fakes.cc
    fakes/socket_fakes.cc
    fakes/transport_net_ib_p2p_fakes.cc
    fakes/transport_net_ib_recovery_fakes.cc
    fakes/transport_net_ib_resiliency_fakes.cc
    fakes/transport_net_ib_wqe_lat_fakes.cc
    ${RCCL_HOST_HIPIFY_DIR}/src/misc/utils.cc
    ${RCCL_HOST_HIPIFY_DIR}/src/transport/net_ib/common.cc
    ${RCCL_HOST_HIPIFY_DIR}/src/transport/net_ib/gdr.cc
    ${RCCL_HOST_HIPIFY_DIR}/src/transport/net_ib/init_tmp.cc
    ${RCCL_HOST_HIPIFY_DIR}/src/transport/net_ib/reg.cc)

rccl_add_transport_micro_binary(rccl-UnitTestsMicroTransportNetIbGdr net_ib/gdr-test.cc
  SOURCES
    fakes/env_fakes.cc
    fakes/gdr_peermem_fakes.cc
    fakes/gdr_probe_fakes.cc
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
    ${RCCL_HOST_HIPIFY_DIR}/src/misc/utils.cc
    ${RCCL_HOST_HIPIFY_DIR}/src/transport/net_ib/common.cc
    ${RCCL_HOST_HIPIFY_DIR}/src/transport/net_ib/init_tmp.cc
    ${RCCL_HOST_HIPIFY_DIR}/src/transport/net_ib/reg.cc)

rccl_add_transport_micro_binary(rccl-UnitTestsMicroTransportNetIbGin net_ib/gin-test.cc
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
    fakes/os_fakes.cc
    fakes/rocmwrap_fakes.cc
    fakes/socket_fakes.cc
    fakes/transport_net_ib_recovery_fakes.cc
    fakes/transport_net_ib_resiliency_fakes.cc
    fakes/transport_net_ib_wqe_lat_fakes.cc
    ${RCCL_HOST_HIPIFY_DIR}/src/misc/utils.cc
    ${RCCL_HOST_HIPIFY_DIR}/src/transport/net_ib/common.cc
    ${RCCL_HOST_HIPIFY_DIR}/src/transport/net_ib/connect_tmp.cc
    ${RCCL_HOST_HIPIFY_DIR}/src/transport/net_ib/gdr.cc
    ${RCCL_HOST_HIPIFY_DIR}/src/transport/net_ib/init_tmp.cc
    ${RCCL_HOST_HIPIFY_DIR}/src/transport/net_ib/p2p_tmp.cc
    ${RCCL_HOST_HIPIFY_DIR}/src/transport/net_ib/reg.cc)

rccl_add_transport_micro_binary(rccl-UnitTestsMicroTransportNetIbInit net_ib/init-test.cc
  SOURCES
    fakes/env_fakes.cc
    fakes/hip_fakes.cc
    fakes/ibvwrap_fakes.cc
    fakes/mlx5dvwrap_fakes.cc
    fakes/nccl_fakes.cc
    fakes/os_fakes.cc
    fakes/socket_fakes.cc
    fakes/transport_net_ib_fakes.cc
    fakes/transport_net_ib_p2p_fakes.cc
    fakes/transport_net_ib_recovery_fakes.cc
    fakes/transport_net_ib_resiliency_fakes.cc
    ${RCCL_HOST_HIPIFY_DIR}/src/misc/utils.cc
    ${RCCL_HOST_HIPIFY_DIR}/src/transport/net_ib/common.cc
    ${RCCL_HOST_HIPIFY_DIR}/src/transport/net_ib/reg.cc)

rccl_add_transport_micro_binary(rccl-UnitTestsMicroTransportNetIbGidInspect net_ib/net-ib-gid-inspect-test.cc
  SOURCES
    fakes/env_fakes.cc
    fakes/gdr_peermem_fakes.cc
    fakes/gdr_probe_fakes.cc
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
    ${RCCL_HOST_HIPIFY_DIR}/src/misc/utils.cc
    ${RCCL_HOST_HIPIFY_DIR}/src/transport/net_ib/common.cc
    ${RCCL_HOST_HIPIFY_DIR}/src/transport/net_ib/gdr.cc
    ${RCCL_HOST_HIPIFY_DIR}/src/transport/net_ib/init_tmp.cc
    ${RCCL_HOST_HIPIFY_DIR}/src/transport/net_ib/reg.cc)

rccl_add_transport_micro_binary(rccl-UnitTestsMicroTransportNetIbP2pResiliencyRecovery
  net_ib/p2p-resiliency-recovery-test.cc
  SOURCES
    fakes/env_fakes.cc
    fakes/gdr_peermem_fakes.cc
    fakes/gdr_probe_fakes.cc
    fakes/hip_fakes.cc
    fakes/ibvwrap_fakes.cc
    fakes/mlx5dvwrap_fakes.cc
    fakes/nccl_fakes.cc
    fakes/os_fakes.cc
    fakes/rocmwrap_fakes.cc
    fakes/socket_fakes.cc
    fakes/transport_net_ib_fakes.cc
    fakes/transport_net_ib_p2p_fakes.cc
    fakes/transport_net_ib_resiliency_fakes.cc
    ${RCCL_HOST_HIPIFY_DIR}/src/misc/utils.cc
    ${RCCL_HOST_HIPIFY_DIR}/src/transport/net_ib/common.cc
    ${RCCL_HOST_HIPIFY_DIR}/src/transport/net_ib/gdr.cc
    ${RCCL_HOST_HIPIFY_DIR}/src/transport/net_ib/init_tmp.cc
    ${RCCL_HOST_HIPIFY_DIR}/src/transport/net_ib/reg.cc)

rccl_add_transport_micro_binary(rccl-UnitTestsMicroTransportNetIbP2pResiliency net_ib/p2p-resiliency-test.cc
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
    fakes/os_fakes.cc
    fakes/rocmwrap_fakes.cc
    fakes/socket_fakes.cc
    fakes/transport_net_ib_recovery_fakes.cc
    fakes/transport_net_ib_wqe_lat_fakes.cc
    ${RCCL_HOST_HIPIFY_DIR}/src/misc/utils.cc
    ${RCCL_HOST_HIPIFY_DIR}/src/transport/net_ib/common.cc
    ${RCCL_HOST_HIPIFY_DIR}/src/transport/net_ib/connect_tmp.cc
    ${RCCL_HOST_HIPIFY_DIR}/src/transport/net_ib/gdr.cc
    ${RCCL_HOST_HIPIFY_DIR}/src/transport/net_ib/init_tmp.cc
    ${RCCL_HOST_HIPIFY_DIR}/src/transport/net_ib/p2p_tmp.cc
    ${RCCL_HOST_HIPIFY_DIR}/src/transport/net_ib/reg.cc)

rccl_add_transport_micro_binary(rccl-UnitTestsMicroTransportNetIbP2p net_ib/net-ib-p2p-test.cc
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
    fakes/os_fakes.cc
    fakes/rocmwrap_fakes.cc
    fakes/socket_fakes.cc
    fakes/transport_net_ib_recovery_fakes.cc
    fakes/transport_net_ib_resiliency_fakes.cc
    fakes/transport_net_ib_wqe_lat_fakes.cc
    ${RCCL_HOST_HIPIFY_DIR}/src/misc/utils.cc
    ${RCCL_HOST_HIPIFY_DIR}/src/transport/net_ib/common.cc
    ${RCCL_HOST_HIPIFY_DIR}/src/transport/net_ib/connect_tmp.cc
    ${RCCL_HOST_HIPIFY_DIR}/src/transport/net_ib/gdr.cc
    ${RCCL_HOST_HIPIFY_DIR}/src/transport/net_ib/init_tmp.cc
    ${RCCL_HOST_HIPIFY_DIR}/src/transport/net_ib/reg.cc)

rccl_add_transport_micro_binary(rccl-UnitTestsMicroTransportNetIbReg net_ib/reg-test.cc
  SOURCES
    fakes/env_fakes.cc
    fakes/gdr_peermem_fakes.cc
    fakes/gdr_probe_fakes.cc
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
    ${RCCL_HOST_HIPIFY_DIR}/src/misc/utils.cc
    ${RCCL_HOST_HIPIFY_DIR}/src/transport/net_ib/common.cc
    ${RCCL_HOST_HIPIFY_DIR}/src/transport/net_ib/gdr.cc
    ${RCCL_HOST_HIPIFY_DIR}/src/transport/net_ib/init_tmp.cc)

rccl_add_transport_micro_binary(rccl-UnitTestsMicroTransportNetIbWqeLatMon net_ib/wqe-lat-mon-test.cc
  SOURCES
    fakes/env_fakes.cc
    fakes/gdr_peermem_fakes.cc
    fakes/gdr_probe_fakes.cc
    fakes/hip_fakes.cc
    fakes/ibvwrap_fakes.cc
    fakes/libc_fakes.cc
    fakes/mlx5dvwrap_fakes.cc
    fakes/nccl_fakes.cc
    fakes/os_fakes.cc
    fakes/rocmwrap_fakes.cc
    fakes/socket_fakes.cc
    fakes/transport_net_ib_fakes.cc
    fakes/transport_net_ib_p2p_fakes.cc
    fakes/transport_net_ib_recovery_fakes.cc
    fakes/transport_net_ib_resiliency_fakes.cc
    ${RCCL_HOST_HIPIFY_DIR}/src/misc/utils.cc
    ${RCCL_HOST_HIPIFY_DIR}/src/transport/net_ib/common.cc
    ${RCCL_HOST_HIPIFY_DIR}/src/transport/net_ib/gdr.cc
    ${RCCL_HOST_HIPIFY_DIR}/src/transport/net_ib/init_tmp.cc
    ${RCCL_HOST_HIPIFY_DIR}/src/transport/net_ib/reg.cc)

rccl_add_transport_micro_binary(rccl-UnitTestsMicroTransportNetIbCastCapability
  net_ib_cast/net-ib-cast-capability-test.cc
  SOURCES
    fakes/env_fakes.cc
    fakes/ibvwrap_fakes.cc
    fakes/nccl_fakes.cc
    fakes/net_ib_cast_common_fakes.cc
    fakes/net_ib_cast_connect_fakes.cc
    fakes/net_ib_cast_init_fakes.cc
    fakes/net_ib_cast_p2p_fakes.cc
    fakes/net_ib_cast_p2p_resiliency_fakes.cc
    fakes/net_ib_cast_p2p_resiliency_recovery_fakes.cc
    fakes/net_ib_cast_qp_sharing_fakes.cc
    fakes/net_ib_cast_scheduler_fakes.cc)

rccl_add_transport_micro_binary(rccl-UnitTestsMicroTransportNetIbCastCommon net_ib_cast/net-ib-cast-common-test.cc
  SOURCES
    fakes/env_fakes.cc
    fakes/gdr_peermem_fakes.cc
    fakes/gdr_probe_fakes.cc
    fakes/hip_fakes.cc
    fakes/ibvwrap_fakes.cc
    fakes/ionicdvwrap_fakes.cc
    fakes/mlx5dvwrap_fakes.cc
    fakes/nccl_fakes.cc
    fakes/net_ib_cast_connect_fakes.cc
    fakes/net_ib_cast_p2p_fakes.cc
    fakes/net_ib_cast_p2p_resiliency_fakes.cc
    fakes/net_ib_cast_p2p_resiliency_recovery_fakes.cc
    fakes/net_ib_cast_qp_sharing_fakes.cc
    fakes/net_ib_cast_scheduler_fakes.cc
    fakes/os_fakes.cc
    fakes/rocmwrap_fakes.cc
    fakes/socket_fakes.cc
    fakes/transport_stubs.cc
    ${RCCL_HOST_HIPIFY_DIR}/src/misc/archinfo.cc
    ${RCCL_HOST_HIPIFY_DIR}/src/misc/utils.cc
    ${RCCL_HOST_HIPIFY_DIR}/src/transport/net_ib_cast/capability.cc
    ${RCCL_HOST_HIPIFY_DIR}/src/transport/net_ib_cast/gdr_tmp.cc
    ${RCCL_HOST_HIPIFY_DIR}/src/transport/net_ib_cast/init_tmp.cc
    ${RCCL_HOST_HIPIFY_DIR}/src/transport/net_ib_cast/reg_tmp.cc
    ${RCCL_HOST_HIPIFY_DIR}/src/transport/net_telemetry.cc)

rccl_add_transport_micro_binary(rccl-UnitTestsMicroTransportNetIbCastGdr net_ib_cast/net-ib-cast-gdr-test.cc
  PROCESS_ISOLATED
  SOURCES
    fakes/env_fakes.cc
    fakes/gdr_peermem_fakes.cc
    fakes/gdr_probe_fakes.cc
    fakes/hip_fakes.cc
    fakes/ibvwrap_fakes.cc
    fakes/ionicdvwrap_fakes.cc
    fakes/libc_fakes.cc
    fakes/mlx5dvwrap_fakes.cc
    fakes/nccl_fakes.cc
    fakes/net_ib_cast_common_fakes.cc
    fakes/net_ib_cast_connect_fakes.cc
    fakes/net_ib_cast_p2p_fakes.cc
    fakes/net_ib_cast_p2p_resiliency_fakes.cc
    fakes/net_ib_cast_p2p_resiliency_recovery_fakes.cc
    fakes/net_ib_cast_qp_sharing_fakes.cc
    fakes/net_ib_cast_scheduler_fakes.cc
    fakes/os_fakes.cc
    fakes/rocmwrap_fakes.cc
    fakes/socket_fakes.cc
    fakes/transport_stubs.cc
    ${RCCL_HOST_HIPIFY_DIR}/src/misc/archinfo.cc
    ${RCCL_HOST_HIPIFY_DIR}/src/misc/utils.cc
    ${RCCL_HOST_HIPIFY_DIR}/src/transport/net_ib_cast/capability.cc
    ${RCCL_HOST_HIPIFY_DIR}/src/transport/net_ib_cast/init_tmp.cc
    ${RCCL_HOST_HIPIFY_DIR}/src/transport/net_telemetry.cc)

rccl_add_transport_micro_binary(rccl-UnitTestsMicroTransportNetIbCastGidInspect
  net_ib_cast/net-ib-cast-gid-inspect-test.cc
  SOURCES
    fakes/env_fakes.cc
    fakes/ibvwrap_fakes.cc
    fakes/nccl_fakes.cc
    fakes/net_ib_cast_common_fakes.cc
    fakes/net_ib_cast_init_fakes.cc
    fakes/net_ib_cast_p2p_fakes.cc
    fakes/net_ib_cast_qp_sharing_fakes.cc
    fakes/net_ib_cast_scheduler_fakes.cc)

rccl_add_transport_micro_binary(rccl-UnitTestsMicroTransportNetIbCastGin net_ib_cast/net-ib-cast-gin-test.cc
  PROCESS_ISOLATED
  SOURCES
    fakes/env_fakes.cc
    fakes/gdr_peermem_fakes.cc
    fakes/gdr_probe_fakes.cc
    fakes/hip_fakes.cc
    fakes/ibvwrap_fakes.cc
    fakes/ionicdvwrap_fakes.cc
    fakes/mlx5dvwrap_fakes.cc
    fakes/nccl_fakes.cc
    fakes/net_ib_cast_connect_fakes.cc
    fakes/net_ib_cast_p2p_fakes.cc
    fakes/net_ib_cast_p2p_resiliency_fakes.cc
    fakes/net_ib_cast_p2p_resiliency_recovery_fakes.cc
    fakes/net_ib_cast_qp_sharing_fakes.cc
    fakes/net_ib_cast_scheduler_fakes.cc
    fakes/os_fakes.cc
    fakes/rocmwrap_fakes.cc
    fakes/socket_fakes.cc
    fakes/transport_stubs.cc
    ${RCCL_HOST_HIPIFY_DIR}/src/misc/archinfo.cc
    ${RCCL_HOST_HIPIFY_DIR}/src/misc/utils.cc
    ${RCCL_HOST_HIPIFY_DIR}/src/transport/net_ib_cast/capability.cc
    ${RCCL_HOST_HIPIFY_DIR}/src/transport/net_ib_cast/common_tmp.cc
    ${RCCL_HOST_HIPIFY_DIR}/src/transport/net_ib_cast/gdr_tmp.cc
    ${RCCL_HOST_HIPIFY_DIR}/src/transport/net_ib_cast/reg_tmp.cc
    ${RCCL_HOST_HIPIFY_DIR}/src/transport/net_telemetry.cc)

rccl_add_transport_micro_binary(rccl-UnitTestsMicroTransportNetIbCastInit net_ib_cast/net-ib-cast-init-test.cc
  PROCESS_ISOLATED
  SOURCES
    fakes/env_fakes.cc
    fakes/gdr_peermem_fakes.cc
    fakes/gdr_probe_fakes.cc
    fakes/hip_fakes.cc
    fakes/ibvwrap_fakes.cc
    fakes/ionicdvwrap_fakes.cc
    fakes/mlx5dvwrap_fakes.cc
    fakes/nccl_fakes.cc
    fakes/net_ib_cast_connect_fakes.cc
    fakes/net_ib_cast_p2p_fakes.cc
    fakes/net_ib_cast_p2p_resiliency_fakes.cc
    fakes/net_ib_cast_p2p_resiliency_recovery_fakes.cc
    fakes/net_ib_cast_qp_sharing_fakes.cc
    fakes/net_ib_cast_scheduler_fakes.cc
    fakes/os_fakes.cc
    fakes/rocmwrap_fakes.cc
    fakes/socket_fakes.cc
    fakes/transport_stubs.cc
    ${RCCL_HOST_HIPIFY_DIR}/src/misc/archinfo.cc
    ${RCCL_HOST_HIPIFY_DIR}/src/misc/utils.cc
    ${RCCL_HOST_HIPIFY_DIR}/src/transport/net_ib_cast/capability.cc
    ${RCCL_HOST_HIPIFY_DIR}/src/transport/net_ib_cast/common_tmp.cc
    ${RCCL_HOST_HIPIFY_DIR}/src/transport/net_ib_cast/gdr_tmp.cc
    ${RCCL_HOST_HIPIFY_DIR}/src/transport/net_ib_cast/reg_tmp.cc
    ${RCCL_HOST_HIPIFY_DIR}/src/transport/net_telemetry.cc)

rccl_add_transport_micro_binary(rccl-UnitTestsMicroTransportNetIbCastMultiplane
  net_ib_cast/net-ib-cast-multiplane-test.cc
  SOURCES
    fakes/env_fakes.cc
    fakes/hip_fakes.cc
    fakes/nccl_fakes.cc
    fakes/net_ib_cast_init_fakes.cc
    fakes/os_fakes.cc
    ${RCCL_HOST_HIPIFY_DIR}/src/graph/xml.cc
    ${RCCL_HOST_HIPIFY_DIR}/src/misc/archinfo.cc
    ${RCCL_HOST_HIPIFY_DIR}/src/misc/utils.cc)

rccl_add_transport_micro_binary(rccl-UnitTestsMicroTransportNetIbCastP2p net_ib_cast/net-ib-cast-p2p-test.cc
  SOURCES
    fakes/env_fakes.cc
    fakes/hip_fakes.cc
    fakes/ibvwrap_fakes.cc
    fakes/nccl_fakes.cc
    fakes/net_ib_cast_common_fakes.cc
    fakes/net_ib_cast_connect_fakes.cc
    fakes/net_ib_cast_init_fakes.cc
    fakes/net_ib_cast_p2p_resiliency_fakes.cc
    fakes/os_fakes.cc
    fakes/socket_fakes.cc
    fakes/transport_stubs.cc
    ${RCCL_HOST_HIPIFY_DIR}/src/misc/utils.cc
    ${RCCL_HOST_HIPIFY_DIR}/src/transport/net_ib_cast/net_ib_ops_fault.cc
    ${RCCL_HOST_HIPIFY_DIR}/src/transport/net_ib_cast/scheduler.cc
    ${RCCL_HOST_HIPIFY_DIR}/src/transport/net_telemetry.cc)

rccl_add_transport_micro_binary(rccl-UnitTestsMicroTransportNetIbCastReg net_ib_cast/net-ib-cast-reg-test.cc
  SOURCES
    fakes/env_fakes.cc
    fakes/ibvwrap_fakes.cc
    fakes/libc_fakes.cc
    fakes/mlx5dvwrap_fakes.cc
    fakes/nccl_fakes.cc
    fakes/net_ib_cast_common_fakes.cc
    fakes/net_ib_cast_init_fakes.cc
    fakes/net_ib_cast_p2p_fakes.cc
    fakes/net_ib_cast_qp_sharing_fakes.cc
    fakes/net_ib_cast_scheduler_fakes.cc)

rccl_add_transport_micro_binary(rccl-UnitTestsMicroTransportNetIbCastScheduler net_ib_cast/net-ib-cast-scheduler-test.cc
  SOURCES
    fakes/env_fakes.cc
    fakes/ibvwrap_fakes.cc
    fakes/libc_fakes.cc
    fakes/nccl_fakes.cc
    fakes/net_ib_cast_common_fakes.cc
    fakes/net_ib_cast_connect_fakes.cc
    fakes/transport_stubs.cc)

rccl_add_transport_micro_binary(rccl-UnitTestsMicroTransportNetIbCastOpsFault net_ib_cast/net-ib-ops-fault-test.cc
  SOURCES
    fakes/env_fakes.cc
    fakes/nccl_fakes.cc)

rccl_add_transport_micro_binary(rccl-UnitTestsMicroTransportNvls nvls-test.cc
  SOURCES
    fakes/env_fakes.cc
    fakes/nccl_fakes.cc)

rccl_add_transport_micro_binary(rccl-UnitTestsMicroTransportP2p p2p-test.cc
  PROCESS_ISOLATED
  SOURCES
    fakes/comm_fakes.cc
    fakes/env_fakes.cc
    fakes/hip_fakes.cc
    fakes/nccl_fakes.cc
    fakes/os_fakes.cc
    fakes/search_fakes.cc          # src/graph/search.cc ncclTopoGetLinkType seam
    ${RCCL_HOST_HIPIFY_DIR}/src/misc/archinfo.cc
    ${RCCL_HOST_HIPIFY_DIR}/src/misc/utils.cc)

rccl_add_transport_micro_binary(rccl-UnitTestsMicroTransportRmaSocketMem rma-socket-mem-test.cc
  SOURCES
    fakes/comm_fakes.cc
    fakes/env_fakes.cc
    fakes/gdrwrap_fakes.cc
    fakes/hip_fakes.cc
    fakes/nccl_fakes.cc)

rccl_add_transport_micro_binary(rccl-UnitTestsMicroTransportRmaSocketProgressRecver rma-socket-progress-recver-test.cc
  SOURCES
    fakes/comm_fakes.cc
    fakes/env_fakes.cc
    fakes/gdrwrap_fakes.cc
    fakes/hip_fakes.cc
    fakes/nccl_fakes.cc
    fakes/os_fakes.cc
    fakes/socket_fakes.cc
    fakes/transport_net_socket_common_fakes.cc
    fakes/transport_net_socket_fakes.cc
    ${RCCL_HOST_HIPIFY_DIR}/src/transport/rma_socket.cc
    ${RCCL_HOST_HIPIFY_DIR}/src/transport/rma_socket_mem.cc
    ${RCCL_HOST_HIPIFY_DIR}/src/transport/rma_socket_setup.cc)

rccl_add_transport_micro_binary(rccl-UnitTestsMicroTransportRmaSocketSetup rma-socket-setup-test.cc
  SOURCES
    fakes/comm_fakes.cc
    fakes/env_fakes.cc
    fakes/hip_fakes.cc
    fakes/nccl_fakes.cc
    fakes/os_fakes.cc
    fakes/socket_fakes.cc
    ${RCCL_HOST_HIPIFY_DIR}/src/transport/rma_socket_mem.cc)

rccl_add_transport_micro_binary(rccl-UnitTestsMicroTransportRmaSocket rma-socket-test.cc
  SOURCES
    fakes/comm_fakes.cc
    fakes/env_fakes.cc
    fakes/gdrwrap_fakes.cc
    fakes/hip_fakes.cc
    fakes/nccl_fakes.cc
    fakes/os_fakes.cc
    fakes/socket_fakes.cc
    fakes/transport_net_socket_common_fakes.cc
    fakes/transport_net_socket_fakes.cc
    ${RCCL_HOST_HIPIFY_DIR}/src/transport/rma_socket_mem.cc
    ${RCCL_HOST_HIPIFY_DIR}/src/transport/rma_socket_progress_recver.cc
    ${RCCL_HOST_HIPIFY_DIR}/src/transport/rma_socket_setup.cc)
