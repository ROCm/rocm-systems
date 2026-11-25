# ---------------------------------------------------------------------------
# src/diagnostics/: one binary per test file, each linking only the fakes its unit needs.
# ---------------------------------------------------------------------------
set(_MICRO_DIAGNOSTICS_DEFS
  DIAG_IB_WRITE_BW_CC_PATH="${RCCL_HOST_HIPIFY_DIR}/src/diagnostics/ib_write_bw.cc"
  DIAG_P2P_CC_PATH="${RCCL_HOST_HIPIFY_DIR}/src/diagnostics/p2p.cc"
  DIAG_P2P_DEVICE_CU_PATH="${RCCL_HOST_HIPIFY_DIR}/src/diagnostics/device/p2p.cu.cpp")

# One src/diagnostics/ test file per binary; see rccl_add_unit_micro_binary().
macro(rccl_add_diagnostics_micro_binary _name _test)
  rccl_add_unit_micro_binary(${_name} diagnostics/${_test} ${ARGN}
    OBJECT_PREFIX rccl-diagnostics-obj
    COMMON_DEFS ${_MICRO_DIAGNOSTICS_DEFS}
    CATEGORIES_YAML "${CMAKE_CURRENT_SOURCE_DIR}/../test_categories_micro_diagnostics.yaml")
endmacro()

rccl_add_diagnostics_micro_binary(rccl-UnitTestsMicroDiagnosticsIbWriteBw diag-ib-write-bw-test.cc
  SOURCES
    fakes/dev_runtime_micro_fakes.cc
    fakes/diagnostics_fakes.cc
    fakes/env_fakes.cc
    fakes/hip_fakes.cc
    fakes/libc_fakes.cc
    fakes/mgmt_task_enq_fakes.cc
    fakes/nccl_device_core_fakes.cc
    fakes/nccl_fakes.cc
    fakes/topo_stubs.cc
    fakes/transport_net_ib_fakes.cc)

rccl_add_diagnostics_micro_binary(rccl-UnitTestsMicroDiagnosticsP2p diag-p2p-test.cc
  SOURCES
    fakes/dev_runtime_micro_fakes.cc
    fakes/diagnostics_p2p_device_fakes.cc
    fakes/env_fakes.cc
    fakes/hip_fakes.cc
    fakes/mgmt_task_enq_fakes.cc
    fakes/nccl_device_core_fakes.cc
    fakes/nccl_fakes.cc
    fakes/topo_stubs.cc
    fakes/transport_p2p_fakes.cc)

# p2p.cu's <<<>>> must not reach hip_fakes.cc's aborting hipLaunchKernel; see diag-p2p-device-test.cc.
set_source_files_properties(diagnostics/diag-p2p-device-test.cc PROPERTIES
  COMPILE_OPTIONS "-fno-hip-new-launch-api;-fuse-cuid=none")
rccl_add_diagnostics_micro_binary(rccl-UnitTestsMicroDiagnosticsP2pDevice diag-p2p-device-test.cc
  SOURCES
    fakes/env_fakes.cc
    fakes/hip_fakes.cc
    fakes/nccl_fakes.cc)
