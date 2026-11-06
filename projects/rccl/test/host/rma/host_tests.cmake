# ---------------------------------------------------------------------------
# src/rma/: one binary per test file, each linking only the fakes its unit needs.
# ---------------------------------------------------------------------------
set(_MICRO_RMA_DEFS
  # rma_tmp.cc, not rma.cc: src/CMakeLists.txt lists plugin/rma.cc first, so
  # add_file_unique() renames this one. Re-check after any source-list reorder.
  RMA_CC_PATH="${RCCL_HOST_HIPIFY_DIR}/src/rma/rma_tmp.cc"
  # rma_ce.cc has no basename collision, so it keeps its own name.
  RMA_CE_CC_PATH="${RCCL_HOST_HIPIFY_DIR}/src/rma/rma_ce.cc"
  RMA_PROXY_CC_PATH="${RCCL_HOST_HIPIFY_DIR}/src/rma/rma_proxy.cc"
  RMA_PROXY_LAUNCH_CC_PATH="${RCCL_HOST_HIPIFY_DIR}/src/rma/rma_proxy_launch.cc"
  RMA_PROXY_PROGRESS_CC_PATH="${RCCL_HOST_HIPIFY_DIR}/src/rma/rma_proxy_progress.cc")

# One src/rma/ test file per binary; see rccl_add_unit_micro_binary().
macro(rccl_add_rma_micro_binary _name _test)
  rccl_add_unit_micro_binary(${_name} rma/${_test} ${ARGN}
    OBJECT_PREFIX rccl-rma-obj
    COMMON_DEFS ${_MICRO_RMA_DEFS}
    CATEGORIES_YAML "${CMAKE_CURRENT_SOURCE_DIR}/../test_categories_micro_rma.yaml")
endmacro()

rccl_add_rma_micro_binary(rccl-UnitTestsMicroRma rma-test.cc
  SOURCES
    fakes/dev_runtime_micro_fakes.cc
    fakes/devcomm_fakes.cc
    fakes/env_fakes.cc
    fakes/hip_fakes.cc
    fakes/mgmt_task_enq_fakes.cc
    fakes/nccl_device_core_fakes.cc
    fakes/nccl_fakes.cc
    fakes/rma_fakes.cc
    fakes/utils_fakes.cc
    ${RCCL_HOST_HIPIFY_DIR}/src/dev_runtime.cc
    ${RCCL_HOST_HIPIFY_DIR}/src/devcomm/devcomm_v22902.cc
    ${RCCL_HOST_HIPIFY_DIR}/src/devcomm/devcomm_v22907.cc
  EXTRA_DEFS
    RCCL_RMA_FAKES_OMIT_ncclRmaInitialized)  # the unit, rma.cc, defines it

rccl_add_rma_micro_binary(rccl-UnitTestsMicroRmaCe rma-ce-test.cc
  SOURCES
    fakes/argcheck_fakes.cc
    fakes/ce_fakes.cc
    fakes/comm_fakes.cc
    fakes/dev_runtime_micro_fakes.cc
    fakes/devcomm_fakes.cc
    fakes/env_fakes.cc
    fakes/hip_fakes.cc
    fakes/mgmt_task_enq_fakes.cc
    fakes/nccl_device_core_fakes.cc
    fakes/nccl_fakes.cc
    fakes/rma_fakes.cc
    fakes/rocmwrap_fakes.cc
    fakes/sym_kernels_fakes.cc
    fakes/transport_p2p_fakes.cc
    fakes/utils_fakes.cc
    ${RCCL_HOST_HIPIFY_DIR}/src/dev_runtime.cc
    ${RCCL_HOST_HIPIFY_DIR}/src/devcomm/devcomm_v22902.cc
    ${RCCL_HOST_HIPIFY_DIR}/src/devcomm/devcomm_v22907.cc
    ${RCCL_HOST_HIPIFY_DIR}/src/rccl_dev_runtime.cc
    ${RCCL_HOST_HIPIFY_DIR}/src/rma/rma_tmp.cc
  EXTRA_DEFS
    RCCL_RMA_FAKES_OMIT_ncclRmaInitialized)  # rma_tmp.cc defines it

rccl_add_rma_micro_binary(rccl-UnitTestsMicroRmaProxy rma-proxy-test.cc
  SOURCES
    fakes/hip_fakes.cc
    fakes/hsa_fakes.cc
    fakes/nccl_fakes.cc
    fakes/os_fakes.cc
    fakes/rma_fakes.cc
    fakes/rocmwrap_fakes.cc
    ${RCCL_HOST_HIPIFY_DIR}/src/misc/utils.cc)

rccl_add_rma_micro_binary(rccl-UnitTestsMicroRmaProxyLaunch rma-proxy-launch-test.cc
  SOURCES
    fakes/hip_fakes.cc
    fakes/nccl_fakes.cc
    ${RCCL_HOST_HIPIFY_DIR}/src/dev_runtime.cc)

rccl_add_rma_micro_binary(rccl-UnitTestsMicroRmaProxyProgress rma-proxy-progress-test.cc
  SOURCES
    fakes/nccl_fakes.cc
    fakes/rma_fakes.cc)
