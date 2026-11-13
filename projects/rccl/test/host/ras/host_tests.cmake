# ---------------------------------------------------------------------------
# src/ras/: one binary per test file, each linking only the fakes and real siblings its test reaches.
# ---------------------------------------------------------------------------
# fakes/ras_comms_fakes.cc copies ras.cc's communicator registry; pin both copies.
foreach(_registry_line
    "std::mutex ncclCommsMutex;"
    "struct ncclComm** ncclComms = nullptr;"
    "int nNcclComms = 0;"
    "bool ncclCommsSorted = false;")
  rccl_assert_source_line("${CMAKE_CURRENT_LIST_DIR}/../../../src/ras/ras.cc" "${_registry_line}"
    "fakes/ras_comms_fakes.cc")
  rccl_assert_source_line("${CMAKE_CURRENT_LIST_DIR}/../fakes/ras_comms_fakes.cc" "${_registry_line}"
    "the ras.cc communicator registry")
endforeach()
set_source_files_properties(
  ${RCCL_HOST_HIPIFY_DIR}/src/ras/diagnostics_checks_common.cc
  PROPERTIES GENERATED TRUE)
set(_MICRO_RAS_DEFS
  CLIENT_SUPPORT_CC_PATH="${RCCL_HOST_HIPIFY_DIR}/src/ras/client_support.cc"
  # collectives_tmp.cc, not collectives.cc: src/CMakeLists.txt lists src/collectives.cc first,
  # so add_file_unique() renames this one. Re-check after any source-list reorder.
  COLLECTIVES_CC_PATH="${RCCL_HOST_HIPIFY_DIR}/src/ras/collectives_tmp.cc"
  RAS_CC_PATH="${RCCL_HOST_HIPIFY_DIR}/src/ras/ras.cc"
  RAS_CLIENT_CC_PATH="${RCCL_HOST_HIPIFY_DIR}/src/ras/client.cc"
  RAS_DIAGNOSTICS_CC_PATH="${RCCL_HOST_HIPIFY_DIR}/src/ras/diagnostics_tmp.cc"
  RAS_DIAGNOSTICS_COMMON_CC_PATH="${RCCL_HOST_HIPIFY_DIR}/src/ras/diagnostics_checks_common.cc"
  RAS_DIAGNOSTICS_ENV_CC_PATH="${RCCL_HOST_HIPIFY_DIR}/src/ras/diagnostics_env.cc"
  RAS_DIAGNOSTICS_GPU_CC_PATH="${RCCL_HOST_HIPIFY_DIR}/src/ras/diagnostics_gpu.cc"
  RAS_DIAGNOSTICS_XID_CC_PATH="${RCCL_HOST_HIPIFY_DIR}/src/ras/diagnostics_xid.cc"
  RAS_PARAM_CC_PATH="${RCCL_HOST_HIPIFY_DIR}/src/ras/ras_param.cc"
  RAS_PEERS_CC_PATH="${RCCL_HOST_HIPIFY_DIR}/src/ras/peers.cc"
  RASNET_CC_PATH="${RCCL_HOST_HIPIFY_DIR}/src/ras/rasnet.cc")

macro(rccl_add_ras_micro_binary _name _test)
  rccl_add_unit_micro_binary(${_name} ras/${_test} ${ARGN}
    OBJECT_PREFIX rccl-ras-obj
    COMMON_DEFS ${_MICRO_RAS_DEFS}
    CATEGORIES_YAML "${CMAKE_CURRENT_SOURCE_DIR}/../test_categories_micro_ras.yaml")
endmacro()

rccl_add_ras_micro_binary(rccl-UnitTestsMicroRasClientSupport client-support-test.cc
  SOURCES
    fakes/collectives_fakes.cc
    fakes/env_fakes.cc
    fakes/hip_fakes.cc
    fakes/libc_fakes.cc
    fakes/nccl_fakes.cc
    fakes/ras_fakes.cc
    fakes/ras_param_fakes.cc)

rccl_add_ras_micro_binary(rccl-UnitTestsMicroRasClient ras-client-test.cc
  SOURCES
    fakes/env_fakes.cc
    fakes/libc_fakes.cc
    fakes/ras_param_fakes.cc)

rccl_add_ras_micro_binary(rccl-UnitTestsMicroRasCollectives ras-collectives-test.cc
  SOURCES
    fakes/env_fakes.cc
    fakes/nccl_fakes.cc)

rccl_add_ras_micro_binary(rccl-UnitTestsMicroRasDiagnosticsCommon ras-diagnostics-common-test.cc
  SOURCES
    fakes/env_fakes.cc
    fakes/nccl_fakes.cc
    fakes/ras_comms_fakes.cc)

rccl_add_ras_micro_binary(rccl-UnitTestsMicroRasDiagnosticsEnv ras-diagnostics-env-test.cc
  SOURCES
    fakes/env_fakes.cc
    fakes/nccl_fakes.cc
    fakes/ras_comms_fakes.cc
    ${RCCL_HOST_HIPIFY_DIR}/src/ras/diagnostics_checks_common.cc)

rccl_add_ras_micro_binary(rccl-UnitTestsMicroRasDiagnosticsGpu ras-diagnostics-gpu-test.cc
  SOURCES
    fakes/amdsmi_fakes.cc
    fakes/env_fakes.cc
    fakes/nccl_fakes.cc
    fakes/ras_comms_fakes.cc
    ${RCCL_HOST_HIPIFY_DIR}/src/ras/diagnostics_checks_common.cc)

rccl_add_ras_micro_binary(rccl-UnitTestsMicroRasDiagnostics ras-diagnostics-test.cc
  SOURCES
    fakes/env_fakes.cc
    fakes/nccl_fakes.cc)

rccl_add_ras_micro_binary(rccl-UnitTestsMicroRasDiagnosticsXid ras-diagnostics-xid-test.cc
  SOURCES
    fakes/nccl_fakes.cc
    ${RCCL_HOST_HIPIFY_DIR}/src/ras/diagnostics_checks_common.cc)

rccl_add_ras_micro_binary(rccl-UnitTestsMicroRasParam ras-param-test.cc
  SOURCES
    fakes/env_fakes.cc
    fakes/nccl_fakes.cc)

rccl_add_ras_micro_binary(rccl-UnitTestsMicroRasPeers ras-peers-test.cc
  SOURCES
    fakes/env_fakes.cc
    fakes/nccl_fakes.cc
    fakes/ras_client_support_fakes.cc
    fakes/ras_message_fakes.cc
    fakes/ras_net_fakes.cc)

rccl_add_ras_micro_binary(rccl-UnitTestsMicroRas ras-test.cc
  SOURCES
    fakes/env_fakes.cc
    fakes/nccl_fakes.cc
    fakes/os_fakes.cc
    fakes/ras_client_support_fakes.cc
    fakes/ras_fakes.cc
    fakes/ras_message_fakes.cc
    fakes/ras_net_fakes.cc
    fakes/ras_param_fakes.cc
    fakes/socket_fakes.cc)

rccl_add_ras_micro_binary(rccl-UnitTestsMicroRasRasnet rasnet-test.cc
  SOURCES
    fakes/env_fakes.cc
    fakes/nccl_fakes.cc)
