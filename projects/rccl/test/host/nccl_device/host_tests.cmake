# ---------------------------------------------------------------------------
# src/nccl_device/: one binary per test file, each linking only the fakes its unit needs.
# ---------------------------------------------------------------------------
set(_MICRO_NCCL_DEVICE_DEFS
  CFT_BARRIER_CC_PATH="${RCCL_HOST_HIPIFY_DIR}/src/nccl_device/cft_barrier.cc"
  CORE_CC_PATH="${RCCL_HOST_HIPIFY_DIR}/src/nccl_device/core.cc"
  GIN_BARRIER_CC_PATH="${RCCL_HOST_HIPIFY_DIR}/src/nccl_device/gin_barrier.cc"
  GIN_SCRATCH_CC_PATH="${RCCL_HOST_HIPIFY_DIR}/src/nccl_device/gin_scratch.cc"
  LL_A2A_CC_PATH="${RCCL_HOST_HIPIFY_DIR}/src/nccl_device/ll_a2a.cc"
  LSA_BARRIER_CC_PATH="${RCCL_HOST_HIPIFY_DIR}/src/nccl_device/lsa_barrier.cc")

# One src/nccl_device/ test file per binary; see rccl_add_unit_micro_binary().
macro(rccl_add_nccl_device_micro_binary _name _test)
  rccl_add_unit_micro_binary(${_name} nccl_device/${_test} ${ARGN}
    OBJECT_PREFIX rccl-nccl_device-obj
    COMMON_DEFS ${_MICRO_NCCL_DEVICE_DEFS}
    CATEGORIES_YAML "${CMAKE_CURRENT_SOURCE_DIR}/../test_categories_micro_nccl_device.yaml")
endmacro()

rccl_add_nccl_device_micro_binary(rccl-UnitTestsMicroNcclDeviceCore core-test.cc
  SOURCES
    fakes/dev_runtime_fakes.cc
    fakes/nccl_stubs.cc)

rccl_add_nccl_device_micro_binary(rccl-UnitTestsMicroNcclDeviceRequirements nccl-device-requirements-test.cc)
