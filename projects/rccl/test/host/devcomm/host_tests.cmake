# ---------------------------------------------------------------------------
# src/devcomm/: one binary per test file, each linking only the fakes its unit needs.
# ---------------------------------------------------------------------------
set(_MICRO_DEVCOMM_DEFS
  DEVCOMM_V22902_CC_PATH="${RCCL_HOST_HIPIFY_DIR}/src/devcomm/devcomm_v22902.cc"
  DEVCOMM_V22907_CC_PATH="${RCCL_HOST_HIPIFY_DIR}/src/devcomm/devcomm_v22907.cc")

# One src/devcomm/ test file per binary; see rccl_add_unit_micro_binary().
macro(rccl_add_devcomm_micro_binary _name _test)
  rccl_add_unit_micro_binary(${_name} devcomm/${_test} ${ARGN}
    OBJECT_PREFIX rccl-devcomm-obj
    COMMON_INCLUDES
      ${RCCL_HOST_HIPIFY_DIR}/src/devcomm              # devcomm_v22902.h, included unqualified
      ${RCCL_HOST_HIPIFY_DIR}/src/include/nccl_device  # device.h includes rccl_ptr.h unqualified
    COMMON_DEFS ${_MICRO_DEVCOMM_DEFS}
    CATEGORIES_YAML "${CMAKE_CURRENT_SOURCE_DIR}/../test_categories_micro_devcomm.yaml")
endmacro()

rccl_add_devcomm_micro_binary(rccl-UnitTestsMicroDevcomm devcomm-test.cc
  SOURCES
    fakes/devcomm_fakes.cc
    fakes/nccl_fakes.cc)
