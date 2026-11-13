# ---------------------------------------------------------------------------
# src/device/: one binary per test file, each linking only the fakes its unit needs.
# ---------------------------------------------------------------------------
set(_MICRO_DEVICE_DEFS
  ONERANK_CC_PATH="${RCCL_HOST_HIPIFY_DIR}/src/device/onerank.cu.cpp")

# One src/device/ test file per binary; see rccl_add_unit_micro_binary().
macro(rccl_add_device_micro_binary _name _test)
  rccl_add_unit_micro_binary(${_name} device/${_test} ${ARGN}
    OBJECT_PREFIX rccl-device-obj
    COMMON_INCLUDES
      ${RCCL_HOST_HIPIFY_DIR}/src/include/nccl_device  # device/op128.h includes rccl_ptr.h unqualified
      ${RCCL_HOST_HIPIFY_DIR}/src/device               # common_kernel.h includes "tdm/tdmCopy.h"
    COMMON_DEFS ${_MICRO_DEVICE_DEFS}
    CATEGORIES_YAML "${CMAKE_CURRENT_SOURCE_DIR}/../test_categories_micro_device.yaml")
endmacro()

rccl_add_device_micro_binary(rccl-UnitTestsMicroDeviceOnerank onerank-test.cc
  SOURCES
    fakes/hip_fakes.cc
    fakes/nccl_fakes.cc)
# onerank.cu defines kernels; embed an empty GPU binary.
file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/empty_gpubinary" "")
target_compile_options(rccl-UnitTestsMicroDeviceOnerank PRIVATE
  "SHELL:-Xclang -fcuda-include-gpubinary -Xclang ${CMAKE_CURRENT_BINARY_DIR}/empty_gpubinary")
