# ---------------------------------------------------------------------------
# src/param/: one binary per test file, each linking only the fakes its unit needs.
# ---------------------------------------------------------------------------
set(_MICRO_PARAM_DEFS
  # param_tmp.cc, not param.cc: src/CMakeLists.txt lists misc/param.cc first, so
  # add_file_unique() renames this one.
  PARAM_CC_PATH="${RCCL_HOST_HIPIFY_DIR}/src/param/param_tmp.cc"
  C_API_CC_PATH="${RCCL_HOST_HIPIFY_DIR}/src/param/c_api.cc"
  MISC_PARAM_CC_PATH="${RCCL_HOST_HIPIFY_DIR}/src/misc/param.cc"
  PARAM_REGISTRY_CC_PATH="${RCCL_HOST_HIPIFY_DIR}/src/param/param_registry.cc")

# One src/param/ test file per binary; see rccl_add_unit_micro_binary().
macro(rccl_add_param_micro_binary _name _test)
  rccl_add_unit_micro_binary(${_name} param/${_test} ${ARGN}
    OBJECT_PREFIX rccl-param-obj
    COMMON_DEFS ${_MICRO_PARAM_DEFS}
    CATEGORIES_YAML "${CMAKE_CURRENT_SOURCE_DIR}/../test_categories_micro_param.yaml")
endmacro()

# No fakes beyond hip_fakes.cc (the runner's GPU-count probe): misc/param.cc defines ncclGetEnv, initEnv and
# ncclLoadParam, as env_fakes.cc does.
rccl_add_param_micro_binary(rccl-UnitTestsMicroParam param-test.cc
  PROCESS_ISOLATED
  SOURCES
    fakes/hip_fakes.cc)

rccl_add_param_micro_binary(rccl-UnitTestsMicroParamRegistry param-registry-test.cc
  SOURCES
    fakes/nccl_fakes.cc)
