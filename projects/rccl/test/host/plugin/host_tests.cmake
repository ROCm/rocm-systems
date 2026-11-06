# ---------------------------------------------------------------------------
# src/plugin/: one binary per test file, each linking only the fakes its unit needs.
# ---------------------------------------------------------------------------
# Real sibling units a test file reaches.
set(_MICRO_PLUGIN_PROFILER_ADAPTERS
  ${RCCL_HOST_HIPIFY_DIR}/src/plugin/profiler/profiler_v1.cc
  ${RCCL_HOST_HIPIFY_DIR}/src/plugin/profiler/profiler_v2.cc
  ${RCCL_HOST_HIPIFY_DIR}/src/plugin/profiler/profiler_v3.cc
  ${RCCL_HOST_HIPIFY_DIR}/src/plugin/profiler/profiler_v4.cc
  ${RCCL_HOST_HIPIFY_DIR}/src/plugin/profiler/profiler_v5.cc
  ${RCCL_HOST_HIPIFY_DIR}/src/plugin/profiler/profiler_v6.cc
  ${RCCL_HOST_HIPIFY_DIR}/src/plugin/profiler/profiler_v7.cc)
set_source_files_properties(${_MICRO_PLUGIN_PROFILER_ADAPTERS} PROPERTIES GENERATED TRUE)
set(_MICRO_PLUGIN_DEFS
  CSV_TUNER_CC_PATH="${RCCL_HOST_HIPIFY_DIR}/src/plugin/tuner/csv_tuner.cc"
  # Source tuner/ dir, so a test can check the generator embedded the CSVs verbatim.
  RCCL_TUNER_CSV_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}/../../tuner"
  # plugin/gin.cc is the first gin.cc in SRC_FILES, so hipify keeps the name.
  GIN_CC_PATH="${RCCL_HOST_HIPIFY_DIR}/src/plugin/gin.cc"
  NET_V6_CC_PATH="${RCCL_HOST_HIPIFY_DIR}/src/plugin/net/net_v6.cc"
  NET_V7_CC_PATH="${RCCL_HOST_HIPIFY_DIR}/src/plugin/net/net_v7.cc"
  NET_V8_CC_PATH="${RCCL_HOST_HIPIFY_DIR}/src/plugin/net/net_v8.cc"
  NET_V9_CC_PATH="${RCCL_HOST_HIPIFY_DIR}/src/plugin/net/net_v9.cc"
  NET_V10_CC_PATH="${RCCL_HOST_HIPIFY_DIR}/src/plugin/net/net_v10.cc"
  NET_V11_CC_PATH="${RCCL_HOST_HIPIFY_DIR}/src/plugin/net/net_v11.cc"
  NET_V12_CC_PATH="${RCCL_HOST_HIPIFY_DIR}/src/plugin/net/net_v12.cc"
  PLUGIN_NET_CC_PATH="${RCCL_HOST_HIPIFY_DIR}/src/plugin/net.cc"
  PLUGIN_OPEN_CC_PATH="${RCCL_HOST_HIPIFY_DIR}/src/plugin/plugin_open.cc"
  PROFILER_CC_PATH="${RCCL_HOST_HIPIFY_DIR}/src/plugin/profiler.cc"
  PROFILER_V1_CC_PATH="${RCCL_HOST_HIPIFY_DIR}/src/plugin/profiler/profiler_v1.cc"
  PROFILER_V2_CC_PATH="${RCCL_HOST_HIPIFY_DIR}/src/plugin/profiler/profiler_v2.cc"
  PROFILER_V3_CC_PATH="${RCCL_HOST_HIPIFY_DIR}/src/plugin/profiler/profiler_v3.cc"
  PROFILER_V4_CC_PATH="${RCCL_HOST_HIPIFY_DIR}/src/plugin/profiler/profiler_v4.cc"
  PROFILER_V5_CC_PATH="${RCCL_HOST_HIPIFY_DIR}/src/plugin/profiler/profiler_v5.cc"
  PROFILER_V6_CC_PATH="${RCCL_HOST_HIPIFY_DIR}/src/plugin/profiler/profiler_v6.cc"
  PROFILER_V7_CC_PATH="${RCCL_HOST_HIPIFY_DIR}/src/plugin/profiler/profiler_v7.cc"
  # src/CMakeLists.txt lists plugin/rma.cc before rma/rma.cc, so it keeps its own name.
  RMA_PLUGIN_CC_PATH="${RCCL_HOST_HIPIFY_DIR}/src/plugin/rma.cc")

# One src/plugin/ test file per binary; see rccl_add_unit_micro_binary().
macro(rccl_add_plugin_micro_binary _name _test)
  rccl_add_unit_micro_binary(${_name} plugin/${_test} ${ARGN}
    OBJECT_PREFIX rccl-plugin-obj
    COMMON_DEFS ${_MICRO_PLUGIN_DEFS}
    CATEGORIES_YAML "${CMAKE_CURRENT_SOURCE_DIR}/../test_categories_micro_plugin.yaml")
endmacro()

rccl_add_plugin_micro_binary(rccl-UnitTestsMicroPluginGinPluginInit gin-plugin-init-test.cc
  SOURCES
    fakes/nccl_fakes.cc)

rccl_add_plugin_micro_binary(rccl-UnitTestsMicroPluginNet plugin-net-test.cc
  PROCESS_ISOLATED
  SOURCES
    fakes/env_fakes.cc
    fakes/hip_fakes.cc                   # hipGetDeviceCount, for the runner's GPU-pool probe
    fakes/nccl_fakes.cc
    fakes/plugin_stubs.cc
    fakes/transport_stubs.cc
  EXTRA_DEFS
    RCCL_TRANSPORT_STUBS_OMIT_ncclGpuGdrSupport)  # net.cc defines it

rccl_add_plugin_micro_binary(rccl-UnitTestsMicroPluginOpen plugin-open-test.cc
  SOURCES
    fakes/nccl_fakes.cc)

# The test file defines the logging, env, plugin-open and dlsym seams itself.
rccl_add_plugin_micro_binary(rccl-UnitTestsMicroPluginProfiler profiler-test.cc
  SOURCES
    fakes/utils_fakes.cc
    ${_MICRO_PLUGIN_PROFILER_ADAPTERS})

rccl_add_plugin_micro_binary(rccl-UnitTestsMicroPluginRmaPluginInit rma-plugin-init-test.cc
  SOURCES
    fakes/nccl_fakes.cc)

rccl_add_plugin_micro_binary(rccl-UnitTestsMicroPluginNetV6 net/net-v6-test.cc
  SOURCES
    fakes/nccl_fakes.cc)

rccl_add_plugin_micro_binary(rccl-UnitTestsMicroPluginNetV7 net/net-v7-test.cc
  SOURCES
    fakes/nccl_fakes.cc)

rccl_add_plugin_micro_binary(rccl-UnitTestsMicroPluginNetV8 net/net-v8-test.cc
  SOURCES
    fakes/nccl_fakes.cc)

rccl_add_plugin_micro_binary(rccl-UnitTestsMicroPluginNetV9 net/net-v9-test.cc
  SOURCES
    fakes/nccl_fakes.cc)

rccl_add_plugin_micro_binary(rccl-UnitTestsMicroPluginNetV10 net/net-v10-test.cc
  SOURCES
    fakes/nccl_fakes.cc)

rccl_add_plugin_micro_binary(rccl-UnitTestsMicroPluginNetV11 net/net-v11-test.cc
  SOURCES
    fakes/nccl_fakes.cc)

rccl_add_plugin_micro_binary(rccl-UnitTestsMicroPluginNetV12 net/net-v12-test.cc
  SOURCES
    fakes/nccl_fakes.cc)

rccl_add_plugin_micro_binary(rccl-UnitTestsMicroPluginProfilerV1 profiler/profiler-v1-test.cc
  SOURCES
    fakes/nccl_fakes.cc
    plugin/profiler/fake_lib_dlsym.cc)

rccl_add_plugin_micro_binary(rccl-UnitTestsMicroPluginProfilerV2 profiler/profiler-v2-test.cc
  SOURCES
    fakes/nccl_fakes.cc
    plugin/profiler/fake_lib_dlsym.cc)

rccl_add_plugin_micro_binary(rccl-UnitTestsMicroPluginProfilerV3 profiler/profiler-v3-test.cc
  SOURCES
    fakes/nccl_fakes.cc
    plugin/profiler/fake_lib_dlsym.cc)

rccl_add_plugin_micro_binary(rccl-UnitTestsMicroPluginProfilerV4 profiler/profiler-v4-test.cc
  SOURCES
    fakes/nccl_fakes.cc
    plugin/profiler/fake_lib_dlsym.cc)

rccl_add_plugin_micro_binary(rccl-UnitTestsMicroPluginProfilerV5 profiler/profiler-v5-test.cc
  SOURCES
    fakes/nccl_fakes.cc
    plugin/profiler/fake_lib_dlsym.cc)

rccl_add_plugin_micro_binary(rccl-UnitTestsMicroPluginProfilerV6 profiler/profiler-v6-test.cc
  SOURCES
    fakes/nccl_fakes.cc
    plugin/profiler/fake_lib_dlsym.cc)

rccl_add_plugin_micro_binary(rccl-UnitTestsMicroPluginProfilerV7 profiler/profiler-v7-test.cc
  SOURCES
    fakes/nccl_fakes.cc
    plugin/profiler/fake_lib_dlsym.cc)

rccl_add_plugin_micro_binary(rccl-UnitTestsMicroPluginTunerCsvTuner tuner/csv-tuner-test.cc
  SOURCES
    fakes/env_fakes.cc
    fakes/nccl_fakes.cc)
