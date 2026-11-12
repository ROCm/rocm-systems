# ---------------------------------------------------------------------------
# src/misc/: one binary per test file, each linking only the fakes its unit needs.
# ---------------------------------------------------------------------------
# The *wrap-test.cc binaries reach their loader through the real *symbols.cc, built behind the libdl seam.
set(_MICRO_MISC_DL_SEAMED
  ${RCCL_HOST_HIPIFY_DIR}/src/misc/ibvsymbols.cc
  ${RCCL_HOST_HIPIFY_DIR}/src/misc/ionicdvsymbols.cc
  ${RCCL_HOST_HIPIFY_DIR}/src/misc/mlx5dvsymbols.cc)
set_source_files_properties(${_MICRO_MISC_DL_SEAMED} PROPERTIES
  GENERATED TRUE
  COMPILE_OPTIONS "-include;fakes/dl_seam.h")
set(_MICRO_MISC_DEFS
  ${_MICRO_DMABUF_EXPORT_DEF}  # standalone: the cuMem DMA-BUF arms in-build gets from rccl
  ENABLE_FAULT_INJECTION       # FAULT_INJECTION defaults ON, so this is the ibvwrap.cc arm that ships
  AMDSMI_WRAP_CC_PATH="${RCCL_HOST_HIPIFY_DIR}/src/misc/amdsmi_wrap.cc"
  API_TRACE_CC_PATH="${RCCL_HOST_HIPIFY_DIR}/src/misc/api_trace.cc"
  ARGCHECK_CC_PATH="${RCCL_HOST_HIPIFY_DIR}/src/misc/argcheck.cc"
  COLL_TRACE_CC_PATH="${RCCL_HOST_HIPIFY_DIR}/src/misc/latency_profiler/CollTrace.cc"
  COLL_TRACE_EVENT_CC_PATH="${RCCL_HOST_HIPIFY_DIR}/src/misc/latency_profiler/CollTraceEvent.cc"
  COLL_TRACE_FUNC_CC_PATH="${RCCL_HOST_HIPIFY_DIR}/src/misc/latency_profiler/CollTraceFunc.cc"
  COLL_TRACE_UTILS_CC_PATH="${RCCL_HOST_HIPIFY_DIR}/src/misc/latency_profiler/CollTraceUtils.cc"
  CRYPT_CC_PATH="${RCCL_HOST_HIPIFY_DIR}/src/misc/crypt.cc"
  DV_PROVIDER_FAKE_IONIC_NAME="$<TARGET_FILE_NAME:rccl-micro-dv-provider-fake-ionic>"
  DV_PROVIDER_FAKE_IONIC_NO_GDA_NAME="$<TARGET_FILE_NAME:rccl-micro-dv-provider-fake-ionic-no-gda>"
  DV_PROVIDER_FAKE_MLX5_NAME="$<TARGET_FILE_NAME:rccl-micro-dv-provider-fake-mlx5>"
  GDR_PEERMEM_CC_PATH="${RCCL_HOST_HIPIFY_DIR}/src/misc/gdr_peermem.cc"
  GDR_PROBE_CC_PATH="${RCCL_HOST_HIPIFY_DIR}/src/misc/gdr_probe.cc"
  IBVSYMBOLS_CC_PATH="${RCCL_HOST_HIPIFY_DIR}/src/misc/ibvsymbols.cc"
  IBVWRAP_CC_PATH="${RCCL_HOST_HIPIFY_DIR}/src/misc/ibvwrap.cc"
  IONICDVSYMBOLS_CC_PATH="${RCCL_HOST_HIPIFY_DIR}/src/misc/ionicdvsymbols.cc"
  IONICDVWRAP_CC_PATH="${RCCL_HOST_HIPIFY_DIR}/src/misc/ionicdvwrap.cc"
  MLX5DVSYMBOLS_CC_PATH="${RCCL_HOST_HIPIFY_DIR}/src/misc/mlx5dvsymbols.cc"
  MLX5DVWRAP_CC_PATH="${RCCL_HOST_HIPIFY_DIR}/src/misc/mlx5dvwrap.cc"
  NVMLWRAP_STUB_CC_PATH="${RCCL_HOST_HIPIFY_DIR}/src/misc/nvmlwrap_stub.cc"
  RECORDER_CC_PATH="${RCCL_HOST_HIPIFY_DIR}/src/misc/recorder.cc"
  ROCMWRAP_CC_PATH="${RCCL_HOST_HIPIFY_DIR}/src/misc/rocmwrap.cc"
  ROCTX_CC_PATH="${RCCL_HOST_HIPIFY_DIR}/src/misc/roctx.cc"
  SHMUTILS_CC_PATH="${RCCL_HOST_HIPIFY_DIR}/src/misc/shmutils.cc"
  SIGNALS_CC_PATH="${RCCL_HOST_HIPIFY_DIR}/src/misc/signals.cc"
  SOCKET_CC_PATH="${RCCL_HOST_HIPIFY_DIR}/src/misc/socket.cc"
  STRONGSTREAM_CC_PATH="${RCCL_HOST_HIPIFY_DIR}/src/misc/strongstream.cc"
  UTILS_CC_PATH="${RCCL_HOST_HIPIFY_DIR}/src/misc/utils.cc")

# The fakes linux-test-common.h drives, for the socket/shmutils/signals tests here and the os/ tests.
rccl_mark_oracle_tus_generated(${RCCL_HOST_HIPIFY_DIR} crypt)
# os/linux-test.cc links the real socket.cc, behind the seams socket-test.cc puts around it.
set_source_files_properties(${RCCL_HOST_HIPIFY_DIR}/src/misc/socket.cc PROPERTIES
  GENERATED TRUE
  COMPILE_OPTIONS "-include;fakes/nccl_fakes.h;-include;fakes/param_redirect.h;-include;fakes/libc_seam.h")
set(_MICRO_LINUX_TEST_SOURCES
  fakes/dl_fakes.cc
  fakes/env_fakes.cc
  fakes/hip_fakes.cc
  fakes/libc_fakes.cc
  fakes/linux_fakes.cc
  fakes/nccl_fakes.cc
  ${RCCL_HOST_HIPIFY_DIR}/src/misc/crypt.cc
  ${RCCL_HOST_HIPIFY_DIR}/src/misc/utils.cc)

# One src/misc/ test file per binary; see rccl_add_unit_micro_binary().
macro(rccl_add_misc_micro_binary _name _test)
  rccl_add_unit_micro_binary(${_name} misc/${_test} ${ARGN}
    OBJECT_PREFIX rccl-misc-obj
    COMMON_INCLUDES
      ${RCCL_HOST_HIPIFY_DIR}/src/transport/net_ib_cast  # net_ib_ops_fault.h, included unqualified
    COMMON_DEFS ${_MICRO_MISC_DEFS}
    CATEGORIES_YAML "${CMAKE_CURRENT_SOURCE_DIR}/../test_categories_micro_misc.yaml")
endmacro()

rccl_add_misc_micro_binary(rccl-UnitTestsMicroMiscAmdsmiWrap amdsmi-wrap-test.cc
  SOURCES
    fakes/hip_fakes.cc
    ${RCCL_HOST_HIPIFY_DIR}/src/misc/utils.cc)  # real busIdToInt64

rccl_add_misc_micro_binary(rccl-UnitTestsMicroMiscApiTrace api-trace-test.cc
  SOURCES
    fakes/debug_fakes.cc
    fakes/recorder_fakes.cc
    fakes/roctx_fakes.cc
  # The rocprofiler-register branch would need librocprofiler-register, so api_trace.cc's
  # RCCL_ROCPROFILER_REGISTER block is not built here and not in this binary's coverage.
  REMOVE_DEFS RCCL_ROCPROFILER_REGISTER=1)

rccl_add_misc_micro_binary(rccl-UnitTestsMicroMiscArgcheck argcheck-test.cc)

rccl_add_misc_micro_binary(rccl-UnitTestsMicroMiscCrypt crypt-test.cc)

rccl_add_misc_micro_binary(rccl-UnitTestsMicroMiscDvsymbols dvsymbols-test.cc
  PROCESS_ISOLATED
  SOURCES
    fakes/dl_fakes.cc
    fakes/hip_fakes.cc                   # hipGetDeviceCount, for the runner's GPU count probe
    fakes/nccl_fakes.cc)

rccl_add_misc_micro_binary(rccl-UnitTestsMicroMiscGdrPeermem gdr-peermem-test.cc
  SOURCES
    fakes/libc_fakes.cc
    fakes/nccl_fakes.cc)

rccl_add_misc_micro_binary(rccl-UnitTestsMicroMiscGdrProbe gdr-probe-test.cc
  SOURCES
    fakes/hip_fakes.cc
    fakes/ibvwrap_fakes.cc
    fakes/nccl_fakes.cc)

rccl_add_misc_micro_binary(rccl-UnitTestsMicroMiscIbvsymbols ibvsymbols-test.cc
  SOURCES
    fakes/dl_fakes.cc
    fakes/env_fakes.cc
    fakes/nccl_fakes.cc)

rccl_add_misc_micro_binary(rccl-UnitTestsMicroMiscIbvwrap ibvwrap-test.cc
  PROCESS_ISOLATED
  SOURCES
    fakes/dl_fakes.cc
    fakes/env_fakes.cc
    fakes/hip_fakes.cc                   # hipGetDeviceCount, for the runner's GPU count probe
    fakes/nccl_fakes.cc
    fakes/net_ib_ops_fault_fakes.cc
    ${RCCL_HOST_HIPIFY_DIR}/src/misc/ibvsymbols.cc
  REMOVE_DEFS ENABLE_QP_TRACKING)  # ibvwrap.cc's create/destroy QP prologue dereferences the fake QP and PD

rccl_add_misc_micro_binary(rccl-UnitTestsMicroMiscIonicdvwrap ionicdvwrap-test.cc
  PROCESS_ISOLATED
  SOURCES
    fakes/dl_fakes.cc
    fakes/hip_fakes.cc                   # hipGetDeviceCount, for the runner's GPU count probe
    fakes/nccl_fakes.cc
    fakes/transport_stubs.cc
    ${RCCL_HOST_HIPIFY_DIR}/src/misc/ionicdvsymbols.cc)

# mlx5dvwrap.cc falls back to ibvwrap.cc's wrap_ibv_reg_dmabuf_mr, which the test drives through ibvSymbols.
set_source_files_properties(${RCCL_HOST_HIPIFY_DIR}/src/misc/ibvwrap.cc PROPERTIES
  GENERATED TRUE
  COMPILE_OPTIONS "-include;fakes/nccl_fakes.h;-include;fakes/param_redirect.h")
rccl_add_misc_micro_binary(rccl-UnitTestsMicroMiscMlx5dvwrap mlx5dvwrap-test.cc
  PROCESS_ISOLATED
  SOURCES
    fakes/dl_fakes.cc
    fakes/env_fakes.cc
    fakes/hip_fakes.cc                   # hipGetDeviceCount, for the runner's GPU count probe
    fakes/nccl_fakes.cc
    fakes/net_ib_ops_fault_fakes.cc
    ${RCCL_HOST_HIPIFY_DIR}/src/misc/ibvsymbols.cc
    ${RCCL_HOST_HIPIFY_DIR}/src/misc/ibvwrap.cc
    ${RCCL_HOST_HIPIFY_DIR}/src/misc/mlx5dvsymbols.cc
  REMOVE_DEFS ENABLE_QP_TRACKING)  # as for MiscIbvwrap

rccl_add_misc_micro_binary(rccl-UnitTestsMicroMiscRecorder recorder-test.cc
  SOURCES
    fakes/debug_fakes.cc)

rccl_add_misc_micro_binary(rccl-UnitTestsMicroMiscRocmwrap rocmwrap-test.cc
  SOURCES
    fakes/hip_fakes.cc
    ${RCCL_HOST_HIPIFY_DIR}/src/misc/archinfo.cc        # real GetGcnArchName/IsArchMatch
    ${RCCL_HOST_HIPIFY_DIR}/src/misc/kernel_config.cc)  # real option matching

rccl_add_misc_micro_binary(rccl-UnitTestsMicroMiscRoctxNvmlStub roctx-nvml-stub-test.cc)
# -roctx builds roctx.cc's ROCTX_ENABLE arm against the stand-in fakes/roctx_include.
if(_MICRO_BUILD_ROCTX_VARIANT)
  rccl_add_misc_micro_binary(rccl-UnitTestsMicroMiscRoctxNvmlStub-roctx roctx-nvml-stub-test.cc
    EXTRA_DEFS ROCTX_ENABLE)
  target_include_directories(rccl-UnitTestsMicroMiscRoctxNvmlStub-roctx BEFORE PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/fakes/roctx_include)
endif()

rccl_add_misc_micro_binary(rccl-UnitTestsMicroMiscShmutils shmutils-test.cc
  SOURCES
    ${_MICRO_LINUX_TEST_SOURCES}
    ${RCCL_HOST_HIPIFY_DIR}/src/os/linux.cc)  # ncclOsShm*

rccl_add_misc_micro_binary(rccl-UnitTestsMicroMiscSignals signals-test.cc
  SOURCES ${_MICRO_LINUX_TEST_SOURCES})

rccl_add_misc_micro_binary(rccl-UnitTestsMicroMiscSocket socket-test.cc
  SOURCES
    ${_MICRO_LINUX_TEST_SOURCES}
    ${RCCL_HOST_HIPIFY_DIR}/src/os/linux.cc)  # ncclOsSocket*, ncclOsFindInterfaces

rccl_add_misc_micro_binary(rccl-UnitTestsMicroMiscStrongstream strongstream-test.cc)

rccl_add_misc_micro_binary(rccl-UnitTestsMicroMiscUtils utils-test.cc)

rccl_add_misc_micro_binary(rccl-UnitTestsMicroMiscLatencyProfilerCollTrace latency_profiler/coll-trace-test.cc
  SOURCES
    fakes/collectives_fakes.cc
    fakes/debug_fakes.cc
    fakes/env_fakes.cc
    fakes/strongstream_stubs.cc)

# Fake ibverbs provider dv libraries for dvsymbols-test.cc; the SONAME is what the builders' dlopen matches.
# The test finds them beside its own executable, in the build tree and in an install.
foreach(_fake ionic ionic-no-gda mlx5)
  string(REGEX REPLACE "-.*" "" _dv "${_fake}")
  add_library(rccl-micro-dv-provider-fake-${_fake} SHARED fakes/dv_provider_fake.cc)
  set_target_properties(rccl-micro-dv-provider-fake-${_fake} PROPERTIES NO_SONAME ON)
  target_compile_options(rccl-micro-dv-provider-fake-${_fake} PRIVATE --offload-host-only)
  target_link_options(rccl-micro-dv-provider-fake-${_fake} PRIVATE -no-hip-rt -Wl,-soname,lib${_dv}.so)
  add_dependencies(rccl-UnitTestsMicroMiscDvsymbols rccl-micro-dv-provider-fake-${_fake})
  if(BUILD_TESTS)
    install(TARGETS rccl-micro-dv-provider-fake-${_fake} LIBRARY DESTINATION ${CMAKE_INSTALL_BINDIR} COMPONENT tests)
  endif()
endforeach()
target_compile_definitions(rccl-micro-dv-provider-fake-ionic-no-gda PRIVATE DV_PROVIDER_FAKE_IONIC_NO_GDA)
target_link_options(rccl-micro-dv-provider-fake-ionic-no-gda PRIVATE
  -Wl,--version-script=${CMAKE_CURRENT_SOURCE_DIR}/fakes/dv_provider_fake_ionic_no_gda.map)
set_property(TARGET rccl-micro-dv-provider-fake-ionic-no-gda APPEND PROPERTY
  LINK_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/fakes/dv_provider_fake_ionic_no_gda.map")
