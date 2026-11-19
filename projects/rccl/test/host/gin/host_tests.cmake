# ---------------------------------------------------------------------------
# src/gin/: one binary per test file, each linking only the fakes its unit needs.
# ---------------------------------------------------------------------------
set(_MICRO_GIN_DEFS
  ${_MICRO_DMABUF_EXPORT_DEF}  # standalone: rocmwrap.h's DMA-BUF gate as rccl compiles it
  GIN_HOST_CC_PATH="${RCCL_HOST_HIPIFY_DIR}/src/gin/gin_host.cc"
  GIN_HOST_PROXY_CC_PATH="${RCCL_HOST_HIPIFY_DIR}/src/gin/gin_host_proxy.cc"
  GPUCONTEXT_CC_PATH="${RCCL_HOST_HIPIFY_DIR}/src/gin/proxy_gpucontext/gpucontext.cc"
  GPUCONTEXT_V1_CC_PATH="${RCCL_HOST_HIPIFY_DIR}/src/gin/proxy_gpucontext/gpucontext_v1.cc"
  GPUCONTEXT_V2_CC_PATH="${RCCL_HOST_HIPIFY_DIR}/src/gin/proxy_gpucontext/gpucontext_v2.cc")

# One src/gin/ test file per binary; see rccl_add_unit_micro_binary().
macro(rccl_add_gin_micro_binary _name _test)
  rccl_add_unit_micro_binary(${_name} gin/${_test} ${ARGN}
    OBJECT_PREFIX rccl-gin-obj
    COMMON_INCLUDES
      ${RCCL_HOST_HIPIFY_DIR}/src/include/plugin
      ${RCCL_HOST_HIPIFY_DIR}/src/include/nccl_device
    COMMON_DEFS ${_MICRO_GIN_DEFS}
    CATEGORIES_YAML "${CMAKE_CURRENT_SOURCE_DIR}/../test_categories_micro_gin.yaml")
endmacro()

rccl_add_gin_micro_binary(rccl-UnitTestsMicroGinHost gin-host-test.cc
  SOURCES
    fakes/bootstrap_stubs.cc
    fakes/nccl_device_core_fakes.cc
    fakes/nccl_fakes.cc
    fakes/os_fakes.cc
    fakes/search_fakes.cc
    fakes/topo_stubs.cc)

# os_fakes.cc: ncclGinProgress references ncclOsCpuCount/ncclOsSetAffinity; no test starts the thread.
rccl_add_gin_micro_binary(rccl-UnitTestsMicroGinHostFinalize gin-host-finalize-test.cc
  SOURCES
    fakes/os_fakes.cc)

# The test redirects rocmwrap.cc's DmaBufEnable param; keep the name and default in step.
foreach(_f "${CMAKE_CURRENT_LIST_DIR}/../../../src/misc/rocmwrap.cc"
           "${CMAKE_CURRENT_LIST_DIR}/gin-host-proxy-test.cc")
  rccl_assert_source_line("${_f}" "NCCL_PARAM(DmaBufEnable, \"DMABUF_ENABLE\", 1)" "gin-host-proxy-test.cc")
endforeach()
rccl_add_gin_micro_binary(rccl-UnitTestsMicroGinHostProxy gin-host-proxy-test.cc
  SOURCES
    fakes/hip_fakes.cc
    fakes/nccl_fakes.cc)
