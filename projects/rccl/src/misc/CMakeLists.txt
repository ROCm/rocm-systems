# Misc sources
set(MISC_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/strongstream.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/socket.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/ibvwrap.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/mlx5dvsymbols.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/mlx5dvwrap.cc
    # >>> RCCL-DELTA: cudawrap.cc is a CUDA driver wrapper; RCCL uses misc/rocmwrap.cc.
    # ${CMAKE_CURRENT_SOURCE_DIR}/cudawrap.cc
    # <<< RCCL-DELTA
    ${CMAKE_CURRENT_SOURCE_DIR}/param.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/ipcsocket.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/utils.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/shmutils.cc
    # >>> RCCL-DELTA: nvmlwrap.cc is NVIDIA NVML; RCCL uses misc/nvmlwrap_stub.cc.
    # ${CMAKE_CURRENT_SOURCE_DIR}/nvmlwrap.cc
    # <<< RCCL-DELTA
    ${CMAKE_CURRENT_SOURCE_DIR}/argcheck.cc
    # >>> RCCL-DELTA: gdrwrap.cc (GPUDirect RDMA copy) is not built by RCCL.
    # ${CMAKE_CURRENT_SOURCE_DIR}/gdrwrap.cc
    # <<< RCCL-DELTA
    ${CMAKE_CURRENT_SOURCE_DIR}/ibvsymbols.cc
    # >>> RCCL-DELTA: git_version.cc is replaced by a build-time generated git_version.cpp.
    # ${CMAKE_CURRENT_SOURCE_DIR}/git_version.cc
    # <<< RCCL-DELTA
)

# >>> RCCL-DELTA: RCCL-only misc sources (no NCCL counterpart).
list(APPEND MISC_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/alt_rsmi.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/archinfo.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/api_trace.c
    ${CMAKE_CURRENT_SOURCE_DIR}/api_trace.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/ionicdvsymbols.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/ionicdvwrap.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/npkit.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/nvmlwrap_stub.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/rocmwrap.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/roctx.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/recorder.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/signals.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/proxy_trace/proxy_trace.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/latency_profiler/CollTrace.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/latency_profiler/CollTraceEvent.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/latency_profiler/CollTraceFunc.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/latency_profiler/CollTraceUtils.cc
)
# AMD SMI wrapper sources. amdsmi is always compiled; rocm_smi is the legacy fallback.
# (The matching amdsmi_wrap.h / rocm_smi_wrap.h headers live under src/include and are
#  listed for hipify in src/CMakeLists.txt.) USE_AMDSMI is defined by the root CMakeLists.
list(APPEND MISC_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/amdsmi_wrap.cc)
if(NOT USE_AMDSMI)
    list(APPEND MISC_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/rocm_smi_wrap.cc)
endif()
# <<< RCCL-DELTA

# Add misc sources to parent scope
set(MISC_SOURCES ${MISC_SOURCES} PARENT_SCOPE)
