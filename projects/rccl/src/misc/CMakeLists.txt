# Misc sources
set(MISC_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/strongstream.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/socket.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/ibvwrap.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/mlx5dvsymbols.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/mlx5dvwrap.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/cudawrap.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/param.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/ipcsocket.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/utils.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/shmutils.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/nvmlwrap.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/argcheck.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/gdrwrap.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/ibvsymbols.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/git_version.cc
)

# >>> RCCL-DELTA: remove NCCL sources that RCCL does not build.
# cudawrap/gdrwrap/nvmlwrap are CUDA/NVML host shims (RCCL uses rocmwrap.cc + nvmlwrap_stub.cc);
# git_version.cc is replaced by RCCL's generated git_version.cpp (update_git_version target).
list(REMOVE_ITEM MISC_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/cudawrap.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/gdrwrap.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/nvmlwrap.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/git_version.cc
)
# <<< RCCL-DELTA

# >>> RCCL-DELTA: RCCL-only sources in src/misc (AMD additions that merge cleanly here).
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
    # latency_profiler (RCCL-only)
    ${CMAKE_CURRENT_SOURCE_DIR}/latency_profiler/CollTrace.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/latency_profiler/CollTraceEvent.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/latency_profiler/CollTraceFunc.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/latency_profiler/CollTraceUtils.cc
)
# <<< RCCL-DELTA

# Add misc sources to parent scope
set(MISC_SOURCES ${MISC_SOURCES} PARENT_SCOPE)
