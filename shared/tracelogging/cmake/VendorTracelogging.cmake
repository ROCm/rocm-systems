# VendorTracelogging.cmake — build the vendored microsoft/tracelogging LTTng
# helper library (lttngh) as a small static library and expose an interface
# target that carries its include path.
#
# This is the TraceLogging backend for the curated ROCm tracepoints. It is a
# BACKEND SWAP under the SAME existing enable gate as classic LTTng-UST
# (HSA_ENABLE_LTTNG_UST / HIP_ENABLE_LTTNG_UST): when the runtime is built
# with the TraceLogging backend, the generated rocm_trace_emit_curated.cpp
# uses TraceLoggingWrite() instead of lttng_ust_do_tracepoint(), and links
# this library plus the vendored LTTng-UST directly.
#
# The upstream LTTng/src/CMakeLists.txt builds lttngh from exactly three .c
# files (LttngHelpers.c, LttngActivityHelpers.c, LttngNetHelpers.c) and an
# INTERFACE target `tracelogging` that just adds the include dir + links
# lttngh. We reproduce that here as a static lib + interface target rather
# than pulling in upstream's find_package/CMakePresets machinery, so it
# slots into the existing vendored-LTTng build tree.
#
# Required input variables (set by includer before include()):
#   TRACELOGGING_VENDORED_SRC        — absolute path of the vendored subtree
#                                      (shared/tracelogging/tracelogging-src)
#   TRACELOGGING_VENDORED_UST_INCLUDE — LTTng-UST include dir (the vendored
#                                      LTTNG_VENDORED_INCLUDE_DIR); needed so
#                                      LttngHelpers.h can find <lttng/*.h>.
#
# Outputs:
#   Target `rocm_tracelogging` — static lttngh lib + include dirs (vendored
#     tracelogging headers and the LTTng-UST headers) as PUBLIC usage
#     requirements. Consumers link this and get TraceLoggingProvider.h.
#   Property TRACELOGGING_VENDORED_INCLUDE_DIR — the vendored include dir.
include_guard(GLOBAL)

if(NOT DEFINED TRACELOGGING_VENDORED_SRC)
    message(FATAL_ERROR "VendorTracelogging.cmake: TRACELOGGING_VENDORED_SRC "
                        "must be set to the vendored tracelogging subtree path.")
endif()
if(NOT DEFINED TRACELOGGING_VENDORED_UST_INCLUDE)
    message(FATAL_ERROR "VendorTracelogging.cmake: TRACELOGGING_VENDORED_UST_INCLUDE "
                        "must be set to the LTTng-UST include directory.")
endif()

set(TRACELOGGING_VENDORED_INCLUDE_DIR "${TRACELOGGING_VENDORED_SRC}/include"
    CACHE INTERNAL "Vendored tracelogging include directory")

add_library(rocm_tracelogging STATIC
    "${TRACELOGGING_VENDORED_SRC}/src/LttngHelpers.c"
    "${TRACELOGGING_VENDORED_SRC}/src/LttngActivityHelpers.c"
    "${TRACELOGGING_VENDORED_SRC}/src/LttngNetHelpers.c")

# Upstream builds these with c_std_99. Keep POSITION_INDEPENDENT_CODE ON so
# the archive links cleanly into the SHARED consumer runtimes
# (libhsa-runtime64.so / libamdhip64.so), independent of the global cache var.
set_target_properties(rocm_tracelogging PROPERTIES
    C_STANDARD 99
    POSITION_INDEPENDENT_CODE ON)

# PUBLIC so consumers that link rocm_tracelogging inherit both the
# TraceLoggingProvider.h include path and the LTTng-UST headers it #includes.
target_include_directories(rocm_tracelogging PUBLIC
    "${TRACELOGGING_VENDORED_INCLUDE_DIR}"
    "${TRACELOGGING_VENDORED_UST_INCLUDE}")

# The vendored LTTng-UST libraries are produced by the lttng_ust_vendored
# ExternalProject (VendorLttng.cmake). lttngh calls into liblttng-ust /
# liblttng-ust-common at runtime, so make sure those are built first. The
# consumer target (hsa-runtime64 / amdhip64) already links PkgConfig::LTTNG_UST
# and depends on lttng_ust_vendored, so we don't re-link the .so here; we only
# guarantee build ordering when the target exists.
if(TARGET lttng_ust_vendored)
    add_dependencies(rocm_tracelogging lttng_ust_vendored)
endif()
