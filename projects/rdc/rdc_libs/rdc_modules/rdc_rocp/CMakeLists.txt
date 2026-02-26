message("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&")
message("                       Cmake RDC Lib-ROCP                          ")
message("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&")

set(SRC_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
set(INC_DIR "${PROJECT_SOURCE_DIR}/include/rdc_modules/rdc_rocp")

set(RDC_ROCP_LIB_COMPONENT "lib${RDC_ROCP_LIB}")

# librdc_rocp.so now uses subprocess-based profiling (RdcRocpSubprocess)
# instead of in-process rocprofiler-sdk (RdcRocpBase + RdcRocpCounterSampler).
set(RDC_ROCP_LIB_SRC_LIST
    "${BOOTSTRAP_LIB_SRC_DIR}/RdcLogger.cc"
    "${BOOTSTRAP_LIB_SRC_DIR}/RdcEntityCodec.cc"
    "${SRC_DIR}/RdcTelemetryLib.cc"
    "${SRC_DIR}/RdcRocpSubprocess.cc"
)
set(RDC_ROCP_LIB_INC_LIST
    "${PROJECT_SOURCE_DIR}/include/rdc/rdc.h"
    "${RDC_LIB_INC_DIR}/RdcDiagnosticLibInterface.h"
    "${RDC_LIB_INC_DIR}/rdc_common.h"
    "${RDC_LIB_INC_DIR}/RdcLogger.h"
    "${RDC_LIB_INC_DIR}/RdcEntityCodec.h"
    "${INC_DIR}/RdcRocpSubprocess.h"
)

if(BUILD_PROFILER)
    message("Build librdc_rocp.so is enabled (subprocess mode).")

    message("RDC_ROCP_LIB_INC_LIST=${RDC_ROCP_LIB_INC_LIST}")

    # librdc_rocp.so no longer links against rocprofiler-sdk directly.
    # It forks/execs the rocpctl binary which handles rocprofiler initialization.
    set(RDC_LIB_MODULES ${RDC_LIB_MODULES} ${RDC_ROCP_LIB} PARENT_SCOPE)
    add_library(${RDC_ROCP_LIB} SHARED ${RDC_ROCP_LIB_SRC_LIST} ${RDC_ROCP_LIB_INC_LIST})
    target_link_libraries(
        ${RDC_ROCP_LIB}
        PRIVATE pthread dl
    )
    target_include_directories(
        ${RDC_ROCP_LIB}
        PRIVATE
            "${PROJECT_SOURCE_DIR}"
            "${PROJECT_SOURCE_DIR}/include"
            "${COMMON_DIR}"
            "${ROCM_DIR}/include"
    )

    # let bootstrap find this module at build time
    set_property(
        TARGET ${BOOTSTRAP_LIB}
        APPEND
        PROPERTY BUILD_RPATH "\$ORIGIN/../rdc_modules/${RDC_ROCP_LIB}"
    )

    set_property(TARGET ${RDC_ROCP_LIB} PROPERTY SOVERSION "${VERSION_MAJOR}")
    set_property(TARGET ${RDC_ROCP_LIB} PROPERTY VERSION "${SO_VERSION_STRING}")

    if("${CMAKE_BUILD_TYPE}" STREQUAL Release)
        add_custom_command(
            TARGET ${RDC_ROCP_LIB}
            POST_BUILD
            COMMAND ${CMAKE_STRIP} ${RDC_ROCP_LIB_COMPONENT}.so
        )
    endif()

    # --- rocpctl standalone binary ---
    # This is the subprocess that librdc_rocp.so forks/execs to collect
    # rocprofiler metrics. It links against rocprofiler-sdk and HIP runtime.
    find_package(rocprofiler-sdk 1.1.0 HINTS ${ROCM_DIR}/lib/cmake CONFIG REQUIRED)
    find_package(hsa-runtime64 NAMES hsa-runtime64 HINTS ${ROCM_DIR}/lib/cmake CONFIG REQUIRED)
    find_package(hip CONFIG REQUIRED HINTS ${ROCM_DIR}/lib/cmake)

    # rocpclient shared library (rocprofiler CounterSampler implementation)
    add_library(rocpclient SHARED
        "${SRC_DIR}/rocpctl/rocpclient.cpp"
        "${SRC_DIR}/rocpctl/RocpCounterSampler.h"
    )
    target_link_libraries(rocpclient
        PRIVATE rocprofiler-sdk::rocprofiler-sdk
    )
    target_include_directories(rocpclient
        PRIVATE
            "${SRC_DIR}/rocpctl"
            "${ROCM_DIR}/include"
    )

    # rocpctl executable (uses HIP runtime API for device enumeration)
    add_executable(rocpctl
        "${SRC_DIR}/rocpctl/rocpctl.cpp"
        "${SRC_DIR}/rocpctl/RocpCounterSampler.h"
    )
    target_link_libraries(rocpctl
        PRIVATE rocpclient hip::host pthread
    )
    target_include_directories(rocpctl
        PRIVATE
            "${SRC_DIR}/rocpctl"
            "${ROCM_DIR}/include"
    )

    # Install rocpctl alongside RDC binaries
    install(TARGETS rocpctl rocpclient
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    )
endif()
