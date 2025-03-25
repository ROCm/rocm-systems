message("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&")
message("                       Cmake RDC Lib-ROCP                          ")
message("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&")

set(SRC_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
set(INC_DIR "${PROJECT_SOURCE_DIR}/include/rdc_modules/rdc_rocp")

set(RDC_ROCP_LIB_COMPONENT "lib${RDC_ROCP_LIB}")
set(RDC_ROCP_LIB_SRC_LIST "${BOOTSTRAP_LIB_SRC_DIR}/RdcLogger.cc" "${SRC_DIR}/RdcTelemetryLib.cc"
                          "${SRC_DIR}/RdcRocpCounterSampler.cc" "${SRC_DIR}/RdcRocpBase.cc")
set(RDC_ROCP_LIB_INC_LIST
    "${PROJECT_SOURCE_DIR}/include/rdc/rdc.h"
    "${RDC_LIB_INC_DIR}/RdcDiagnosticLibInterface.h"
    "${RDC_LIB_INC_DIR}/rdc_common.h"
    "${RDC_LIB_INC_DIR}/RdcLogger.h"
    "${SRC_DIR}/../../rdc/src/SmiUtils.cc"
    "${INC_DIR}/RdcRocpBase.h"
    "${INC_DIR}/RdcRocpCounterSampler.h")

if(BUILD_PROFILER)
    message("Build librdc_rocp.so is enabled, make sure ROCmTools is installed.")

    message("RDC_ROCP_LIB_INC_LIST=${RDC_ROCP_LIB_INC_LIST}")

    find_package(rocprofiler-sdk 1.0.0 HINTS ${ROCM_DIR}/lib/cmake CONFIGURE REQUIRED)
    find_package(
        hsa-runtime64
        NAMES
        hsa-runtime64
        HINTS
        ${ROCM_DIR}/lib/cmake
        CONFIGURE
        REQUIRED)
    set(RDC_LIB_MODULES
        ${RDC_LIB_MODULES} ${RDC_ROCP_LIB}
        PARENT_SCOPE)
    add_library(${RDC_ROCP_LIB} SHARED ${RDC_ROCP_LIB_SRC_LIST} ${RDC_ROCP_LIB_INC_LIST})
    target_link_libraries(
        ${RDC_ROCP_LIB} PRIVATE hsa-runtime64::hsa-runtime64 rocprofiler-sdk::rocprofiler-sdk
                                pthread dl amd_smi)
    target_include_directories(
        ${RDC_ROCP_LIB}
        PRIVATE "${PROJECT_SOURCE_DIR}" "${PROJECT_SOURCE_DIR}/include" "${COMMON_DIR}"
                "${AMD_SMI_INCLUDE_DIR}" "${ROCM_DIR}/include" "${ROCM_DIR}/include/hsa")

    # let bootstrap find this module at build time
    # at install-time the directory structure is a lot simpler
    set_property(
        TARGET ${BOOTSTRAP_LIB}
        APPEND
        PROPERTY BUILD_RPATH "\$ORIGIN/../rdc_modules/${RDC_ROCP_LIB}")

    # Set the VERSION and SOVERSION values
    set_property(TARGET ${RDC_ROCP_LIB} PROPERTY SOVERSION "${VERSION_MAJOR}")
    set_property(TARGET ${RDC_ROCP_LIB} PROPERTY VERSION "${SO_VERSION_STRING}")

    # If the library is a release, strip the target library
    if("${CMAKE_BUILD_TYPE}" STREQUAL Release)
        add_custom_command(
            TARGET ${RDC_ROCP_LIB}
            POST_BUILD
            COMMAND ${CMAKE_STRIP} ${RDC_ROCP_LIB_COMPONENT}.so)
    endif()
endif()
