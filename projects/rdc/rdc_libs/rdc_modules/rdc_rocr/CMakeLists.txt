message("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&")
message("                       Cmake RDC Lib-ROCR                          ")
message("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&")

set(SRC_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
set(INC_DIR "${PROJECT_SOURCE_DIR}/include/rdc_modules/rdc_rocr")

set(RDC_ROCR_LIB_COMPONENT "lib${RDC_ROCR_LIB}")
set(RDC_ROCR_LIB_SRC_LIST
    "${BOOTSTRAP_LIB_SRC_DIR}/RdcLogger.cc"
    "${SRC_DIR}/ComputeQueueTest.cc"
    "${SRC_DIR}/MemoryAccess.cc"
    "${SRC_DIR}/MemoryTest.cc"
    "${SRC_DIR}/RdcDiagnosticLib.cc"
    "${SRC_DIR}/RdcRocrBase.cc"
    "${SRC_DIR}/TestBase.cc"
    "${SRC_DIR}/base_rocr_utils.cc"
    "${SRC_DIR}/common.cc")

set(RDC_ROCR_LIB_INC_LIST
    "${INC_DIR}/MemoryAccess.h"
    "${INC_DIR}/MemoryTest.h"
    "${INC_DIR}/RdcRocrBase.h"
    "${INC_DIR}/TestBase.h"
    "${INC_DIR}/base_rocr_utils.h"
    "${INC_DIR}/common.h"
    "${PROJECT_SOURCE_DIR}/include/rdc/rdc.h"
    "${RDC_LIB_INC_DIR}/RdcLogger.h"
    "${RDC_LIB_INC_DIR}/RdcDiagnosticLibInterface.h"
    "${RDC_LIB_INC_DIR}/rdc_common.h")

if(BUILD_RUNTIME)
    message("Build librdc_rocr.so is enabled, make sure the Rocm run time is installed.")

    message("RDC_ROCR_LIB_INC_LIST=${RDC_ROCR_LIB_INC_LIST}")

    find_package(
        hsa-runtime64
        NAMES
        hsa-runtime64
        HINTS
        ${ROCM_DIR}/lib/cmake
        CONFIGURE
        REQUIRED)
    set(RDC_LIB_MODULES
        ${RDC_LIB_MODULES} ${RDC_ROCR_LIB}
        PARENT_SCOPE)
    add_library(${RDC_ROCR_LIB} SHARED ${RDC_ROCR_LIB_SRC_LIST} ${RDC_ROCR_LIB_INC_LIST})
    target_link_libraries(${RDC_ROCR_LIB} ${RDC_LIB} ${BOOTSTRAP_LIB} hsa-runtime64::hsa-runtime64
                          pthread dl)
    target_include_directories(
        ${RDC_ROCR_LIB} PRIVATE "${PROJECT_SOURCE_DIR}" "${PROJECT_SOURCE_DIR}/include"
                                "${COMMON_DIR}" "${AMD_SMI_INCLUDE_DIR}" "${ROCM_DIR}/include")

    # let bootstrap find this module at build time
    # at install-time the directory structure is a lot simpler
    set_property(
        TARGET ${BOOTSTRAP_LIB}
        APPEND
        PROPERTY BUILD_RPATH "\$ORIGIN/../rdc_modules/${RDC_ROCR_LIB}")

    # Set the VERSION and SOVERSION values
    set_property(TARGET ${RDC_ROCR_LIB} PROPERTY SOVERSION "${VERSION_MAJOR}")
    set_property(TARGET ${RDC_ROCR_LIB} PROPERTY VERSION "${SO_VERSION_STRING}")

    # If the library is a release, strip the target library
    if("${CMAKE_BUILD_TYPE}" STREQUAL Release)
        add_custom_command(
            TARGET ${RDC_ROCR_LIB}
            POST_BUILD
            COMMAND ${CMAKE_STRIP} ${RDC_ROCR_LIB_COMPONENT}.so)
    endif()
endif()
