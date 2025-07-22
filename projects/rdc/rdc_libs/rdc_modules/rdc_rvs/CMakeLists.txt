message("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&")
message("                       Cmake RDC Lib-RVS                          ")
message("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&")

set(SRC_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
set(INC_DIR "${PROJECT_SOURCE_DIR}/include/rdc_modules/rdc_rvs")

set(RDC_RVS_LIB_COMPONENT "lib${RDC_RVS_LIB}")
set(RDC_RVS_LIB_SRC_LIST
    "${BOOTSTRAP_LIB_SRC_DIR}/RdcLogger.cc"
    "${SRC_DIR}/RvsBase.cc"
    "${SRC_DIR}/RdcDiagnosticLib.cc"
)
set(RDC_RVS_LIB_INC_LIST
    "${PROJECT_SOURCE_DIR}/include/rdc/rdc.h"
    "${RDC_LIB_INC_DIR}/RdcDiagnostic.h"
    "${RDC_LIB_INC_DIR}/RdcDiagnosticLibInterface.h"
    "${RDC_LIB_INC_DIR}/rdc_common.h"
    "${RDC_LIB_INC_DIR}/RdcLogger.h"
    "${INC_DIR}/RvsBase.h"
)

if(BUILD_RVS)
    message("Build librdc_rvs.so is enabled, make sure RVS is installed.")

    message("RDC_RVS_LIB_INC_LIST=${RDC_RVS_LIB_INC_LIST}")

    # below provides rvs::rvs package
    include(Findrvs)

    # needed to find extra packages
    list(APPEND CMAKE_PREFIX_PATH ${ROCM_DIR})
    find_package(rvs REQUIRED)

    set(RDC_LIB_MODULES ${RDC_LIB_MODULES} ${RDC_RVS_LIB} PARENT_SCOPE)
    add_library(${RDC_RVS_LIB} SHARED ${RDC_RVS_LIB_SRC_LIST} ${RDC_RVS_LIB_INC_LIST})
    target_link_libraries(
        ${RDC_RVS_LIB}
        PRIVATE ${RDC_LIB} ${BOOTSTRAP_LIB} rvs::rvs pthread dl
    )
    target_include_directories(
        ${RDC_RVS_LIB}
        PRIVATE
            "${PROJECT_SOURCE_DIR}"
            "${PROJECT_SOURCE_DIR}/include"
            "${COMMON_DIR}"
            "${AMD_SMI_INCLUDE_DIR}"
            "${ROCM_DIR}/include"
            "${ROCM_DIR}/include/hsa"
            "${ROCM_VALIDATION_SUITE_INCLUDE_DIR}"
    )

    # let bootstrap find this module at build time
    # at install-time the directory structure is a lot simpler
    set_property(
        TARGET ${BOOTSTRAP_LIB}
        APPEND
        PROPERTY BUILD_RPATH "\$ORIGIN/../rdc_modules/${RDC_RVS_LIB}"
    )

    # Set the VERSION and SOVERSION values
    set_property(TARGET ${RDC_RVS_LIB} PROPERTY SOVERSION "${VERSION_MAJOR}")
    set_property(TARGET ${RDC_RVS_LIB} PROPERTY VERSION "${SO_VERSION_STRING}")

    # If the library is a release, strip the target library
    if("${CMAKE_BUILD_TYPE}" STREQUAL Release)
        add_custom_command(
            TARGET ${RDC_RVS_LIB}
            POST_BUILD
            COMMAND ${CMAKE_STRIP} ${RDC_RVS_LIB_COMPONENT}.so
        )
    endif()

    # Install RVS config files into /opt/rocm/share/rdc/conf/rvs/
    install(
        DIRECTORY "${SRC_DIR}/conf/"
        DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/${RDC}/conf/rvs/
        COMPONENT ${SERVER_COMPONENT}
    )
endif()
