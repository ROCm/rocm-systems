message("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&")
message("                       Cmake RDC Lib-Bootstrap                     ")
message("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&")

set(SRC_DIR "${CMAKE_CURRENT_SOURCE_DIR}/src")
set(INC_DIR "${RDC_LIB_INC_DIR}")
# need source dir for rocr
set(BOOTSTRAP_LIB_SRC_DIR "${SRC_DIR}" PARENT_SCOPE)

set(BOOTSTRAP_LIB_COMPONENT "lib${BOOTSTRAP_LIB}")
set(BOOTSTRAP_LIB_SRC_LIST
    "${COMMON_DIR}/rdc_fields_supported.cc"
    "${SRC_DIR}/RdcBootStrap.cc"
    "${SRC_DIR}/RdcLibraryLoader.cc"
    "${SRC_DIR}/RdcLogger.cc"
    "${SRC_DIR}/RdcEntityCodec.cc"
)
set(BOOTSTRAP_LIB_INC_LIST
    "${COMMON_DIR}/rdc_fields_supported.h"
    "${INC_DIR}/RdcHandler.h"
    "${INC_DIR}/RdcLibraryLoader.h"
    "${INC_DIR}/RdcLogger.h"
    "${INC_DIR}/RdcEntityCodec.h"
    "${INC_DIR}/rdc_common.h"
    "${PROJECT_SOURCE_DIR}/include/rdc/rdc.h"
)
message("BOOTSTRAP_LIB_INC_LIST=${BOOTSTRAP_LIB_INC_LIST}")

add_library(${BOOTSTRAP_LIB} SHARED ${BOOTSTRAP_LIB_SRC_LIST} ${BOOTSTRAP_LIB_INC_LIST})
target_link_libraries(${BOOTSTRAP_LIB} pthread dl)
target_include_directories(
    ${BOOTSTRAP_LIB}
    PRIVATE
        "${PROJECT_SOURCE_DIR}"
        "${PROJECT_SOURCE_DIR}/include"
        "${COMMON_DIR}"
        "${AMD_SMI_INCLUDE_DIR}"
        "${ROCM_DIR}/include"
)

target_include_directories(
    ${BOOTSTRAP_LIB}
    PUBLIC
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

# Set the VERSION and SOVERSION values
set_property(TARGET ${BOOTSTRAP_LIB} PROPERTY SOVERSION "${VERSION_MAJOR}")
set_property(TARGET ${BOOTSTRAP_LIB} PROPERTY VERSION "${SO_VERSION_STRING}")
set_target_properties(${BOOTSTRAP_LIB} PROPERTIES INSTALL_RPATH "\$ORIGIN:\$ORIGIN/rdc")

# If the library is a release, strip the target library
if("${CMAKE_BUILD_TYPE}" STREQUAL Release)
    add_custom_command(
        TARGET ${BOOTSTRAP_LIB}
        POST_BUILD
        COMMAND ${CMAKE_STRIP} ${BOOTSTRAP_LIB_COMPONENT}.so
    )
endif()
