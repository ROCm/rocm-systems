message("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&")
message("                  CMake AMD SMI C/C++ Library                      ")
message("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&")

message("")
message("Build Configuration:")
message("--------Proj Src Dir: " ${PROJECT_SOURCE_DIR})

## Include common cmake modules
include(utils)

set(SO_VERSION_GIT_TAG_PREFIX "amdsmi_so_ver")

set(SRC_DIR "amd_smi")
set(INC_DIR "${PROJECT_SOURCE_DIR}/include/amd_smi")

set(SRC_LIST
    "${SRC_DIR}/amd_smi.cc"
    "${SRC_DIR}/amd_smi_common.cc"
    "${SRC_DIR}/amd_smi_utils.cc"
    "${SRC_DIR}/amd_smi_uuid.cc"
    "${CMN_SRC_LIST}"
)

file(GLOB IMPL_HEADERS "${INC_DIR}/impl/*.h")
set(INC_LIST
    "${IMPL_HEADERS}"
    "${INC_DIR}/amdsmi.h"
    "${PROJECT_SOURCE_DIR}"
)

message("Package version: ${AMDSMI_PACKAGE_VERSION}")
set(SO_VERSION_STRING "${AMDSMI_PACKAGE_VERSION}")
message("SOVERSION: ${SO_VERSION_STRING}")

add_library(${AMD_SMI} ${SRC_LIST} ${INC_LIST})
target_compile_definitions(${AMD_SMI} PRIVATE
    AMDSMI_LIB_VERSION_MAJOR=${CPACK_PACKAGE_VERSION_MAJOR}
    AMDSMI_LIB_VERSION_MINOR=${CPACK_PACKAGE_VERSION_MINOR}
    AMDSMI_LIB_VERSION_RELEASE=${CPACK_PACKAGE_VERSION_PATCH}
    AMDSMI_LIB_VERSION_STRING=\"${${AMD_SMI_LIBS_TARGET}_VERSION_STRING}\"
)
target_link_libraries(${AMD_SMI} PRIVATE
    -Wl,--whole-archive thunk_proxy dxg_shared -Wl,--no-whole-archive
    rt
    Threads::Threads
    ${CMAKE_DL_LIBS}
    ${FILESYSTEM_LIB}
)
target_include_directories(${AMD_SMI} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/../
    ${PROJECT_SOURCE_DIR}/common/shared_mutex
    ${PROJECT_SOURCE_DIR}/../include
)
target_include_directories(${AMD_SMI} SYSTEM PRIVATE
    ${WIN_SDK}
)
if(WIN_SDK)
    add_compile_definitions(LINUX __AMD64__)
endif()

# use the target_include_directories() command to specify the include directories for the target
target_include_directories(${AMD_SMI} PUBLIC 
    "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
                                             "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>")

## Set the VERSION and SOVERSION values
set_property(TARGET ${AMD_SMI} PROPERTY SOVERSION "${CPACK_PACKAGE_VERSION_MAJOR}")
set_property(TARGET ${AMD_SMI} PROPERTY VERSION "${SO_VERSION_STRING}")

## If the library is a release, strip the target library
if("${CMAKE_BUILD_TYPE}" STREQUAL Release)
    if(${BUILD_SHARED_LIBS}) #stripping only for .so
        add_custom_command(TARGET ${AMD_SMI} POST_BUILD COMMAND ${CMAKE_STRIP} lib${AMD_SMI}.so.${SO_VERSION_STRING})
    endif()
endif()

## Add the install directives for the runtime library.
if(BUILD_SHARED_LIBS)
install(
    TARGETS ${AMD_SMI}
    EXPORT amd_smiTargets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT dev)
endif()

install(
    TARGETS ${AMD_SMI}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT asan)

install(
    FILES ${PROJECT_SOURCE_DIR}/include/amd_smi/amdsmi.h
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/amd_smi
    COMPONENT dev)

install(
    FILES ${IMPL_HEADERS} 
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/amd_smi/impl/
    COMPONENT dev)

file(GLOB ESMI_HEADERS "${PROJECT_SOURCE_DIR}/esmi_ib_library/include/e_smi/*.h")
install(
    FILES ${ESMI_HEADERS}
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/e_smi/
    COMPONENT dev)

