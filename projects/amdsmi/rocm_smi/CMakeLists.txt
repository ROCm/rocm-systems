message("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&")
message("                    CMake ROCm SMI (Library)                       ")
message("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&")

message("")
message("Build Configuration:")
message("--------Proj Src Dir: " ${PROJECT_SOURCE_DIR})

set(ROCM_SMI "rocm_smi")
set(ROCM_SMI_COMPONENT "lib${ROCM_SMI}")
set(ROCM_SMI_TARGET "${ROCM_SMI}64")

# Used by ESMI
set(SMI_SRC_LIST ${CMN_SRC_LIST})

## Include common cmake modules
include(utils)

################# Determine the library version #########################
set(SO_VERSION_GIT_TAG_PREFIX "rsmi_so_ver")

# VERSION_* variables should be set by get_version_from_tag
message("Package version: ${PKG_VERSION_STR}")

# Debian package specific variables
# Set a default value for the package version
get_version_from_tag("1.0.0.0" ${SO_VERSION_GIT_TAG_PREFIX} GIT)

# VERSION_* variables should be set by get_version_from_tag
if(${ROCM_PATCH_VERSION})
    set(VERSION_PATCH ${ROCM_PATCH_VERSION})
    set(SO_VERSION_STRING "${VERSION_MAJOR}.${VERSION_MINOR}.${VERSION_PATCH}")
else()
    set(SO_VERSION_STRING "${VERSION_MAJOR}.${VERSION_MINOR}")
endif()
set(${ROCM_SMI}_VERSION_MAJOR "${CPACK_PACKAGE_VERSION_MAJOR}")
set(${ROCM_SMI}_VERSION_MINOR "${CPACK_PACKAGE_VERSION_MINOR}")
set(${ROCM_SMI}_VERSION_PATCH "${CPACK_PACKAGE_VERSION_PATCH}")
set(${ROCM_SMI}_VERSION_BUILD "0")
set(${ROCM_SMI}_VERSION_HASH "${PKG_VERSION_HASH}")
message("SOVERSION: ${SO_VERSION_STRING}")

# Create a configure file to get version info from within library
configure_file("src/${ROCM_SMI_TARGET}Config.in"
               "${CMAKE_CURRENT_SOURCE_DIR}/include/rocm_smi/${ROCM_SMI_TARGET}Config.h")

add_library(${ROCM_SMI_TARGET} ${CMN_SRC_LIST} ${CMN_INC_LIST})
target_link_libraries(${ROCM_SMI_TARGET} PRIVATE
    rt
    Threads::Threads
    ${CMAKE_DL_LIBS}
)
target_include_directories(${ROCM_SMI_TARGET} PRIVATE 
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/common/shared_mutex
    ${DRM_INCLUDE_DIRS}
    ${DRM_AMDGPU_INCLUDE_DIRS}
)

# use the target_include_directories() command to specify the include directories for the target
target_include_directories(${ROCM_SMI_TARGET} PUBLIC "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
                                                     "$<INSTALL_INTERFACE:${ROCM_SMI}/include>")

## Set the VERSION and SOVERSION values
set_property(TARGET ${ROCM_SMI_TARGET} PROPERTY SOVERSION "${VERSION_MAJOR}")
set_property(TARGET ${ROCM_SMI_TARGET} PROPERTY VERSION "${SO_VERSION_STRING}")

## If the library is a release, strip the target library
if("${CMAKE_BUILD_TYPE}" STREQUAL Release)
    if(${BUILD_SHARED_LIBS}) #stripping only for .so
        add_custom_command(TARGET ${ROCM_SMI_TARGET} POST_BUILD COMMAND ${CMAKE_STRIP}
                                                                        lib${ROCM_SMI_TARGET}.so.${SO_VERSION_STRING})
    endif()
endif()
