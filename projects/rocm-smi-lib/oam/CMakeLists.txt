#
# Minimum version of cmake required
#

message("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&")
message("                    CMake OAM (Library)                            ")
message("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&")

## Verbose output.
set(CMAKE_VERBOSE_MAKEFILE on)

# Required Defines first:

message("")
message("Build Configuration:")
# message("-----------BuildType: " ${CMAKE_BUILD_TYPE})
# message("------------Compiler: " ${CMAKE_CXX_COMPILER})
# message("-------------Version: " ${CMAKE_CXX_COMPILER_VERSION})
message("--------Proj Src Dir: " ${PROJECT_SOURCE_DIR})
# message("--------Proj Bld Dir: " ${PROJECT_BINARY_DIR})
# message("--------Proj Lib Dir: " ${PROJECT_BINARY_DIR}/lib)
# message("--------Proj Exe Dir: " ${PROJECT_BINARY_DIR}/bin)
# message("--------RSMI Lib Dir: " ${RSMI_LIB_DIR})
# message("--------RSMI Inc Dir: " ${OAM_INC_DIR})
# message("")

set(OAM_ROOT "${PROJECT_SOURCE_DIR}/oam")
set(OAM_NAME "oam")
set(OAM_COMPONENT "lib${OAM_NAME}")
set(OAM_TARGET "${OAM_NAME}")

################# Determine the library version #########################
set(SO_VERSION_GIT_TAG_PREFIX "oam_so_ver")

# VERSION_* variables should be set by get_version_from_tag
message("Package version: ${PKG_VERSION_STR}")

# Debian package specific variables
# Set a default value for the package version
get_version_from_tag("1.0.0.0" ${SO_VERSION_GIT_TAG_PREFIX} GIT)

# VERSION_* variables should be set by get_version_from_tag
if ( ${ROCM_PATCH_VERSION} )
    set ( VERSION_PATCH ${ROCM_PATCH_VERSION})
    set(SO_VERSION_STRING "${VERSION_MAJOR}.${VERSION_MINOR}.${VERSION_PATCH}")
else()
    set(SO_VERSION_STRING "${VERSION_MAJOR}.${VERSION_MINOR}")
endif ()
set(${OAM_NAME}_VERSION_MAJOR "${VERSION_MAJOR}")
set(${OAM_NAME}_VERSION_MINOR "${VERSION_MINOR}")
set(${OAM_NAME}_VERSION_PATCH "0")
set(${OAM_NAME}_VERSION_BUILD "0")
message("SOVERSION: ${SO_VERSION_STRING}")


# Create a configure file to get version info from within library
configure_file(
  "${OAM_ROOT}/src/${OAM_TARGET}Config.in"
  "${OAM_ROOT}/include/oam/${OAM_TARGET}Config.h")

set(OAM_SRC_DIR "src")
set(OAM_INC_DIR "include")
set(OAM_DOCS_DIR "docs")

set(OAM_SRC_LIST ${CMN_SRC_LIST} "${OAM_SRC_DIR}/amd_oam.cc")

set(OAM_INC_LIST ${COMMON_INC_DIR} "${OAM_INC_DIR}")
set(OAM_EXAMPLE_EXE "oam_ex")

add_executable(${OAM_EXAMPLE_EXE} "example/oam_example.c")
target_include_directories(${OAM_EXAMPLE_EXE} PRIVATE ${OAM_INC_LIST})
target_link_libraries(${OAM_EXAMPLE_EXE} ${OAM_TARGET})
add_library(${OAM_TARGET} ${CMN_SRC_LIST} ${OAM_SRC_LIST}
                                              ${CMN_INC_LIST} ${OAM_INC_LIST})
target_link_libraries(${OAM_TARGET} PRIVATE pthread rt dl)
target_include_directories(${OAM_TARGET} PRIVATE
   ${CMAKE_CURRENT_SOURCE_DIR}/include ${COMMON_PROJ_ROOT}/common/shared_mutex)

## Set the VERSION and SOVERSION values
set_property(TARGET ${OAM_TARGET} PROPERTY
             SOVERSION "${VERSION_MAJOR}")
set_property(TARGET ${OAM_TARGET} PROPERTY
             VERSION "${SO_VERSION_STRING}")

## If the library is a release, strip the target library
if ("${CMAKE_BUILD_TYPE}" STREQUAL Release)
    if(${BUILD_SHARED_LIBS}) #striping only for .so
        add_custom_command(
             TARGET ${OAM_TARGET}
             POST_BUILD COMMAND ${CMAKE_STRIP} lib${OAM_TARGET}.so)
    endif()
endif ()

# use the target_include_directories() command to specify the include directories for the target
target_include_directories(${OAM_TARGET}
                           PUBLIC
                           "$<BUILD_INTERFACE:${DRM_INCLUDE_DIRS}>"
                           "$<BUILD_INTERFACE:${AMDGPU_DRM_INCLUDE_DIRS}>"
                           "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
                           "$<INSTALL_INTERFACE:{OAM_NAME}/include>")

## Add the install directives for the runtime library.
install(TARGETS ${OAM_TARGET}
          EXPORT rocm_smiTargets
          LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
          COMPONENT dev)
install(TARGETS ${OAM_TARGET}
          LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
          COMPONENT asan)

install(FILES ${COMMON_SRC_ROOT}/oam/include/oam/oam_mapi.h
              ${COMMON_SRC_ROOT}/oam/include/oam/amd_oam.h
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/oam
        COMPONENT dev)

# Generate Doxygen documentation
if (DOXYGEN_FOUND)
  configure_file(${OAM_DOCS_DIR}/docs/rsmi_doxygen.cfg
                                   ${OAM_DOCS_DIR}/Doxyfile @ONLY)
  add_custom_target(doc
         ${DOXYGEN_EXECUTABLE} ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile
         WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
          COMMENT "Generating AMD OAM API documentation with Doxygen" VERBATIM)
endif(DOXYGEN_FOUND)

