#
# Minimum version of cmake required
#

message("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&")
message("                    CMake ROCm SMI (Library)                       ")
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
# message("--------RSMI Inc Dir: " ${RSMI_INC_DIR})
# message("")

set(ROCM_SMI "rocm_smi")
set(ROCM_SMI_COMPONENT "lib${ROCM_SMI}")
set(ROCM_SMI_TARGET "${ROCM_SMI}64")

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
if ( ${ROCM_PATCH_VERSION} )
    set ( VERSION_PATCH ${ROCM_PATCH_VERSION})
    set(SO_VERSION_STRING "${VERSION_MAJOR}.${VERSION_MINOR}.${VERSION_PATCH}")
else()
    set(SO_VERSION_STRING "${VERSION_MAJOR}.${VERSION_MINOR}")
endif ()
set(${ROCM_SMI}_VERSION_MAJOR "${VERSION_MAJOR}")
set(${ROCM_SMI}_VERSION_MINOR "${VERSION_MINOR}")
set(${ROCM_SMI}_VERSION_PATCH "0")
set(${ROCM_SMI}_VERSION_BUILD "0")
message("SOVERSION: ${SO_VERSION_STRING}")


# Create a configure file to get version info from within library
configure_file(
  "${PROJECT_SOURCE_DIR}/src/${ROCM_SMI_TARGET}Config.in"
  "${PROJECT_SOURCE_DIR}/include/rocm_smi/${ROCM_SMI_TARGET}Config.h")

set(RSMI_SRC_DIR "src")
set(RSMI_INC_DIR "include")
set(RSMI_DOCS_DIR "docs")

# Add any rocm_smi_lib specific source files here
set(SMI_SRC_LIST ${CMN_SRC_LIST})

# Add any rocm_smi_lib specific headers here
set(SMI_INC_LIST "")

set(SMI_EXAMPLE_EXE "rocm_smi_ex")

add_executable(${SMI_EXAMPLE_EXE} "example/rocm_smi_example.cc")
target_link_libraries(${SMI_EXAMPLE_EXE} ${ROCM_SMI_TARGET})
add_library(${ROCM_SMI_TARGET} SHARED ${CMN_SRC_LIST} ${SMI_SRC_LIST}
                                              ${CMN_INC_LIST} ${SMI_INC_LIST})
target_link_libraries(${ROCM_SMI_TARGET} pthread rt)
target_include_directories(${ROCM_SMI_TARGET} PRIVATE
          ${CMAKE_CURRENT_SOURCE_DIR} ${COMMON_PROJ_ROOT}/common/shared_mutex)

## Set the VERSION and SOVERSION values
set_property(TARGET ${ROCM_SMI_TARGET} PROPERTY
             SOVERSION "${VERSION_MAJOR}")
set_property(TARGET ${ROCM_SMI_TARGET} PROPERTY
             VERSION "${SO_VERSION_STRING}")

## If the library is a release, strip the target library
if ("${CMAKE_BUILD_TYPE}" STREQUAL Release)
    add_custom_command(
          TARGET ${ROCM_SMI_TARGET}
          POST_BUILD COMMAND ${CMAKE_STRIP} lib${ROCM_SMI_TARGET}.so)
endif ()

## Add symlinks from top level ROCm lib dir to rocm-smi lib so files
add_custom_target ( so-link ALL WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
                    COMMAND ${CMAKE_COMMAND} -E create_symlink
                          ../${ROCM_SMI}/lib/${ROCM_SMI_LIB_NAME}.so so-link )
add_custom_target ( so-major-link ALL WORKING_DIRECTORY
                  ${CMAKE_CURRENT_BINARY_DIR} COMMAND ${CMAKE_COMMAND}
                  -E create_symlink
                  ../${ROCM_SMI}/lib/${ROCM_SMI_LIB_NAME}.so.${VERSION_MAJOR}
                                                               so-major-link )

install ( FILES ${CMAKE_CURRENT_BINARY_DIR}/so-link DESTINATION lib RENAME
                                                     ${ROCM_SMI_LIB_NAME}.so )
install ( FILES ${CMAKE_CURRENT_BINARY_DIR}/so-major-link DESTINATION lib
                             RENAME ${ROCM_SMI_LIB_NAME}.so.${VERSION_MAJOR} )

## Add the install directives for the runtime library.
install(TARGETS ${ROCM_SMI_TARGET}
          LIBRARY DESTINATION ${ROCM_SMI}/lib COMPONENT ${ROCM_SMI_COMPONENT})
install(FILES ${COMMON_SRC_ROOT}/include/rocm_smi/rocm_smi.h
                                        DESTINATION rocm_smi/include/rocm_smi)
install(FILES ${COMMON_SRC_ROOT}/include/rocm_smi/kfd_ioctl.h
                                        DESTINATION rocm_smi/include/rocm_smi)
# Generate Doxygen documentation
find_package(Doxygen)
find_package(LATEX COMPONENTS PDFLATEX)

if (DOXYGEN_FOUND AND LATEX_FOUND)
  set (RSMI_MANUAL_NAME "ROCm_SMI_Manual")
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/docs/amd_smi_doxygen.cfg
                                   ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile @ONLY)

  add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/latex/refman.tex
        COMMAND ${DOXYGEN_EXECUTABLE} ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile
        DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/docs/amd_smi_doxygen.cfg
                "${COMMON_SRC_ROOT}/include/rocm_smi/rocm_smi.h"
         WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
  add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/latex/refman.pdf
         COMMAND make  > /dev/null
         COMMAND cp  ${CMAKE_CURRENT_BINARY_DIR}/latex/refman.pdf
                  ${CMAKE_CURRENT_SOURCE_DIR}/docs/${RSMI_MANUAL_NAME}_new.pdf
         DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/latex/refman.tex
         WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/latex)

  add_custom_target(docs DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/latex/refman.pdf)

  add_dependencies(${ROCM_SMI_TARGET} docs)
  install(FILES ${CMAKE_CURRENT_BINARY_DIR}/latex/refman.pdf
                         DESTINATION ${ROCM_SMI}/docs/${RSMI_MANUAL_NAME}.pdf)
  install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/docs/README.md
                          DESTINATION ${ROCM_SMI}/docs/)
else()
  message("Doxygen or Latex is not found. Will not generate documents.")
endif(DOXYGEN_FOUND AND LATEX_FOUND)

