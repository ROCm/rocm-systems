############################### GENERAL ###############################

set(CPACK_PACKAGE_NAME ${PROJECT_NAME})
if(DEFINED ROCPROFILER_PACKAGE_EXTRA_NAME)
    set(CPACK_PACKAGE_NAME "${PROJECT_NAME}-${ROCPROFILER_PACKAGE_EXTRA_NAME}")
endif()
set(CPACK_PACKAGE_VENDOR "Advanced Micro Devices, Inc.")
set(CPACK_PACKAGE_DESCRIPTION_SUMMARY "Thread Trace Decoder for ROCProf ATT Output")
set(CPACK_PACKAGE_PROJECT_VERSION_MAJOR "${PROJECT_VERSION_MAJOR}")
set(CPACK_PACKAGE_PROJECT_VERSION_MINOR "${PROJECT_VERSION_MINOR}")
set(CPACK_PACKAGE_PROJECT_VERSION_PATCH "${PROJECT_VERSION_PATCH}")
set(CPACK_PACKAGE_CONTACT "ROCm Profiler Support <dl.ROCm-Profiler.support@amd.com>")
set(CPACK_RESOURCE_FILE_LICENSE "${TTD_PROJECT_SOURCE_DIR}/LICENSE")

if(DEFINED CPACK_PACKAGE_INSTALL_DIRECTORY)
    set(CPACK_SET_DESTDIR true)
    set(CPACK_INSTALL_PREFIX ${CPACK_PACKAGE_INSTALL_DIRECTORY})
endif()

if(DEFINED ENV{CPACK_PACKAGE_FILE_NAME})
    set(CPACK_PACKAGE_FILE_NAME $ENV{CPACK_PACKAGE_FILE_NAME})
endif()

if(NOT DEFINED CPACK_GENERATOR)
    set(CPACK_GENERATOR "STGZ" "DEB" "RPM" "TGZ")
endif()

set(CPACK_PACKAGE_VERSION
    "${CPACK_PACKAGE_PROJECT_VERSION_MAJOR}.${CPACK_PACKAGE_PROJECT_VERSION_MINOR}.${CPACK_PACKAGE_PROJECT_VERSION_PATCH}"
)

############################### DEBIAN ###############################

## Debian package specific variables
set(CPACK_DEBIAN_PACKAGE_HOMEPAGE "https://github.com/ROCm/rocprof-trace-decoder")

# Enable Component Mode & install settings.
set(CPACK_DEB_COMPONENT_INSTALL ON)
set(CPACK_DEBIAN_RUNTIME_PACKAGE_NAME "${PROJECT_NAME}")
set(CPACK_DEBIAN_ASAN_PACKAGE_NAME "${PROJECT_NAME}-asan")
set(CPACK_DEBIAN_TESTS_PACKAGE_NAME "${PROJECT_NAME}-tests")

############################### RPM ###############################

#Disable build id for rocprofiler as its creating transaction error
# set ( CPACK_RPM_SPEC_MORE_DEFINE "%define _build_id_links none
#                                     %global __strip ${CPACK_STRIP_EXECUTABLE}
#                                     %global __objdump ${CPACK_OBJDUMP_EXECUTABLE}
#                                     %global __objcopy ${CPACK_OBJCOPY_EXECUTABLE}
#                                     %global __readelf ${CPACK_READELF_EXECUTABLE}")

## 'dist' breaks manual builds on debian systems due to empty Provides
# execute_process( COMMAND rpm --eval %{?dist}
#                  RESULT_VARIABLE PROC_RESULT
#                  OUTPUT_VARIABLE EVAL_RESULT
#                  OUTPUT_STRIP_TRAILING_WHITESPACE )

# if ( PROC_RESULT EQUAL "0" AND NOT EVAL_RESULT STREQUAL "" )
#   string ( APPEND CPACK_RPM_PACKAGE_RELEASE "%{?dist}" )
# endif()
# set ( CPACK_RPM_FILE_NAME "RPM-DEFAULT" )
# if ( DEFINED CPACK_PACKAGING_INSTALL_PREFIX )
#     set ( CPACK_RPM_EXCLUDE_FROM_AUTO_FILELIST_ADDITION "${CPACK_PACKAGING_INSTALL_PREFIX}" )
# endif ( )

# Enable Component Mode & install settings.
set(CPACK_RPM_COMPONENT_INSTALL ON)
set(CPACK_RPM_RUNTIME_PACKAGE_NAME "${PROJECT_NAME}")
set(CPACK_RPM_ASAN_PACKAGE_NAME "${PROJECT_NAME}-asan")
set(CPACK_RPM_TESTS_PACKAGE_NAME "${PROJECT_NAME}-tests")

include(CPack)
