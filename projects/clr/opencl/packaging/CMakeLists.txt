cmake_minimum_required(VERSION 3.16.8)
project(rocm-opencl)

set(CPACK_COMPONENTS_ALL binary dev)
if(BUILD_ICD)
set(CPACK_COMPONENTS_ALL "${CPACK_COMPONENTS_ALL}" icd)
endif()
if(BUILD_TESTS)
set(CPACK_COMPONENTS_ALL "${CPACK_COMPONENTS_ALL}" ocltst)
endif()

if(ENABLE_ASAN_PACKAGING)
  set(CPACK_COMPONENTS_ALL asan)
endif()

set(CPACK_DEB_COMPONENT_INSTALL ON)
set(CPACK_RPM_COMPONENT_INSTALL ON)

install(TARGETS clinfo DESTINATION ${CMAKE_INSTALL_BINDIR} COMPONENT binary)
install(TARGETS amdocl DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT binary)
install(TARGETS amdocl DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT asan)
install(FILES ${opencl_SOURCE_DIR}/LICENSE.md DESTINATION ${CMAKE_INSTALL_DOCDIR} COMPONENT binary)
install(FILES ${opencl_SOURCE_DIR}/LICENSE.md DESTINATION ${CMAKE_INSTALL_DOCDIR}-asan COMPONENT asan)

install(DIRECTORY ${opencl_SOURCE_DIR}/khronos/headers/opencl2.2/CL
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR} COMPONENT dev
        USE_SOURCE_PERMISSIONS
        PATTERN cl_d3d10.h EXCLUDE
        PATTERN cl_d3d11.h EXCLUDE
        PATTERN cl_dx9_media_sharing.h EXCLUDE )

if(BUILD_ICD)
  install(TARGETS OpenCL DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT icd )
  install(TARGETS OpenCL DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT asan )
  install(FILES ${opencl_SOURCE_DIR}/khronos/icd/LICENSE DESTINATION ${CMAKE_INSTALL_DATADIR}/doc/rocm-ocl-icd COMPONENT icd)
endif()


# Generic CPACK variables
set(CPACK_GENERATOR "DEB;RPM" CACHE STRING "Package types to build")
set(CPACK_PACKAGE_CONTACT "ROCm OpenCL Support <rocm-opencl.support@amd.com>")
set(CPACK_PACKAGE_VENDOR "Advanced Micro Devices, Inc.")
if(ENABLE_ASAN_PACKAGING)
  set(CPACK_PACKAGE_DESCRIPTION "OpenCL: AddressSanitizer Instrumented libraries for Open Computing Language on ROCclr")
else()
  set(CPACK_PACKAGE_DESCRIPTION "OpenCL: Open Computing Language on ROCclr")
endif()

#########Binary-Package###############
set(CPACK_DEBIAN_BINARY_PACKAGE_NAME "rocm-opencl")
set(CPACK_RPM_BINARY_PACKAGE_NAME "rocm-opencl")

# Backward compatibility support for deprecated package: rocm-opencl-icd-loader
set(CPACK_DEBIAN_BINARY_PACKAGE_REPLACES "rocm-opencl-icd-loader")
set(CPACK_DEBIAN_BINARY_PACKAGE_CONFLICTS "rocm-opencl-icd-loader")
set(CPACK_RPM_BINARY_PACKAGE_OBSOLETES "rocm-opencl-icd-loader")

set(CPACK_DEBIAN_BINARY_PACKAGE_CONTROL_EXTRA "${CMAKE_BINARY_DIR}/opencl/packages/opencl/postinst;${CMAKE_BINARY_DIR}/opencl/packages/opencl/postrm")
# RPM CPACK variables
set(CPACK_RPM_BINARY_POST_INSTALL_SCRIPT_FILE "${CMAKE_BINARY_DIR}/opencl/packages/opencl/rpm_post")
set(CPACK_RPM_BINARY_POST_UNINSTALL_SCRIPT_FILE "${CMAKE_BINARY_DIR}/opencl/packages/opencl/rpm_postun")

## Set ASAN package name
set(CPACK_DEBIAN_ASAN_PACKAGE_NAME "rocm-opencl-asan")
set(CPACK_RPM_ASAN_PACKAGE_NAME "rocm-opencl-asan")

# Debian CPACK variables
set(CPACK_BINARY_DEB "ON")
if(DEFINED ENV{CPACK_DEBIAN_PACKAGE_RELEASE})
  set(CPACK_DEBIAN_PACKAGE_RELEASE $ENV{CPACK_DEBIAN_PACKAGE_RELEASE})
else()
  set(CPACK_DEBIAN_PACKAGE_RELEASE "local")
endif()
message("Using CPACK_DEBIAN_PACKAGE_RELEASE ${CPACK_DEBIAN_PACKAGE_RELEASE}")

set(CPACK_DEBIAN_FILE_NAME "DEB-DEFAULT")
if(BUILD_ICD)
  set(CPACK_DEBIAN_BINARY_PACKAGE_DEPENDS "libelf-dev, comgr, hsa-rocr, rocm-core")
  set(CPACK_DEBIAN_ASAN_PACKAGE_DEPENDS "libelf-dev, comgr-asan, hsa-rocr-asan, rocm-core-asan")
else()
  set(CPACK_DEBIAN_BINARY_PACKAGE_DEPENDS "libelf-dev, comgr, hsa-rocr, rocm-core, ocl-icd-libopencl1")
  set(CPACK_DEBIAN_ASAN_PACKAGE_DEPENDS "libelf-dev, comgr-asan, hsa-rocr-asan, rocm-core-asan")
endif()

# RPM CPACK variables
set(CPACK_BINARY_RPM "ON")
if(BUILD_ICD)
  set(CPACK_RPM_BINARY_PACKAGE_REQUIRES "comgr, hsa-rocr, rocm-core")
  set(CPACK_RPM_ASAN_PACKAGE_REQUIRES "comgr-asan, hsa-rocr-asan, rocm-core-asan")
else()
  set(OCL_ICD_PACKAGE "ocl-icd")
  if(${CMAKE_SYSTEM_NAME} STREQUAL "Linux")
    file(READ "/etc/os-release" OS_RELEASE_CONTENT)
    string(TOLOWER "${OS_RELEASE_CONTENT}" OS_RELEASE_CONTENT_LOWER)
    if(OS_RELEASE_CONTENT_LOWER MATCHES "suse" OR OS_RELEASE_CONTENT_LOWER MATCHES "sles")
      set(OCL_ICD_PACKAGE "ocl-icd-devel")
    endif()
  endif()
  set(CPACK_RPM_BINARY_PACKAGE_REQUIRES "comgr, hsa-rocr, rocm-core, ${OCL_ICD_PACKAGE}")
  set(CPACK_RPM_ASAN_PACKAGE_REQUIRES "comgr-asan, hsa-rocr-asan, rocm-core-asan")
endif()

#Unable to set CPACK_RPM_BIANRY_PACKAGE_LICENSE to control individual pacakge license
#Hence combining the license for BINARY,DEV,ICD to one
set(CPACK_RPM_PACKAGE_LICENSE "MIT and ASL 2.0")

#########Dev-Package###############
set(CPACK_DEBIAN_DEV_PACKAGE_NAME "rocm-opencl-dev")
set(CPACK_RPM_DEV_PACKAGE_NAME "rocm-opencl-devel")

if(BUILD_ICD)
  set(CPACK_DEBIAN_DEV_PACKAGE_DEPENDS "mesa-common-dev, rocm-opencl, hsa-rocr-dev, rocm-core")
  set(CPACK_RPM_DEV_PACKAGE_REQUIRES "rocm-opencl, hsa-rocr-devel, rocm-core")
else()
  set(CPACK_DEBIAN_DEV_PACKAGE_DEPENDS "mesa-common-dev, rocm-opencl, hsa-rocr-dev, rocm-core, ocl-icd-opencl-dev")
  set(CPACK_RPM_DEV_PACKAGE_REQUIRES "rocm-opencl, hsa-rocr-devel, rocm-core, ocl-icd-devel")
endif()
#############################
# ICD Loader
#############################

# Debian CPACK variables
if(BUILD_ICD)
  set(CPACK_ICD_DEB "ON")
  set(CPACK_DEBIAN_ICD_PACKAGE_NAME "rocm-ocl-icd")

  set(CPACK_DEBIAN_ICD_PACKAGE_CONTROL_EXTRA "${CMAKE_BINARY_DIR}/opencl/packages/rocm-ocl-icd/postinst;${CMAKE_BINARY_DIR}/opencl/packages/rocm-ocl-icd/postrm")
  set(CPACK_DEBIAN_ICD_PACKAGE_DEPENDS "rocm-core")

# RPM CPACK variables
  set(CPACK_ICD_RPM "ON")
  set(CPACK_RPM_ICD_PACKAGE_NAME "rocm-ocl-icd")



  set(CPACK_RPM_ICD_POST_INSTALL_SCRIPT_FILE "${CMAKE_BINARY_DIR}/opencl/packages/rocm-ocl-icd/rpm_post")
  set(CPACK_RPM_ICD_POST_UNINSTALL_SCRIPT_FILE "${CMAKE_BINARY_DIR}/opencl/packages/rocm-ocl-icd/rpm_postun")
  set(CPACK_RPM_ICD_PACKAGE_REQUIRES "rocm-core")
endif()

if(BUILD_TESTS)
  set(CPACK_OCLTST_DEB "ON")
  set(CPACK_DEBIAN_OCLTST_PACKAGE_NAME "rocm-ocltst")
  set(CPACK_OCLTST_RPM "ON")
  set(CPACK_RPM_OCLTST_PACKAGE_NAME "rocm-ocltst")
endif()

if(DEFINED ENV{CPACK_RPM_PACKAGE_RELEASE})
  set(CPACK_RPM_PACKAGE_RELEASE $ENV{CPACK_RPM_PACKAGE_RELEASE})
else()
  set(CPACK_RPM_PACKAGE_RELEASE "local")
endif()
message("Using CPACK_RPM_PACKAGE_RELEASE ${CPACK_RPM_PACKAGE_RELEASE}")


set(CPACK_RPM_FILE_NAME "RPM-DEFAULT")
set(CPACK_RPM_PACKAGE_AUTOREQPROV " no")

## 'dist' breaks manual builds on debian systems due to empty Provides
execute_process(COMMAND rpm --eval %{?dist}
                 RESULT_VARIABLE PROC_RESULT
                 OUTPUT_VARIABLE EVAL_RESULT
                 OUTPUT_STRIP_TRAILING_WHITESPACE)
message("RESULT_VARIABLE ${PROC_RESULT} OUTPUT_VARIABLE: ${EVAL_RESULT}")

if(PROC_RESULT EQUAL "0" AND NOT EVAL_RESULT STREQUAL "")
  string(APPEND CPACK_RPM_PACKAGE_RELEASE "%{?dist}")
endif()
# Remove dependency on rocm-core if -DROCM_DEP_ROCMCORE=ON not given to cmake
if(NOT ROCM_DEP_ROCMCORE)
    string(REGEX REPLACE ",? ?rocm-core" "" CPACK_RPM_BINARY_PACKAGE_REQUIRES ${CPACK_RPM_BINARY_PACKAGE_REQUIRES})
    string(REGEX REPLACE ",? ?rocm-core" "" CPACK_DEBIAN_BINARY_PACKAGE_DEPENDS ${CPACK_DEBIAN_BINARY_PACKAGE_DEPENDS})
    string(REGEX REPLACE ",? ?rocm-core" "" CPACK_RPM_DEV_PACKAGE_REQUIRES ${CPACK_RPM_DEV_PACKAGE_REQUIRES})
    string(REGEX REPLACE ",? ?rocm-core" "" CPACK_DEBIAN_DEV_PACKAGE_DEPENDS ${CPACK_DEBIAN_DEV_PACKAGE_DEPENDS})
    if(BUILD_ICD)
        string(REGEX REPLACE ",? ?rocm-core" "" CPACK_RPM_ICD_PACKAGE_REQUIRES ${CPACK_RPM_ICD_PACKAGE_REQUIRES})
        string(REGEX REPLACE ",? ?rocm-core" "" CPACK_DEBIAN_ICD_PACKAGE_DEPENDS ${CPACK_DEBIAN_ICD_PACKAGE_DEPENDS})
    endif()
endif()

include(CPack)
