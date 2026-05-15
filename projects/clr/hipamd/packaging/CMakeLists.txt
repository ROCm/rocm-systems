# Copyright (c) 2020 - 2022 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

cmake_minimum_required(VERSION 3.16.8)

#set components for HIP
set(CPACK_COMPONENTS_ALL binary dev doc runtime-nvidia)

# ASAN Package requires only libraries and license file
if(ENABLE_ASAN_PACKAGING)
   set(CPACK_COMPONENTS_ALL asan)
endif()
###############Install Required files for all compnents########

#Enable Component Install
set(CPACK_RPM_COMPONENT_INSTALL ON)
set(CPACK_DEB_COMPONENT_INSTALL ON)

###Set License####
set(CPACK_RESOURCE_FILE_LICENSE ${hip_SOURCE_DIR}/LICENSE.md)
install(FILES ${CPACK_RESOURCE_FILE_LICENSE} DESTINATION ${CMAKE_INSTALL_DOCDIR} COMPONENT binary)
# install license file in share/doc/hip-asan folder
install(FILES ${CPACK_RESOURCE_FILE_LICENSE} DESTINATION ${CMAKE_INSTALL_DOCDIR}-asan COMPONENT asan)
set(CPACK_RPM_PACKAGE_LICENSE "MIT")
#Begin binary files install
if(HIP_PLATFORM STREQUAL "amd" )
   # Set component name and  library type
   set(COMPONENT_NAME "binary")
   set(CMAKE_LIB_TYPE "LIBRARY")

   if(NOT BUILD_SHARED_LIBS)
      # Change library type for static builds
      set(CMAKE_LIB_TYPE "ARCHIVE")
   endif()

   if(ENABLE_ASAN_PACKAGING)
      # Change component name for ASAN builds
      set(COMPONENT_NAME "asan")
   endif()

   # Install libraries
   install(TARGETS amdhip64
              ${CMAKE_LIB_TYPE}
              DESTINATION ${CMAKE_INSTALL_LIBDIR}
              COMPONENT ${COMPONENT_NAME})
   install(TARGETS hiprtc
              ${CMAKE_LIB_TYPE}
              DESTINATION ${CMAKE_INSTALL_LIBDIR}
              COMPONENT ${COMPONENT_NAME})
   install(TARGETS hiprtc-builtins
              ${CMAKE_LIB_TYPE}
              DESTINATION ${CMAKE_INSTALL_LIBDIR}
              COMPONENT ${COMPONENT_NAME})

#TODO:This do not belong in BINARY package.
#Keeping it as is for now
install(FILES ${CMAKE_BINARY_DIR}/hipamd/share/hip/.hipInfo DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT binary)

install ( EXPORT hip-targets  FILE hip-targets.cmake NAMESPACE hip:: DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/hip  COMPONENT dev)

install(FILES ${CMAKE_BINARY_DIR}/hipamd/src/hip-lang-config.cmake ${CMAKE_BINARY_DIR}/hipamd/src/hip-lang-config-version.cmake DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/hip-lang COMPONENT dev)
install ( EXPORT hip-lang-targets  FILE hip-lang-targets.cmake NAMESPACE hip-lang:: DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/hip-lang  COMPONENT dev)

install(FILES ${CMAKE_BINARY_DIR}/hipamd/hiprtc-config.cmake ${CMAKE_BINARY_DIR}/hipamd/hiprtc-config-version.cmake DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/hiprtc COMPONENT dev)
install ( EXPORT hiprtc-targets  FILE hiprtc-targets.cmake NAMESPACE hiprtc:: DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/hiprtc  COMPONENT dev)

endif()#End HIP_PLATFORM = "amd"
#End bianry files install

#Begin dev files install
#Install bin files from HIP_COMMON_DIR
file(GLOB BIN_FILES ${HIP_COMMON_DIR}/bin/*)
if(NOT WIN32)
    list(FILTER BIN_FILES EXCLUDE REGEX ".bat$")
endif()
foreach(binFile ${BIN_FILES})
    install(PROGRAMS ${binFile} DESTINATION ${CMAKE_INSTALL_BINDIR} COMPONENT dev)
endforeach()
unset(BIN_FILES)

#Install bin files from hip_SOURCE_DIR
file(GLOB BIN_FILES ${hip_SOURCE_DIR}/bin/*)
if(NOT WIN32)
    list(FILTER BIN_FILES EXCLUDE REGEX ".bat$")
endif()
foreach(binFile ${BIN_FILES})
    install(PROGRAMS ${binFile} DESTINATION ${CMAKE_INSTALL_BINDIR} COMPONENT dev)
endforeach()

install(DIRECTORY ${HIP_COMMON_DIR}/include DESTINATION . COMPONENT dev)
install(DIRECTORY ${hip_SOURCE_DIR}/include/hip/amd_detail
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/hip COMPONENT dev)
if(DEFINED HIPNV_DIR)
    install(DIRECTORY ${HIPNV_DIR}/include/hip/nvidia_detail
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/hip COMPONENT dev)
endif()
if(HIP_PLATFORM STREQUAL "amd" )
install(FILES ${CMAKE_BINARY_DIR}/hipamd/include/hip/amd_detail/hip_prof_str.h
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/hip/amd_detail COMPONENT dev)
endif()
install(FILES ${CMAKE_BINARY_DIR}/hipamd/include/hip/hip_version.h
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/hip COMPONENT dev)
install(FILES ${CMAKE_BINARY_DIR}/hipamd/share/hip/version DESTINATION ${CMAKE_INSTALL_DATADIR}/hip COMPONENT dev)
# .hipVersion is added to satisfy Windows compute build.
#TODO to be removed
if(WIN32)
    install(FILES ${CMAKE_BINARY_DIR}/hipamd/share/hip/version DESTINATION ${CMAKE_INSTALL_BINDIR} RENAME .hipVersion COMPONENT dev)
endif()
install(DIRECTORY ${HIP_COMMON_DIR}/cmake/ DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/hip COMPONENT dev)
install(FILES ${CMAKE_BINARY_DIR}/hipamd/hip-config.cmake ${CMAKE_BINARY_DIR}/hipamd/hip-config-version.cmake DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/hip COMPONENT dev)
install(FILES ${CMAKE_BINARY_DIR}/hipamd/hip-config-amd.cmake DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/hip COMPONENT dev)
install(FILES ${CMAKE_BINARY_DIR}/hipamd/hip-config-nvidia.cmake DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/hip COMPONENT dev)
#End dev files install

#Begin doc files install
find_program(DOXYGEN_EXE doxygen)
if(DOXYGEN_EXE)
    if(EXISTS "${HIP_COMMON_DIR}/docs/doxygen-input/doxy.cfg")
        add_custom_target(build_doxygen ALL
            COMMAND HIP_PATH=${HIP_COMMON_DIR} doxygen ${HIP_COMMON_DIR}/docs/doxygen-input/doxy.cfg)
    elseif(EXISTS "${HIP_COMMON_DIR}/docs/.doxygen/Doxyfile")
        add_custom_target(build_doxygen ALL
            COMMAND HIP_PATH=${HIP_COMMON_DIR} doxygen ${HIP_COMMON_DIR}/docs/.doxygen/Doxyfile)
    else()
        message(FATAL_ERROR "Unable to find doxygen config file")
    endif()
    install(DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/RuntimeAPI/html
            DESTINATION ${CMAKE_INSTALL_DOCDIR}/RuntimeAPI COMPONENT doc)
endif()
#End doc files install

##################################
# Packaging steps COMMON Variables
##################################
set(CPACK_PACKAGE_VENDOR "Advanced Micro Devices, Inc.")
set(CPACK_PACKAGE_CONTACT "HIP Support <hip.support@amd.com>")
set(CPACK_PACKAGE_DESCRIPTION_SUMMARY "HIP:Heterogenous-computing Interface for Portability")
set(CPACK_PACKAGE_VERSION_MAJOR ${HIP_VERSION_MAJOR})
set(CPACK_PACKAGE_VERSION_MINOR ${HIP_VERSION_MINOR})
set(CPACK_PACKAGE_VERSION_PATCH ${HIP_VERSION_PATCH})
set(CPACK_PACKAGE_VERSION ${HIP_VERSION_MAJOR}.${HIP_VERSION_MINOR}.${HIP_PACKAGING_VERSION_PATCH})
set(CPACK_GENERATOR "TGZ;DEB;RPM" CACHE STRING "Package types to build")

set(CPACK_RPM_EXCLUDE_FROM_AUTO_FILELIST_ADDITION "/opt")
if (CPACK_RPM_PACKAGE_RELEASE MATCHES "local" )
  #If building locally default value will cause build failure
  #DEBUG SYMBOL pacaking require SOURCE_DIR to be small
  set(CPACK_RPM_BUILD_SOURCE_DIRS_PREFIX ${CPACK_INSTALL_PREFIX})
endif()

# Eventhough hip-runtime package has libraries,it was not in the package provides list,
# since CPACK_RPM_PACKAGE_AUTOREQPROV was set to "no".
# Use AUTOREQ,(rather than AUTOREQPROV) so that package will also provides the libraries
set(CPACK_RPM_PACKAGE_AUTOREQ " no")
set(CPACK_RPM_FILE_NAME "RPM-DEFAULT")

set(CPACK_DEBIAN_FILE_NAME "DEB-DEFAULT")

set(CPACK_SOURCE_GENERATOR "TGZ")


#Begin Binary Packaging setting

set(CPACK_BINARY_DEB "ON")
set(CPACK_BINARY_RPM "ON")

set(CPACK_DEBIAN_BINARY_PACKAGE_NAME "hip-runtime-amd")
set(CPACK_RPM_BINARY_PACKAGE_NAME "hip-runtime-amd")

set(CPACK_COMPONENT_BINARY_DESCRIPTION "HIP:Heterogenous-computing Interface for Portability [RUNTIME - AMD]")
# Set Runtime Package  dependencies
set(HIP_RUNTIME_ROCM_PKG_DEPENDENCIES "hsa-rocr, rocminfo, comgr, rocm-core")
# Add rocprofiler-register dependencies
if(HIP_ENABLE_ROCPROFILER_REGISTER)
   set(HIP_RUNTIME_ROCM_PKG_DEPENDENCIES "${HIP_RUNTIME_ROCM_PKG_DEPENDENCIES}, rocprofiler-register")
endif()

set(HIP_RUNTIME_DEB_STDPKG_DEPENDENCIES "libnuma1, libstdc++6, libc6")

if(DEB10_DISTRO)
  # On Debian Buster only: libgcc_s.so.1 is in the package libgcc1
  set(HIP_RUNTIME_DEB_STDPKG_DEPENDENCIES "${HIP_RUNTIME_DEB_STDPKG_DEPENDENCIES}, libgcc1")
else()
  set(HIP_RUNTIME_DEB_STDPKG_DEPENDENCIES "${HIP_RUNTIME_DEB_STDPKG_DEPENDENCIES}, libgcc-s1")
endif()

set(CPACK_DEBIAN_BINARY_PACKAGE_DEPENDS "${HIP_RUNTIME_ROCM_PKG_DEPENDENCIES}, ${HIP_RUNTIME_DEB_STDPKG_DEPENDENCIES}")

set(CPACK_DEBIAN_BINARY_PACKAGE_PROVIDES "hip-rocclr (= ${CPACK_PACKAGE_VERSION})")
set(CPACK_DEBIAN_BINARY_PACKAGE_REPLACES "hip-rocclr (= ${CPACK_PACKAGE_VERSION})")

set(CPACK_RPM_BINARY_PACKAGE_ARCHITECTURE "${CMAKE_SYSTEM_PROCESSOR}")
string(REPLACE "-" "_" HIP_BASE_VERSION ${CPACK_PACKAGE_VERSION})

if(EL7_DISTRO)
  # centos: In centos using parenthesis is causing error. So set the specific dependencies
  set(HIP_RUNTIME_RPM_STDPKG_DEPENDENCIES "glibc, numactl-libs, libstdc++, libgcc")
else()
  # RHEL/SLES
  set(HIP_RUNTIME_RPM_STDPKG_DEPENDENCIES "glibc, (numactl-libs or libnuma1), (libstdc++ or libstdc++6)")
  if(DEB10_DISTRO)
    # On Debian Buster only: libgcc_s.so.1 is in the package libgcc1
    set(HIP_RUNTIME_RPM_STDPKG_DEPENDENCIES "${HIP_RUNTIME_RPM_STDPKG_DEPENDENCIES}, libgcc1")
  else()
    set(HIP_RUNTIME_RPM_STDPKG_DEPENDENCIES "${HIP_RUNTIME_RPM_STDPKG_DEPENDENCIES}, (libgcc or libgcc_s1)" )
  endif()
endif()

set(CPACK_RPM_BINARY_PACKAGE_REQUIRES "${HIP_RUNTIME_ROCM_PKG_DEPENDENCIES}, ${HIP_RUNTIME_RPM_STDPKG_DEPENDENCIES}")

set(CPACK_RPM_BINARY_PACKAGE_PROVIDES "hip-rocclr = ${HIP_BASE_VERSION}")
set(CPACK_RPM_BINARY_PACKAGE_OBSOLETES "hip-rocclr = ${HIP_BASE_VERSION}")
#End Binary Packaging setting

#Begin dev Packaging setting
set(CPACK_DEV_DEB "ON")
set(CPACK_DEV_RPM "ON")

set(CPACK_DEBIAN_DEV_PACKAGE_NAME "hip-dev")
set(CPACK_RPM_DEV_PACKAGE_NAME "hip-devel")

set(CPACK_COMPONENT_DEV_DESCRIPTION "HIP: Heterogenous-computing Interface for Portability [DEVELOPMENT]")

configure_file(hip-devel.postinst ${CMAKE_CURRENT_BINARY_DIR}/dev/postinst @ONLY)
configure_file(hip-devel.prerm    ${CMAKE_CURRENT_BINARY_DIR}/dev/prerm @ONLY)
set(CPACK_DEBIAN_DEV_PACKAGE_CONTROL_EXTRA "${CMAKE_CURRENT_BINARY_DIR}/dev/postinst;${CMAKE_CURRENT_BINARY_DIR}/dev/prerm")
# Dev/devel package dependencies
set(HIP_DEV_ROCM_PKG_DEPENDENCIES "hip-runtime-amd, rocm-llvm, rocm-core")
set(CPACK_DEBIAN_DEV_PACKAGE_RECOMMENDS "perl (>= 5.0), libfile-copy-recursive-perl, libfile-listing-perl, libfile-which-perl, liburi-perl")
set(CPACK_DEBIAN_DEV_PACKAGE_DEPENDS "libc6, file, ${HIP_DEV_ROCM_PKG_DEPENDENCIES}, hsa-rocr-dev, hipcc")
set(CPACK_DEBIAN_DEV_PACKAGE_PROVIDES "hip-base")
set(CPACK_DEBIAN_DEV_PACKAGE_REPLACES "hip-base")

set(CPACK_RPM_DEV_POST_INSTALL_SCRIPT_FILE "${CMAKE_CURRENT_BINARY_DIR}/dev/postinst")
set(CPACK_RPM_DEV_PRE_UNINSTALL_SCRIPT_FILE "${CMAKE_CURRENT_BINARY_DIR}/dev/prerm")
set(CPACK_RPM_DEV_PACKAGE_SUGGESTS "perl >= 5.0, perl-File-Which, perl-File-Listing, perl-URI")
set(CPACK_RPM_DEV_PACKAGE_REQUIRES "file, ${HIP_DEV_ROCM_PKG_DEPENDENCIES}, hsa-rocr-devel, hipcc")

set(CPACK_RPM_DEV_PACKAGE_PROVIDES "hip-base")
set(CPACK_RPM_DEV_PACKAGE_OBSOLETES "hip-base")
#End dev Packaging setting

#Begin doc Packaging setting
set(CPACK_DOC_DEB "ON")
set(CPACK_DOC_RPM "ON")
set(CPACK_DEBIAN_DOC_PACKAGE_NAME "hip-doc")
set(CPACK_RPM_DOC_PACKAGE_NAME "hip-doc")
set(CPACK_COMPONENT_DOC_DESCRIPTION "HIP: Heterogenous-computing Interface for Portability [DOCUMENTATION]")

set(CPACK_DEBIAN_DOC_PACKAGE_DEPENDS "hip-dev (= ${CPACK_PACKAGE_VERSION}-${CPACK_DEBIAN_PACKAGE_RELEASE}), rocm-core")
set(CPACK_DEBIAN_DOC_PACKAGE_PROVIDES "hip-doc")

string(REPLACE "-" "_" HIP_BASE_VERSION ${CPACK_PACKAGE_VERSION})
set(CPACK_RPM_DOC_PACKAGE_REQUIRES "hip-devel = ${HIP_BASE_VERSION}-${CPACK_RPM_PACKAGE_RELEASE}, rocm-core")

#End doc Packaging setting

#Begin runtime-nvidia Packaging setting
set(CPACK_RUNTIME-NVIDIA_DEB "ON")
set(CPACK_RUNTIME-NVIDIA_RPM "ON")
set(CPACK_DEBIAN_RUNTIME-NVIDIA_PACKAGE_NAME "hip-runtime-nvidia")
set(CPACK_RPM_RUNTIME-NVIDIA_PACKAGE_NAME "hip-runtime-nvidia")
set(CPACK_COMPONENT_RUNTIME-NVIDIA_DESCRIPTION "HIP: Heterogenous-computing Interface for Portability [RUNTIME-NVIDIA]")

set(CPACK_DEBIAN_RUNTIME-NVIDIA_PACKAGE_DEPENDS "cuda (>= 7.5), rocm-core, hipcc-nvidia")
set(CPACK_DEBIAN_RUNTIME-NVIDIA_PACKAGE_PROVIDES "hip-nvcc")
set(CPACK_DEBIAN_RUNTIME-NVIDIA_PACKAGE_REPLACES "hip-nvcc")

set(CPACK_RPM_RUNTIME-NVIDIA_PACKAGE_PROVIDES "hip-nvcc")
set(CPACK_RPM_RUNTIME-NVIDIA_PACKAGE_OBSOLETES "hip-nvcc")
set(CPACK_RPM_RUNTIME-NVIDIA_PACKAGE_REQUIRES "cuda >= 7.5, rocm-core, hipcc-nvidia")

# Begin asan Packaging setting
set(CPACK_ASAN_DEB "ON")
set(CPACK_ASAN_RPM "ON")
set(CPACK_DEBIAN_ASAN_PACKAGE_NAME "hip-runtime-amd-asan")
set(CPACK_RPM_ASAN_PACKAGE_NAME "hip-runtime-amd-asan")
set(CPACK_COMPONENT_ASAN_DESCRIPTION "HIP:Heterogenous-computing Interface for Portability [AddressSanitizer libraries]")
set(HIP_ASAN_ROCM_PKG_DEPENDENCIES "hsa-rocr-asan, rocminfo, comgr-asan, rocm-llvm, rocm-core-asan")
set(CPACK_DEBIAN_ASAN_PACKAGE_DEPENDS "${HIP_ASAN_ROCM_PKG_DEPENDENCIES}, ${HIP_RUNTIME_DEB_STDPKG_DEPENDENCIES}")
set(CPACK_RPM_ASAN_PACKAGE_REQUIRES "${HIP_ASAN_ROCM_PKG_DEPENDENCIES}, ${HIP_RUNTIME_RPM_STDPKG_DEPENDENCIES}")
#End asan Packaging setting

# Remove dependency on rocm-core if -DROCM_DEP_ROCMCORE=ON not given to cmake
if(NOT ROCM_DEP_ROCMCORE)

    string(REGEX REPLACE ",? ?rocm-core" "" CPACK_RPM_BINARY_PACKAGE_REQUIRES ${CPACK_RPM_BINARY_PACKAGE_REQUIRES})
    string(REGEX REPLACE ",? ?rocm-core" "" CPACK_DEBIAN_BINARY_PACKAGE_DEPENDS ${CPACK_DEBIAN_BINARY_PACKAGE_DEPENDS})
    string(REGEX REPLACE ",? ?rocm-core" "" CPACK_RPM_DEV_PACKAGE_REQUIRES ${CPACK_RPM_DEV_PACKAGE_REQUIRES})
    string(REGEX REPLACE ",? ?rocm-core" "" CPACK_DEBIAN_DEV_PACKAGE_DEPENDS ${CPACK_DEBIAN_DEV_PACKAGE_DEPENDS})
    string(REGEX REPLACE ",? ?rocm-core" "" CPACK_RPM_DOC_PACKAGE_REQUIRES ${CPACK_RPM_DOC_PACKAGE_REQUIRES})
    string(REGEX REPLACE ",? ?rocm-core" "" CPACK_DEBIAN_DOC_PACKAGE_DEPENDS ${CPACK_DEBIAN_DOC_PACKAGE_DEPENDS})
    string(REGEX REPLACE ",? ?rocm-core" "" CPACK_RPM_RUNTIME-NVIDIA_PACKAGE_REQUIRES ${CPACK_RPM_RUNTIME-NVIDIA_PACKAGE_REQUIRES})
    string(REGEX REPLACE ",? ?rocm-core" "" CPACK_DEBIAN_RUNTIME-NVIDIA_PACKAGE_DEPENDS ${CPACK_DEBIAN_RUNTIME-NVIDIA_PACKAGE_DEPENDS})
    string(REGEX REPLACE ",? ?rocm-core-asan" "" CPACK_RPM_ASAN_PACKAGE_REQUIRES ${CPACK_RPM_ASAN_PACKAGE_REQUIRES})
    string(REGEX REPLACE ",? ?rocm-core-asan" "" CPACK_DEBIAN_ASAN_PACKAGE_DEPENDS ${CPACK_DEBIAN_ASAN_PACKAGE_DEPENDS})
endif()
# package name and dependencies for static package
if(NOT BUILD_SHARED_LIBS)
   set(CPACK_RPM_STATIC_PACKAGE_NAME "hip-static-devel")
   set(CPACK_DEBIAN_STATIC_PACKAGE_NAME "hip-static-dev")
   set(CPACK_COMPONENT_STATIC_DESCRIPTION "HIP:Heterogenous-computing Interface for Portability [Static Libraries - AMD]")

   set(CPACK_RPM_STATIC_PACKAGE_REQUIRES "${CPACK_RPM_DEV_PACKAGE_REQUIRES}, ${HIP_RUNTIME_RPM_STDPKG_DEPENDENCIES}")
   string(REGEX REPLACE ",? ?hsa-rocr-devel" "" CPACK_RPM_STATIC_PACKAGE_REQUIRES ${CPACK_RPM_STATIC_PACKAGE_REQUIRES})
   string(REGEX REPLACE ",? ?hipcc" "" CPACK_RPM_STATIC_PACKAGE_REQUIRES ${CPACK_RPM_STATIC_PACKAGE_REQUIRES})
   string(REGEX REPLACE ",? ?hip-runtime-amd" "" CPACK_RPM_STATIC_PACKAGE_REQUIRES ${CPACK_RPM_STATIC_PACKAGE_REQUIRES})
   string(APPEND CPACK_RPM_STATIC_PACKAGE_REQUIRES  ", hsa-rocr-static-devel, hipcc-static-devel, rocminfo-static-devel, comgr-static-devel")

   set(CPACK_DEBIAN_STATIC_PACKAGE_DEPENDS "${CPACK_DEBIAN_DEV_PACKAGE_DEPENDS}, ${HIP_RUNTIME_RPM_STDPKG_DEPENDENCIES}")
   string(REGEX REPLACE ",? ?hsa-rocr-dev" "" CPACK_DEBIAN_STATIC_PACKAGE_DEPENDS ${CPACK_DEBIAN_STATIC_PACKAGE_DEPENDS})
   string(REGEX REPLACE ",? ?hipcc" "" CPACK_DEBIAN_STATIC_PACKAGE_DEPENDS ${CPACK_DEBIAN_STATIC_PACKAGE_DEPENDS})
   string(REGEX REPLACE ",? ?hip-runtime-amd" "" CPACK_DEBIAN_STATIC_PACKAGE_DEPENDS ${CPACK_DEBIAN_STATIC_PACKAGE_DEPENDS})
   string(APPEND CPACK_DEBIAN_STATIC_PACKAGE_DEPENDS  ", hsa-rocr-static-dev, hipcc-static-dev, rocminfo-static-dev, comgr-static-dev")
endif()
include(CPack)
#static package generation
# Group binary and dev component to single package
if(NOT BUILD_SHARED_LIBS)
    cpack_add_component_group("static")
    cpack_add_component( binary  GROUP static )
    cpack_add_component( dev GROUP static )
endif()
