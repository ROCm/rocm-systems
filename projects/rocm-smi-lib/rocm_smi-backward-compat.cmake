# Copyright (c) 2022 Advanced Micro Devices, Inc. All Rights Reserved.
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

set(RSMI_BUILD_DIR ${CMAKE_CURRENT_BINARY_DIR})
set(RSMI_WRAPPER_DIR ${RSMI_BUILD_DIR}/wrapper_dir)
set(RSMI_WRAPPER_INC_DIR ${RSMI_WRAPPER_DIR}/include/${ROCM_SMI})
set(OAM_TARGET_NAME "oam")
set(OAM_WRAPPER_INC_DIR ${RSMI_WRAPPER_DIR}/include/${OAM_TARGET_NAME})
set(RSMI_WRAPPER_LIB_DIR ${RSMI_WRAPPER_DIR}/${ROCM_SMI}/lib)
set(OAM_WRAPPER_LIB_DIR ${RSMI_WRAPPER_DIR}/${OAM_TARGET_NAME}/lib)
## package headers
set(PUBLIC_RSMI_HEADERS
    rocm_smi.h
    ${ROCM_SMI_TARGET}Config.h
    kfd_ioctl.h)
set(OAM_HEADERS
    oam_mapi.h
    amd_oam.h)

#Function to generate header template file
function(create_header_template)
    file(WRITE ${RSMI_WRAPPER_DIR}/header.hpp.in "/*
    Copyright (c) 2022 Advanced Micro Devices, Inc. All rights reserved.

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the \"Software\"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in
    all copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED \"AS IS\", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
   THE SOFTWARE.
   */\n\n#ifndef @include_guard@\n#define @include_guard@ \n\n#pragma message(\"This file is deprecated. Use file from include path /opt/rocm-ver/include/ and prefix with ${ROCM_SMI}\")\n@include_statements@ \n\n#endif")
endfunction()

#use header template file and generate wrapper header files
function(generate_wrapper_header)
  file(MAKE_DIRECTORY ${RSMI_WRAPPER_INC_DIR})
  #Generate wrapper header files from  the list
  foreach(header_file ${PUBLIC_RSMI_HEADERS})
    # set include guard
    get_filename_component(INC_GAURD_NAME ${header_file} NAME_WE)
    string(TOUPPER ${INC_GAURD_NAME} INC_GAURD_NAME)
    set(include_guard "${include_guard}COMGR_WRAPPER_INCLUDE_${INC_GAURD_NAME}_H")
    #set #include statement
    get_filename_component(file_name ${header_file} NAME)
    set(include_statements "${include_statements}#include \"../../../include/${ROCM_SMI}/${file_name}\"\n")
    configure_file(${RSMI_WRAPPER_DIR}/header.hpp.in ${RSMI_WRAPPER_INC_DIR}/${file_name})
    unset(include_guard)
    unset(include_statements)
  endforeach()

#OAM Wrpper Header file generation
  file(MAKE_DIRECTORY ${OAM_WRAPPER_INC_DIR})
  #Generate wrapper header files from  the list
  foreach(header_file ${OAM_HEADERS})
    # set include guard
    get_filename_component(INC_GAURD_NAME ${header_file} NAME_WE)
    string(TOUPPER ${INC_GAURD_NAME} INC_GAURD_NAME)
    set(include_guard "${include_guard}COMGR_WRAPPER_INCLUDE_${INC_GAURD_NAME}_H")
    #set #include statement
    get_filename_component(file_name ${header_file} NAME)
    set(include_statements "${include_statements}#include \"../../../include/${OAM_TARGET_NAME}/${file_name}\"\n")
    configure_file(${RSMI_WRAPPER_DIR}/header.hpp.in ${OAM_WRAPPER_INC_DIR}/${file_name})
    unset(include_guard)
    unset(include_statements)
  endforeach()

endfunction()

#function to create symlink to binaries
function(create_binary_symlink)
  file(MAKE_DIRECTORY ${RSMI_WRAPPER_DIR}/bin)

  file(GLOB binary_files ${COMMON_SRC_ROOT}/python_smi_tools/*.py )
  foreach(binary_file ${binary_files})
    get_filename_component(file_name ${binary_file} NAME)
    add_custom_target(link_${file_name} ALL
                  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
                  COMMAND ${CMAKE_COMMAND} -E create_symlink
                  ../../libexec/${ROCM_SMI}/${file_name} ${RSMI_WRAPPER_DIR}/bin/${file_name})
  endforeach()
  #soft link rocm_smi binary
  add_custom_target(link_${ROCM_SMI} ALL
                   WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
                   COMMAND ${CMAKE_COMMAND} -E create_symlink
                   ../../bin/${ROCM_SMI} ${RSMI_WRAPPER_DIR}/bin/${ROCM_SMI})
endfunction()

#function to create symlink to libraries
function(create_library_symlink)

  file(MAKE_DIRECTORY ${RSMI_WRAPPER_LIB_DIR})
  if(BUILD_SHARED_LIBS)

    #get rsmi lib versions
    set(SO_VERSION_GIT_TAG_PREFIX "rsmi_so_ver")
    get_version_from_tag("1.0.0.0" ${SO_VERSION_GIT_TAG_PREFIX} GIT)
    if(${ROCM_PATCH_VERSION})
      set(VERSION_PATCH ${ROCM_PATCH_VERSION})
      set(SO_VERSION_STRING "${VERSION_MAJOR}.${VERSION_MINOR}.${VERSION_PATCH}")
    else()
      set(SO_VERSION_STRING "${VERSION_MAJOR}.${VERSION_MINOR}")
    endif()

    #link RSMI library files
    set(LIB_RSMI "${ROCM_SMI_LIB_NAME}.so")
    set(library_files "${LIB_RSMI}"  "${LIB_RSMI}.${VERSION_MAJOR}" "${LIB_RSMI}.${SO_VERSION_STRING}")
  else()
    set(LIB_RSMI "${ROCM_SMI_LIB_NAME}.a")
    set(library_files "${LIB_RSMI}")
  endif()

  foreach(file_name ${library_files})
     add_custom_target(link_${file_name} ALL
                  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
                  COMMAND ${CMAKE_COMMAND} -E create_symlink
                  ../../lib/${file_name} ${RSMI_WRAPPER_LIB_DIR}/${file_name})
  endforeach()

  file(MAKE_DIRECTORY ${OAM_WRAPPER_LIB_DIR})
  if(BUILD_SHARED_LIBS)

    #get OAM lib versions
    set(SO_VERSION_GIT_TAG_PREFIX "oam_so_ver")
    get_version_from_tag("1.0.0.0" ${SO_VERSION_GIT_TAG_PREFIX} GIT)
    if(${ROCM_PATCH_VERSION})
      set(VERSION_PATCH ${ROCM_PATCH_VERSION})
      set(SO_VERSION_STRING "${VERSION_MAJOR}.${VERSION_MINOR}.${VERSION_PATCH}")
    else()
      set(SO_VERSION_STRING "${VERSION_MAJOR}.${VERSION_MINOR}")
    endif()

    #link OAM library files
    set(LIB_OAM "lib${OAM_TARGET_NAME}.so")
    set(library_files "${LIB_OAM}"  "${LIB_OAM}.${VERSION_MAJOR}" "${LIB_OAM}.${SO_VERSION_STRING}")
  else()
    set(LIB_OAM "lib${OAM_TARGET_NAME}.a")
    set(library_files "${LIB_OAM}")
  endif()

  foreach(file_name ${library_files})
     add_custom_target(link_${file_name} ALL
                  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
                  COMMAND ${CMAKE_COMMAND} -E create_symlink
                  ../../lib/${file_name} ${OAM_WRAPPER_LIB_DIR}/${file_name})
  endforeach()

endfunction()

#Creater a template for header file
create_header_template()
#Use template header file and generater wrapper header files
generate_wrapper_header()
install(DIRECTORY ${RSMI_WRAPPER_INC_DIR} DESTINATION ${ROCM_SMI}/include)
install(DIRECTORY ${OAM_WRAPPER_INC_DIR} DESTINATION ${OAM_TARGET_NAME}/include)
# Create symlink to binaries
create_binary_symlink()
install(DIRECTORY ${RSMI_WRAPPER_DIR}/bin  DESTINATION ${ROCM_SMI})
# Create symlink to library files
create_library_symlink()
install(DIRECTORY ${RSMI_WRAPPER_LIB_DIR} DESTINATION ${ROCM_SMI} COMPONENT lib${ROCM_SMI})
install(DIRECTORY ${OAM_WRAPPER_LIB_DIR} DESTINATION ${OAM_TARGET_NAME} COMPONENT lib${OAM_TARGET_NAME} )
