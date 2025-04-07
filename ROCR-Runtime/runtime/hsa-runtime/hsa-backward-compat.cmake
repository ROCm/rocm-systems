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

set(HSA_WRAPPER_DIR ${CMAKE_CURRENT_BINARY_DIR}/wrapper_dir)
set(HSA_WRAPPER_INC_DIR ${HSA_WRAPPER_DIR}/include/hsa)
set(HSA_WRAPPER_LIB_DIR ${HSA_WRAPPER_DIR}/lib)

#Function to generate header template file
function(create_header_template)
    file(WRITE ${HSA_WRAPPER_DIR}/header.hpp.in "/*
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
   */

#ifndef @include_guard@
#define @include_guard@

#ifndef ROCM_HEADER_WRAPPER_WERROR
#define ROCM_HEADER_WRAPPER_WERROR @deprecated_error@
#endif
#if ROCM_HEADER_WRAPPER_WERROR  /* ROCM_HEADER_WRAPPER_WERROR 1 */
#error \"@file_name@ has moved to @CMAKE_INSTALL_PREFIX@/@CMAKE_INSTALL_INCLUDEDIR@/hsa and package include paths have changed.\\nInclude as \\\"hsa/@file_name@\\\" when using cmake packages.\"
#else  /* ROCM_HEADER_WRAPPER_WERROR 0 */
#if defined(__GNUC__)
#warning \"@file_name@ has moved to @CMAKE_INSTALL_PREFIX@/@CMAKE_INSTALL_INCLUDEDIR@/hsa and package include paths have changed.\\nInclude as \\\"hsa/@file_name@\\\" when using cmake packages.\"
#else
#pragma message(\"@file_name@ has moved to @CMAKE_INSTALL_PREFIX@/@CMAKE_INSTALL_INCLUDEDIR@/hsa and package include paths have changed.\\nInclude as \\\"hsa/@file_name@\\\" when using cmake packages.\")
#endif
#endif  /* ROCM_HEADER_WRAPPER_WERROR */
@include_statements@

#endif")
endfunction()

#use header template file and generate wrapper header files
function(generate_wrapper_header)
  file(MAKE_DIRECTORY ${HSA_WRAPPER_INC_DIR})
  file(GLOB include_files ${CMAKE_CURRENT_SOURCE_DIR}/inc/*.h)
  #Generate wrapper header files
  foreach(header_file ${include_files})
    # set include guard
    get_filename_component(INC_GUARD_NAME ${header_file} NAME_WE)
    string(TOUPPER ${INC_GUARD_NAME} INC_GUARD_NAME)
    set(include_guard "${include_guard}HSA_RUNTIME_WRAPPER_INC_${INC_GUARD_NAME}_H")
    #set #include statement
    get_filename_component(file_name ${header_file} NAME)
    set(include_statements "${include_statements}#include \"../../../${CMAKE_INSTALL_INCLUDEDIR}/hsa/${file_name}\"\n")
    configure_file(${HSA_WRAPPER_DIR}/header.hpp.in ${HSA_WRAPPER_INC_DIR}/${file_name})

    unset(include_guard)
    unset(include_statements)
  endforeach()

endfunction()

#function to create symlink to libraries
function(create_library_symlink)
  file(MAKE_DIRECTORY ${HSA_WRAPPER_LIB_DIR})
  if(BUILD_SHARED_LIBS)
    set(LIB_NAME "${CORE_RUNTIME_LIBRARY}.so")
    set(MAJ_VERSION "${VERSION_MAJOR}")
    set(library_files "${LIB_NAME}"  "${LIB_NAME}.${MAJ_VERSION}" )
  else()
    set(LIB_NAME "${CORE_RUNTIME_LIBRARY}.a")
    set(library_files "${LIB_NAME}")
  endif()

  foreach(file_name ${library_files})
     add_custom_target(link_${file_name} ALL
                  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
                  COMMAND ${CMAKE_COMMAND} -E create_symlink
                  ../../${CMAKE_INSTALL_LIBDIR}/${file_name} ${HSA_WRAPPER_LIB_DIR}/${file_name})
  endforeach()
endfunction()

#Creater a template for header file
create_header_template()
#Use template header file and generater wrapper header files
generate_wrapper_header()
install(DIRECTORY ${HSA_WRAPPER_INC_DIR} DESTINATION hsa/include COMPONENT dev)
#Create symlinks for library files
create_library_symlink()
install(DIRECTORY ${HSA_WRAPPER_LIB_DIR} DESTINATION hsa  COMPONENT binary)
