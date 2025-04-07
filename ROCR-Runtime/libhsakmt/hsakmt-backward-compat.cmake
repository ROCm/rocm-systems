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

set(HSAKMT_WRAPPER_DIR ${CMAKE_CURRENT_BINARY_DIR}/wrapper_dir)
set(HSAKMT_WRAPPER_INC_DIR ${HSAKMT_WRAPPER_DIR}/include)
#Function to generate header template file
function(create_header_template)
    file(WRITE ${HSAKMT_WRAPPER_DIR}/header.hpp.in "/*
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
#error \"@file_name@ has moved to @CMAKE_INSTALL_PREFIX@/@CMAKE_INSTALL_INCLUDEDIR@/hsakmt and package include paths have changed.\\nInclude as \\\"hsakmt/@file_name@\\\" when using cmake packages.\"
#else  /* ROCM_HEADER_WRAPPER_WERROR 0 */
#if defined(__GNUC__)
#warning \"@file_name@ has moved to @CMAKE_INSTALL_PREFIX@/@CMAKE_INSTALL_INCLUDEDIR@/hsakmt and package include paths have changed.\\nInclude as \\\"hsakmt/@file_name@\\\" when using cmake packages.\"
#else
#pragma message(\"@file_name@ has moved to @CMAKE_INSTALL_PREFIX@/@CMAKE_INSTALL_INCLUDEDIR@/hsakmt and package include paths have changed.\\nInclude as \\\"hsakmt/@file_name@\\\" when using cmake packages.\")
#endif
#endif  /* ROCM_HEADER_WRAPPER_WERROR */

@include_statements@

#endif")
endfunction()

#use header template file and generate wrapper header files
function(generate_wrapper_header)
  file(MAKE_DIRECTORY ${HSAKMT_WRAPPER_INC_DIR})
  #find all header files from include folder
  file(GLOB include_files ${CMAKE_CURRENT_SOURCE_DIR}/include/*.h)
  #generate wrapper header files
  foreach(header_file ${include_files})
    # set include guard
    get_filename_component(INC_GUARD_NAME ${header_file} NAME_WE)
    string(TOUPPER ${INC_GUARD_NAME} INC_GUARD_NAME)
    set(include_guard "${include_guard}HSAKMT_WRAPPER_INCLUDE_${INC_GUARD_NAME}_H")
    # set include statements
    get_filename_component(file_name ${header_file} NAME)
    set(include_statements "${include_statements}#include \"hsakmt/${file_name}\"\n")
    configure_file(${HSAKMT_WRAPPER_DIR}/header.hpp.in ${HSAKMT_WRAPPER_INC_DIR}/${file_name})
    unset(include_guard)
    unset(include_statements)
  endforeach()
endfunction()

#Creater a template for header file
create_header_template()
#Use template header file and generater wrapper header files
generate_wrapper_header()
install(DIRECTORY ${HSAKMT_WRAPPER_INC_DIR} DESTINATION . COMPONENT devel PATTERN "linux" EXCLUDE)
