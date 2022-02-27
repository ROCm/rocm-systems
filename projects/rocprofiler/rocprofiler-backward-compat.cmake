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

set(ROCPROF_BUILD_DIR ${CMAKE_CURRENT_BINARY_DIR})
set(ROCPROF_WRAPPER_DIR ${ROCPROF_BUILD_DIR}/wrapper_dir)
set(ROCPROF_WRAPPER_INC_DIR ${ROCPROF_WRAPPER_DIR}/include)
set(ROCPROF_WRAPPER_BIN_DIR ${ROCPROF_WRAPPER_DIR}/bin)
set(ROCPROF_WRAPPER_LIB_DIR ${ROCPROF_WRAPPER_DIR}/lib)
set(ROCPROF_WRAPPER_TOOL_DIR ${ROCPROF_WRAPPER_DIR}/tool)

#Function to generate header template file
function(create_header_template)
    file(WRITE ${ROCPROF_WRAPPER_DIR}/header.hpp.in "/*
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
   */\n\n#ifndef @include_guard@\n#define @include_guard@ \n\n#pragma message(\"This file is deprecated. Use file from include path /opt/rocm-ver/include/ and prefix with rocprofiler\")\n@include_statements@ \n\n#endif")
endfunction()

#use header template file and generate wrapper header files
function(generate_wrapper_header)
  file(MAKE_DIRECTORY ${ROCPROF_WRAPPER_INC_DIR})
  #find all header files from inc
  file(GLOB include_files ${CMAKE_CURRENT_SOURCE_DIR}/inc/*.h)
  #Convert the list of files into #includes
  foreach(header_file ${include_files})
     #set include  guard
    get_filename_component(INC_GAURD_NAME ${header_file} NAME_WE)
    string(TOUPPER ${INC_GAURD_NAME} INC_GAURD_NAME)
    set(include_guard "${include_guard}ROCPROF_WRAPPER_INCLUDE_${INC_GAURD_NAME}_H")
     #set include statement
    get_filename_component(file_name ${header_file} NAME)
    set(include_statements "${include_statements}#include \"../../include/${ROCPROFILER_NAME}/${file_name}\"\n")
    configure_file(${ROCPROF_WRAPPER_DIR}/header.hpp.in ${ROCPROF_WRAPPER_INC_DIR}/${file_name})
    unset(include_guard)
    unset(include_statements)
  endforeach()

  #Only single file from  ${CMAKE_CURRENT_SOURCE_DIR}/src/core/activity.h is packaged. So drectly using that file name
  set(file_name "activity.h")
  #set include  guard
  get_filename_component(INC_GAURD_NAME ${file_name} NAME_WE)
  string(TOUPPER ${INC_GAURD_NAME} INC_GAURD_NAME)
  set(include_guard "${include_guard}ROCPROF_WRAPPER_INCLUDE_${INC_GAURD_NAME}_H")
  set(include_statements "${include_statements}#include \"../../include/${ROCPROFILER_NAME}/${file_name}\"\n")
  configure_file(${ROCPROF_WRAPPER_DIR}/header.hpp.in ${ROCPROF_WRAPPER_INC_DIR}/${file_name})
endfunction()

#function to create symlink to binaries
function(create_binary_symlink)
  file(MAKE_DIRECTORY ${ROCPROF_WRAPPER_BIN_DIR})
  #create symlink for rocprof
  set(file_name "rocprof")
  add_custom_target(link_${file_name} ALL
                 WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
                    COMMAND ${CMAKE_COMMAND} -E create_symlink
                    ../../bin/${file_name} ${ROCPROF_WRAPPER_BIN_DIR}/${file_name})

endfunction()

#function to create symlink to libraries
function(create_library_symlink)
  file(MAKE_DIRECTORY ${ROCPROF_WRAPPER_LIB_DIR})
  set(LIB_ROCPROF "${ROCPROFILER_LIBRARY}.so")
  set(MAJ_VERSION "${LIB_VERSION_MAJOR}")
  set(SO_VERSION "${LIB_VERSION_STRING}")
  set(library_files "${LIB_ROCPROF}"  "${LIB_ROCPROF}.${MAJ_VERSION}" "${LIB_ROCPROF}.${SO_VERSION}")

  foreach(file_name ${library_files})
    add_custom_target(link_${file_name} ALL
                  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
                  COMMAND ${CMAKE_COMMAND} -E create_symlink
                  ../../lib/${file_name} ${ROCPROF_WRAPPER_LIB_DIR}/${file_name})
  endforeach()
  #create symlink to rocprofiler/tool/libtool.so
  # With File reorg,tool renamed to rocprof-tool
  file(MAKE_DIRECTORY ${ROCPROF_WRAPPER_TOOL_DIR})
  set(LIB_TOOL "libtool.so")
  set(LIB_ROCPROFTOOL "librocprof-tool.so")
  add_custom_target(link_${LIB_TOOL} ALL
                   WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
                   COMMAND ${CMAKE_COMMAND} -E create_symlink
                   ../../lib/${ROCPROFILER_NAME}/${LIB_ROCPROFTOOL} ${ROCPROF_WRAPPER_TOOL_DIR}/${LIB_TOOL})
  #create symlink to test binary
  #since its saved in lib folder , the code for the same is added here
  # With File reorg ,binary name changed from ctrl to rocprof-ctrl
  set(TEST_CTRL "ctrl")
  set(TEST_ROCPROFCTRL "rocprof-ctrl")
  add_custom_target(link_${TEST_CTRL} ALL
                    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
                    COMMAND ${CMAKE_COMMAND} -E create_symlink
                    ../../lib/${ROCPROFILER_NAME}/${TEST_ROCPROFCTRL} ${ROCPROF_WRAPPER_TOOL_DIR}/${TEST_CTRL})
endfunction()

#Creater a template for header file
create_header_template()
#Use template header file and generater wrapper header files
generate_wrapper_header()
install(DIRECTORY ${ROCPROF_WRAPPER_INC_DIR} DESTINATION ${ROCPROFILER_NAME})
# Create symlink to binaries
create_binary_symlink()
install(DIRECTORY ${ROCPROF_WRAPPER_BIN_DIR} DESTINATION ${ROCPROFILER_NAME})
create_library_symlink()
install(DIRECTORY ${ROCPROF_WRAPPER_LIB_DIR} DESTINATION ${ROCPROFILER_NAME})
#install tools directory
install(DIRECTORY ${ROCPROF_WRAPPER_TOOL_DIR} DESTINATION ${ROCPROFILER_NAME})
