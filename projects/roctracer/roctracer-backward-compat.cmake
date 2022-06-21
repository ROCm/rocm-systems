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

set(ROCT_BUILD_DIR ${CMAKE_CURRENT_BINARY_DIR})
set(ROCT_WRAPPER_DIR ${ROCT_BUILD_DIR}/wrapper_dir)
set(ROCT_WRAPPER_INC_DIR ${ROCT_WRAPPER_DIR}/include)
set(ROCT_WRAPPER_LIB_DIR ${ROCT_WRAPPER_DIR}/lib)
#set(ROCT_WRAPPER_TOOL_DIR ${ROCT_WRAPPER_DIR}/tool)

#Function to set actual file contents in wrapper files
#Some components grep for the contents in the file
function(set_file_contents input_file)
    set(hashzero_check "#if 0  //Area for original file dump\n
/* The following is a copy of the original file for the benefit of build systems which grep for values
 * in this file rather than preprocess it. This is just for backward compatibility */")

    file(READ ${input_file} file_contents)
    set(original_contents "${hashzero_check}\n
${file_contents}
#endif")
    get_filename_component(file_name ${input_file} NAME)
    configure_file(${PROJECT_SOURCE_DIR}/header_template.hpp.in ${ROCT_WRAPPER_INC_DIR}/${file_name})
endfunction()

#use header template file and generate wrapper header files
function(generate_wrapper_header)
  file(MAKE_DIRECTORY ${ROCT_WRAPPER_INC_DIR}/ext)
  #Get the header files from PUBLIC_HEADERS variable
  foreach(header_file ${PUBLIC_HEADERS})
    #set include  guard
    get_filename_component(INC_GAURD_NAME ${header_file} NAME_WE)
    string(TOUPPER ${INC_GAURD_NAME} INC_GAURD_NAME)
    set(include_guard "ROCTRACER_WRAPPER_INCLUDE_${INC_GAURD_NAME}_H")
    #set include statements
    get_filename_component(file_name ${header_file} NAME)
    get_filename_component ( header_subdir ${header_file} DIRECTORY )
    if(header_subdir)
      set(include_statements "#include \"../../../${CMAKE_INSTALL_INCLUDEDIR}/${PROJECT_NAME}/${header_subdir}/${file_name}\"\n")
      configure_file(${PROJECT_SOURCE_DIR}/header_template.hpp.in ${ROCT_WRAPPER_INC_DIR}/${header_subdir}/${file_name})
    else()
      set(include_statements "#include \"../../${CMAKE_INSTALL_INCLUDEDIR}/${PROJECT_NAME}/${file_name}\"\n")
      if(${file_name} STREQUAL "roctracer.h")
        set_file_contents(${PROJECT_SOURCE_DIR}/inc/${file_name})
      else()
        configure_file(${PROJECT_SOURCE_DIR}/header_template.hpp.in ${ROCT_WRAPPER_INC_DIR}/${file_name})
      endif()
    endif()
  endforeach()

  foreach(header_file ${GENERATED_HEADERS})
    #set include  guard
    get_filename_component(INC_GAURD_NAME ${header_file} NAME_WE)
    string(TOUPPER ${INC_GAURD_NAME} INC_GAURD_NAME)
    set(include_guard "ROCTRACER_WRAPPER_INCLUDE_${INC_GAURD_NAME}_H")
    #set include statements
    get_filename_component(file_name ${header_file} NAME)
    set(include_statements "#include \"../../${CMAKE_INSTALL_INCLUDEDIR}/${PROJECT_NAME}/${file_name}\"\n")
    configure_file(${PROJECT_SOURCE_DIR}/header_template.hpp.in ${ROCT_WRAPPER_INC_DIR}/${file_name})

  endforeach()

endfunction()

#function to create symlink to libraries
function(create_library_symlink)
  file(MAKE_DIRECTORY ${ROCT_WRAPPER_LIB_DIR})
  set(LIB_ROCT "libroctracer64.so")
  set(MAJ_VERSION "${PROJECT_VERSION_MAJOR}")
  set(SO_VERSION "${PROJECT_VERSION}")
  set(library_files "${LIB_ROCT}")  #"${LIB_ROCT}.${MAJ_VERSION}" "${LIB_ROCT}.${SO_VERSION}")

  set(LIB_ROCTX64 "libroctx64.so")
  set(library_files "${library_files}" "${LIB_ROCTX64}") # "${LIB_ROCTX64}.${MAJ_VERSION}" "${LIB_ROCTX64}.${SO_VERSION}" )

  foreach(file_name ${library_files})
    add_custom_target(link_${file_name} ALL
                  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
                  COMMAND ${CMAKE_COMMAND} -E create_symlink
                  ../../${CMAKE_INSTALL_LIBDIR}/${file_name} ${ROCT_WRAPPER_LIB_DIR}/${file_name})
  endforeach()
  #set softlink for roctracer/tool/libtracer_tool.so
  #The libray name is changed to libroctracer_tool.so with file reorg changes
  #file(MAKE_DIRECTORY ${ROCT_WRAPPER_TOOL_DIR})
  #set(LIB_TRACERTOOL "libtracer_tool.so")
  #set(LIB_ROCTRACERTOOL "libroctracer_tool.so")
  #add_custom_target(link_${LIB_TRACERTOOL} ALL
  #                WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
  #                COMMAND ${CMAKE_COMMAND} -E create_symlink
  #                ../../lib/${PROJECT_NAME}/${LIB_ROCTRACERTOOL} ${ROCT_WRAPPER_TOOL_DIR}/${LIB_TRACERTOOL})
endfunction()

#Use template header file and generater wrapper header files
generate_wrapper_header()
install(DIRECTORY ${ROCT_WRAPPER_INC_DIR} DESTINATION ${PROJECT_NAME} COMPONENT runtime)
create_library_symlink()
install(DIRECTORY ${ROCT_WRAPPER_LIB_DIR} DESTINATION ${PROJECT_NAME} COMPONENT runtime)
#install soft link to tool
#install(DIRECTORY ${ROCT_WRAPPER_TOOL_DIR} DESTINATION ${PROJECT_NAME})
