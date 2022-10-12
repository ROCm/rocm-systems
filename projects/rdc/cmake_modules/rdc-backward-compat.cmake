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

set(RDC_WRAPPER_DIR ${CMAKE_CURRENT_BINARY_DIR}/wrapper_dir)
set(RDC_WRAPPER_INC_DIR ${RDC_WRAPPER_DIR}/include/rdc)
set(RDC_WRAPPER_LIB_DIR ${RDC_WRAPPER_DIR}/lib)
set(RDC_WRAPPER_BIN_DIR ${RDC_WRAPPER_DIR}/bin)
set(RDC_SRC_INC_DIR ${PROJECT_SOURCE_DIR}/include/rdc)

#use header template file and generate wrapper header files
function(generate_wrapper_header)
  file(MAKE_DIRECTORY ${RDC_WRAPPER_INC_DIR})
  #Only rdc.h header file used in packaging
  # set include guard
  set(include_guard "RDC_WRAPPER_INCLUDE_RDC_H")
  #set #include statement
  set(file_name "rdc.h")
  set(header_name ${file_name})
  set(include_statements "#include \"../../../${CMAKE_INSTALL_INCLUDEDIR}/rdc/${file_name}\"\n")
  configure_file(${PROJECT_SOURCE_DIR}/src/header_template.hpp.in ${RDC_WRAPPER_INC_DIR}/${file_name})
endfunction()

# function to create symlink to libraries
function(create_library_symlink)
  file(MAKE_DIRECTORY ${RDC_WRAPPER_LIB_DIR})
  set(LIB_RDC "librdc.so")
  set(LIB_RDC_BOOTSTRAP "librdc_bootstrap.so")
  set(LIB_RDC_CLIENT "librdc_client.so")
  set(MAJ_VERSION "${VERSION_MAJOR}")
  set(SO_VERSION "${VERSION_MAJOR}.${VERSION_MINOR}")
  set(library_files "${LIB_RDC}"  "${LIB_RDC}.${MAJ_VERSION}" "${LIB_RDC}.${SO_VERSION}")
  set(library_files "${library_files}" "${LIB_RDC_BOOTSTRAP}"  "${LIB_RDC_BOOTSTRAP}.${MAJ_VERSION}" "${LIB_RDC_BOOTSTRAP}.${SO_VERSION}" )
  set(library_files "${library_files}" "${LIB_RDC_CLIENT}"  "${LIB_RDC_CLIENT}.${MAJ_VERSION}" "${LIB_RDC_CLIENT}.${SO_VERSION}" )

  foreach(file_name ${library_files})
     add_custom_target(link_${file_name} ALL
                  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
                  COMMAND ${CMAKE_COMMAND} -E create_symlink
                  ../../${CMAKE_INSTALL_LIBDIR}/${file_name} ${RDC_WRAPPER_LIB_DIR}/${file_name})
  endforeach()
  # Symlink for private libraries
  set(LIB_RDC_ROCR "librdc_rocr.so")
  set(LIB_RDC_RAS "librdc_ras.so")
  set(LIB_RDC_CLIENT_SMI "librdc_client_smi.so")
  set(library_files "${LIB_RDC_ROCR}"  "${LIB_RDC_ROCR}.${MAJ_VERSION}" "${LIB_RDC_ROCR}.${SO_VERSION}" )
  set(library_files "${library_files}" "${LIB_RDC_CLIENT_SMI}"  "${LIB_RDC_CLIENT_SMI}.${MAJ_VERSION}" "${LIB_RDC_CLIENT_SMI}.${SO_VERSION}" )
  set(library_files "${library_files}" "${LIB_RDC_RAS}")

  foreach(file_name ${library_files})
     add_custom_target(link_${file_name} ALL
                  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
                  COMMAND ${CMAKE_COMMAND} -E create_symlink
                  ../../${CMAKE_INSTALL_LIBDIR}/${RDC}/${file_name} ${RDC_WRAPPER_LIB_DIR}/${file_name})
  endforeach()
  # create symlink to grpc directory
  # this helps avoid using LD_LIBRARY_PATH
  if(BUILD_STANDALONE)
    set(file_name "grpc")
    add_custom_target(link_${file_name} ALL
                    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
                    COMMAND ${CMAKE_COMMAND} -E create_symlink
                    ../../${CMAKE_INSTALL_LIBDIR}/${RDC}/${file_name} ${RDC_WRAPPER_LIB_DIR}/${file_name})
  endif()
  # create symlink to rdc.service
  set(file_name "rdc.service")
  add_custom_target(link_${file_name} ALL
                  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
                  COMMAND ${CMAKE_COMMAND} -E create_symlink
                  ../../${CMAKE_INSTALL_LIBEXECDIR}/${RDC}/${file_name} ${RDC_WRAPPER_LIB_DIR}/${file_name})
endfunction()

# function to create symlink to binaries
function(create_binary_symlink)
  file(MAKE_DIRECTORY ${RDC_WRAPPER_BIN_DIR})
  # create symlink for rdcd and rdci
  set(binary_files "rdcd" "rdci")
  foreach(file_name ${binary_files})
     add_custom_target(link_${file_name} ALL
                  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
                  COMMAND ${CMAKE_COMMAND} -E create_symlink
                  ../../${CMAKE_INSTALL_BINDIR}/${file_name} ${RDC_WRAPPER_BIN_DIR}/${file_name})
  endforeach()
endfunction()

# Use template header file and generate wrapper header files
generate_wrapper_header()
install(DIRECTORY ${RDC_WRAPPER_INC_DIR} DESTINATION DESTINATION ${RDC_INSTALL_PREFIX}/${RDC}/include COMPONENT ${CLIENT_COMPONENT})

# Create symlink to library files
create_library_symlink()
install(DIRECTORY ${RDC_WRAPPER_LIB_DIR} DESTINATION ${RDC_INSTALL_PREFIX}/${RDC} COMPONENT ${CLIENT_COMPONENT})
# Create symlink to library binaries
create_binary_symlink()
install(DIRECTORY ${RDC_WRAPPER_BIN_DIR} DESTINATION ${RDC_INSTALL_PREFIX}/${RDC} COMPONENT ${CLIENT_COMPONENT})
