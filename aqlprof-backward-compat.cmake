

cmake_minimum_required(VERSION 3.16.8)

set(AQLPROF_BUILD_DIR ${CMAKE_CURRENT_BINARY_DIR})
set(AQLPROF_WRAPPER_DIR ${AQLPROF_BUILD_DIR}/wrapper_dir)
set(AQLPROF_WRAPPER_LIB_DIR ${AQLPROF_WRAPPER_DIR}/lib)

#function to create symlink to libraries
function(create_library_symlink)
  file(MAKE_DIRECTORY ${AQLPROF_WRAPPER_LIB_DIR})
  set(LIB_AQLPROF "${AQLPROFILE_LIBRARY}.so")
  set(MAJ_VERSION "${LIB_VERSION_MAJOR}")
  set(SO_VERSION "${LIB_VERSION_STRING}")
  set(library_files "${LIB_AQLPROF}"  "${LIB_AQLPROF}.${MAJ_VERSION}" "${LIB_AQLPROF}.${SO_VERSION}")

  foreach(file_name ${library_files})
     add_custom_target(link_${file_name} ALL
                  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
                  COMMAND ${CMAKE_COMMAND} -E create_symlink
                  ../../${CMAKE_INSTALL_LIBDIR}/${file_name} ${AQLPROF_WRAPPER_LIB_DIR}/${file_name})
  endforeach()
endfunction()

# Create symlink to library files
create_library_symlink()
install(DIRECTORY ${AQLPROF_WRAPPER_LIB_DIR} DESTINATION ${AQLPROFILE_NAME} COMPONENT ${AQLPROFILE_LIBRARY})
