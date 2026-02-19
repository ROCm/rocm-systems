
####### Expanded from @PACKAGE_INIT@ by configure_package_config_file() #######
####### Any changes to this file will be overwritten by the next CMake run ####
####### The input file was hip-config.cmake.in                            ########

get_filename_component(PACKAGE_PREFIX_DIR "${CMAKE_CURRENT_LIST_DIR}/../../../" ABSOLUTE)

macro(set_and_check _var _file)
  set(${_var} "${_file}")
  if(NOT EXISTS "${_file}")
    message(FATAL_ERROR "File or directory ${_file} referenced by variable ${_var} does not exist !")
  endif()
endmacro()

macro(check_required_components _NAME)
  foreach(comp ${${_NAME}_FIND_COMPONENTS})
    if(NOT ${_NAME}_${comp}_FOUND)
      if(${_NAME}_FIND_REQUIRED_${comp})
        set(${_NAME}_FOUND FALSE)
      endif()
    endif()
  endforeach()
endmacro()

####################################################################################

# hip-config.cmake
# CMake configuration for remote HIP client library

include("${CMAKE_CURRENT_LIST_DIR}/hip-targets.cmake")

# Provide compatibility variables
set(HIP_FOUND TRUE)
set(HIP_VERSION "1.0.0")
set(HIP_PLATFORM "amd")
set(HIP_RUNTIME "remote")

# Check for required components
set(_HIP_NOTFOUND_MESSAGE)
foreach(_comp ${hip_FIND_COMPONENTS})
  if(_comp STREQUAL "hip")
    # hip component is always available
  else()
    set(_HIP_NOTFOUND_MESSAGE "${_HIP_NOTFOUND_MESSAGE}  - ${_comp}\n")
  endif()
endforeach()

if(_HIP_NOTFOUND_MESSAGE)
  message(WARNING "Remote HIP does not provide the following components:\n${_HIP_NOTFOUND_MESSAGE}")
endif()

# Provide hip::hip alias for compatibility
if(TARGET amdhip64 AND NOT TARGET hip::hip)
  add_library(hip::hip ALIAS amdhip64)
endif()

check_required_components(hip)
