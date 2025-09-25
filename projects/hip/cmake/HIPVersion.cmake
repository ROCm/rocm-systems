include_guard(GLOBAL)

function(hip_compute_version)
  find_package(Git QUIET)
  if(NOT GIT_FOUND)
    return()
  endif()

  execute_process(
    COMMAND ${GIT_EXECUTABLE} describe --tags --match "hip-version_*" --abbrev=0
    WORKING_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}
    RESULT_VARIABLE _tag_res
    OUTPUT_VARIABLE _tag
    OUTPUT_STRIP_TRAILING_WHITESPACE
  )

  if(_tag_res EQUAL 0 AND _tag)
    string(REGEX MATCH "hip-version_([0-9]+)\\.([0-9]+)\\.([0-9]+)" _m "${_tag}")
    if(_m)
      set(HIP_VERSION_MAJOR "${CMAKE_MATCH_1}" PARENT_SCOPE)
      set(HIP_VERSION_MINOR "${CMAKE_MATCH_2}" PARENT_SCOPE)
      set(HIP_VERSION_PATCH "${CMAKE_MATCH_3}" PARENT_SCOPE)
      message(STATUS "Computed HIP version: ${CMAKE_MATCH_1}.${CMAKE_MATCH_2}.${CMAKE_MATCH_3}")
      return()
    endif()
  endif()

  message(WARNING "HIP version not found!")
endfunction()
