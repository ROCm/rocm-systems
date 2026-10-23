# Configure-time drift checks for fake parameter defaults whose production
# values exist only as RCCL_PARAM/NCCL_PARAM macro arguments. The original
# sources are guaranteed to be available here, unlike at installed-test run
# time, so a changed default becomes an immediate configuration error.
function(rccl_assert_source_line _file _expected _what)
  if(NOT EXISTS "${_file}")
    message(FATAL_ERROR "wrap-test drift guard: ${_file} not found at configure time")
  endif()
  file(STRINGS "${_file}" _lines)
  set(_found FALSE)
  foreach(_line IN LISTS _lines)
    string(STRIP "${_line}" _line)
    string(FIND "${_line}" "${_expected}" _pos)
    if(NOT _pos EQUAL -1)
      set(_found TRUE)
      break()
    endif()
  endforeach()
  if(NOT _found)
    message(FATAL_ERROR
      "wrap-test drift guard: ${_file} no longer contains\n"
      "    ${_expected}\n"
      "Update ${_what}, then update this check.")
  endif()
endfunction()
