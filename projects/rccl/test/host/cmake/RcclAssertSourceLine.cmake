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

# Assert an exact (possibly multiline) source fragment appears a fixed number
# of times. Use this when one line alone is not specific enough to identify the
# declaration whose copied default a fake mirrors.
function(rccl_assert_source_text_count _file _expected _expected_count _what)
  if(NOT EXISTS "${_file}")
    message(FATAL_ERROR "wrap-test drift guard: ${_file} not found at configure time")
  endif()
  file(READ "${_file}" _source)
  string(LENGTH "${_source}" _source_length)
  string(LENGTH "${_expected}" _expected_length)
  if(_expected_length EQUAL 0)
    message(FATAL_ERROR "wrap-test drift guard: empty expected text for ${_what}")
  endif()
  string(REPLACE "${_expected}" "" _without_expected "${_source}")
  string(LENGTH "${_without_expected}" _without_expected_length)
  math(EXPR _found_count
    "(${_source_length} - ${_without_expected_length}) / ${_expected_length}")
  if(NOT _found_count EQUAL _expected_count)
    message(FATAL_ERROR
      "wrap-test drift guard: ${_file} contains ${_found_count}, expected ${_expected_count}, copies of\n"
      "    ${_expected}\n"
      "Update ${_what}, then update this check.")
  endif()
endfunction()
