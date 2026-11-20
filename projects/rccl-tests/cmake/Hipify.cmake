find_package(Git QUIET)
if(GIT_EXECUTABLE)
  get_filename_component(_git_bin "${GIT_EXECUTABLE}" DIRECTORY)
endif()
find_program(PERL_EXECUTABLE NAMES perl perl.exe
  HINTS "${_git_bin}/../usr/bin" "${_git_bin}/../../usr/bin")
find_package(Perl REQUIRED)
find_program(HIPIFY_PERL_EXECUTABLE NAMES hipify-perl
  HINTS "${ROCM_PATH}/bin" "${ROCM_PATH}/libexec/hipify")
if(NOT HIPIFY_PERL_EXECUTABLE)
  message(FATAL_ERROR "Set HIPIFY_PERL_EXECUTABLE to the selected SDK's hipify-perl script.")
endif()

function(rccl_tests_hipify output input)
  get_filename_component(_output_dir "${output}" DIRECTORY)
  add_custom_command(OUTPUT "${output}"
    COMMAND "${CMAKE_COMMAND}" -E make_directory "${_output_dir}"
    COMMAND "${PERL_EXECUTABLE}" "${HIPIFY_PERL_EXECUTABLE}" -quiet-warnings "${input}" -o "${output}"
    DEPENDS "${input}" "${HIPIFY_PERL_EXECUTABLE}" "${PERL_EXECUTABLE}"
    VERBATIM)
endfunction()
