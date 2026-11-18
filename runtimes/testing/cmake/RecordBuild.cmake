# SPDX-License-Identifier: MIT

foreach(_required TEST_EXECUTABLE RECORD SOURCE_ROOT COMPILER)
    if(NOT DEFINED ${_required} OR "${${_required}}" STREQUAL "")
        message(FATAL_ERROR "Build record requires ${_required}")
    endif()
endforeach()
file(SHA256 "${TEST_EXECUTABLE}" _binary_hash)
find_program(_git git REQUIRED)
execute_process(
    COMMAND "${_git}" -C "${SOURCE_ROOT}" rev-parse HEAD
    OUTPUT_VARIABLE _revision
    OUTPUT_STRIP_TRAILING_WHITESPACE
    COMMAND_ERROR_IS_FATAL ANY
)
execute_process(
    COMMAND
        "${_git}" -C "${SOURCE_ROOT}" status --porcelain --untracked-files=all
        -- .
    OUTPUT_VARIABLE _status
    COMMAND_ERROR_IS_FATAL ANY
)
set(_record
    "revision=${_revision}\ncompiler=${COMPILER}\nexecutable_sha256=${_binary_hash}\nRUSTFLAGS=$ENV{RUSTFLAGS}\nCARGO_ENCODED_RUSTFLAGS=$ENV{CARGO_ENCODED_RUSTFLAGS}\nsource_status:\n${_status}\ninputs:\n"
)
foreach(_root IN ITEMS "${SOURCE_ROOT}" ${EXTRA_INPUT_ROOTS})
    if(IS_DIRECTORY "${_root}")
        file(GLOB_RECURSE _inputs LIST_DIRECTORIES false "${_root}/*")
    else()
        set(_inputs "${_root}")
    endif()
    foreach(_input IN LISTS _inputs)
        file(SHA256 "${_input}" _hash)
        string(APPEND _record "${_hash}  ${_input}\n")
    endforeach()
endforeach()
get_filename_component(_directory "${RECORD}" DIRECTORY)
file(MAKE_DIRECTORY "${_directory}")
file(WRITE "${RECORD}" "${_record}")
