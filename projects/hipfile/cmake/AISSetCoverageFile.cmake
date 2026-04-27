# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT

function(ais_set_coverage_file test_list_name directory)
    set(test_counter 0)
    set(test_list "${${test_list_name}}")
    foreach(test IN LISTS test_list)
        set_tests_properties(
            ${test}
            PROPERTIES ENVIRONMENT_MODIFICATION "LLVM_PROFILE_FILE=set:${directory}/${test_list_name}.${test_counter}.profraw")
        math(EXPR test_counter "${test_counter}+1")
    endforeach()
endfunction()
