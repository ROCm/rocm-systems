set(RJ_AFL_ROOT
    "${PROJECT_SOURCE_DIR}/third_party/AFLplusplus"
    CACHE PATH
    "Optional AFL++ checkout used by rocjitsu fuzz tests"
)

if(RJ_AFL_CXX_EXECUTABLE AND NOT EXISTS "${RJ_AFL_CXX_EXECUTABLE}")
    unset(RJ_AFL_CXX_EXECUTABLE CACHE)
endif()
if(RJ_AFL_FUZZ_EXECUTABLE AND NOT EXISTS "${RJ_AFL_FUZZ_EXECUTABLE}")
    unset(RJ_AFL_FUZZ_EXECUTABLE CACHE)
endif()

find_program(
    RJ_AFL_CXX_EXECUTABLE
    NAMES afl-clang-fast++
    HINTS "${RJ_AFL_ROOT}"
    PATH_SUFFIXES . bin
)
find_program(
    RJ_AFL_FUZZ_EXECUTABLE
    NAMES afl-fuzz
    HINTS "${RJ_AFL_ROOT}"
    PATH_SUFFIXES . bin
)

if(NOT RJ_AFL_CXX_EXECUTABLE OR NOT RJ_AFL_FUZZ_EXECUTABLE)
    message(
        STATUS
        "AFL hello_world smoke test disabled (afl-clang-fast++ or afl-fuzz not found)"
    )
    return()
endif()

set(RJ_FUZZ_HELLO_WORLD_BIN "${CMAKE_CURRENT_BINARY_DIR}/bin/hello_world_afl")
set(RJ_FUZZ_HELLO_WORLD_PERSISTENT_BIN
    "${CMAKE_CURRENT_BINARY_DIR}/bin/hello_world_afl_persistent"
)
set(RJ_FUZZ_HELLO_WORLD_INPUT_DIR
    "${CMAKE_CURRENT_BINARY_DIR}/hello_world_inputs"
)
set(RJ_FUZZ_HELLO_WORLD_OUTPUT_DIR
    "${CMAKE_CURRENT_BINARY_DIR}/hello_world_findings"
)
set(RJ_FUZZ_HELLO_WORLD_PERSISTENT_OUTPUT_DIR
    "${CMAKE_CURRENT_BINARY_DIR}/hello_world_persistent_findings"
)

file(MAKE_DIRECTORY "${RJ_FUZZ_HELLO_WORLD_INPUT_DIR}")
file(WRITE "${RJ_FUZZ_HELLO_WORLD_INPUT_DIR}/seed.txt" "hello\n")

add_custom_command(
    OUTPUT "${RJ_FUZZ_HELLO_WORLD_BIN}"
    COMMAND ${CMAKE_COMMAND} -E make_directory "${CMAKE_CURRENT_BINARY_DIR}/bin"
    COMMAND
        ${CMAKE_COMMAND} -E env --unset=AFL_CXX --unset=MAKEFLAGS --unset=MFLAGS
        "${RJ_AFL_CXX_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/hello_world.cpp"
        -std=c++20 -o "${RJ_FUZZ_HELLO_WORLD_BIN}"
    DEPENDS hello_world.cpp
    COMMENT "Building AFL-instrumented hello_world smoke test"
    VERBATIM
)

add_custom_target(hello_world_afl DEPENDS "${RJ_FUZZ_HELLO_WORLD_BIN}")

add_custom_command(
    OUTPUT "${RJ_FUZZ_HELLO_WORLD_PERSISTENT_BIN}"
    COMMAND ${CMAKE_COMMAND} -E make_directory "${CMAKE_CURRENT_BINARY_DIR}/bin"
    COMMAND
        ${CMAKE_COMMAND} -E env --unset=AFL_CXX --unset=MAKEFLAGS --unset=MFLAGS
        "${RJ_AFL_CXX_EXECUTABLE}" -DRJ_AFL_PERSISTENT_MODE
        "${CMAKE_CURRENT_SOURCE_DIR}/hello_world.cpp" -std=c++20 -o
        "${RJ_FUZZ_HELLO_WORLD_PERSISTENT_BIN}"
    DEPENDS hello_world.cpp
    COMMENT "Building AFL persistent-mode hello_world smoke test"
    VERBATIM
)

add_custom_target(
    hello_world_afl_persistent
    DEPENDS "${RJ_FUZZ_HELLO_WORLD_PERSISTENT_BIN}"
)

configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/run_hello_world_afl.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/run_hello_world_afl.cmake"
    @ONLY
)
configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/run_hello_world_afl_persistent.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/run_hello_world_afl_persistent.cmake"
    @ONLY
)

add_test(
    NAME "RocFuzz.HelloWorldAFL"
    COMMAND
        ${CMAKE_COMMAND} -P
        "${CMAKE_CURRENT_BINARY_DIR}/run_hello_world_afl.cmake"
)
set_tests_properties("RocFuzz.HelloWorldAFL" PROPERTIES TIMEOUT 30)
add_test(
    NAME "RocFuzz.HelloWorldAFL.Persistent"
    COMMAND
        ${CMAKE_COMMAND} -P
        "${CMAKE_CURRENT_BINARY_DIR}/run_hello_world_afl_persistent.cmake"
)
set_tests_properties("RocFuzz.HelloWorldAFL.Persistent" PROPERTIES TIMEOUT 30)
