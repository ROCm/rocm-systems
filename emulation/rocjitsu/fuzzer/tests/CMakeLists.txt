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

set(RJ_AFL_PRELOAD_LIBRARY
    "${CMAKE_BINARY_DIR}/fuzzer/afl-dbi/librocjitsu_afl_preload.so"
)

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

find_path(
    RJ_HIP_INCLUDE_DIR
    NAMES hip/hip_runtime_api.h
    HINTS
    ENV ROCM_PATH
    /opt/rocm
    PATH_SUFFIXES include
)
find_path(
    RJ_HIPBLASLT_INCLUDE_DIR
    NAMES hipblaslt/hipblaslt.h
    HINTS
    ENV ROCM_PATH
    /opt/rocm
    PATH_SUFFIXES include
)
find_library(
    RJ_HIP_RUNTIME_LIBRARY
    NAMES amdhip64
    HINTS
    ENV ROCM_PATH
    /opt/rocm
    PATH_SUFFIXES lib lib64
)
find_library(
    RJ_HIPRTC_LIBRARY
    NAMES hiprtc
    HINTS
    ENV ROCM_PATH
    /opt/rocm
    PATH_SUFFIXES lib lib64
)
find_library(
    RJ_HIPBLASLT_LIBRARY
    NAMES hipblaslt
    HINTS
    ENV ROCM_PATH
    /opt/rocm
    PATH_SUFFIXES lib lib64
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

if(
    NOT RJ_HIP_INCLUDE_DIR
    OR NOT RJ_HIP_RUNTIME_LIBRARY
    OR NOT RJ_HIPRTC_LIBRARY
)
    message(
        STATUS
        "two-vector-add AFL smoke test disabled (HIP headers/runtime or hiprtc not found)"
    )
else()
    get_filename_component(
        RJ_HIP_RUNTIME_LIBRARY_DIR
        "${RJ_HIP_RUNTIME_LIBRARY}"
        DIRECTORY
    )
    get_filename_component(
        RJ_HIPRTC_LIBRARY_DIR
        "${RJ_HIPRTC_LIBRARY}"
        DIRECTORY
    )

    set(RJ_FUZZ_TWO_VECTOR_ADD_BIN
        "${CMAKE_CURRENT_BINARY_DIR}/bin/two_vector_add_afl"
    )
    set(RJ_FUZZ_TWO_VECTOR_ADD_PERSISTENT_BIN
        "${CMAKE_CURRENT_BINARY_DIR}/bin/two_vector_add_afl_persistent"
    )
    set(RJ_FUZZ_TWO_VECTOR_ADD_INPUT_DIR
        "${CMAKE_CURRENT_SOURCE_DIR}/seeds/two_vector_add"
    )
    set(RJ_FUZZ_TWO_VECTOR_ADD_OUTPUT_DIR
        "${CMAKE_CURRENT_BINARY_DIR}/two_vector_add_findings"
    )
    set(RJ_FUZZ_TWO_VECTOR_ADD_PERSISTENT_OUTPUT_DIR
        "${CMAKE_CURRENT_BINARY_DIR}/two_vector_add_persistent_findings"
    )
    set(RJ_FUZZ_TWO_VECTOR_ADD_LIBRARY_PATH
        "${RJ_HIPRTC_LIBRARY_DIR}:${RJ_HIP_RUNTIME_LIBRARY_DIR}:$ENV{LD_LIBRARY_PATH}"
    )
    set(RJ_FUZZ_TWO_VECTOR_ADD_ENV
        "LD_LIBRARY_PATH=${RJ_FUZZ_TWO_VECTOR_ADD_LIBRARY_PATH}"
    )
    set(RJ_FUZZ_TWO_VECTOR_ADD_COMPILE_FLAGS
        -std=c++20
        -D__HIP_PLATFORM_AMD__
        "-I${CMAKE_CURRENT_SOURCE_DIR}"
        "-I${RJ_HIP_INCLUDE_DIR}"
        "${CMAKE_CURRENT_SOURCE_DIR}/two_vector_add_fuzz.cpp"
        "${RJ_HIPRTC_LIBRARY}"
        "${RJ_HIP_RUNTIME_LIBRARY}"
        -Wl,-rpath,${RJ_HIPRTC_LIBRARY_DIR}
        -Wl,-rpath,${RJ_HIP_RUNTIME_LIBRARY_DIR}
        -l${CMAKE_DL_LIBS}
    )

    add_custom_command(
        OUTPUT "${RJ_FUZZ_TWO_VECTOR_ADD_BIN}"
        COMMAND
            ${CMAKE_COMMAND} -E make_directory "${CMAKE_CURRENT_BINARY_DIR}/bin"
        COMMAND
            ${CMAKE_COMMAND} -E env --unset=AFL_CXX --unset=MAKEFLAGS
            --unset=MFLAGS "${RJ_AFL_CXX_EXECUTABLE}"
            ${RJ_FUZZ_TWO_VECTOR_ADD_COMPILE_FLAGS} -o
            "${RJ_FUZZ_TWO_VECTOR_ADD_BIN}"
        DEPENDS two_vector_add_fuzz.cpp rj_fuzz_input.h
        COMMENT "Building AFL-instrumented two-vector-add smoke test"
        VERBATIM
    )
    add_custom_target(
        two_vector_add_afl
        DEPENDS "${RJ_FUZZ_TWO_VECTOR_ADD_BIN}"
    )

    add_custom_command(
        OUTPUT "${RJ_FUZZ_TWO_VECTOR_ADD_PERSISTENT_BIN}"
        COMMAND
            ${CMAKE_COMMAND} -E make_directory "${CMAKE_CURRENT_BINARY_DIR}/bin"
        COMMAND
            ${CMAKE_COMMAND} -E env --unset=AFL_CXX --unset=MAKEFLAGS
            --unset=MFLAGS "${RJ_AFL_CXX_EXECUTABLE}" -DRJ_AFL_PERSISTENT_MODE
            ${RJ_FUZZ_TWO_VECTOR_ADD_COMPILE_FLAGS} -o
            "${RJ_FUZZ_TWO_VECTOR_ADD_PERSISTENT_BIN}"
        DEPENDS two_vector_add_fuzz.cpp rj_fuzz_input.h
        COMMENT "Building AFL persistent-mode two-vector-add smoke test"
        VERBATIM
    )
    add_custom_target(
        two_vector_add_afl_persistent
        DEPENDS "${RJ_FUZZ_TWO_VECTOR_ADD_PERSISTENT_BIN}"
    )

    configure_file(
        "${CMAKE_CURRENT_SOURCE_DIR}/run_two_vector_add_afl.cmake.in"
        "${CMAKE_CURRENT_BINARY_DIR}/run_two_vector_add_afl.cmake"
        @ONLY
    )
    configure_file(
        "${CMAKE_CURRENT_SOURCE_DIR}/run_two_vector_add_afl_persistent.cmake.in"
        "${CMAKE_CURRENT_BINARY_DIR}/run_two_vector_add_afl_persistent.cmake"
        @ONLY
    )

    add_test(
        NAME "RocFuzz.TwoVectorAddAFL"
        COMMAND
            ${CMAKE_COMMAND} -P
            "${CMAKE_CURRENT_BINARY_DIR}/run_two_vector_add_afl.cmake"
    )
    set_tests_properties("RocFuzz.TwoVectorAddAFL" PROPERTIES TIMEOUT 60)
    add_test(
        NAME "RocFuzz.TwoVectorAddAFL.Persistent"
        COMMAND
            ${CMAKE_COMMAND} -P
            "${CMAKE_CURRENT_BINARY_DIR}/run_two_vector_add_afl_persistent.cmake"
    )
    set_tests_properties(
        "RocFuzz.TwoVectorAddAFL.Persistent"
        PROPERTIES TIMEOUT 60
    )
endif()

if(
    NOT RJ_HIP_INCLUDE_DIR
    OR NOT RJ_HIPBLASLT_INCLUDE_DIR
    OR NOT RJ_HIP_RUNTIME_LIBRARY
    OR NOT RJ_HIPBLASLT_LIBRARY
)
    message(
        STATUS
        "hipBLASLt AFL smoke test disabled (HIP or hipBLASLt headers/libraries not found)"
    )
    return()
endif()

get_filename_component(
    RJ_HIP_RUNTIME_LIBRARY_DIR
    "${RJ_HIP_RUNTIME_LIBRARY}"
    DIRECTORY
)
get_filename_component(
    RJ_HIPBLASLT_LIBRARY_DIR
    "${RJ_HIPBLASLT_LIBRARY}"
    DIRECTORY
)

set(RJ_FUZZ_HIPBLASLT_TRANSFORM_BIN
    "${CMAKE_CURRENT_BINARY_DIR}/bin/hipblaslt_transform_afl"
)
set(RJ_FUZZ_HIPBLASLT_TRANSFORM_PERSISTENT_BIN
    "${CMAKE_CURRENT_BINARY_DIR}/bin/hipblaslt_transform_afl_persistent"
)
set(RJ_FUZZ_HIPBLASLT_TRANSFORM_INPUT_DIR
    "${CMAKE_CURRENT_SOURCE_DIR}/seeds/hipblaslt_transform"
)
set(RJ_FUZZ_HIPBLASLT_TRANSFORM_OUTPUT_DIR
    "${CMAKE_CURRENT_BINARY_DIR}/hipblaslt_transform_findings"
)
set(RJ_FUZZ_HIPBLASLT_TRANSFORM_PERSISTENT_OUTPUT_DIR
    "${CMAKE_CURRENT_BINARY_DIR}/hipblaslt_transform_persistent_findings"
)

set(RJ_FUZZ_HIPBLASLT_TRANSFORM_ENV
    "LD_LIBRARY_PATH=${RJ_HIPBLASLT_LIBRARY_DIR}:${RJ_HIP_RUNTIME_LIBRARY_DIR}:$ENV{LD_LIBRARY_PATH}"
)

set(RJ_FUZZ_HIPBLASLT_TRANSFORM_COMPILE_FLAGS
    -std=c++20
    -D__HIP_PLATFORM_AMD__
    "-I${CMAKE_CURRENT_SOURCE_DIR}"
    "-I${RJ_HIP_INCLUDE_DIR}"
    "-I${RJ_HIPBLASLT_INCLUDE_DIR}"
    "${CMAKE_CURRENT_SOURCE_DIR}/hipblaslt_transform_fuzz.cpp"
    "${RJ_HIPBLASLT_LIBRARY}"
    "${RJ_HIP_RUNTIME_LIBRARY}"
    -Wl,-rpath,${RJ_HIPBLASLT_LIBRARY_DIR}
    -Wl,-rpath,${RJ_HIP_RUNTIME_LIBRARY_DIR}
)

add_custom_command(
    OUTPUT "${RJ_FUZZ_HIPBLASLT_TRANSFORM_BIN}"
    COMMAND ${CMAKE_COMMAND} -E make_directory "${CMAKE_CURRENT_BINARY_DIR}/bin"
    COMMAND
        ${CMAKE_COMMAND} -E env --unset=AFL_CXX --unset=MAKEFLAGS --unset=MFLAGS
        "${RJ_AFL_CXX_EXECUTABLE}" ${RJ_FUZZ_HIPBLASLT_TRANSFORM_COMPILE_FLAGS}
        -o "${RJ_FUZZ_HIPBLASLT_TRANSFORM_BIN}"
    DEPENDS hipblaslt_transform_fuzz.cpp rj_fuzz_input.h
    COMMENT "Building AFL-instrumented hipBLASLt transform smoke test"
    VERBATIM
)

add_custom_target(
    hipblaslt_transform_afl
    DEPENDS "${RJ_FUZZ_HIPBLASLT_TRANSFORM_BIN}"
)

add_custom_command(
    OUTPUT "${RJ_FUZZ_HIPBLASLT_TRANSFORM_PERSISTENT_BIN}"
    COMMAND ${CMAKE_COMMAND} -E make_directory "${CMAKE_CURRENT_BINARY_DIR}/bin"
    COMMAND
        ${CMAKE_COMMAND} -E env --unset=AFL_CXX --unset=MAKEFLAGS --unset=MFLAGS
        "${RJ_AFL_CXX_EXECUTABLE}" -DRJ_AFL_PERSISTENT_MODE
        ${RJ_FUZZ_HIPBLASLT_TRANSFORM_COMPILE_FLAGS} -o
        "${RJ_FUZZ_HIPBLASLT_TRANSFORM_PERSISTENT_BIN}"
    DEPENDS hipblaslt_transform_fuzz.cpp rj_fuzz_input.h
    COMMENT "Building AFL persistent-mode hipBLASLt transform smoke test"
    VERBATIM
)

add_custom_target(
    hipblaslt_transform_afl_persistent
    DEPENDS "${RJ_FUZZ_HIPBLASLT_TRANSFORM_PERSISTENT_BIN}"
)

configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/run_hipblaslt_transform_afl.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/run_hipblaslt_transform_afl.cmake"
    @ONLY
)
configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/run_hipblaslt_transform_afl_persistent.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/run_hipblaslt_transform_afl_persistent.cmake"
    @ONLY
)

add_test(
    NAME "RocFuzz.HipblasLtTransformAFL"
    COMMAND
        ${CMAKE_COMMAND} -P
        "${CMAKE_CURRENT_BINARY_DIR}/run_hipblaslt_transform_afl.cmake"
)
set_tests_properties("RocFuzz.HipblasLtTransformAFL" PROPERTIES TIMEOUT 60)
add_test(
    NAME "RocFuzz.HipblasLtTransformAFL.Persistent"
    COMMAND
        ${CMAKE_COMMAND} -P
        "${CMAKE_CURRENT_BINARY_DIR}/run_hipblaslt_transform_afl_persistent.cmake"
)
set_tests_properties(
    "RocFuzz.HipblasLtTransformAFL.Persistent"
    PROPERTIES TIMEOUT 60
)
