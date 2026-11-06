# Pure-processing tests: no capture code.
add_executable(processing_tests
    test_elf_symbols.cpp
    test_symbol_name.cpp
    test_frame_filters.cpp
    test_sample_postprocessor.cpp
    test_module_map_symbolization.cpp
    test_lld_pt_load_mismatch.cpp)
target_link_libraries(processing_tests PRIVATE
    cpu_snap::processing cpu_snap::core dl
    GTest::gtest GTest::gtest_main)
target_compile_options(processing_tests PRIVATE -Wall -Wextra -Wpedantic -O2 -g -fno-omit-frame-pointer)
gtest_discover_tests(processing_tests PROPERTIES TIMEOUT 30)

# lld PT_LOAD offset/vaddr mismatch fixture for test_lld_pt_load_mismatch.cpp.
# Skipped gracefully (test reports GTEST_SKIP) when no lld is found.
find_program(CPU_SNAP_LLD_EXECUTABLE NAMES ld.lld
    HINTS /home/amd/therock/selected/lib/llvm/bin /opt/rocm/llvm/bin)
if(CPU_SNAP_LLD_EXECUTABLE)
    get_filename_component(CPU_SNAP_LLD_DIR "${CPU_SNAP_LLD_EXECUTABLE}" DIRECTORY)
    set(CPU_SNAP_LLD_MISMATCH_LIB
        "${CMAKE_CURRENT_BINARY_DIR}/liblld_mismatch_fixture.so")
    add_custom_command(
        OUTPUT "${CPU_SNAP_LLD_MISMATCH_LIB}"
        COMMAND "${CMAKE_CXX_COMPILER}" -shared -fPIC -O0 -g
                -B "${CPU_SNAP_LLD_DIR}" -fuse-ld=lld -Wl,-z,noseparate-code
                -o "${CPU_SNAP_LLD_MISMATCH_LIB}"
                "${CMAKE_CURRENT_SOURCE_DIR}/fixtures/lld_mismatch_lib.cpp"
        DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/fixtures/lld_mismatch_lib.cpp"
        COMMENT "Building lld PT_LOAD offset/vaddr mismatch fixture")
    add_custom_target(lld_mismatch_fixture DEPENDS "${CPU_SNAP_LLD_MISMATCH_LIB}")
    add_dependencies(processing_tests lld_mismatch_fixture)
    target_compile_definitions(processing_tests PRIVATE
        CPU_SNAP_HAVE_LLD_MISMATCH_LIB=1
        CPU_SNAP_LLD_MISMATCH_LIB_PATH="${CPU_SNAP_LLD_MISMATCH_LIB}")
else()
    message(STATUS "ld.lld not found; ProcMapsLldMismatch tests will be skipped")
endif()

# Stripped-library fixture for the symbol-size-check regression test
# (test_stripped_symbol_size.cpp): built with debug info, then stripped down
# to .dynsym only (no .symtab, no .debug_info).
set(CPU_SNAP_STRIPPED_SYMBOL_LIB_UNSTRIPPED
    "${CMAKE_CURRENT_BINARY_DIR}/libstripped_symbol_fixture_unstripped.so")
set(CPU_SNAP_STRIPPED_SYMBOL_LIB
    "${CMAKE_CURRENT_BINARY_DIR}/libstripped_symbol_fixture.so")
set(CPU_SNAP_STRIPPED_SYMBOL_MAIN_OBJ
    "${CMAKE_CURRENT_BINARY_DIR}/stripped_symbol_fixture.o")
set(CPU_SNAP_STRIPPED_SYMBOL_HIDDEN_OBJ
    "${CMAKE_CURRENT_BINARY_DIR}/stripped_symbol_fixture_hidden.o")
add_custom_command(
    OUTPUT "${CPU_SNAP_STRIPPED_SYMBOL_LIB}"
    COMMAND "${CMAKE_CXX_COMPILER}" -fPIC -O1 -fno-inline -g -c
            -o "${CPU_SNAP_STRIPPED_SYMBOL_MAIN_OBJ}"
            "${CMAKE_CURRENT_SOURCE_DIR}/fixtures/stripped_symbol_fixture.cpp"
    COMMAND "${CMAKE_CXX_COMPILER}" -fPIC -O1 -fno-inline -g -c
            -o "${CPU_SNAP_STRIPPED_SYMBOL_HIDDEN_OBJ}"
            "${CMAKE_CURRENT_SOURCE_DIR}/fixtures/stripped_symbol_fixture_hidden.cpp"
    # Link order matters: the main object (exported fn_a/fn_b) first keeps
    # those symbols at lower addresses than the hidden-visibility function
    # linked after it, so the hidden function lands in the gap AFTER the
    # last known .dynsym symbol (the scenario under test), not before it.
    COMMAND "${CMAKE_CXX_COMPILER}" -shared -fPIC -g
            -o "${CPU_SNAP_STRIPPED_SYMBOL_LIB_UNSTRIPPED}"
            "${CPU_SNAP_STRIPPED_SYMBOL_MAIN_OBJ}"
            "${CPU_SNAP_STRIPPED_SYMBOL_HIDDEN_OBJ}"
    COMMAND "${CMAKE_STRIP}" --strip-all
            -o "${CPU_SNAP_STRIPPED_SYMBOL_LIB}"
            "${CPU_SNAP_STRIPPED_SYMBOL_LIB_UNSTRIPPED}"
    DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/fixtures/stripped_symbol_fixture.cpp"
            "${CMAKE_CURRENT_SOURCE_DIR}/fixtures/stripped_symbol_fixture_hidden.cpp"
    COMMENT "Building + stripping symbol-size-check fixture")
add_custom_target(stripped_symbol_fixture DEPENDS "${CPU_SNAP_STRIPPED_SYMBOL_LIB}")

# Symbolizer tests: -no-pie so the known-workload function addresses are
# stable against the leaf-IP range check (mirrors sampling_no_pie_tests'
# rationale for its own symbolizer test, before the move).
add_executable(processing_no_pie_tests
    test_symbolizer_format.cpp
    test_stripped_symbol_size.cpp)
target_link_libraries(processing_no_pie_tests PRIVATE
    cpu_snap::processing dl
    GTest::gtest GTest::gtest_main)
target_compile_options(processing_no_pie_tests PRIVATE
    -Wall -Wextra -Wpedantic -O2 -g
    -fno-omit-frame-pointer -fno-optimize-sibling-calls -no-pie)
target_link_options(processing_no_pie_tests PRIVATE -no-pie)
add_dependencies(processing_no_pie_tests stripped_symbol_fixture)
target_compile_definitions(processing_no_pie_tests PRIVATE
    CPU_SNAP_STRIPPED_SYMBOL_LIB_PATH="${CPU_SNAP_STRIPPED_SYMBOL_LIB}")
gtest_discover_tests(processing_no_pie_tests PROPERTIES TIMEOUT 30)
