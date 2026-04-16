# Unit tests (GoogleTest framework)

#===----------------------------------------------------------------------===//
# GoogleTest-based tests
#===----------------------------------------------------------------------===//

# Include fixtures directory
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/fixtures)

# Descriptor Updater tests
add_executable(descriptor_gtest DescriptorUpdaterGTest.cpp)
target_link_libraries(descriptor_gtest
  aegisbit
  GTest::gtest_main
)
target_include_directories(descriptor_gtest PRIVATE
  ${CMAKE_SOURCE_DIR}/include
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${LLVM_INCLUDE_DIRS}
)
add_test(NAME DescriptorGTest COMMAND descriptor_gtest)

# Disassembler tests
add_executable(disassembler_gtest DisassemblerGTest.cpp)
target_link_libraries(disassembler_gtest
  aegisbit
  GTest::gtest_main
)
target_include_directories(disassembler_gtest PRIVATE
  ${CMAKE_SOURCE_DIR}/include
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${LLVM_INCLUDE_DIRS}
)
add_test(NAME DisassemblerGTest COMMAND disassembler_gtest)

# CFG Builder tests
add_executable(cfg_gtest CFGBuilderGTest.cpp)
target_link_libraries(cfg_gtest
  aegisbit
  GTest::gtest_main
)
target_include_directories(cfg_gtest PRIVATE
  ${CMAKE_SOURCE_DIR}/include
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${LLVM_INCLUDE_DIRS}
)
add_test(NAME CFGGTest COMMAND cfg_gtest)

# Instruction Builder tests
add_executable(instruction_builder_gtest InstructionBuilderGTest.cpp)
target_link_libraries(instruction_builder_gtest
  aegisbit
  GTest::gtest_main
)
target_include_directories(instruction_builder_gtest PRIVATE
  ${CMAKE_SOURCE_DIR}/include
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${LLVM_INCLUDE_DIRS}
)
add_test(NAME InstructionBuilderGTest COMMAND instruction_builder_gtest)

# RuntimeConfig tests
add_executable(runtime_config_gtest RuntimeConfigGTest.cpp)
target_link_libraries(runtime_config_gtest
  aegisbit
  GTest::gtest_main
)
target_include_directories(runtime_config_gtest PRIVATE
  ${CMAKE_SOURCE_DIR}/include
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${LLVM_INCLUDE_DIRS}
)
add_test(NAME RuntimeConfigGTest COMMAND runtime_config_gtest)

# Instrumentation planner tests (typed plan selection and validation)
add_executable(instrumentation_planner_gtest InstrumentationPlannerGTest.cpp)
target_link_libraries(instrumentation_planner_gtest
  aegisbit
  GTest::gtest_main
)
target_include_directories(instrumentation_planner_gtest PRIVATE
  ${CMAKE_SOURCE_DIR}/include
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${LLVM_INCLUDE_DIRS}
)
add_test(NAME InstrumentationPlannerGTest COMMAND instrumentation_planner_gtest)

# HSA Interceptor tests
add_executable(hsa_interceptor_gtest HSAInterceptorGTest.cpp)
target_link_libraries(hsa_interceptor_gtest
  aegisbit
  GTest::gtest_main
)
target_include_directories(hsa_interceptor_gtest PRIVATE
  ${CMAKE_SOURCE_DIR}/include
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${LLVM_INCLUDE_DIRS}
)
add_test(NAME HSAInterceptorGTest COMMAND hsa_interceptor_gtest)

# Dispatch Integration tests
add_executable(dispatch_integration_gtest DispatchIntegrationGTest.cpp)
target_link_libraries(dispatch_integration_gtest
  aegisbit
  GTest::gtest_main
)
target_include_directories(dispatch_integration_gtest PRIVATE
  ${CMAKE_SOURCE_DIR}/include
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${LLVM_INCLUDE_DIRS}
)
add_test(NAME DispatchIntegrationGTest COMMAND dispatch_integration_gtest)

# HSA Memory Pool tests (requires GPU)
add_executable(hsa_pool_gtest HSAPoolGTest.cpp)
target_link_libraries(hsa_pool_gtest
  aegisbit
  GTest::gtest_main
)
target_include_directories(hsa_pool_gtest PRIVATE
  ${CMAKE_SOURCE_DIR}/include
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${LLVM_INCLUDE_DIRS}
)
add_test(NAME HSAPoolGTest COMMAND hsa_pool_gtest)

# TrampolineBridge tests
add_executable(trampoline_bridge_gtest TrampolineBridgeGTest.cpp)
target_link_libraries(trampoline_bridge_gtest PRIVATE aegisbit GTest::gtest_main)
target_include_directories(trampoline_bridge_gtest PRIVATE
  ${CMAKE_SOURCE_DIR}/include
  ${CMAKE_SOURCE_DIR}/test/unit
  ${LLVM_INCLUDE_DIRS}
)
add_test(NAME TrampolineBridgeGTest COMMAND trampoline_bridge_gtest)

# Coalescing Analyzer tests (element size inference, per-site analysis)
add_executable(coalescing_analyzer_gtest CoalescingAnalyzerGTest.cpp)
target_link_libraries(coalescing_analyzer_gtest PRIVATE aegisbit GTest::gtest_main)
target_include_directories(coalescing_analyzer_gtest PRIVATE
  ${CMAKE_SOURCE_DIR}/include
  ${LLVM_INCLUDE_DIRS}
)
add_test(NAME CoalescingAnalyzerGTest COMMAND coalescing_analyzer_gtest)

# Island Allocator tests (pure layout logic, no disassembler)
add_executable(island_allocator_gtest IslandAllocatorGTest.cpp)
target_link_libraries(island_allocator_gtest PRIVATE aegisbit GTest::gtest_main)
target_include_directories(island_allocator_gtest PRIVATE
  ${CMAKE_SOURCE_DIR}/include
  ${CMAKE_SOURCE_DIR}/test/unit
  ${LLVM_INCLUDE_DIRS}
)
add_test(NAME IslandAllocatorGTest COMMAND island_allocator_gtest)

# Jump heuristics tests (relay forcing and pre-kernel padding)
add_executable(jump_heuristics_gtest JumpHeuristicsGTest.cpp)
target_link_libraries(jump_heuristics_gtest PRIVATE aegisbit GTest::gtest_main)
target_include_directories(jump_heuristics_gtest PRIVATE
  ${CMAKE_SOURCE_DIR}/include
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${LLVM_INCLUDE_DIRS}
)
add_test(NAME JumpHeuristicsGTest COMMAND jump_heuristics_gtest)

# Relay Emitter tests (stub generation and fixup validation)
add_executable(relay_emitter_gtest RelayEmitterGTest.cpp)
target_link_libraries(relay_emitter_gtest PRIVATE aegisbit GTest::gtest_main)
target_include_directories(relay_emitter_gtest PRIVATE
  ${CMAKE_SOURCE_DIR}/include
  ${CMAKE_SOURCE_DIR}/test/unit
  ${LLVM_INCLUDE_DIRS}
)
add_test(NAME RelayEmitterGTest COMMAND relay_emitter_gtest)

# Trampoline Stress tests (parameterized sweep, universal invariants)
add_executable(trampoline_stress_gtest TrampolineStressGTest.cpp)
target_link_libraries(trampoline_stress_gtest PRIVATE aegisbit GTest::gtest_main)
target_include_directories(trampoline_stress_gtest PRIVATE
  ${CMAKE_SOURCE_DIR}/include
  ${CMAKE_SOURCE_DIR}/test/unit
  ${LLVM_INCLUDE_DIRS}
)
add_test(NAME TrampolineStressGTest COMMAND trampoline_stress_gtest)

# Trampoline Correctness tests (encoding, register identity, SGPR conflict)
add_executable(trampoline_correctness_gtest TrampolineCorrectnessGTest.cpp)
target_link_libraries(trampoline_correctness_gtest PRIVATE aegisbit GTest::gtest_main)
target_include_directories(trampoline_correctness_gtest PRIVATE
  ${CMAKE_SOURCE_DIR}/include
  ${CMAKE_SOURCE_DIR}/test/unit
  ${LLVM_INCLUDE_DIRS}
)
add_test(NAME TrampolineCorrectnessGTest COMMAND trampoline_correctness_gtest)

# NoteMetadataHandler tests
add_executable(note_metadata_gtest NoteMetadataHandlerGTest.cpp)
target_link_libraries(note_metadata_gtest PRIVATE aegisbit GTest::gtest_main)
target_include_directories(note_metadata_gtest PRIVATE
  ${CMAKE_SOURCE_DIR}/include
  ${LLVM_INCLUDE_DIRS}
)
add_test(NAME NoteMetadataGTest COMMAND note_metadata_gtest)

# CodeObjectParser tests (ELF detection, GPU arch, size inference contract)
add_executable(codeobj_parser_gtest CodeObjectParserGTest.cpp)
target_link_libraries(codeobj_parser_gtest PRIVATE aegisbit GTest::gtest_main)
target_include_directories(codeobj_parser_gtest PRIVATE
  ${CMAKE_SOURCE_DIR}/include
  ${LLVM_INCLUDE_DIRS}
)
add_test(NAME CodeObjParserGTest COMMAND codeobj_parser_gtest)

# CodeObjectHandler tests (load, query, modify, build code objects)
add_executable(codeobj_handler_gtest CodeObjectHandlerGTest.cpp)
target_link_libraries(codeobj_handler_gtest PRIVATE aegisbit GTest::gtest_main)
target_include_directories(codeobj_handler_gtest PRIVATE
  ${CMAKE_SOURCE_DIR}/include
  ${CMAKE_SOURCE_DIR}/test/unit
  ${LLVM_INCLUDE_DIRS}
)
add_test(NAME CodeObjHandlerGTest COMMAND codeobj_handler_gtest)

# PayloadCompiler tests (IR building and AMDGPU code generation)
add_executable(payload_compiler_gtest PayloadCompilerGTest.cpp)
target_link_libraries(payload_compiler_gtest PRIVATE aegisbit GTest::gtest_main)
target_include_directories(payload_compiler_gtest PRIVATE
  ${CMAKE_SOURCE_DIR}/include
  ${LLVM_INCLUDE_DIRS}
)
add_test(NAME PayloadCompilerGTest COMMAND payload_compiler_gtest)

# SourceMapper tests (DWARF source mapping and SourceLocation helpers)
add_executable(source_mapper_gtest SourceMapperGTest.cpp)
target_link_libraries(source_mapper_gtest PRIVATE aegisbit GTest::gtest_main)
target_include_directories(source_mapper_gtest PRIVATE
  ${CMAKE_SOURCE_DIR}/include
  ${CMAKE_SOURCE_DIR}/test/unit
  ${LLVM_INCLUDE_DIRS}
)
add_test(NAME SourceMapperGTest COMMAND source_mapper_gtest)

# KernelPatcher tests (creation, cache, ScratchRegisters allocation)
add_executable(kernel_patcher_gtest KernelPatcherGTest.cpp)
target_link_libraries(kernel_patcher_gtest PRIVATE aegisbit GTest::gtest_main)
target_include_directories(kernel_patcher_gtest PRIVATE
  ${CMAKE_SOURCE_DIR}/include
  ${CMAKE_SOURCE_DIR}/test/unit
  ${LLVM_INCLUDE_DIRS}
)
add_test(NAME KernelPatcherGTest COMMAND kernel_patcher_gtest)

# Patched ELF layout invariant tests (island overlap, descriptor consistency)
add_executable(patched_elf_layout_gtest PatchedELFLayoutGTest.cpp)
target_link_libraries(patched_elf_layout_gtest PRIVATE aegisbit GTest::gtest_main)
target_compile_definitions(patched_elf_layout_gtest PRIVATE
  MEGA_GATHER_FIXTURE_PATH="${CMAKE_CURRENT_SOURCE_DIR}/fixtures/mega_gather_gfx950.bin"
)
target_include_directories(patched_elf_layout_gtest PRIVATE
  ${CMAKE_SOURCE_DIR}/include
  ${CMAKE_SOURCE_DIR}/test/unit
  ${LLVM_INCLUDE_DIRS}
)
add_test(NAME PatchedELFLayoutGTest COMMAND patched_elf_layout_gtest)

# Heap corruption reproduction (FLASH_ATTN_EXT code object)
add_executable(heap_corruption_repro_gtest HeapCorruptionReproGTest.cpp)
target_link_libraries(heap_corruption_repro_gtest PRIVATE aegisbit GTest::gtest_main)
target_compile_definitions(heap_corruption_repro_gtest PRIVATE
  FLASH_ATTN_FIXTURE_PATH="${CMAKE_CURRENT_SOURCE_DIR}/fixtures/flash_attn_gfx950.bin"
)
target_include_directories(heap_corruption_repro_gtest PRIVATE
  ${CMAKE_SOURCE_DIR}/include
  ${CMAKE_SOURCE_DIR}/test/unit
  ${LLVM_INCLUDE_DIRS}
)
add_test(NAME HeapCorruptionReproGTest COMMAND heap_corruption_repro_gtest)

# CanonicalKernelAccum tests (pure data, per-PC accumulation)
add_executable(canonical_kernel_accum_gtest CanonicalKernelAccumGTest.cpp)
target_link_libraries(canonical_kernel_accum_gtest PRIVATE aegisbit GTest::gtest_main)
target_include_directories(canonical_kernel_accum_gtest PRIVATE
  ${CMAKE_SOURCE_DIR}/include
  ${LLVM_INCLUDE_DIRS}
)
add_test(NAME CanonicalKernelAccumGTest COMMAND canonical_kernel_accum_gtest)

# ProfilingResultsSink tests (canonical-by-PC ingest accumulation)
add_executable(profiling_results_sink_gtest ProfilingResultsSinkGTest.cpp)
target_link_libraries(profiling_results_sink_gtest PRIVATE aegisbit GTest::gtest_main)
target_include_directories(profiling_results_sink_gtest PRIVATE
  ${CMAKE_SOURCE_DIR}/include
  ${LLVM_INCLUDE_DIRS}
)
add_test(NAME ProfilingResultsSinkGTest COMMAND profiling_results_sink_gtest)

# VariantCoverageAccumulator tests (pure coverage bookkeeping)
add_executable(variant_coverage_accumulator_gtest VariantCoverageAccumulatorGTest.cpp)
target_link_libraries(variant_coverage_accumulator_gtest PRIVATE aegisbit GTest::gtest_main)
target_include_directories(variant_coverage_accumulator_gtest PRIVATE
  ${CMAKE_SOURCE_DIR}/include
  ${LLVM_INCLUDE_DIRS}
)
add_test(NAME VariantCoverageAccumulatorGTest COMMAND variant_coverage_accumulator_gtest)

# VariantSelector tests (pure round-robin variant picking)
add_executable(variant_selector_gtest VariantSelectorGTest.cpp)
target_link_libraries(variant_selector_gtest PRIVATE aegisbit GTest::gtest_main)
target_include_directories(variant_selector_gtest PRIVATE
  ${CMAKE_SOURCE_DIR}/include
  ${LLVM_INCLUDE_DIRS}
)
add_test(NAME VariantSelectorGTest COMMAND variant_selector_gtest)

# InstrumentationReplay end-to-end test (multi-variant patching)
add_executable(instrumentation_replay_gtest InstrumentationReplayGTest.cpp)
target_link_libraries(instrumentation_replay_gtest PRIVATE aegisbit GTest::gtest_main)
target_compile_definitions(instrumentation_replay_gtest PRIVATE
  MEGA_GATHER_FIXTURE_PATH="${CMAKE_CURRENT_SOURCE_DIR}/fixtures/mega_gather_gfx950.bin"
)
target_include_directories(instrumentation_replay_gtest PRIVATE
  ${CMAKE_SOURCE_DIR}/include
  ${CMAKE_SOURCE_DIR}/test/unit
  ${LLVM_INCLUDE_DIRS}
)
add_test(NAME InstrumentationReplayGTest COMMAND instrumentation_replay_gtest)

# TracingEngine tests (singleton, init state, stats)
add_executable(tracing_engine_gtest TracingEngineGTest.cpp)
target_link_libraries(tracing_engine_gtest PRIVATE aegisbit GTest::gtest_main)
target_include_directories(tracing_engine_gtest PRIVATE
  ${CMAKE_SOURCE_DIR}/include
  ${LLVM_INCLUDE_DIRS}
)
add_test(NAME TracingEngineGTest COMMAND tracing_engine_gtest)

# Enable verbose test output for GoogleTest
set_tests_properties(DescriptorGTest DisassemblerGTest CFGGTest InstructionBuilderGTest RuntimeConfigGTest InstrumentationPlannerGTest HSAInterceptorGTest DispatchIntegrationGTest HSAPoolGTest TrampolineBridgeGTest TrampolineCorrectnessGTest CoalescingAnalyzerGTest NoteMetadataGTest CodeObjParserGTest CodeObjHandlerGTest PayloadCompilerGTest SourceMapperGTest KernelPatcherGTest PatchedELFLayoutGTest TracingEngineGTest IslandAllocatorGTest JumpHeuristicsGTest RelayEmitterGTest TrampolineStressGTest HeapCorruptionReproGTest CanonicalKernelAccumGTest ProfilingResultsSinkGTest VariantCoverageAccumulatorGTest VariantSelectorGTest InstrumentationReplayGTest PROPERTIES
  ENVIRONMENT "GTEST_COLOR=1"
)

#===----------------------------------------------------------------------===//
# Exploratory tests (not part of CI)
#===----------------------------------------------------------------------===//

# Exploration: VALU instruction requirements
add_executable(explore_valu ExploreVALUTest.cpp)
target_link_libraries(explore_valu aegisbit)
target_include_directories(explore_valu PRIVATE
  ${CMAKE_SOURCE_DIR}/include
  ${LLVM_INCLUDE_DIRS}
)
