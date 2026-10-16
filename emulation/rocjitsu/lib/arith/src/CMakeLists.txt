# Build the static arithmetic library and select optional CPU backends.
# Public include paths and implementation-only compiler settings are defined here.

add_library(rj_arith STATIC
  common/cpu.cpp
  common/cpu_decode.cpp
  register/dpp.cpp
  fp/arithmetic.cpp
  fp/half_fma.cpp
  wmma/wmma.cpp
  wmma/wmma_replicated.cpp
)

target_include_directories(rj_arith
  PUBLIC ${rj_arith_source_dir}/include
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
)
if(UNIX)
  target_link_libraries(rj_arith PRIVATE m)
endif()
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  target_compile_options(rj_arith PRIVATE -Wall -Wextra -Werror -ffp-contract=off)
endif()
if(RJ_ARITH_ENABLE_X86_64)
  target_compile_definitions(rj_arith PRIVATE RJ_ARITH_X86_64_QUERY=1)
  include(${CMAKE_CURRENT_SOURCE_DIR}/x86_64.cmake)
endif()
