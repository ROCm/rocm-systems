# Build and register the standalone CPU-feature diagnostic.

add_executable(cpuinfo cpuinfo.c)
target_link_libraries(cpuinfo PRIVATE rj_arith)
if(CMAKE_C_COMPILER_ID MATCHES "GNU|Clang")
  target_compile_options(cpuinfo PRIVATE -Wall -Wextra -Werror -pedantic)
endif()
add_test(NAME "rj_arith.cpuinfo" COMMAND cpuinfo)

