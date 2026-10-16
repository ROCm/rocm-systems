# Compile each public header independently as C and C++, without a library include path.

# Standalone C99/C++ checks use absolute includes and no rj-arith include path.
file(GLOB detail_headers CONFIGURE_DEPENDS "${rj_arith_source_dir}/include/rj_arith/detail/*.h")
foreach(header_path IN ITEMS "${rj_arith_source_dir}/include/rj_arith/rj_arith.h" ${detail_headers})
  get_filename_component(header "${header_path}" NAME_WE)
  foreach(extension IN ITEMS c cpp)
    if(header STREQUAL "rj_arith")
      set(target "rj_arith_umbrella_header_${extension}_test")
      set(template umbrella.c)
    elseif(header STREQUAL "rj_arith_common")
      set(target "rj_arith_common_header_${extension}_test")
      set(template common.c)
    else()
      set(target "${header}_header_${extension}_test")
      set(template detail.c)
    endif()
    set(source "${CMAKE_CURRENT_BINARY_DIR}/${header}_header_test.${extension}")
    configure_file("${template}" "${source}" @ONLY)
    add_library(${target} OBJECT "${source}")
    set_target_properties(${target} PROPERTIES
      C_STANDARD 99 C_STANDARD_REQUIRED ON
    )
    if(CMAKE_C_COMPILER_ID MATCHES "GNU|Clang" AND CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
      target_compile_options(${target} PRIVATE -Wall -Wextra -Werror -pedantic)
    endif()
  endforeach()
endforeach()
