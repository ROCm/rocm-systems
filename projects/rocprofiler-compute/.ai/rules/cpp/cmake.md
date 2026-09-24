# CMake

Modern, target-based CMake. The tree already works this way, so the main job is
not to regress it.

## Principles

1. Think in targets, not variables.
2. Say what you want, not how to build it.
3. Do not pollute the global scope. Settings belong on a target.
4. Use generator expressions for anything that depends on the configuration.
5. Use `PUBLIC`, `PRIVATE`, and `INTERFACE` deliberately.

## Target commands, not global ones

```cmake
# Do not
include_directories(${PROJECT_SOURCE_DIR}/include)
add_definitions(-DMY_DEFINE)
link_directories(${SOME_LIB_DIR})

# Do
target_include_directories(mylib PUBLIC include/)
target_compile_definitions(mylib PRIVATE MY_DEFINE)
target_link_libraries(mylib PUBLIC somelib)
```

## Visibility

- `PRIVATE`: only this target needs it.
- `INTERFACE`: only consumers need it.
- `PUBLIC`: both.

Get this right. A dependency that appears in a public header is `PUBLIC`.
An implementation detail is `PRIVATE`. `src/lib/rocprofiler_compute_tool/CMakeLists.txt`
links `compression` as `PRIVATE` for that reason, and says so in a comment.

## List sources explicitly

Never `file(GLOB)` for sources. A glob does not invalidate the build when a file
appears, so the build silently goes stale. List headers alongside sources so
they show up in IDEs.

## Language standard

Set the standard once for the tree, as `src/lib/CMakeLists.txt` does. Raise it
for one target with `target_compile_features`, which is how
`torch_trace_collector` gets C++20. Do not put `-std=c++XX` in
`target_compile_options`.

Guard a raised standard on compiler support and fail with a clear message rather
than a link error later.

## Compile options

```cmake
target_compile_options(mylib PRIVATE
    $<$<CXX_COMPILER_ID:GNU,Clang>:-Wall -Wextra>
)
```

## Dependencies

Prefer `find_package` with imported targets:

```cmake
find_package(Boost 1.70 REQUIRED COMPONENTS filesystem)
target_link_libraries(myapp PRIVATE Boost::filesystem)
```

Vendored dependencies live under `src/lib/external/` and come in through
`add_subdirectory`.

## Header-only libraries

```cmake
add_library(myheaderlib INTERFACE)
target_include_directories(myheaderlib INTERFACE include/)
```

`gsl_assert` and `synchronized` under `src/lib/utils/` are the examples here.

## Tests

Test targets go in a `tests/` subdirectory, added conditionally:

```cmake
if(ENABLE_TESTS)
    add_subdirectory(tests)
endif()
```

Register them with `add_test` so `ctest` finds them.

## Checklist

- [ ] No `include_directories`, `add_definitions`, or `link_directories`
- [ ] Every `target_link_libraries` and `target_include_directories` has a
      visibility keyword, and the choice is correct
- [ ] Sources listed explicitly, no `file(GLOB)`
- [ ] Configuration-dependent flags use generator expressions
- [ ] Tests behind `ENABLE_TESTS`
