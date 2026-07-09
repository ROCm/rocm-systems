# DEVCOMM sources (paths relative to PROJECT_SOURCE_DIR to match SRC_FILES
# convention, so they flow through the hipify pipeline like every other source).
set(DEVCOMM_SOURCES
    src/devcomm/devcomm_v22902.h
    src/devcomm/devcomm_v22902.cc
    src/devcomm/devcomm_v22907.cc
    src/devcomm/devcomm_v23000.cc
)

# The devcomm headers (devcomm_v22902.h) are included as "devcomm_v22902.h" by
# the .cc files. After hipify the copies land under HIPIFY_DIR/src/devcomm, so
# pass that path to the parent scope for target_include_directories(rccl ...).
set(DEVCOMM_INCLUDE_DIRS
    "${CMAKE_BINARY_DIR}/hipify/src/devcomm"
    PARENT_SCOPE)

# Propagate sources to parent scope so src/CMakeLists.txt can use ${DEVCOMM_SOURCES}.
set(DEVCOMM_SOURCES ${DEVCOMM_SOURCES} PARENT_SCOPE)
