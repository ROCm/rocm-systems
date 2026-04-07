#include <cstdint>

// Provide ncclDebugLevel and ncclDebugMask symbols for test binaries.
// In Release builds these are hidden inside librccl.so, but the NCCL 2.28.9
// INFO/WARN/TRACE macros reference them directly via extern declarations.
int ncclDebugLevel = -1;
uint64_t ncclDebugMask = 0;
