/***************************************************************************
 Copyright (c) 2022-2023, Advanced Micro Devices, Inc. All rights reserved.
***************************************************************************/

/**
 * Setting this to 1 makes the application use only single-precision floating-point data.
 * Set this to 0 in order to use double-precision floating-point data instead.
 */
#define USE_FLOAT 0

/**
 * This is the default domain size (when not explicitly stated with "-d" in the
 * command-line arguments).
 */
#define DEFAULT_DOMAIN_SIZE 4096

/**
 * This is the minimum acceptable domain size in any of the 2 dimensions.
 */
#define MIN_DOM_SIZE 1

/**
 * Global domain dimensions
 */
#define X_MIN -0.5
#define X_MAX 0.5
#define Y_MIN -0.5
#define Y_MAX 0.5

/**
 * Side Indices
 */
#define NSIDES     4
#define SIDE_DOWN  0
#define SIDE_RIGHT 1
#define SIDE_UP    2
#define SIDE_LEFT  3

/**
 * This is the Jacobi tolerance threshold. The run is considered to have converged when
 * the maximum residue falls below this value.
 */
#define JACOBI_TOLERANCE 1.0E-5F

/**
 * This is the Jacobi iteration count limit. The Jacobi run will never cycle more than
 * this, even if it has not converged when finishing the last allowed iteration.
 */
#define JACOBI_MAX_LOOPS 1000

/**
 * This is the status value that indicates a successful operation.
 */
#define STATUS_OK 0

/**
 * This is the status value that indicates an error.
 */
#define STATUS_ERR -1

#if USE_FLOAT
#    define dfloat     float
#    define MPI_DFLOAT MPI_FLOAT
#else
#    define dfloat     double
#    define MPI_DFLOAT MPI_DOUBLE
#endif

#define uint64     unsigned long long
#define MPI_UINT64 MPI_UNSIGNED_LONG_LONG

#define SafeHipCall(call) CheckHipCall(call, #call, __FILE__, __LINE__)
#define SafeHostFree(block)                                                              \
    {                                                                                    \
        if(block) free(block);                                                           \
    }
#define SafeDevFree(block)                                                               \
    {                                                                                    \
        if(block) SafeHipCall(hipFree(block));                                           \
    }
#define OnePrintf(allow, ...)                                                            \
    {                                                                                    \
        if(allow) printf(__VA_ARGS__);                                                   \
    }
#define OneErrPrintf(allow, ...)                                                         \
    {                                                                                    \
        if(allow) fprintf(stderr, __VA_ARGS__);                                          \
    }
#define HasNeighbor(neighbors, dir) (neighbors[dir] != MPI_PROC_NULL)

#define CheckHipCall(command, commandName, fileName, line)                               \
    if(command != hipSuccess)                                                            \
    {                                                                                    \
        fprintf(stderr,                                                                  \
                "Error: \"%s\" for call \"%s\""                                          \
                "in file \"%s\" at line %d. Terminating...\n",                           \
                hipGetErrorString(command), commandName, fileName, line);                \
        MPI_Abort(MPI_COMM_WORLD, STATUS_ERR);                                           \
    }
