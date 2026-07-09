/* Test fixture — a minimal HIP-like header for verifier tests.
 * Lets us assert verifier behavior without depending on a real
 * /opt/rocm/include layout. Self-contained: defines size_t locally
 * so libclang doesn't need system include resolution.
 *
 * Type shapes mirror real HIP:
 *   - hipStream_t is pointer-to-opaque-struct (so the verifier's
 *     handle-pattern matcher recognizes 'ihipStream_t' in the
 *     canonical type), matching the real hipamd hipStream_t typedef.
 *   - hipMemcpyKind is a real enum (so libclang's type spelling is
 *     'enum hipMemcpyKind' and matches the verifier's enum-DSL rule). */
#ifndef FAKE_HIP_HEADER_H_
#define FAKE_HIP_HEADER_H_

typedef unsigned long size_t;

typedef enum { hipSuccess = 0, hipErrorOutOfMemory = 2 } hipError_t;

struct ihipStream_t;
typedef struct ihipStream_t* hipStream_t;

typedef enum hipMemcpyKind {
    hipMemcpyHostToHost = 0,
    hipMemcpyHostToDevice = 1,
    hipMemcpyDeviceToHost = 2,
    hipMemcpyDeviceToDevice = 3,
    hipMemcpyDefault = 4
} hipMemcpyKind;

hipError_t hipMemcpyAsync(void* dst, const void* src, size_t sizeBytes,
                          hipMemcpyKind kind, hipStream_t stream);

hipError_t hipMalloc(void** ptr, size_t size);

hipError_t hipDeviceSynchronize(void);

#endif
