// AIRUNTIME-28 benchmark support: the production dispatch geometry.
//
// A copy kernel's performance depends as much on how it is dispatched as on what
// it stores, so every experiment must dispatch the way the runtime does. This
// mirrors KernelBlitManager::shaderCopyBuffer (clr/rocclr/device/rocm/rocblit.cpp)
// argument for argument: element count rounded up, global work size clamped to
// limit_blit_wg_ x local size then aligned up, and end_ptr excluding the scalar
// tail. Getting any of these wrong changes the number of grid-stride iterations
// per thread, which is a larger effect than anything under test.
#ifndef AIRUNTIME28_GEOMETRY_H_
#define AIRUNTIME28_GEOMETRY_H_

#include <hip/hip_runtime.h>

#include <algorithm>
#include <vector>

#include "check.h"
#include "config.h"
#include "variants.h"

namespace bench {

inline u64 alignUp(u64 x, u64 a) { return (x + a - 1) / a * a; }

struct Geometry {
  dim3 grid, block;
  u32 remainder = 0;
  u32 alignedSize = kMaxAlignment;
  u32 nextChunk = 0;
  u32 workgroupSize = static_cast<u32>(kLocalWorkSize);
  u64 endPtr = 0;
};

// blitWg is limit_blit_wg_, which the runtime sets to the CU count.
inline Geometry makeGeometry(const VariantInfo& vi, const u8* dst, u64 size, u32 blitWg) {
  Geometry g;
  g.alignedSize = vi.alignedSizeArg;
  g.remainder = static_cast<u32>(size % vi.alignedSizeArg);

  // Production: size.c[0] /= aligned_size; size.c[0] += (remainder != 0) ? 1 : 0;
  const u64 elements = size / vi.geomElemBytes + ((size % vi.geomElemBytes) ? 1 : 0);

  u64 globalWorkSize = std::min<u64>(static_cast<u64>(blitWg) * kLocalWorkSize, elements);
  globalWorkSize = alignUp(globalWorkSize, kLocalWorkSize);

  g.workgroupSize = static_cast<u32>(kLocalWorkSize);
  g.nextChunk = static_cast<u32>(globalWorkSize);
  g.block = dim3(static_cast<u32>(kLocalWorkSize));
  g.grid = dim3(static_cast<u32>(globalWorkSize / kLocalWorkSize));
  g.endPtr = reinterpret_cast<u64>(dst) + size - g.remainder;
  return g;
}

// Enqueue one variant's copy with production geometry. This is the call every
// experiment makes; nothing else should be constructing a Geometry.
inline void enqueueCopy(int variant, hipStream_t stream, const u8* src, u8* dst, u64 size,
                        u32 blitWg) {
  const Geometry g = makeGeometry(kVariants[variant], dst, size, blitWg);
  launchVariant(variant, g.grid, g.block, stream, src, dst, size, g.remainder, g.alignedSize,
                g.endPtr, g.nextChunk, g.workgroupSize);
}

// Byte-exactness. A variant that skips part of the buffer would look fast for an
// uninteresting reason, and the tail path is exactly where a width change is
// likely to go wrong: the size below is deliberately not a multiple of 16.
inline void verifyVariant(int variant, u32 blitWg, u64 size = 4 * kMiB + 12345) {
  std::vector<u8> pattern(size);
  for (u64 i = 0; i < size; ++i) pattern[i] = static_cast<u8>((i * 31 + 7) & 0xFF);

  u8* src = nullptr;
  u8* dst = nullptr;
  HIP_CHECK(hipMalloc(&src, size));
  HIP_CHECK(hipMalloc(&dst, size));
  HIP_CHECK(hipMemcpy(src, pattern.data(), size, hipMemcpyHostToDevice));
  HIP_CHECK(hipMemset(dst, 0xCD, size));

  enqueueCopy(variant, nullptr, src, dst, size, blitWg);
  HIP_CHECK(hipDeviceSynchronize());

  std::vector<u8> back(size, 0);
  HIP_CHECK(hipMemcpy(back.data(), dst, size, hipMemcpyDeviceToHost));

  u64 mismatches = 0;
  u64 firstBad = 0;
  for (u64 i = 0; i < size; ++i) {
    if (back[i] != pattern[i]) {
      if (mismatches == 0) firstBad = i;
      ++mismatches;
    }
  }
  BENCH_ASSERT(mismatches == 0, "variant %s copied %llu/%llu bytes wrongly, first at %llu",
               kVariants[variant].name, static_cast<unsigned long long>(mismatches),
               static_cast<unsigned long long>(size), static_cast<unsigned long long>(firstBad));

  HIP_CHECK(hipFree(src));
  HIP_CHECK(hipFree(dst));
}

}  // namespace bench

#endif  // AIRUNTIME28_GEOMETRY_H_
