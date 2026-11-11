/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef LL128_NARROW_FLAG_H_
#define LL128_NARROW_FLAG_H_

#include <hip/hip_runtime.h>
#include <stdint.h>

// Flag-lane layout for narrow LL128 flags (RCCL_LL128_FLAG_BYTES = 4, 2 or 1; see rccl_ll128_flag.h).
//
// Each lane owns ThreadChunks 16-byte wire chunks per slice, one per row of WarpSz lanes, and every LineLanes
// consecutive lanes of a row form one line whose last lane is the flag lane. Data lanes are as with 8-byte flags.
// A flag lane's wire chunk carries WireData data bytes, then the flag. Its user words form a byte stream of
// StreamBytes that fills its chunks in order (packFlagChunk / unpack), so 8-byte types stay whole and are reduced in
// user layout:
//   - an even number of whole words is interleaved across the row's flag lanes, every pair 16-byte aligned;
//   - an odd number is contiguous, so some pairs start on an odd word;
//   - a stream that ends in half a word is split between the flag lanes of lines 2p and 2p + 1, which swap halves
//     after every receive.
//
// Templated on the line geometry rather than reading NCCL_LL128_LINEELEMS / WARP_SIZE so the unit tests can check
// every layout on the CPU.
template <int FlagBytes, int LineElems, int ElemsPerThread, int WarpSz>
struct Ll128NarrowLayout {
  static constexpr int LineLanes = LineElems / 2;
  static constexpr int ThreadChunks = ElemsPerThread / 2;
  static constexpr int FlagLanesPerRow = WarpSz / LineLanes;
  static constexpr int DataLanesPerRow = WarpSz - FlagLanesPerRow;
  static constexpr int WireData = 16 - FlagBytes;
  static constexpr int StreamBytes = ThreadChunks * WireData;
  static constexpr int StreamDw = StreamBytes / 4;
  // Whole user words per flag lane. With a split word, the flag lanes of lines 2p and 2p + 1 both hold it.
  static constexpr int FlagWords = StreamBytes / 8;
  static constexpr bool SplitWord = StreamBytes % 8 != 0;
  static constexpr bool OddStarts = SplitWord || FlagWords % 2 != 0;
  static constexpr int FlagShift = 64 - 8 * FlagBytes;
  // User words per slice.
  static constexpr int DataWords = ThreadChunks * WarpSz / LineLanes * (LineElems * 8 - FlagBytes) / 8;
  static_assert(FlagBytes == 4 || FlagBytes == 2 || FlagBytes == 1, "narrow flags only");
  static_assert(StreamBytes % 4 == 0, "flag-lane stream must be whole dwords");
  static_assert(!SplitWord || (FlagLanesPerRow % 2 == 0 && FlagWords + 1 <= 2 * ThreadChunks),
                "a split word needs pairs of flag lanes and a register for it");

  static __host__ __device__ constexpr bool isFlagLane(int lane) {
    return lane % LineLanes == LineLanes - 1;
  }
  // Of a split word, only the odd line's flag lane stores it.
  static __host__ __device__ constexpr bool isOddFlagLine(int lane) {
    return SplitWord && (lane / LineLanes) % 2 == 1;
  }
  // Slice-relative user word index of register pair g: regs[2g], regs[2g+1] hold user words ix, ix+1.
  static __host__ __device__ constexpr int userWordIx(int g, int lane) {
    constexpr int Base = 2 * ThreadChunks * DataLanesPerRow;
    const int line = lane / LineLanes;
    if (!isFlagLane(lane)) return 2 * (g * DataLanesPerRow + lane - line);
    if (!OddStarts) return Base + 2 * ((g < FlagWords / 2 - 1 ? g : FlagWords / 2 - 1) * FlagLanesPerRow + line);
    if (!SplitWord) return Base + FlagWords * line + 2 * g;
    return Base + (2 * FlagWords + 1) * (line / 2) + FlagWords * (line % 2) + 2 * g;
  }
  // Bit k set: this lane stores word k of pair g. Words a lane holds but does not store are duplicates.
  static __host__ __device__ constexpr int storedWords(int g, int lane) {
    if (!isFlagLane(lane)) return 3;
    const int words = FlagWords + (isOddFlagLine(lane) ? 1 : 0);
    return (2 * g < words ? 1 : 0) | (2 * g + 1 < words ? 2 : 0);
  }

  template <int N>
  static __host__ __device__ uint32_t dwordOf(uint64_t const (&x)[N], int k) {
    return k >= 2 * N ? 0u : k % 2 ? uint32_t(x[k / 2] >> 32) : uint32_t(x[k / 2]);
  }
  // Bytes [sh, sh + 4) of the 8 bytes hi:lo.
  static __host__ __device__ uint32_t funnel(uint32_t lo, uint32_t hi, int sh) {
#ifdef __HIP_DEVICE_COMPILE__
    return sh == 0 ? lo : __builtin_amdgcn_alignbyte(hi, lo, sh);
#else
    return sh == 0 ? lo : uint32_t((uint64_t(hi) << 32 | lo) >> (8 * sh));
#endif
  }

  // Wire chunk c of a flag lane holding user words u: bytes [c * WireData, (c + 1) * WireData) of its stream, then
  // the flag. With a split word X (the even line's u[FlagWords], the odd line's u[0]), the even line's stream ends
  // with lo(X); the odd line skips lo(X) and sends the high dword of its last word in its place, at position 0.
  template <int N>
  static __host__ __device__ void packFlagChunk(uint64_t const (&u)[N], int c, bool odd, uint64_t flag, uint64_t& w0,
                                                uint64_t& w1) {
    auto s = [&](int i) -> uint32_t {
      if (i >= StreamDw) return 0u;
      if (SplitWord && i == 0) return odd ? dwordOf(u, 2 * FlagWords + 1) : dwordOf(u, 0);
      return dwordOf(u, i);
    };
    uint32_t p[4];
#pragma unroll
    for (int j = 0; j < 4; j++) {
      const int b = c * WireData + 4 * j;
      p[j] = funnel(s(b / 4), s(b / 4 + 1), b % 4);
    }
    constexpr int DataBits = 32 - 8 * FlagBytes;
    p[3] = DataBits == 0 ? uint32_t(flag) : (p[3] & ((1u << DataBits) - 1)) | (uint32_t(flag) << DataBits);
    w0 = p[0] | uint64_t(p[1]) << 32;
    w1 = p[2] | uint64_t(p[3]) << 32;
  }

  // Turn a received slice back into user words, in place. Only flag lanes change, but every lane must call it:
  // with a split word, swapXor(v) must return v from lane ^ LineLanes (a warp shuffle on the device).
  template <int N, typename SwapXor>
  static __host__ __device__ void unpack(uint64_t (&x)[N], bool flagLane, bool odd, SwapXor&& swapXor) {
    static_assert(N == 2 * ThreadChunks, "slice registers must match the layout");
    uint32_t d[2 * N];
#pragma unroll
    for (int k = 0; k < 2 * N; k++) d[k] = dwordOf(x, k);
    uint32_t s[StreamDw];
#pragma unroll
    for (int i = 0; i < StreamDw; i++) {
      const int c0 = 4 * i / WireData, o = 4 * i % WireData;
      const int k = 4 * c0 + o / 4;
      uint32_t v = funnel(d[k], k + 1 < 2 * N ? d[k + 1] : 0u, o % 4);
      if (o + 4 > WireData) {
        const int n = WireData - o;
        v = (v & ((1u << (8 * n)) - 1)) | (d[4 * (c0 + 1)] << (8 * n));
      }
      s[i] = v;
    }
#pragma unroll
    for (int i = 0; i < StreamDw; i++) d[i] = flagLane ? s[i] : d[i];
    if (SplitWord) {
      const uint32_t other = swapXor(odd ? s[1] : s[StreamDw - 1]);
      if (flagLane) {
        d[0] = odd ? other : s[0];
        d[2 * FlagWords + 1] = odd ? s[0] : other;
      }
    }
#pragma unroll
    for (int k = 0; k < N; k++) x[k] = d[2 * k] | uint64_t(d[2 * k + 1]) << 32;
  }
};

#endif // LL128_NARROW_FLAG_H_
