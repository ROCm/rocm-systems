// AIRUNTIME-28 benchmark support: the constants every experiment shares.
//
// These were previously copy-pasted per program, which is how the bootstrap
// resample count came to differ between experiments (2000 in one, 3000 in the
// rest) and how every footprint ended up sized against a cache size we later
// measured to be wrong. One definition each, here.
#ifndef AIRUNTIME28_CONFIG_H_
#define AIRUNTIME28_CONFIG_H_

// Widths spelled the way the production OpenCL kernel spells them. u64x2 is a
// native ext_vector_type rather than HIP's ulong2, because HIP's is a struct and
// __builtin_nontemporal_* rejects it; the native vector reproduces the OpenCL
// codegen exactly.
//
// Note what that rejection is and is not. The builtin wants "a pointer to
// integer, float, pointer, or a vector of such types", so it objects to the type
// being a struct, not to it being 128 bits wide. Declaring the vector natively is
// the whole fix: these are HIP translation units and they emit
// global_store_b128 ... th:TH_STORE_NT, which isa_check.sh verifies. Narrowing
// the access to get the hint - which the ticket assumed was necessary - is not
// required in HIP either, and would cost 77% (see the isolated-copy table).
typedef unsigned long u64;
typedef unsigned int u32;
typedef unsigned char u8;
typedef unsigned long u64x2 __attribute__((ext_vector_type(2)));

namespace bench {

constexpr u64 kKiB = 1024ULL;
constexpr u64 kMiB = 1024ULL * 1024ULL;
constexpr u64 kGiB = 1024ULL * 1024ULL * 1024ULL;

// ---------------------------------------------------------------------------
// Cache facts for gfx1250, measured rather than queried
// ---------------------------------------------------------------------------
// The driver reports 4 MiB via hipDeviceProp_t::l2CacheSize, rocminfo and
// amd-smi. All three read the same KFD topology record, whose geometry fields
// (cache_line_size, association, latency) are all zero - an unpopulated stub.
//
// The capacity sweep (experiments/cache_capacity) measures a ~11.5 TB/s plateau
// across 24-80 MiB against a 4.3 TB/s HBM floor, with the curve crossing the
// midpoint between them at 128 MiB. So the capacity is bracketed at 96-128 MiB.
// 96 MiB is used here: it is the conservative end, and it matches the
// architecture notes' "96 MB per AID". Being conservative is the right direction,
// because this constant decides which footprints an experiment calls
// cache-resident, and overstating it would mean claiming a working set fits when
// it does not. Size working sets against this, never against the driver figure.
constexpr u64 kGL2Bytes = 96 * kMiB;
constexpr u64 kDriverReportedL2Bytes = 4 * kMiB;  // kept only to report the discrepancy

// Aggregate near cache (L0 + GL1). The capacity sweep shows the first knee
// between 12 and 16 MiB; KFD publishes 566 level-1 entries totalling 26.9 MiB,
// not all of which one access stream can use.
constexpr u64 kNearCacheBytes = 16 * kMiB;

// ---------------------------------------------------------------------------
// Flush sizing
// ---------------------------------------------------------------------------
// Every "cold" measurement depends on this being large enough to evict GL2. The
// original 256 MiB was chosen believing GL2 was 4 MiB, a 64x margin; against the
// measured ~96 MiB it is 2.7x, which is why the sweep was needed.
//
// experiments/flush_sensitivity (2026-08-28) finds results insensitive to it
// across 128 MiB - 2 GiB: cold dependent-load latency is 738-742 ns/hop at every
// size and equal to the warm reference, and copy time varies by 2%. That is not
// because 128 MiB is comfortably adequate; it is because the dispatch boundary
// already invalidates GL2 (see FINDING-gl2-residency.md), so the flush has
// nothing left to do. 1 GiB is kept anyway - 10.7x measured GL2, costing ~0.2 ms
// per sample - so that no result depends on that finding continuing to hold.
constexpr u64 kFlushBytes = 1 * kGiB;

// ---------------------------------------------------------------------------
// Production dispatch geometry, from KernelBlitManager::shaderCopyBuffer
// ---------------------------------------------------------------------------
// aligned ? 512 : 1024; every experiment here uses aligned buffers.
constexpr u64 kLocalWorkSize = 512;
// kMaxAlignment = 2 * sizeof(uint64_t)
constexpr unsigned kMaxAlignment = 16;

// ---------------------------------------------------------------------------
// Statistics
// ---------------------------------------------------------------------------
constexpr int kBootstrapResamples = 3000;
constexpr unsigned kBootstrapSeed = 20260828u;  // fixed so intervals are reproducible
constexpr int kDefaultIters = 40;
constexpr int kDefaultWarmup = 8;

}  // namespace bench

#endif  // AIRUNTIME28_CONFIG_H_
