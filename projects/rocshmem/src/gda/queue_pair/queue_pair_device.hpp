// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef LIBRARY_SRC_GDA_QUEUE_PAIR_DEVICE_HPP_
#define LIBRARY_SRC_GDA_QUEUE_PAIR_DEVICE_HPP_

/**
 * @file queue_pair_device.hpp
 *
 * @section DESCRIPTION
 * An IB QueuePair (SQ and CQ) that the device can use to perform network
 * operations. Most important rocSHMEM operations are performed by this
 * class.
 */

#include <cstddef>
#include <cstdint>

#include <limits>
#include <type_traits>

#include <hip/hip_runtime.h>

#include "constants.hpp"
#include "containers/free_list.hpp"
#include "gda/endian.hpp"
#include "log.hpp"
#include "util.hpp"

#include "queue_pair_common.hpp"
#include "queue_pair_interface.hpp"
#include "queue_pair_option.hpp"
#include "queue_pair_shmem.hpp"

namespace rocshmem {

/*
 * @brief Information required for initializing a Queue Pair.
 */
struct QueuePairInitInfo {
  uintptr_t heap_laddr;
  uintptr_t heap_raddr;
  size_t    heap_size;
  uint32_t  heap_lkey;
  uint32_t  heap_rkey;

  uint64_t             *fetching_atomic;
  uint64_t             *nonfetching_atomic;
  uint32_t             fetching_atomic_lkey;
  uint32_t             nonfetching_atomic_lkey;
  FreeList<uint64_t*>  *fetching_atomic_freelist;
  const BufferInfo     *local_buffers;
  size_t               num_user_buffers;
  const SymmBufferInfo *symm_buffers;
  const int            *symm_count;
};

/*
 * @brief CRTP base class for Provider-specific Queue Pair implementations.
 */
template <typename Provider>
class alignas(ALIGNMENT) QueuePairDevice : public QueuePairSHMEM<Provider> {
/**
 * @name Provider-Defined Members
 *
 * Members that are defined based on specializations of QueuePairTraits<Provider>.
 * Each Provider must supply their own definitions.
 *
 * @{
 */
public:
  /**
   * @brief Type alias for QueuePairTraits<Provider>.
   */
  using typename QueuePairInterface<Provider>::Traits;

  /**
   * @brief Enumeration of the opcodes for Write, Read, Fetch-Add, and Compare-and-Swap.
   *
   * A definition must be provided by the QueuePairTraits<Provider> specialization
   * of each Provider subclass.
   */
  using typename QueuePairInterface<Provider>::OpCode;

  /**
   * @brief Constant defining the endianness required by the provider. Used for e.g. lkey and rkey.
   *
   * A definition must be provided by the QueuePairTraits<Provider> specialization
   * of each Provider subclass.
   */
  static constexpr endian::Order ProviderEndianness = Traits::Endianness;
/**@}*/



/**
 * @name Non-static Data Members
 *
 * @{
 */
protected:
  /*
   * @brief Cached heap info for this QP.
   *
   * heap.local_base is heap_bases[my_pe], captured once at setup.
   * heap.remote_base is heap_bases[dest_pe], captured once at setup.
   * heap.length is the heap size.
   * heap.lkey is the LKey for local_base.
   * heap.rkey is the RKey for remote_base.
   *
   * This lets the (common) heap translation run as pure register arithmetic
   * with no additional indirection or memory load (besides *this).
   */
  SymmBufferInfo heap;

  // Used in most WQEs
  uint32_t qp_num;
  uint32_t fetching_atomic_idx{0};

  // Used in atomic WQEs
  uint64_t* fetching_atomic;
  uint64_t* nonfetching_atomic;

  uint32_t fetching_atomic_lkey;
  uint32_t nonfetching_atomic_lkey;

  static constexpr size_t FETCHING_ATOMIC_CNT{1024};
  static_assert(FETCHING_ATOMIC_CNT % WF_SIZE == 0);
  using FreeListT = FreeList<uint64_t*>;
  FreeListT* fetching_atomic_freelist;

private:
  // Used by get_laddr
  const BufferInfo* local_buffers;
  size_t num_user_buffers;

  /**
   * @brief Contiguous slice of registration entries for this QP's fixed
   * (dest_pe, nic_idx), indexed by registration slot [0, *symm_count).
   *
   * Points into the backend's flat entry table; every entry is already
   * specialized to this QP, so a match yields the remote address and keys with
   * no further dereference. Null when symmetric registration is unavailable.
   */
  const SymmBufferInfo *symm_buffers;

  /**
   * @brief Shared registration count (number of live entries per slice).
   *
   * Single device int shared by all QPs; register/unregister publish updates
   * here. One load per non-heap lookup (null when registration unavailable).
   */
  const int *symm_count;
/**@}*/



/**
 * @name Internal
 *
 * @{
 */
protected:
  /*
   * @brief Provider is a friend of QueuePairDevice<Provider>.
   */
  friend Provider;

  template <AMOFetchType Fetch>
  __device__ __forceinline__ constexpr uint64_t* get_atomic_addr() const {
    static_assert(Fetch != AMOFetchType::NonBlocking, "non-blocking AMOs not yet implemented");
    if constexpr (Fetch == AMOFetchType::Blocking) {
      return fetching_atomic;
    } else if constexpr (Fetch == AMOFetchType::NonFetching) {
      return nonfetching_atomic;
    }
  }

  template <AMOFetchType Fetch>
  __device__ __forceinline__ constexpr uint32_t get_atomic_lkey() const {
    static_assert(Fetch != AMOFetchType::NonBlocking, "non-blocking AMOs not yet implemented");
    if constexpr (Fetch == AMOFetchType::Blocking) {
      return fetching_atomic_lkey;
    } else if constexpr (Fetch == AMOFetchType::NonFetching) {
      return nonfetching_atomic_lkey;
    }
  }

private:
  __device__ Provider& provider() {
    return static_cast<Provider&>(*this);
  }
/**@}*/



/**
 * @name Constructors and Destructors
 *
 * @{
 */
protected:
  /**
   * @brief Constructor.
   *
   * @param[in] qpn Queue Pair number.
   * @param[in] init_info Initialization information for this QP.
   */
  __host__ explicit QueuePairDevice(uint32_t qpn, QueuePairInitInfo&& init_info)
    : heap{.local_base  = init_info.heap_laddr,
           .remote_base = init_info.heap_raddr,
           .length      = init_info.heap_size,
           .lkey        = to_provider_endianness(init_info.heap_lkey),
           .rkey        = to_provider_endianness(init_info.heap_rkey)},
      qp_num{qpn},
      fetching_atomic{init_info.fetching_atomic},
      nonfetching_atomic{init_info.nonfetching_atomic},
      fetching_atomic_lkey{to_provider_endianness(init_info.fetching_atomic_lkey)},
      nonfetching_atomic_lkey{to_provider_endianness(init_info.nonfetching_atomic_lkey)},
      fetching_atomic_freelist{init_info.fetching_atomic_freelist},
      local_buffers{init_info.local_buffers},
      num_user_buffers{init_info.num_user_buffers},
      symm_buffers{init_info.symm_buffers},
      symm_count{init_info.symm_count} { }

  /**
   * @brief Convenience Constructor for GIN. Unspecified values are nullptr or 0.
   *
   * A QueuePairDevice constructed in this manner
   * must ONLY call explicit-key APIs such as those defined by QueuePairInterface.
   * Additionally, only non-fetching atomics may be used:
   * fetching (blocking) atomics require that initialization of the array of fetch locations.
   *
   * @param[in] qpn Queue Pair number.
   * @param[in] nonfetching_atomic Single location used for the return values of nonfetching AMOs.
   * @param[in] nonfetching_atomic_lkey LKey for nonfetching_atomic.
   */
  __host__ explicit QueuePairDevice(uint32_t qpn,
                                    uint64_t *nonfetching_atomic, uint32_t nonfetching_atomic_lkey)
    : QueuePairDevice{qpn, QueuePairInitInfo{
          .heap_laddr = 0, .heap_raddr = 0, .heap_size = 0, .heap_lkey = 0, .heap_rkey = 0,
          .fetching_atomic      = nullptr, .nonfetching_atomic      = nonfetching_atomic,
          .fetching_atomic_lkey = 0,       .nonfetching_atomic_lkey = nonfetching_atomic_lkey,
          .fetching_atomic_freelist = nullptr,
          .local_buffers = nullptr, .num_user_buffers = 0,
          .symm_buffers  = nullptr, .symm_count       = nullptr}} { }

  __host__ QueuePairDevice(const QueuePairDevice& other)            = delete;
  __host__ QueuePairDevice& operator=(const QueuePairDevice& other) = delete;
  __host__ QueuePairDevice(QueuePairDevice&& other) noexcept        = default;
  __host__ QueuePairDevice& operator=(QueuePairDevice&& other)      = default;
  __host__ ~QueuePairDevice()                                       = default;
/**@}*/



/**
 * @name Address Management
 *
 * @{
 */
public:
  /**
   * @brief Resolve the local (origin) virtual address and LKey of a symmetric address.
   *
   * Attempts to resolve the address in the following order of locations:
   *   1. The default symmetric heap.
   *   2. All user-registered local buffers.
   *   3. All user-registered symmetric buffers.
   *
   * @param[in] addr Symmetric address to resolve.
   *
   * @return {laddr, lkey} for addr or {0, std::numeric_limits<uint32_t>::max()} if not found.
   * Endianness of returned LKey value is ProviderEndianness.
   */
  __device__ __forceinline__
  std::tuple<uintptr_t, uint32_t> get_laddr_info(const void *addr, bool inlined = false) const;

  /**
   * @brief Resolve the remote (target) virtual address and RKey of a symmetric address.
   *
   * Attempts to resolve the address in the following order of locations:
   *   1. The default symmetric heap.
   *   2. All user-registered symmetric buffers.
   *
   * @param[in] addr Symmetric address to resolve.
   *
   * @return {raddr, rkey} for addr or {0, std::numeric_limits<uint32_t>::max()} if not found.
   * Endianness of returned RKey value is ProviderEndianness.
   */
  __device__ __forceinline__
  std::tuple<uintptr_t, uint32_t> get_raddr_info(const void *addr) const;

  /*
   * @brief Query whether data can be inlined into a WQE.
   *
   * @param[in] size Size of data carried by the WQE.
   * @tparam Op OpCode for this WQE.
   *
   * @return True if size bytes of data can be inlined into a WQE with OpCode Op, else false.
   */
  template <OpCode Op>
  static __host__ __device__ __forceinline__ constexpr bool can_inline(size_t size) {
    if constexpr (Op == OpCode::RDMA_WRITE) {
      return size <= Traits::InlineThreshold;
    } else {
      return false;
    }
  }

  /*
   * @brief Copy an inline RMA payload of `size` bytes (1..InlineThreshold) from
   * `laddr` into the WQE inline data segment at `dst`.
   *
   * Dispatches the runtime size to a constant-width copy so each power-of-two
   * width lowers to a single native load/store (one or two wide VMEM ops) rather
   * than a byte-by-byte loop, while reading exactly `size` bytes (no read past
   * the source buffer). Width arms wider than the provider's InlineThreshold are
   * discarded at compile time, so a provider never emits copy code for widths it
   * cannot inline. Non-power-of-two sizes fall back to a correct byte copy.
   */
  static __device__ __forceinline__ void copy_inline_payload(uint8_t* dst, uintptr_t laddr, size_t size) {
    constexpr size_t kMax = Traits::InlineThreshold;
    static_assert(kMax <= 32, "inline payload fast-path only covers widths up to 32B");
    const uint8_t* src = reinterpret_cast<const uint8_t*>(laddr);

    // Fast path: exact power-of-two width -> one native load/store (the hot case,
    // e.g. aligned 4B/8B scalar _p). Arms wider than kMax are pruned at compile time.
    if constexpr (kMax >= 1)  { if (size == 1)  { __builtin_memcpy(dst, src, 1);  return; } }
    if constexpr (kMax >= 2)  { if (size == 2)  { __builtin_memcpy(dst, src, 2);  return; } }
    if constexpr (kMax >= 4)  { if (size == 4)  { __builtin_memcpy(dst, src, 4);  return; } }
    if constexpr (kMax >= 8)  { if (size == 8)  { __builtin_memcpy(dst, src, 8);  return; } }
    if constexpr (kMax >= 16) { if (size == 16) { __builtin_memcpy(dst, src, 16); return; } }
    if constexpr (kMax >= 32) { if (size == 32) { __builtin_memcpy(dst, src, 32); return; } }

    // Residual (non-power-of-two) sizes: cover with two overlapping constant-width
    // copies of the first and last K bytes (K = largest power of two < size). Both
    // widths are compile-time constant -> single wide ops, never a byte loop, and
    // the overlap stays within [0, size) so nothing is read past the source.
    if constexpr (kMax >= 16) { if (size > 16) { __builtin_memcpy(dst, src, 16); __builtin_memcpy(dst + size - 16, src + size - 16, 16); return; } }
    if constexpr (kMax >= 8)  { if (size > 8)  { __builtin_memcpy(dst, src, 8);  __builtin_memcpy(dst + size - 8,  src + size - 8,  8);  return; } }
    if constexpr (kMax >= 4)  { if (size > 4)  { __builtin_memcpy(dst, src, 4);  __builtin_memcpy(dst + size - 4,  src + size - 4,  4);  return; } }
    if constexpr (kMax >= 2)  { if (size > 2)  { __builtin_memcpy(dst, src, 2);  __builtin_memcpy(dst + size - 2,  src + size - 2,  2);  return; } }
    __builtin_memcpy(dst, src, 1);  // size == 1 remainder
  }

  /**
   * @brief Convert value to ProviderEndianness, byteswapping if necessary.
   *
   * @param[in] val Value to convert.
   * @tparam T Type of val.
   *
   * @return endian::from_native<ProviderEndianness, T>(val)
   */
  template <typename T>
  static __host__ __device__ __forceinline__ constexpr T to_provider_endianness(T val) {
    return endian::from_native<ProviderEndianness, T>(val);
  }
/**@}*/
};



template <typename Provider>
__device__ __forceinline__ std::tuple<uintptr_t, uint32_t>
QueuePairDevice<Provider>::get_laddr_info(const void *addr, bool inlined) const {
  uintptr_t laddr = reinterpret_cast<uintptr_t>(addr);
  /* Check for inlined RDMA_WRITE */
  if (inlined) {
    return {laddr, 0};
  }

  /* Check symmetric heap */
  if (is_ptr_in_range(heap.local_base, heap.length, laddr)) [[likely]] {
    return {laddr, heap.lkey};
  }

  /* Check user-registered local buffers */
  for (size_t i = 0; i < num_user_buffers; i++) {
    if (is_ptr_in_range(local_buffers[i].addr, local_buffers[i].length, laddr)) {
      return {laddr, local_buffers[i].lkey};
    }
  }

  /* Check user-registered symmetric buffers */
  if (symm_count) {
    for (int i = 0; i < *symm_count; i++) {
      if (is_ptr_in_range(symm_buffers[i].local_base, symm_buffers[i].length, laddr)) {
        return {laddr, symm_buffers[i].lkey};
      }
    }
  }

  LOGD_ERROR_ABORT("Valid <laddr, lkey> for address %p not found", addr);
  return {0, std::numeric_limits<uint32_t>::max()};
}

template <typename Provider>
__device__ __forceinline__
std::tuple<uintptr_t, uint32_t> QueuePairDevice<Provider>::get_raddr_info(const void *addr) const {
  uintptr_t laddr = reinterpret_cast<uintptr_t>(addr);
  /* Check symmetric heap */
  if (is_ptr_in_range(heap.local_base, heap.length, laddr)) [[likely]] {
    uintptr_t raddr = heap.remote_base + (laddr - heap.local_base);
    return {raddr, heap.rkey};
  }

  /* Check user-registered symmetric buffers */
  if (symm_count) {
    for (int i = 0; i < *symm_count; i++) {
      if (is_ptr_in_range(symm_buffers[i].local_base, symm_buffers[i].length, laddr)) {
        uintptr_t raddr = symm_buffers[i].remote_base + (laddr - symm_buffers[i].local_base);
        return {raddr, symm_buffers[i].rkey};
      }
    }
  }

  LOGD_ERROR_ABORT("Valid <raddr, rkey> for address %p not found", addr);
  return {0, std::numeric_limits<uint32_t>::max()};
}

}  // namespace rocshmem

#endif  // LIBRARY_SRC_GDA_QUEUE_PAIR_DEVICE_HPP_
