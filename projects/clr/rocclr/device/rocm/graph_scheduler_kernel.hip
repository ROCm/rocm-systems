#include <stdint.h>

#define QUEUE_BASE_ADDR_OFF    8
#define QUEUE_SIZE_OFF         24
#define QUEUE_WRITE_ID_OFF     56
#define QUEUE_READ_ID_OFF      128

// ============================================================================
// Device-side data structures (layout must match host-side filling)
// ============================================================================

struct BlockDescriptor {
    uint32_t packet_offset;     // Byte offset from cmd_buffer_base
    uint32_t packet_count;      // Number of 64-byte AQL packets in this block
};

enum CondOp : uint32_t {
    COND_EQ = 0, COND_NE = 1,
    COND_LT = 2, COND_LE = 3,
    COND_GT = 4, COND_GE = 5
};

struct ExecutionState {
    uint64_t cmd_buffer_base;        // Pointer to packet data area
    uint64_t block_table_ptr;        // Pointer to BlockDescriptor[]
    uint32_t block_count;
    uint32_t current_block;
    uint64_t queue_ptr;              // Queue A (control) - amd_queue_t*
    uint64_t completion_signal_ptr;  // &amd_signal_t->value
    uint64_t kernarg_pool_ptr;
    uint64_t doorbell_ptr;           // Queue A hardware_doorbell_ptr
    uint64_t work_queue_ptr;         // Queue B (work) - where body packets go
    uint64_t work_doorbell_ptr;      // Queue B hardware_doorbell_ptr
    uint64_t cond_init_ptr;          // Cond variable to init (0 = skip)
    uint64_t cond_init_value;        // Value to write before first block
    // Pre-resolved queue metadata for graphCondBranchWhile (avoids per-iter lookups)
    uint64_t cached_queue_base;      // Queue ring buffer base address
    uint32_t cached_queue_size;      // Queue ring buffer size (slots)
    uint32_t _reserved0;
};

// ============================================================================
// Shared device function: copy one block's packets to the HW queue
// ============================================================================

static inline __device__ void issue_block_to_queue(
    const uint8_t* src, uint32_t pkt_count, uint8_t* qptr,
    volatile uint64_t* doorbell) {
  if (pkt_count == 0) return;

  uint8_t*           base   = *(uint8_t**)(qptr + QUEUE_BASE_ADDR_OFF);
  uint32_t           q_size = *(uint32_t*)(qptr + QUEUE_SIZE_OFF);
  uint32_t           q_mask = q_size - 1;
  volatile uint64_t* wr_id  = (volatile uint64_t*)(qptr + QUEUE_WRITE_ID_OFF);
  volatile uint64_t* rd_id  = (volatile uint64_t*)(qptr + QUEUE_READ_ID_OFF);

  uint64_t wr = __atomic_fetch_add(wr_id, (uint64_t)pkt_count, __ATOMIC_RELAXED);

  while (wr + pkt_count - *rd_id > q_size) {}

  uint32_t saved_header = *(const volatile uint32_t*)src;
  *(volatile uint32_t*)src = 0;

  uint32_t start_slot        = (uint32_t)(wr & q_mask);
  uint32_t slots_before_wrap = q_size - start_slot;
  uint32_t first_count  = (pkt_count <= slots_before_wrap) ? pkt_count : slots_before_wrap;
  uint32_t second_count = pkt_count - first_count;

  __builtin_memcpy(base + start_slot * 64, src, first_count * 64);
  if (second_count > 0) {
    __builtin_memcpy(base, src + first_count * 64, second_count * 64);
  }

  *(volatile uint32_t*)src = saved_header;

  __builtin_amdgcn_fence(__ATOMIC_RELEASE, "agent");

  __atomic_store_n((uint32_t*)(base + start_slot * 64),
                   saved_header, __ATOMIC_RELEASE);

  if (doorbell) {
    *doorbell = wr + pkt_count - 1;
  }
}

// ============================================================================
// Flat scheduler (backward compatible, single flat buffer)
// ============================================================================

extern "C" __attribute__((amdgpu_flat_work_group_size(1, 1)))
__global__ void __amd_rocclr_graphSchedulerHIP(
    const uint8_t* __restrict__ cmd_buffer,
    uint32_t total_packet_count,
    uint8_t* __restrict__ queue_ptr) {

  uint8_t* qptr = queue_ptr ? queue_ptr
                             : (uint8_t*)__builtin_amdgcn_queue_ptr();

  uint8_t* base = *(uint8_t**)(qptr + QUEUE_BASE_ADDR_OFF);
  uint32_t q_size = *(uint32_t*)(qptr + QUEUE_SIZE_OFF);
  uint32_t q_mask = q_size - 1;
  volatile uint64_t* wr_id = (volatile uint64_t*)(qptr + QUEUE_WRITE_ID_OFF);
  volatile uint64_t* rd_id = (volatile uint64_t*)(qptr + QUEUE_READ_ID_OFF);

  uint64_t wr = __atomic_fetch_add(wr_id, (uint64_t)total_packet_count,
                                    __ATOMIC_RELAXED);

  while (wr + total_packet_count - *rd_id > q_size) {}

  uint32_t saved_header = *(volatile uint32_t*)cmd_buffer;
  *(volatile uint32_t*)cmd_buffer = 0;

  uint32_t start_slot = (uint32_t)(wr & q_mask);
  uint32_t slots_before_wrap = q_size - start_slot;
  uint32_t first_count = (total_packet_count <= slots_before_wrap)
                          ? total_packet_count : slots_before_wrap;
  uint32_t second_count = total_packet_count - first_count;

  __builtin_memcpy(base + start_slot * 64, cmd_buffer, first_count * 64);

  if (second_count > 0) {
    __builtin_memcpy(base, cmd_buffer + first_count * 64, second_count * 64);
  }

  __builtin_amdgcn_fence(__ATOMIC_RELEASE, "agent");

  __atomic_store_n((uint32_t*)(base + start_slot * 64),
                   saved_header, __ATOMIC_RELEASE);

  *(volatile uint32_t*)cmd_buffer = saved_header;
}

// ============================================================================
// Block-based scheduler: initial entry point dispatched from host.
// ============================================================================

extern "C" __attribute__((amdgpu_flat_work_group_size(1, 1)))
__global__ void __amd_rocclr_graphBlockIssue(ExecutionState* state) {
  // Initialize conditional variable on device before first block dispatch.
  // Keeps cond_ptr in coarse-grained VRAM -- no host-side PCIe write needed.
  if (state->cond_init_ptr) {
    *(volatile uint64_t*)state->cond_init_ptr = state->cond_init_value;
  }

  uint32_t blk_idx = state->current_block;
  const BlockDescriptor* table = (const BlockDescriptor*)state->block_table_ptr;
  const uint8_t* src = (const uint8_t*)state->cmd_buffer_base
                       + table[blk_idx].packet_offset;
  uint32_t pkt_count = table[blk_idx].packet_count;

  issue_block_to_queue(src, pkt_count, (uint8_t*)state->queue_ptr,
                       (volatile uint64_t*)state->doorbell_ptr);
}

// ============================================================================
// CFG: Unconditional branch — set current_block and issue next block
// ============================================================================

extern "C" __attribute__((amdgpu_flat_work_group_size(1, 1)))
__global__ void __amd_rocclr_graphBranch(
    ExecutionState* state,
    uint32_t target_block) {
  state->current_block = target_block;

  const BlockDescriptor* table = (const BlockDescriptor*)state->block_table_ptr;
  const uint8_t* src = (const uint8_t*)state->cmd_buffer_base
                       + table[target_block].packet_offset;
  uint32_t pkt_count = table[target_block].packet_count;

  issue_block_to_queue(src, pkt_count, (uint8_t*)state->queue_ptr,
                       (volatile uint64_t*)state->doorbell_ptr);
}

// ============================================================================
// CFG: Conditional branch — evaluate *ref_ptr <cond> value, pick block
// ============================================================================

static inline __device__ bool evaluate_cond(
    uint64_t lhs, CondOp cond, uint64_t rhs) {
  switch (cond) {
    case COND_EQ: return lhs == rhs;
    case COND_NE: return lhs != rhs;
    case COND_LT: return (int64_t)lhs <  (int64_t)rhs;
    case COND_LE: return (int64_t)lhs <= (int64_t)rhs;
    case COND_GT: return (int64_t)lhs >  (int64_t)rhs;
    case COND_GE: return (int64_t)lhs >= (int64_t)rhs;
    default:      return false;
  }
}

// kInlineReturn: when a target equals this sentinel, graphCondBranch signals
// completion inline instead of issuing another block. Eliminates the need
// for a separate RETURN block in simple WHILE/IF graphs.
#define kInlineReturn 0xFFFFFFFFu

extern "C" __attribute__((amdgpu_flat_work_group_size(1, 1)))
__global__ void __amd_rocclr_graphCondBranch(
    ExecutionState* state,
    const uint64_t* __restrict__ ref_ptr,
    uint32_t cond,
    uint64_t value,
    uint32_t true_block,
    uint32_t false_block) {
  uint64_t ref_val = *ref_ptr;
  uint32_t next = evaluate_cond(ref_val, (CondOp)cond, value)
                      ? true_block : false_block;

  if (next == kInlineReturn) {
    volatile int64_t* sig = (volatile int64_t*)state->completion_signal_ptr;
    if (sig) {
      __builtin_amdgcn_fence(__ATOMIC_RELEASE, "");
      __atomic_store_n(sig, 0, __ATOMIC_RELEASE);
    }
    return;
  }

  state->current_block = next;

  const BlockDescriptor* table = (const BlockDescriptor*)state->block_table_ptr;
  const uint8_t* src = (const uint8_t*)state->cmd_buffer_base
                       + table[next].packet_offset;
  uint32_t pkt_count = table[next].packet_count;

  issue_block_to_queue(src, pkt_count, (uint8_t*)state->queue_ptr,
                       (volatile uint64_t*)state->doorbell_ptr);
}

// ============================================================================
// CFG: Fast-path conditional branch for WHILE self-loops.
// All invariant lookups (block table, queue struct, source header) are
// pre-resolved into kernel arguments or ExecutionState cached fields,
// eliminating ~9 memory loads per iteration.
// ============================================================================

extern "C" __attribute__((amdgpu_flat_work_group_size(1, 1)))
__global__ void __amd_rocclr_graphCondBranchWhile(
    ExecutionState* state,
    const uint64_t* __restrict__ ref_ptr,
    uint32_t cond,
    uint64_t value,
    const uint8_t* __restrict__ body_src,
    uint32_t body_pkt_count) {

  uint64_t ref_val = *ref_ptr;
  if (!evaluate_cond(ref_val, (CondOp)cond, value)) {
    volatile int64_t* sig = (volatile int64_t*)state->completion_signal_ptr;
    if (sig) {
      __builtin_amdgcn_fence(__ATOMIC_RELEASE, "");
      __atomic_store_n(sig, 0, __ATOMIC_RELEASE);
    }
    return;
  }

  // Queue metadata from pre-resolved cached fields (avoids queue struct indirection).
  uint8_t*           base   = (uint8_t*)state->cached_queue_base;
  uint32_t           q_size = state->cached_queue_size;
  uint32_t           q_mask = q_size - 1;
  uint8_t*           qptr   = (uint8_t*)state->queue_ptr;
  volatile uint64_t* wr_id  = (volatile uint64_t*)(qptr + QUEUE_WRITE_ID_OFF);
  volatile uint64_t* rd_id  = (volatile uint64_t*)(qptr + QUEUE_READ_ID_OFF);

  uint64_t wr = __atomic_fetch_add(wr_id, (uint64_t)body_pkt_count,
                                   __ATOMIC_RELAXED);
  while (wr + body_pkt_count - *rd_id > q_size) {}

  uint32_t saved_header = *(const volatile uint32_t*)body_src;
  *(volatile uint32_t*)body_src = 0;

  uint32_t start_slot        = (uint32_t)(wr & q_mask);
  uint32_t slots_before_wrap = q_size - start_slot;
  uint32_t first_count  = (body_pkt_count <= slots_before_wrap)
                            ? body_pkt_count : slots_before_wrap;
  uint32_t second_count = body_pkt_count - first_count;

  __builtin_memcpy(base + start_slot * 64, body_src, first_count * 64);
  if (second_count > 0) {
    __builtin_memcpy(base, body_src + first_count * 64, second_count * 64);
  }

  *(volatile uint32_t*)body_src = saved_header;

  __builtin_amdgcn_fence(__ATOMIC_RELEASE, "agent");

  __atomic_store_n((uint32_t*)(base + start_slot * 64),
                   saved_header, __ATOMIC_RELEASE);

  *((volatile uint64_t*)state->doorbell_ptr) = wr + body_pkt_count - 1;
}

// ============================================================================
// CFG: Return — signal completion, end the GPU-side chain
// ============================================================================

extern "C" __attribute__((amdgpu_flat_work_group_size(1, 1)))
__global__ void __amd_rocclr_graphReturn(ExecutionState* state) {
  volatile int64_t* sig = (volatile int64_t*)state->completion_signal_ptr;
  if (sig) {
    __builtin_amdgcn_fence(__ATOMIC_RELEASE, "");
    __atomic_store_n(sig, 0, __ATOMIC_RELEASE);
  }
}

// ============================================================================
// Fused WHILE loop: persistent kernel that issues body packets, waits for
// completion by polling read_id, checks the condition, and loops.
// Eliminates 2 AQL dispatch round-trips (graphCondBranch + graphBranch) per
// iteration by keeping the control flow entirely within this single kernel.
// ============================================================================

// Sentinel value: set cond_ptr to this before issuing body.
// Body kernel overwrites with 0 or 1, so seeing != SENTINEL means body is done.
#define COND_SENTINEL 0xFFFFFFFFFFFFFFFFULL

extern "C" __attribute__((amdgpu_flat_work_group_size(1, 1)))
__global__ void __amd_rocclr_graphWhileLoop(
    ExecutionState* state,
    volatile uint64_t* __restrict__ cond_ptr,
    uint32_t body_block_idx) {

  const BlockDescriptor* table = (const BlockDescriptor*)state->block_table_ptr;
  const uint8_t* cmd_base = (const uint8_t*)state->cmd_buffer_base;

  // Body packets go to Queue B (work queue), NOT the control queue.
  uint8_t* wq = (uint8_t*)state->work_queue_ptr;
  volatile uint64_t* w_doorbell = (volatile uint64_t*)state->work_doorbell_ptr;

  uint8_t*           q_base = *(uint8_t**)(wq + QUEUE_BASE_ADDR_OFF);
  uint32_t           q_size = *(uint32_t*)(wq + QUEUE_SIZE_OFF);
  uint32_t           q_mask = q_size - 1;
  volatile uint64_t* wr_id  = (volatile uint64_t*)(wq + QUEUE_WRITE_ID_OFF);
  volatile uint64_t* rd_id  = (volatile uint64_t*)(wq + QUEUE_READ_ID_OFF);

  const uint8_t* body_src = cmd_base + table[body_block_idx].packet_offset;
  uint32_t body_pkt_count = table[body_block_idx].packet_count;

  while (*cond_ptr != 0) {
    // Set sentinel before issuing body. The body kernel will overwrite
    // cond_ptr with 0 or 1; polling for != SENTINEL is our completion check.
    *cond_ptr = COND_SENTINEL;
    __builtin_amdgcn_fence(__ATOMIC_RELEASE, "agent");

    uint64_t wr = __atomic_fetch_add(wr_id, (uint64_t)body_pkt_count, __ATOMIC_RELAXED);

    while (wr + body_pkt_count - *rd_id > q_size) {
      __builtin_amdgcn_s_sleep(1);
    }

    uint32_t saved_header = *(const volatile uint32_t*)body_src;
    *(volatile uint32_t*)body_src = 0;

    uint32_t start_slot        = (uint32_t)(wr & q_mask);
    uint32_t slots_before_wrap = q_size - start_slot;
    uint32_t first_count  = (body_pkt_count <= slots_before_wrap)
                              ? body_pkt_count : slots_before_wrap;
    uint32_t second_count = body_pkt_count - first_count;

    __builtin_memcpy(q_base + start_slot * 64, body_src, first_count * 64);
    if (second_count > 0) {
      __builtin_memcpy(q_base, body_src + first_count * 64, second_count * 64);
    }

    *(volatile uint32_t*)body_src = saved_header;

    __builtin_amdgcn_fence(__ATOMIC_RELEASE, "agent");
    __atomic_store_n((uint32_t*)(q_base + start_slot * 64),
                     saved_header, __ATOMIC_RELEASE);

    if (w_doorbell) {
      *w_doorbell = wr + body_pkt_count - 1;
    }

    // Tight-poll cond_ptr. Body kernel writes 0 or 1 as its last op.
    // cond_ptr is in fine-grained device memory (VRAM) for fast GPU reads.
    while (*cond_ptr == COND_SENTINEL) {
      __builtin_amdgcn_s_sleep(1);
    }

    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent");
  }

  // Loop exited: signal completion
  volatile int64_t* sig = (volatile int64_t*)state->completion_signal_ptr;
  if (sig) {
    __builtin_amdgcn_fence(__ATOMIC_RELEASE, "");
    __atomic_store_n(sig, 0, __ATOMIC_RELEASE);
  }
}
