.. meta::
  :description: rocSHMEM intra-kernel networking runtime for AMD dGPUs on the ROCm platform.
  :keywords: rocSHMEM, API, ROCm, documentation, HIP, Networking, Communication

.. _rocshmem-api-memory-ordering:

---------------------------
Memory ordering routines
---------------------------

ROCSHMEM_FENCE
--------------

.. cpp:function:: __device__ void rocshmem_fence()
.. cpp:function:: __device__ void rocshmem_fence(int pe)
.. cpp:function:: __device__ void rocshmem_ctx_fence(rocshmem_ctx_t ctx)
.. cpp:function:: __device__ void rocshmem_ctx_fence(rocshmem_ctx_t ctx, int pe)

 :param ctx: Context with which to perform this operation.
 :param pe:  Destination ``pe``.
 :returns:   None.

**Description:**
This routine ensures order between messages in this context to follow OpenSHMEM semantics.

ROCSHMEM_THREADFENCE_SYSTEM
---------------------------

.. cpp:function:: __device__ void rocshmem_threadfence_system()
.. cpp:function:: __device__ void rocshmem_ctx_threadfence_system(rocshmem_ctx_t ctx)

 :param ctx: Context with which to perform this operation.
 :returns:   None.

**Description:**
This routine issues a system-scope memory fence, making the calling thread's
prior memory operations visible to the host and to peer devices. It orders
memory accesses but, unlike ``rocshmem_quiet``, does not wait for outstanding
rocSHMEM operations to complete. Use it when combining rocSHMEM transfers with
direct load/store access to peer memory obtained from ``rocshmem_ptr``.

ROCSHMEM_QUIET
--------------

.. cpp:function:: __device__ void rocshmem_ctx_quiet(rocshmem_ctx_t ctx)
.. cpp:function:: __device__ void rocshmem_quiet()

 :param ctx: Context with which to perform this operation.
 :returns:   None.

**Description:**
This routine completes all previous operations posted to this context.

ROCSHMEM_PE_QUIET
-----------------

.. cpp:function:: __device__ void rocshmem_ctx_pe_quiet(shmem_ctx_t ctx, const int *target_pes, size_t npes)
.. cpp:function:: __device__ void rocshmem_pe_quiet(const int *target_pes, size_t npes)

 :param ctx: Context with which to perform this operation.
 :param target_pes: Address of target PE array where the operations need to be completed
 :param npes: The number of PEs in the target PE array
 :returns:   None.

**Description:**
This routine completes all previous operations posted to this context
for the PEs in the `target_pes` array.
