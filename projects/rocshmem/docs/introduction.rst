.. meta::
  :description: rocSHMEM intra-kernel networking runtime for AMD GPUs on the ROCm platform.
  :keywords: rocSHMEM, API, ROCm, documentation, HIP, Networking, Communication

.. _rocshmem-introduction:

---------------------------
What is rocSHMEM?
---------------------------

The ROCm OpenSHMEM (rocSHMEM) is an intra-kernel networking library that provides GPU-centric networking through an OpenSHMEM-like interface. It simplifies application code complexity and enables finer communication and computation overlap than traditional host-driven networking. rocSHMEM uses a single symmetric heap allocated on GPU memories.

The rocSHMEM programming model
-------------------------------

Defining how OpenSHMEM applications interact with GPUs remains an
ongoing active discussion within the OpenSHMEM community, and the OpenSHMEM
specification has yet to coalesce on this topic.
rocSHMEM extends beyond the OpenSHMEM specification to add semantics that
support GPU kernel communication while maintaining close resemblance to
the original OpenSHMEM specification semantics. 

Applications using :doc:`HIP <hip:index>` can interface with rocSHMEM.
Using the HIP programming model,
rocSHMEM provides ``__host__`` APIs for host code,
and ``__device__`` APIs for GPU kernels.
Device APIs without special suffixes or infixes , for example, ``_wg`` or ``_wave``, 
must be called by a single thread.
GPU specific ``_wg`` and ``_wave`` APIs are designed to be called by multiple GPU threads
and will block until the calling scope completes.
These APIs can be called in divergent code paths, but this is not recommended.

Wavefront APIs
==============
Wavefront APIs are those with the ``_wave`` suffix.
The parameters in which these routines are called must be
the same for every thread in the wavefront.
The behavior is undefined if any thread calls these routines with different parameters. These APIs will block until the calling wavefront is complete.

Workgroup APIs
==============
The workgroup APIs have the ``_wg`` suffix or ``_wg_`` infix.
The parameters in which these routines are called must be
the same for every thread in the workgroup.
The behavior is undefined if any thread calls these routines with different parameters. These APIs will block until the calling workgroup is complete.
