# ROCm OpenSHMEM (rocSHMEM)

The ROCm OpenSHMEM (rocSHMEM) runtime is part of an AMD and AMD Research
initiative to provide GPU-centric networking through an OpenSHMEM-like interface.
This intra-kernel networking library simplifies application
code complexity and enables more fine-grained communication/computation
overlap than traditional host-driven networking.
rocSHMEM uses a single symmetric heap that is allocated on GPU memories.

There are currently three backends for rocSHMEM;
IPC, Reverse Offload (RO), and GDA.
The backends primarily differ in their implementations of intra-kernel networking.

The IPC backend implements communication primitives using load/store operations issued from the GPU.

The Reverse Offload (RO) backend has the GPU runtime forward rocSHMEM networking operations
to the host-side runtime, which calls into a traditional MPI or OpenSHMEM
implementation. This forwarding of requests is transparent to the
programmer, who only sees the GPU-side interface.

The GPU Direct Async (GDA) backend allows for rocSHMEM to issue communication operations to the NIC directly from the device-side code, without involving a CPU proxy.
within the GPU.
During initialization we prepare network resources for each NIC vendor using the vendor-appropriate
Direct Verbs APIs.
When calling the device-side rocSHMEM API, the device threads are used to construct Work Queue Entries (WQEs) and post the communication to the send queues of the NIC directly.
Completion Queues (CQs) are polled from the device-side code as well.

The RO and GDA backend is provided as-is with limited support from AMD or AMD Research.

## Installation and using rocSHMEM

For information on how to install and use rocSHMEM,
[please see our documentation](https://rocm.docs.amd.com/projects/rocSHMEM/en/latest/).
