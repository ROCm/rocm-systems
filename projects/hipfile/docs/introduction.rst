Introduction
============
In recent years, the demand for high-performance data movement between
storage and GPU memory has grown rapidly, driven by the increasing
scale of AI training, scientific simulations, and data analytics
workloads. In response to this evolving need within heterogeneous
computing environments, AMD introduces hipFile, a cuFile-equivalent
API framework designed to enable direct data paths between NVMe
devices and GPU memory, significantly reducing CPU overhead and
improving IO throughput.

hipFile provides developers with an interface for performing
high-performance IO operations between storage devices and AMD
GPUs. By bypassing traditional CPU memory staging buffers,
hipFile allows applications to achieve lower latency and higher
bandwidth when transferring large datasets into GPU memory. This
direct data path integration complements the broader ROCm stack
- seamlessly interoperating with HIP kernels, HIP streams, and
RDMA-enabled storage systems - to support end-to-end acceleration
for IO-bound workflows.
