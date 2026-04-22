.. meta::
  :description: rocDecode memory types
  :keywords: parse video, parse, decode, video decoder, video decoding, rocDecode, AMD, ROCm, memory types

********************************************************************
rocDecode surface data memory locations
********************************************************************

Surface data memory refers to the memory used by rocDecode for decoded frames and processing results. There are three locations where surface data memory can be stored: device memory, host memory, and internal memory.

Device memory refers to GPU memory. It's optimized for operations performed by the GPU, avoiding unnecessary memory transfers between the device and the host. It's used for standalone GPU processing and high-performance computing tasks where multiple operations are performed on the same data.


Host memory refers to CPU memory. It's suitable for when the memory needs to be accessed or manipulated by CPU-side applications or when data needs to be transferred between systems.

Internal memory refers to intermediate GPU memory that is shared between operators. It's optimized for operator chaining within GPU workflows. It keeps data localized on the GPU so it can be accessed by subsequent operations, reducing latency and improving throughput. For example, in image processing pipelines, the results of a resizing operator can directly feed into a filtering operator without needing to copy data to the host between each step. This optimization is especially useful for large datasets and real-time applications.


The ``OutputSurfaceMemoryType_enum`` enum type defines ``OUT_SURFACE_MEM_DEV_COPIED``, ``OUT_SURFACE_MEM_HOST_COPIED``, and ``OUT_SURFACE_MEM_DEV_INTERNAL``, for the three different types of memory locations.  ``OUT_SURFACE_MEM_DEV_COPIED`` indicates device, or GPU, memory. ``OUT_SURFACE_MEM_HOST_COPIED`` indicates host, or CPU, memory. And ``OUT_SURFACE_MEM_DEV_INTERNAL`` indicates intermediate GPU memory.

``OUT_SURFACE_MEM_DEV_COPIED`` is not supported when the FFmpeg decoder is used.

A fourth enum, ``OUT_SURFACE_MEM_NOT_MAPPED``, is used only for performance purposes. The decoded frames are not available when this memory type is used.

