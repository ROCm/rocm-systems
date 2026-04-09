IO Path
=======

Currently, there are two IO paths in hipFile: the fastpath, where data is
transferred directly between the GPU and the storage device, and the fallback path, where
data is staged through a CPU bounce buffer. hipFile selects the path best suited
for each IO request based on the characteristics of the operation and the
capabilities of the underlying system.

Fastpath
--------

If an IO request meets the criteria for the fastpath, data will be transferred
directly between the GPU and the storage device. The fastpath is the most
efficient path for IO requests that can use it, as it eliminates the overhead of
staging data through CPU memory. However, not all IO requests will be eligible
for the fastpath.

The following criteria must be met for the fastpath to be used for an IO request:
* The storage device backing the file must be able to perform peer-to-peer DMA directly to the GPU
* the filesystem backing the file must support direct IO
* the memory type of the buffer must be ``hipMemoryTypeDevice``
* The IO operation's file offset, buffer offset, and size must satisfy direct IO
  alignment requirements (see statx(2)).

When these criteria are met, the IO request uses hipFile's fastpath. The IO
request passes through HIP Runtime, ROCR Runtime, HSA, and THUNK layers before
being handled by the amdgpu kernel driver. The amdgpu driver validates the
request, pins the GPU buffer, and then submits the IO request to the virtual
filesystem layer.

Fallback Path
-------------

If an IO request does not meet the criteria for the fastpath, hipFile uses the
fallback path. The fallback path stages IO requests through a CPU bounce buffer.
The fallback path is less efficient than the fastpath due to the additional data
copy between GPU and CPU memory, but it is universally supported for all device
memory IO requests regardless of file, buffer, or alignment characteristics.

In the case of a read operation, hipFile first reads data from the file into a
CPU buffer and then copies the CPU buffer into the GPU buffer. For a write
operation, hipFile first copies data from the GPU buffer into a CPU buffer and
then writes the CPU buffer to the file.
