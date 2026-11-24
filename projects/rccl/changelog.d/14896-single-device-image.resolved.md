* With the device linker on, `librccl.so` carries one device image, registered once,
  instead of one per source file that mixes host and device code. Under `--rocshmem`,
  rocSHMEM's device state has one copy instead of two. A static `librccl.a` now
  contains RCCL's and rocSHMEM's device code, so programs that link it no longer
  depend on paths in RCCL's build tree.
