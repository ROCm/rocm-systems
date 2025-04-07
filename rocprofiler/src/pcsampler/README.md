# ROCProfiler PC sampling

## Known issues

* Abnormal process termination with running GPU kernels while PC sampling is
  active may leak resources, eventually necessitating a GPU reset or system
  reboot.  This is a consequence of the current user-space PC sampling
  implementation, which must coexist with AMD GPU driver activity in the Linux
  kernel.

* The `ioctl` used to synchronize with kfd (part of the AMD GPU driver) may pose
  a bottleneck on systems with multiple HSA GPU agents.
