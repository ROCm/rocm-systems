* With the device linker on, device code is no longer compiled with
  `-gline-tables-only`, so `librccl.so` is smaller (gfx90a: 21.4 MB to 16.3 MB).
  Device code has no source-line information except in Debug builds, which still
  get full debug information.
