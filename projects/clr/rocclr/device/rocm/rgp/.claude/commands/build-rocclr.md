# Build rocclr

Build the rocclr target from the hipamd build tree.

```bash
cd /c/github-emu/hipamd && cmake --build . --config Release -j 6 --target rocclr 2>&1 | tail -30
```

If the build fails due to CMake reconfiguration errors (missing source files), check that all
paths in `ROCclrHSA.cmake` and `rocclr.vcxproj` use `device/rocm/rgp/` for RGP files.
