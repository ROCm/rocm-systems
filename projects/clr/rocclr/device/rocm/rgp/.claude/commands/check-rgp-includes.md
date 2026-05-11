# Check RGP include paths

Verify all #include references to RGP headers use the correct `device/rocm/rgp/` prefix.
Any matches without the `rgp/` prefix indicate a stale path that will cause build failures.

```bash
grep -rn "#include.*\(rocgpuopen\|roctracesession\|rocubertracesvc\|rocdriverutils\|rocurilocator\)" \
  /c/github-emu/rocm-systems/projects/clr \
  /c/github-emu/hipamd/rocclr/rocclr.vcxproj 2>/dev/null | grep -v "rgp/"
```

If any results appear, those files have stale paths — update them to add `rgp/` before the filename.

Also verify CMake has correct paths:
```bash
grep -n "rocgpuopen\|roctracesession\|rocubertracesvc\|rocurilocator\|g_service" \
  /c/github-emu/rocm-systems/projects/clr/rocclr/cmake/ROCclrHSA.cmake
```

All should show `device/rocm/rgp/`.
