# Build and install amdhip64 + rocclr

Full install build (amdhip64 + rocclr + hip runtime).

```bash
cd /c/github-emu/hipamd && cmake --build . --config Release -j 6 --target install 2>&1 | tail -50
```

Installs to `C:/github-emu/install/`.
