# rocencCaps

Prints every rocEncode capability for every codec on every visible device.

```bash
mkdir build && cd build
cmake ../
make -j$(nproc)
./rocenccaps            # every device
./rocenccaps -d 0       # one device
```

A device with no encoder -- Instinct MI300-class parts and Navi 24
(VCN 3.0.33) -- reports zero codecs and exits 0. That is the expected
result on decode-only silicon, not a failure.

Set `ROCENC_LOG_LEVEL=3` (or 4) for the VA-API probe trace, including the
DRM node that was selected and the Mesa vendor string.
