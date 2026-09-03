# Sample Videos

These files are synthetic test videos used as a fallback when the rocDecode
sample dataset (`share/rocdecode/video/`) is absent from the ROCm installation
(e.g. non-test/nightly builds).

## Files

| File | Codec | Resolution | FPS | Duration |
|------|-------|------------|-----|----------|
| `test_AV1_1.mp4` | AV1 | 320x240 | 30 | 10 s |
| `test_AV1_2.mp4` | AV1 | 320x240 | 30 | 10 s |

Two files are provided so that the videodecode example creates two
`rocDecCreateVideoParser` instances, matching the validation expectations.

## Generation

Generated with ffmpeg using a synthetic solid-color source (no third-party
content). The files are freely usable without license restrictions.

```bash
ffmpeg -y -f lavfi -i 'color=c=blue:size=320x240:duration=10:rate=30' \
  -c:v libaom-av1 -cpu-used 8 -b:v 0 -crf 50 test_AV1_1.mp4
cp test_AV1_1.mp4 test_AV1_2.mp4
```
