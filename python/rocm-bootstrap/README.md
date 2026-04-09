# rocm-bootstrap

GPU detection and target groupings for the ROCm Python ecosystem.

`rocm-bootstrap` provides pure-Python AMD GPU detection (no ROCm installation
required), a canonical 3-level GFX target hierarchy, and a WheelNext variant
plugin so that `uv`/`pip` can automatically select the right device-specific
wheel.

## Installation

```bash
pip install rocm-bootstrap
```

## CLI

The `rocm-bootstrap-detect` command detects AMD GPUs on the current system.

```bash
# Unique target names, one per line (default)
rocm-bootstrap-detect --unique
rocm-bootstrap-detect -u

# Bundle hierarchy: target sub-family family
rocm-bootstrap-detect --hierarchy

# Human-readable with generic ISA and hierarchy details
rocm-bootstrap-detect --verbose
rocm-bootstrap-detect -v
```

Example outputs:

```
$ rocm-bootstrap-detect --unique
gfx1201
gfx1100

$ rocm-bootstrap-detect --hierarchy
gfx1201 gfx12_0 gfx12
gfx1100 gfx11_0 gfx11

$ rocm-bootstrap-detect --verbose
Detected 2 AMD GPU(s):

  Node 1: gfx1201  major=12 minor=0 stepping=1  PCI=0x7550
    sub-family: gfx12_0 (GFX12.0 (RDNA 4))  generic: gfx12-generic
    family: gfx12 (GFX12)

  Node 2: gfx1100  major=11 minor=0 stepping=0  PCI=0x7448
    sub-family: gfx11_0 (GFX11.0 (RDNA 3))
    family: gfx11 (GFX11 (RDNA 3))  generic: gfx11-generic
```

### Environment variables

- `ROCM_BOOTSTRAP_FORCE_GFX_ARCH` - Comma-separated target list to use
  instead of detection (e.g., `gfx942,gfx1100`).
- `ROCM_BOOTSTRAP_DISABLE_DETECTION` - Set to `1` to disable detection
  entirely.

## Python API

### GPU detection

```python
from rocm_bootstrap import detect_gpus, detect_gfx_targets

# All detected GPUs (may include duplicates for multi-GPU)
gpus = detect_gpus()
for gpu in gpus:
    print(gpu.target.name, gpu.node_id, gpu.pci_id)

# Deduplicated targets
targets = detect_gfx_targets()
# [GfxTarget(name='gfx1201', ...), GfxTarget(name='gfx1100', ...)]
```

### Target hierarchy

Every GFX target maps to a 3-level packaging chain:

```python
from rocm_bootstrap import packaging_chain, lookup_target

target_b, sf_b, fam_b = packaging_chain("gfx1151")
# target_b.key  = "gfx1151"   (PackagingLevel.TARGET)
# sf_b.key      = "gfx11_5"   (PackagingLevel.SUB_FAMILY)
# fam_b.key     = "gfx11"     (PackagingLevel.FAMILY)
```

Look up individual targets and bundles:

```python
from rocm_bootstrap import lookup_target, lookup_bundle

t = lookup_target("gfx942")
# GfxTarget(name='gfx942', major=9, minor=4, stepping=2, xnack=SUPPORTED)

b = lookup_bundle("gfx11_5")
# TargetBundle(key='gfx11_5', level=SUB_FAMILY, ...)
# b.members -> (gfx1150, gfx1151, gfx1152, gfx1153)
# b.llvm_generic -> None  (gfx11-generic is at family level)
```

Enumerate all bundles:

```python
from rocm_bootstrap import all_bundles, PackagingLevel

families = all_bundles(PackagingLevel.FAMILY)
# 4 bundles: gfx9, gfx10, gfx11, gfx12

sub_families = all_bundles(PackagingLevel.SUB_FAMILY)
# 9 bundles: gfx9_0, gfx9_4, gfx9_5, gfx10_1, ...
```

### Package naming

```python
from rocm_bootstrap import bundle_names, device_dist_name, lookup_bundle

b = lookup_bundle("gfx11_5")
names = bundle_names(b)
# names.dist_name   = "gfx11-5"   (pip/wheel safe)
# names.module_name = "gfx11_5"   (Python identifier)

device_dist_name("rocm-sdk-device", b)
# "rocm-sdk-device-gfx11-5"
```

### Sysfs decoding

```python
from rocm_bootstrap import parse_gfx_target_version

t = parse_gfx_target_version(120001)
# GfxTarget(name='gfx1201', ...)
```

## Testing

Tests ship with the package and can run on real hardware or in CI with
faked sysfs content:

```bash
pip install rocm-bootstrap[dev]
pytest --pyargs rocm_bootstrap.tests
```
