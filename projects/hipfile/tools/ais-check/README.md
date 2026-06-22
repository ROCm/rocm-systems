# ais-check

`ais-check` checks whether a system has the components needed to use AMD
Infinity Storage (AIS), the GPU-direct storage feature that hipFile relies on.
Run it before using hipFile to confirm that AIS is available on the machine.

## Usage

```
./ais-check [-q | --quiet] [-v | --verbose]
```

The program takes no required arguments. By default it prints the support
status of each component. Pass `-v`/`--verbose` to additionally list the
discovered HIP libraries (this list varies by system and some entries may refer
to the same library). Pass `-q`/`--quiet` to suppress all regular output and
rely on the exit code alone (useful in scripts).

## What it checks

AIS requires three independent components to be present and AIS-capable. The
tool checks all three:

1. **Kernel P2PDMA support** — the running kernel must be built with
   `CONFIG_PCI_P2PDMA=y`. This is read from the kernel config (`/boot/config-*`,
   the module build config, or `/proc/config.gz`).
2. **HIP runtime** — at least one installed HIP runtime library
   (`libamdhip64.so`) must export the AIS entry points (`hipAmdFileRead` and
   `hipAmdFileWrite`). The tool searches the usual ROCm locations and the
   dynamic linker cache.
3. **amdgpu driver** — the loaded kernel driver must provide AIS file I/O,
   detected by looking for the `kfd_ais_rw_file` symbol in `/proc/kallsyms`.

## Exit codes

| Code | Meaning |
|------|---------|
| `0`  | All three components are present and support AIS. |
| non-zero | One or more components are missing or do not support AIS. |

## Example output

```
Linux myhost 6.6.0 #1 SMP x86_64

AIS support in:
	Kernel P2PDMA support   : True
	HIP runtime             : True
	amdgpu                  : True
```

With `-v`/`--verbose`, the discovered HIP libraries are also listed:

```
Linux myhost 6.6.0 #1 SMP x86_64

Found these HIP libraries (some may refer to the same library):
	/opt/rocm/lib/libamdhip64.so.6 (AIS supported)

AIS support in:
	Kernel P2PDMA support   : True
	HIP runtime             : True
	amdgpu                  : True
```

## Notes

- Diagnostic messages (for example, missing kernel config files or an
  unreadable `/proc/kallsyms`) are written to standard error and do not affect
  the exit code on their own.
- Reading `/proc/kallsyms` may require elevated privileges; if it cannot be read
  the amdgpu check reports no support.
- Only the standard library is used, so no extra Python packages are required.
  Python 3 is needed to run the tool.
