# AMD SMI OpenSpec

This directory is a written baseline of how AMD SMI **already behaves**. It is
not a roadmap and not a design document. Each capability under `specs/` states
a contract that today's code satisfies, so that a proposed change can be
expressed as a delta against something written down rather than against
whatever the reader happens to remember.

Managed with [OpenSpec](https://github.com/Fission-AI/OpenSpec). The CLI is
optional for reading; the files are plain markdown.

```sh
cd projects/amdsmi
openspec list                    # capabilities and active changes
openspec validate --all --strict # structure gate, must be clean
```

## What is in here

Eight capabilities, 64 requirements, 173 scenarios — all delivery: how AMD
SMI is configured, built and reaches a user. Four channels ship the same tree
in four different shapes, and most packaging bugs come from conflating them.

| Capability | Owns |
| ---------- | ---- |
| `amdsmi-install-layout` | The prefix-relative `bin` → `libexec` → `share` → `lib` tree every channel inherits |
| `amdsmi-python-loader` | How a Python process finds the module and the native library, in every channel |
| `amdsmi-python-wheel` | The standalone `pip install amdsmi` wheel and its SONAME-isolated library |
| `amdsmi-python-system-package` | The `amd-smi-lib` deb/rpm: site-packages detection, dependencies, scriptlets |
| `amdsmi-therock-subproject` | AMD SMI built as a CMake subproject inside TheRock |
| `amdsmi-therock-artifact` | The `core-amdsmi` artifact and what each component captures |
| `amdsmi-rocm-python-distribution` | The `rocm-sdk` pip channel |
| `amdsmi-rocm-os-packages` | The `amdrocm-amdsmi` native packages and tarballs |

Capabilities cross-reference each other by bare id in brackets, for example
[amdsmi-python-loader]. One fact lives in exactly one capability; the others
point at it.

## Reading it in a browser

Flat markdown hides the shape of the set. `tools/openspec_viewer.py` renders it
as one self-contained HTML page — capability index, requirement and scenario
hierarchy, cross-reference diagram, instant search, deep links:

```sh
cd projects/amdsmi
python3 tools/openspec_viewer.py              # writes to $TMPDIR
python3 tools/openspec_viewer.py --check      # CI structure gate
```

Standard library only, no build step, and the page works offline from a
`file://` URL. Output never lands inside the repository.

## Proposing a change

Behavior changes go through `openspec/changes/<change-id>/` with a proposal,
the delta specs, and tasks; the delta is archived into `specs/` once it lands.
See the OpenSpec documentation for the workflow.

Two rules specific to this project:

1. **A requirement must be traceable to code.** If you cannot point at the file
   that makes a statement true, it does not belong in a spec.
2. **A scenario must encode a real failure mode.** A scenario that only
   restates its requirement is noise. The ones here exist because someone hit,
   or could hit, that exact case: a silent misplacement, an upgrade path, a
   coexistence conflict, a platform difference.
