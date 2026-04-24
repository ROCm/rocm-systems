---
description: "Use when: working with generator.py, update_wrapper.sh, amdsmi_wrapper.py, code generation, wrapper regeneration."
---
# Tools & Generators

| Tool | Purpose |
|------|---------|
| `tools/generator.py` | Parses `amdsmi.h`, emits ctypes wrapper code |
| `tools/update_wrapper.sh` | Regenerates `py-interface/amdsmi_wrapper.py` (uses Docker + `generator.py`) |
| `tools/update_rust_wrapper.sh` | Regenerates Rust bindings |
| `tools/run-clang-tidy.sh` | Runs clang-tidy on C++ sources |

**Never manually edit `py-interface/amdsmi_wrapper.py`** — always regenerate via `tools/update_wrapper.sh` or `cmake -DBUILD_WRAPPER=ON`.
