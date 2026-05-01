# Copilot Instructions — AMD SMI

See [CLAUDE.md](../CLAUDE.md) for project overview, critical rules, and behavioral guidelines.
See [.github/CONTRIBUTING.md](CONTRIBUTING.md) for PR workflow and coding standards.

## Quick Reference

- **Never edit `py-interface/amdsmi_wrapper.py` manually** — regenerate with `tools/update_wrapper.sh`
- **PRs target `develop`** branch
- **Pre-commit must pass**: `pre-commit run --all-files`
- **Version** defined in `include/amd_smi/amdsmi.h`
- **Excluded from linting**: `docs/`, `build/`, `esmi_ib_library/`, `third_party/`
