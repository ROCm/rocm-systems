---
myst:
    html_meta:
        "description": "How to regenerate ISA simulation and DBT source files from AMD Machine-Readable ISA XML using the rocJITsu amdisa Python library."
        "keywords": "rocJITsu, ROCm, ISA, codegen, DBT, amdisa, regenerate, code generation"
---::: note
Note
Regenerating ISA and DBT source files requires the `amdisa` Python
codegen library and AMD Machine-Readable ISA XML specification files.
Documenting the full regeneration workflow requires the contents of
`lib/python/amdisa/__main__.py` (the CLI entry point),
`lib/python/amdisa/codegen/_generator.py` (the generator
implementation), `lib/python/amdisa/codegen/config.py` (codegen
configuration), and `docs/codegen.md` (the existing codegen pipeline
documentation), which are not available in sufficient detail to ground
the specific command-line flags (`--multi`, `--gen-isas`, `--gen-dbt`,
`--isa-output`, `--dbt-output`), XML file locations, or the exact
regeneration commands.

To complete this page, the following files need to be available with
their full contents:

-   `lib/python/amdisa/__main__.py` --- CLI argument definitions and
    entry-point logic
-   `lib/python/amdisa/codegen/_generator.py` --- generator class with
    `--gen-isas` and `--gen-dbt` handling
-   `lib/python/amdisa/codegen/config.py` --- codegen configuration
    including output path defaults
-   `docs/codegen.md` --- existing developer documentation covering the
    regeneration commands and workflow
