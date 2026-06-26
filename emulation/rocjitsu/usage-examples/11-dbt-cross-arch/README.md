# Example 11: Dynamic Binary Translation (DBT)

## Objective

Use DBT to run code compiled for one GPU architecture on another.

## Key Concepts

- CDNA4 (gfx950) → CDNA3 (gfx942) translation
- ISA cross-compatibility
- DBT verification
- Performance comparison

## Files

- `src/dbt_example.cpp` - Simple kernel for DBT
- `Makefile`

## Quick Start

```bash
make ARCH=gfx950  # Compile for CDNA4
make run-dbt      # Translate and run on CDNA3 config
```

## Environment Variables

```bash
RJ_DBT_TARGET_ISA=gfx942  # Target architecture
RJ_DBT_LOG=1              # Enable DBT logging
```

## Use Cases

- Test code without target hardware
- Validate cross-architecture compatibility
- Debug ISA-specific issues