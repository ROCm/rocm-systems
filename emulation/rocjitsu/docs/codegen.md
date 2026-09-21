# ISA Code Generation

Instruction decoders, execution bodies, legalization tables, and encoding
translators are generated from the
[Machine-Readable ISA (MR ISA)](https://gpuopen.com/machine-readable-isa/)
XML specification via the `amdisa` Python library in `lib/python/amdisa/`.

## amdisa modules

| Module | Purpose |
|---|---|
| `parser.py` | Parse MR ISA XML specs into `IsaSpec` objects |
| `gpuisa.py` | Core data structures (`IsaSpec`, `Instruction`, `InstEncoding`, `Operand`) |
| `isa_profile.py` | Per-ISA profile constants and encoding rules |
| `semantics.py` | Derive instruction semantics from mnemonics |
| `cross_isa.py` | Cross-ISA instruction overlap analysis |
| `codegen/` | Generate C++ decoders, encoders, and instruction execute bodies |
| `legalization.py` | Generate cross-ISA legalization tables (Action classification) |
| `legalization_codegen.py` | Emit C++20 `InstructionLegalization[]` legalization table headers |
| `encoding_translator_codegen.py` | Emit C++20 neutral field structs + decode/encode functions |

## Installation

Create or activate a virtual environment, then install in editable mode from
the rocjitsu project root together with pre-commit:

```bash
python -m pip install -e lib/python/ pre-commit
```

The helper deliberately takes both Python and pre-commit from the same active
virtual environment. It prepends this checkout's `lib/python/` to `PYTHONPATH`,
so the generator implementation comes from the same checkout as the helper
while its installed dependencies come from the environment.

## MR ISA location

```
rocm-systems/shared/machine-readable-isa/isa/
```

## ISA additions

Repository-owned ISA additions can be supplied without modifying the public MR
ISA XML. Each ISA additions document has this shape:

```xml
<IsaAdditions Id="example-additions"
              BaseArchitecture="AMD CDNA 5"
              BaseSchemaVersion="1.2.0">
  <EncodingIdentifierAdditions>
    <EncodingIdentifierAddition>
      <EncodingName>ENC_EXAMPLE</EncodingName>
      <Opcode>7</Opcode>
      <EncodingIdentifier Radix="2">...complete identifier...</EncodingIdentifier>
    </EncodingIdentifierAddition>
  </EncodingIdentifierAdditions>
  <InstructionAdditions>
    <Instruction>
      <!-- One complete MR ISA <Instruction> node. -->
    </Instruction>
  </InstructionAdditions>
</IsaAdditions>
```

Apply one or more additions documents with a logical ISA name. Repeated
documents for one ISA are processed in command-line and document order:

```bash
python -m amdisa \
  cdna5:/path/to/amdgpu_isa_cdna5.xml \
  --isa-additions cdna5:/path/to/first-additions.xml \
  --isa-additions cdna5:/path/to/second-additions.xml \
  --isa-output /path/to/isa-output \
  --dbt-output /path/to/dbt-output
```

The `EncodingIdentifierAdditions` element is optional. `InstructionAdditions`
is required and may be empty. The additions contract is intentionally narrower
than a general XML patch:

- Only complete `<Instruction>` nodes and complete identifiers for an existing
  encoding's `<EncodingIdentifiers>` list are accepted. An additions document
  cannot replace or delete base definitions, identifier masks, microcode
  formats, encoding conditions, operand types, or data formats.
- Added instructions must reference encodings, encoding conditions, operand
  types, and operand fields already defined by the applicable base encoding
  layout. Implied-literal operands outside that layout are accepted only when
  their normalized field, type, direction, and parser-role flags match an
  implied-literal contract established by the base XML.
- An identifier addition names its encoding and expected opcode. Its radix and
  width must match the base encoding; its fixed bits must match an existing
  identifier layout; and its decoded opcode must equal the declared opcode.
  Duplicate and colliding decode slots are rejected. Each identifier must be
  owned by an added instruction. An implied-literal identifier must have the
  same instruction owner and opcode in its parent encoding.
- Additions document IDs and instruction names must be valid and unique.
  Encoding/opcode ownership collisions between different instructions,
  repeated instruction forms, multiple active forms that generate the same C++
  instruction symbol, out-of-range opcodes, missing references, unknown root
  data, and base architecture/schema mismatches are errors. Repetition of one
  instruction's slot under distinct base-established MR-ISA encoding conditions
  is allowed only when profile filtering leaves at most one generated form
  active. Every added instruction must retain at least one active form.
- Every input document and contained addition validates before the parser-owned
  tree is changed. Validated identifiers are merged before normal
  encoding/decode-table parsing, then instructions are appended in deterministic
  file/document order. A bad later file cannot leave a partially merged
  specification, and an active added instruction whose final opcode pointer
  would remain unpopulated is an error.
- An empty additions document leaves generated output byte-identical to the base
  XML.

The parsed `IsaSpec.applied_additions` and each added
`Instruction.source_addition` retain the source document's provenance for later
generator stages.
Provenance does not by itself define target availability: a separate variant
model must also be able to restrict encoding forms already present in the base
XML.

Production additions must be independently auditable from public inputs. Keep a
colocated provenance manifest or equivalent record that names a pinned public
source revision, the exact definition and test-vector locations, the public
base-XML template used for every field, and the derivation of each identifier.
When a repository verifier accompanies the delta, the production generation
script must run it before parsing the XML and fail closed if the delta contains
unrecorded metadata or no longer reconstructs from those inputs. The CDNA5
gfx1251 manifest and verifier next to its delta are the reference pattern.

## ISA variant capabilities

An optional JSON manifest assigns semantic capability bits to concrete targets
within one generated ISA family. It is deliberately separate from additive XML
deltas: the XML describes which instruction forms can be decoded, while the
manifest describes which concrete targets may use them. For example:

```json
{
  "schema_version": 1,
  "features": ["example_instruction", "example_encoding_form"],
  "variants": {
    "gfx_base": [],
    "gfx_plus": ["example_instruction", "example_encoding_form"]
  },
  "instructions": [
    {"feature": "example_instruction", "names": ["V_EXAMPLE"]}
  ],
  "model_only_instructions": ["V_EXAMPLE"],
  "encodings": [
    {
      "feature": "example_encoding_form",
      "encoding": "VOP1_VOP_DPP16",
      "instructions": ["V_OTHER"]
    }
  ]
}
```

Feature names describe reusable ISA properties rather than GPU revisions. The
array order assigns the generated feature bit positions; within one schema
version, existing entries must not be reordered or removed. The
loader validates the complete manifest before changing the parsed `IsaSpec`.
Unknown features, instructions, or encodings; duplicate requirements; malformed
names; more than 32 instruction features; and generated C++ name collisions
fail closed. Instruction requirements
apply to every form of a mnemonic. Encoding requirements apply only when that
runtime encoding modifier is present, so an ordinary form can remain legal when
its DPP16 form is target-specific. Schema version 1 deliberately accepts only
DPP16 encoding requirements; other encoding kinds fail validation until their
runtime construction paths implement the same fail-closed contract.

The generator emits immutable feature masks in `isa_features.h`. A
target-specific decoder compares the decoded instruction's combined
instruction-plus-encoding requirement with its concrete target mask before
returning it. Entries in `model_only_instructions` are emitted without execute
declarations, definitions, execution IDs, or callbacks; they can be decoded and
inspected but cannot accidentally acquire placeholder execution semantics.

Attach at most one manifest to each logical ISA name. It is applied after the
base XML and all ISA additions are parsed, and before semantics derivation and
code generation:

```bash
python -m amdisa cdna5:/path/to/amdgpu_isa_cdna5.xml \
  --isa-additions cdna5:/path/to/cdna5-additions.xml \
  --isa-variants cdna5:/path/to/cdna5-variants.json \
  --isa-output /path/to/isa-output \
  --dbt-output /path/to/dbt-output
```

## Generated file locations

| Generated files | Location | Generator |
|---|---|---|
| ISA decoders, encoders, execute bodies, and `insts.h` | `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/generated/<output-directory>/` | `codegen/` |
| Shared execute templates | `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/generated/shared/` | `codegen/` |
| ISA files generated at a custom path | A caller-selected tree such as `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/custom/generated/` | `codegen/` |
| Cross-ISA legalization tables | `lib/rocjitsu/src/rocjitsu/code/dbt/generated/` | `legalization_codegen.py` |
| Encoding decode/encode functions | `lib/rocjitsu/src/rocjitsu/code/dbt/generated/` | `encoding_translator_codegen.py` |

GPUOpen's public CDNA5 MR ISA uses the architecture name `AMD CDNA 5` and the
filename `amdgpu_isa_cdna5.xml`. rocjitsu's logical generator key and
configuration architecture are `cdna5`. Concrete GPU/runtime, ELF, and public
DBT identities remain `gfx1250` or `gfx1251`; an architecture-only lookup uses
the provider's explicit `gfx1250` default. Filesystem directories, generated
and hand-written C++ namespace (`rocjitsu::cdna5`), and CMake provider targets
continue to use `cdna5`.

Hand-written per-ISA files (`isa.h`, `mma_exec.h`, `addr_calc.h/.cpp`) remain
under `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/<output-directory>/` and are
not overwritten by the generator. CDNA5 hand-written files are under
`arch/amdgpu/cdna5/`. Hand-written shared headers remain under
`lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/shared/`.

The encoding translator engine (`code/dbt/encoding_translator.h`) is
hand-written and shared across all ISA pairs. Only the per-pair
decode/encode functions and neutral field structs are auto-generated.

## CLI reference

```
python -m amdisa [--isa-additions NAME:XML] [--isa-variants NAME:JSON]
                 [--gen-isas] [--gen-dbt]
                 [--isa-output DIR] [--include-root DIR]
                 [--dbt-output DIR] [NAME:]XML ...
```

| Option | Description |
|---|---|
| `[NAME:]XML ...` | One or more ISA XMLs. A recognized name selects its semantic profile. An unrecognized explicit name is treated as a custom generated identity, with its semantic profile detected from the XML |
| `--isa-additions NAME:XML` | Apply a validated ISA additions file to the named ISA; may be repeated |
| `--isa-variants NAME:JSON` | Attach one validated target-capability manifest to the named ISA |
| `--gen-isas` | Generate ISA C++ files (decoders, encodings, execute bodies) |
| `--gen-dbt` | Generate DBT legalization tables and encoding translators |
| `--isa-output DIR` | Output path for generated ISA C++ files |
| `--include-root DIR` | Compiler include root used to spell relocatable generated includes |
| `--dbt-output DIR` | Output directory for DBT tables (defaults to `--isa-output`) |

When neither `--gen-isas` nor `--gen-dbt` is specified, both are
generated. DBT generation is skipped when only one ISA is provided.

`--isa-output` controls where files are written. Generated include prefixes
are derived from that path. If `--include-root` is provided, `--isa-output`
must be beneath it and includes are emitted relative to it. Otherwise, absolute
include paths are emitted so that a relative output path does not depend on the
generator's working directory.

The recognized profile names include `cdna1` through `cdna5`, `rdna1` through
`rdna4`, and both `rdna3.5` and `rdna3_5`. An unrecognized explicit name is
treated as a custom generated identity; its semantic profile is detected from
the XML.

## Regenerating everything

The repository helper derives the repository, shared MR ISA, and generated
output directories from its own location. It uses the checked-in public CDNA5
MR ISA alongside the other XML inputs.
Activate a Python virtual environment containing the generator dependencies and
`pre-commit`, then run this command from the `rocm-systems` repository root:

```bash
./emulation/rocjitsu/scripts/generate-amdisa.sh
```

The helper can be invoked from any working directory when given by an
appropriate relative or absolute path. It discovers the active environment
through `VIRTUAL_ENV` or the active Python interpreter, then formats changed
generated files through the repository's pre-commit configuration.

The manual commands below are useful for focused generator development and are
run from the rocjitsu project root. Set `MRISA` to the shared MR ISA directory:

```bash
MRISA=../../shared/machine-readable-isa/isa

python -m amdisa \
    cdna1:$MRISA/amdgpu_isa_cdna1.xml \
    cdna2:$MRISA/amdgpu_isa_cdna2.xml \
    cdna3:$MRISA/amdgpu_isa_cdna3.xml \
    cdna4:$MRISA/amdgpu_isa_cdna4.xml \
    rdna1:$MRISA/amdgpu_isa_rdna1.xml \
    rdna2:$MRISA/amdgpu_isa_rdna2.xml \
    rdna3:$MRISA/amdgpu_isa_rdna3.xml \
    rdna3_5:$MRISA/amdgpu_isa_rdna3_5.xml \
    rdna4:$MRISA/amdgpu_isa_rdna4.xml \
    cdna5:$MRISA/amdgpu_isa_cdna5.xml \
  --isa-output lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/generated \
  --include-root lib/rocjitsu/src \
  --dbt-output lib/rocjitsu/src/rocjitsu/code/dbt/generated

find lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/generated lib/rocjitsu/src/rocjitsu/code/dbt/generated \
  \( -name '*.h' -o -name '*.cpp' \) -exec clang-format -i {} +
```

## Regenerating ISA files only

```bash
python -m amdisa \
    cdna1:$MRISA/amdgpu_isa_cdna1.xml \
    cdna2:$MRISA/amdgpu_isa_cdna2.xml \
    cdna3:$MRISA/amdgpu_isa_cdna3.xml \
    cdna4:$MRISA/amdgpu_isa_cdna4.xml \
    rdna1:$MRISA/amdgpu_isa_rdna1.xml \
    rdna2:$MRISA/amdgpu_isa_rdna2.xml \
    rdna3:$MRISA/amdgpu_isa_rdna3.xml \
    rdna3_5:$MRISA/amdgpu_isa_rdna3_5.xml \
    rdna4:$MRISA/amdgpu_isa_rdna4.xml \
    cdna5:$MRISA/amdgpu_isa_cdna5.xml \
  --gen-isas \
  --isa-output lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/generated \
  --include-root lib/rocjitsu/src

find lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/generated \
  \( -name '*.cpp' -o -name '*.h' \) -exec clang-format -i {} +
```

## Regenerating DBT files only

```bash
python -m amdisa \
    cdna1:$MRISA/amdgpu_isa_cdna1.xml \
    cdna2:$MRISA/amdgpu_isa_cdna2.xml \
    cdna3:$MRISA/amdgpu_isa_cdna3.xml \
    cdna4:$MRISA/amdgpu_isa_cdna4.xml \
    rdna1:$MRISA/amdgpu_isa_rdna1.xml \
    rdna2:$MRISA/amdgpu_isa_rdna2.xml \
    rdna3:$MRISA/amdgpu_isa_rdna3.xml \
    rdna3_5:$MRISA/amdgpu_isa_rdna3_5.xml \
    rdna4:$MRISA/amdgpu_isa_rdna4.xml \
    cdna5:$MRISA/amdgpu_isa_cdna5.xml \
  --gen-dbt \
  --dbt-output lib/rocjitsu/src/rocjitsu/code/dbt/generated

find lib/rocjitsu/src/rocjitsu/code/dbt/generated \
  \( -name '*.cpp' -o -name '*.h' \) -exec clang-format -i {} +
```

## Workflow

When modifying ISA semantics or adding instruction support:

1. Edit the authoritative Python input: the generator orchestration in
   `lib/python/amdisa/codegen/_generator.py`, execution emitters under
   `lib/python/amdisa/codegen/execute/`, instruction classification in
   `lib/python/amdisa/semantics.py`, or compatibility inventory logic in
   `lib/python/amdisa/parser.py` as appropriate. Never edit generated C++.
2. Regenerate with `scripts/generate-amdisa.sh` or the CLI as shown above
3. If you regenerated manually, format the generated files with `clang-format`
   (the helper formats changed generated files for you)
4. Stage ALL generated files before committing
