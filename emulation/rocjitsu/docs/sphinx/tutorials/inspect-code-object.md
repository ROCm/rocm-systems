---
myst:
    html_meta:
        "description": "Tutorial walking through code object inspection and instruction disassembly using the rocJITsu C API."
        "keywords": "rocJITsu, ROCm, code object, disassemble, basic block, C API, GPU, ISA"
---
# Inspect and disassemble a code object with the C API

This tutorial walks through a complete workflow for loading a GPU
executable, extracting a code object, decomposing it into basic blocks,
iterating over instructions, and disassembling each instruction into
human-readable text. You use the rocJITsu code object C API throughout,
and at the end you tear down every handle with the correct
retain/release/destroy sequence.

This workflow is useful when you need to inspect the contents of a
compiled HIP fat binary or a standalone device ELF without running the
GPU virtual machine --- for example, during offline analysis, tooling
integration, or debugging a dynamic binary translation pipeline.

See [API reference: code object](../reference/api-code-object.md) for the full
API reference.

## Load the executable

Start by loading an executable from disk. `rj_code_executable_create`
accepts the path to an x86 HIP fat binary or a standalone AMDGPU device
ELF and produces an opaque executable handle.

``` c
#include "rocjitsu/code/rj_code.h"

rj_code_executable_t *exec = NULL;
rj_status_t st = rj_code_executable_create("/path/to/vector_add", &exec);
if (st != ROCJITSU_STATUS_SUCCESS) {
    fprintf(stderr, "rj_code_executable_create failed: %d\n", st);
    return 1;
}
```

The returned handle has a reference count of zero. The caller owns it
and is responsible for destroying it when finished.

## Extract a code object

An executable might contain code objects for several GPU targets. Use
`rj_code_executable_get_code_object` to retrieve a code object for a
specific target and index. The call allocates a new handle with a
reference count of one.

``` c
rj_code_object_t *obj = NULL;
st = rj_code_executable_get_code_object(exec,
                                        ROCJITSU_CODE_TARGET_GFX942,
                                        /*index=*/0, &obj);
if (st != ROCJITSU_STATUS_SUCCESS) {
    fprintf(stderr, "rj_code_executable_get_code_object failed: %d\n", st);
    rj_code_executable_destroy(exec);
    return 1;
}
```

`ROCJITSU_CODE_TARGET_GFX942` selects the gfx942 (CDNA3) target. Other
targets such as `ROCJITSU_CODE_TARGET_GFX950` and
`ROCJITSU_CODE_TARGET_GFX1200` are available --- see the
`rj_code_target_id_t` enumeration in
[API reference: code object](../reference/api-code-object.md).

## Build the basic block list

`rj_code_basic_block_list_create` decodes the `.text` sections of the
code object and constructs a list of basic blocks --- straight-line
instruction sequences that end at a branch, program terminator, or
fall-through to another block.

``` c
rj_code_basic_block_list_t *bbs = NULL;
st = rj_code_basic_block_list_create(obj, ROCJITSU_CODE_TARGET_GFX942, &bbs);
if (st != ROCJITSU_STATUS_SUCCESS) {
    fprintf(stderr, "rj_code_basic_block_list_create failed: %d\n", st);
    rj_code_object_destroy(obj);
    rj_code_object_release(obj);
    rj_code_executable_destroy(exec);
    return 1;
}
```

The `target_id` argument tells the decoder which ISA to use. It should
match the target you selected when extracting the code object.

## Iterate blocks and disassemble instructions

Walk each basic block, then iterate over the instructions inside it
using `rj_code_basic_block_first_inst` and `rj_code_inst_next`. Each
instruction can be disassembled into a text buffer with
`rj_code_inst_disassemble`.

``` c
uint32_t num_blocks = rj_code_basic_block_list_size(bbs);
char dis_buf[256];

for (uint32_t i = 0; i < num_blocks; i++) {
    rj_code_basic_block_t *block = NULL;
    rj_code_basic_block_list_get(bbs, i, &block);

    const rj_code_inst_t *inst = rj_code_basic_block_first_inst(block);
    while (inst != NULL) {
        rj_code_inst_disassemble(inst, dis_buf, sizeof(dis_buf));
        printf("  [%uB] %s\n", rj_code_inst_size(inst), dis_buf);
        inst = rj_code_inst_next(inst);
    }

    rj_code_basic_block_destroy(block);
    rj_code_basic_block_release(block);
}
```

A few things to note in this loop:

-   `rj_code_basic_block_list_get` allocates a handle with reference
    count one. You must call both `rj_code_basic_block_destroy` and
    `rj_code_basic_block_release` when you are done with it.
-   `rj_code_basic_block_first_inst` returns `NULL` for an empty block.
    `rj_code_inst_next` returns `NULL` after the last instruction.
-   You can also query per-instruction metadata such as
    `rj_code_inst_mnemonic`, `rj_code_inst_size`, and
    `rj_code_inst_flags`. The flags bitmask uses values from
    `rj_code_inst_flags_t` --- for example, `RJ_CODE_INST_BRANCH` and
    `RJ_CODE_INST_PROGRAM_TERMINATOR`.

## Clean up handles

Every opaque handle in the code object API follows the same lifecycle:
call `*_destroy` to mark it for destruction, then `*_release` to drop
the reference. If the reference count is already zero at destroy time,
the memory is freed immediately; otherwise it is freed when the last
release brings the count to zero.

``` c
rj_code_basic_block_list_destroy(bbs);

rj_code_object_destroy(obj);
rj_code_object_release(obj);

rj_code_executable_destroy(exec);
```

The basic block list was created with a reference count of zero, so
calling `rj_code_basic_block_list_destroy` frees it directly. The code
object was created by `rj_code_executable_get_code_object` with a
reference count of one, so you need both the destroy and the release.

## Next steps

-   To translate a code object between GPU architectures, see
    [Translate a CDNA4 kernel to RDNA4 using rj_dbt_translate](translate-cdna4-to-rdna4.md),
    which uses `rj_code_translate`.
-   To analyze a code object for wait-count hazards, see
    [Analyze a code object for wait hazards with waitcheck](waitcheck-analyze-code-object.md).
-   For the full code object and instruction API reference, see
    [API reference: code object](../reference/api-code-object.md).
-   For an overview of how the code layer fits into the rocJITsu
    architecture, see [Architecture and component layers](../conceptual/architecture.md).
