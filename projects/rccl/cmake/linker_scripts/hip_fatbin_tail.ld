/* hip_fatbin_tail.ld
 *
 * Move the merged .hip_fatbin output section to the very end of librccl.so.
 *
 * Why
 * ---
 * The bundled per-arch device image embedded in librccl.so (.hip_fatbin)
 * routinely grows to hundreds of megabytes and, with the full default
 * GPU_TARGETS list and uncompressed embedding, can exceed INT32_MAX.
 *
 * In lld's default orphan placement, .hip_fatbin (SHF_ALLOC, !SHF_WRITE,
 * !SHF_EXECINSTR) lands in the read-only loadable cluster, between .rodata
 * and .eh_frame.  That puts a multi-hundred-MB blob between .text and the
 * subsequent data sections (.eh_frame, .data.rel.ro, .got, .data, .bss, ...)
 * and breaks small-code-model host relocations:
 *   R_X86_64_PC32 / R_X86_64_PLT32   .text -> .rodata / .eh_frame / .data*
 *   R_X86_64_GOTPCRELX               .text -> .got slot on the far side
 *   R_X86_64_32S                     literal offsets in .eh_frame / .data.rel.ro
 *
 * Nothing in .hip_fatbin is the target of a small-model relocation from
 * host code in the usual sense.  The HIP runtime reaches it via the wrapper
 * struct in .hipFatBinSegment (which holds a 64-bit pointer materialized
 * via R_X86_64_RELATIVE / R_X86_64_64 in PIE/.so output).  The constructor
 * __hip_module_ctor references &__hip_fatbin_wrapper, not &__hip_fatbin.
 *
 * So if .hip_fatbin lives at the tail of the image, the text<->data island
 * shrinks back to roughly its pre-fatbin size and small-code-model
 * relocations stay in range.  The only long-distance reference is the
 * 64-bit pointer in the wrapper, which doesn't care about distance.
 *
 * How
 * ---
 * This file uses lld's (and ld.bfd's) INSERT AFTER directive: the default
 * linker script is preserved and we only splice .hip_fatbin in at a new
 * position.  KEEP() prevents --gc-sections from dropping the blob (the
 * relocations from .hipFatBinSegment already keep it live, but be explicit).
 * ALIGN(0x1000) preserves the 4 KiB alignment the runtime expects.
 *
 * Applies to the host link of librccl.so for both ENABLE_DEVICE_LINKER=ON
 * (custom device-linker pipeline) and ENABLE_DEVICE_LINKER=OFF (-fgpu-rdc
 * --hip-link path).  The offload device-link does not see this script.
 */
SECTIONS
{
  .hip_fatbin : ALIGN(0x1000) { KEEP(*(.hip_fatbin)) }
}
INSERT AFTER .bss;
