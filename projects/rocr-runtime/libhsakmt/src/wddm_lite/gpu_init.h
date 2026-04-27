/*
 * GPU hardware initialization for wddm_lite backend.
 *
 * Performs IP discovery (reading the binary table from VRAM) and
 * GMC initialization (programming MMHUB/GFXHUB registers) through
 * the WDDM escape interface.
 *
 * This replaces kernel-side init — all register programming is done
 * from userspace via READ_REG32/WRITE_REG32 escapes.
 */

#ifndef GPU_INIT_H_INCLUDED
#define GPU_INIT_H_INCLUDED

#include "wddm_lite_escape.h"

struct WddmLiteDevice;

/* Maximum IP blocks we expect from discovery */
#define GPU_MAX_IP_BLOCKS 64

/* Hardware IDs from Linux amdgpu discovery.h */
#define GPU_HWID_GC     11
#define GPU_HWID_SDMA0  29
#define GPU_HWID_SDMA1  30
#define GPU_HWID_MMHUB  34
#define GPU_HWID_NBIO   14
#define GPU_HWID_MP0     1  /* PSP */
#define GPU_HWID_MP1     2  /* SMU */
#define GPU_HWID_IH     24
#define GPU_HWID_OSSSYS  8

/* IP block info from discovery table */
struct GpuIpBlock {
    USHORT  hw_id;
    UCHAR   instance;
    UCHAR   major, minor, revision;
    ULONG   base_addr[6];  /* DWORD offsets per base_index */
    UCHAR   num_base_addr;
};

/* IP discovery results */
struct GpuIpDiscovery {
    struct GpuIpBlock blocks[GPU_MAX_IP_BLOCKS];
    ULONG   num_blocks;

    /* Frequently-used base addresses (DWORD offsets) */
    ULONG   mmhub_base;    /* MMHUB base_index 0 */
    ULONG   gc_base;       /* GC base_index 0 */
    ULONG   gc_base1;      /* GC base_index 1 (for CP/MES regs) */
    ULONG   sdma0_base;    /* SDMA0 base_index 0 */
    ULONG   nbio_base;     /* NBIO base_index 0 */
    ULONG   nbio_base1;    /* NBIO base_index 1 (RSMU indirect) */
    ULONG   nbio_base2;    /* NBIO base_index 2 (HDP remap etc.) */
    ULONG   ih_base;       /* IH base_index 0 */
    ULONG   mp0_base;      /* MP0 (PSP) base_index 0 */
    ULONG   mp1_base;      /* MP1 (SMU) base_index 0 */
    ULONG   mp1_base1;     /* MP1 (SMU) base_index 1 */
};

/* GMC configuration state */
struct GpuGmcConfig {
    /* VRAM layout (from MMHUB FB_LOCATION registers) */
    ULONGLONG   vram_start;     /* MC address of VRAM start */
    ULONGLONG   vram_end;       /* MC address of VRAM end */

    /* GART (Graphics Address Remapping Table) */
    ULONGLONG   gart_start;     /* MC address of GART start */
    ULONGLONG   gart_end;       /* MC address of GART end */
    ULONGLONG   gart_size;      /* GART size in bytes (default 512MB) */

    /* Page table */
    ULONGLONG   gart_table_bus_addr;    /* Bus addr of GART page table */
    PVOID       gart_table_cpu_addr;    /* CPU mapping of GART table */
    PVOID       gart_table_handle;      /* DMA alloc handle */

    /* Dummy page for fault handling */
    ULONGLONG   dummy_page_bus_addr;
    PVOID       dummy_page_cpu_addr;
    PVOID       dummy_page_handle;

    /* GART slot allocator */
    ULONG       gart_next_slot;     /* Next free GART page index (bump) */
    ULONG       gart_total_slots;   /* Total GART pages */

    BOOLEAN     initialized;
};

/* GFX engine state */
struct GpuGfxState {
    BOOLEAN     rlc_ready;          /* RLC autoload completed */
    BOOLEAN     mec_enabled;        /* MEC (compute) engine running */
    BOOLEAN     sh_mem_configured;  /* SH_MEM_CONFIG set for all VMIDs */
    ULONG       cp_stat;            /* Last CP_STAT readback */
    ULONG       rlc_bootload_status;/* Last RLC bootload status */

    /* Firmware ucode entry points (from RS64 v2 headers, >> 2 for register) */
    ULONGLONG   pfp_ucode_start;    /* PFP program counter start */
    ULONGLONG   me_ucode_start;     /* ME program counter start */
    ULONGLONG   mec_ucode_start;    /* MEC program counter start */
    ULONGLONG   mes_ucode_start;    /* MES (scheduler, pipe 0) start */
    ULONGLONG   mes_kiq_ucode_start;/* MES KIQ (pipe 1) start */

    /* Saved MEC firmware bytes (for GART-based loading when AUTOLOAD fails) */
    UCHAR      *mec_fw_code;        /* Firmware code section */
    ULONG       mec_fw_code_size;
    UCHAR      *mec_fw_data;        /* Firmware data/stack section */
    ULONG       mec_fw_data_size;
};

/* Compute queue state (direct MMIO programming) */
#define GPU_MAX_COMPUTE_QUEUES  8

struct GpuComputeQueue {
    BOOLEAN     active;
    ULONG       me;         /* ME index (1 for MEC) */
    ULONG       pipe;       /* Pipe index */
    ULONG       queue;      /* Queue index within pipe */
    ULONGLONG   ring_addr;  /* GPU address of ring buffer */
    ULONG       ring_size;  /* Ring size in bytes */
    ULONGLONG   rptr_addr;  /* RPTR writeback address */
    ULONGLONG   wptr_addr;  /* WPTR poll address */
    ULONGLONG   eop_addr;   /* EOP buffer address */
    ULONG       eop_size;   /* EOP buffer size */
    ULONG       doorbell_offset; /* Doorbell DWORD offset for register */
    ULONG       doorbell_index;  /* Doorbell index for BAR byte offset */
    /* MQD in GPU-accessible memory */
    void       *mqd_cpu_addr;    /* CPU pointer to MQD DMA buffer */
    ULONGLONG   mqd_bus_addr;    /* Bus address (physical) */
    ULONGLONG   mqd_gpu_addr;    /* GART-mapped GPU address */
    ULONGLONG   mqd_alloc_handle; /* Allocation handle for free */
};

/* Extended device state for hardware init */
struct GpuHwState {
    struct GpuIpDiscovery   ip;
    struct GpuGmcConfig     gmc;
    struct GpuGfxState      gfx;
    struct GpuComputeQueue  queues[GPU_MAX_COMPUTE_QUEUES];
    ULONG                   num_active_queues;
    BOOLEAN                 ip_discovery_done;
    BOOLEAN                 gmc_initialized;
    BOOLEAN                 gfx_initialized;
    BOOLEAN                 gfxoff_disabled;
    BOOLEAN                 psp_sos_alive;
    ULONG                   hdp_flush_addr;  /* DWORD offset for HDP flush (from remap) */
};

/*
 * Run IP discovery by reading the discovery table from VRAM.
 * Populates dev->hw.ip with block info and base addresses.
 * Returns 0 on success, -1 on failure.
 */
int gpu_ip_discovery(struct WddmLiteDevice *dev);

/*
 * Load PSP SOS firmware from a file on disk.
 * Parses the firmware binary, loads each SOS component via the
 * PSP bootloader command interface (C2PMSG_35/36), and waits
 * for SOS to come alive.
 *
 * fw_path: path to the decompressed psp_14_0_3_sos.bin file.
 * Requires IP discovery to have been run first.
 * Returns 0 on success, -1 on failure.
 */
int gpu_psp_load_sos(struct WddmLiteDevice *dev, const char *fw_path);

/*
 * Load all GPU firmware via PSP GPCOM ring and trigger RLC autoload.
 *
 * Creates the PSP ring, loads firmware in order:
 *   SMU → SDMA → PFP → ME → MEC → RLC → AUTOLOAD_RLC
 *
 * fw_dir: directory containing firmware .bin files.
 * Requires PSP SOS to be alive (call gpu_psp_load_sos first).
 * Returns 0 on success, -1 on failure.
 */
int gpu_psp_load_all_fw(struct WddmLiteDevice *dev, const char *fw_dir);

/* Legacy name — calls gpu_psp_load_all_fw internally */
int gpu_psp_load_smu_fw(struct WddmLiteDevice *dev, const char *fw_dir);

/*
 * Trigger RLC autoload after firmware staging and GFXOFF disable.
 * Must be called AFTER gpu_psp_load_all_fw() and gpu_disable_gfxoff().
 * The RLC cannot distribute firmware while GC is in GFXOFF.
 */
int gpu_psp_trigger_autoload(struct WddmLiteDevice *dev);

/* Initialize GFXHUB with GART before AUTOLOAD.
 * Called after firmware staging wakes GC. */
void gpu_gfxhub_init_for_autoload(struct WddmLiteDevice *dev);

/*
 * Initialize GMC (GPU Memory Controller).
 * Programs MMHUB and GFXHUB registers for GART, system aperture,
 * TLB, L2 cache, and VM context 0.
 * Requires IP discovery to have been run first.
 * Returns 0 on success, -1 on failure.
 */
int gpu_gmc_init(struct WddmLiteDevice *dev);

/*
 * Clean up GMC resources (free GART table and dummy page).
 */
void gpu_gmc_cleanup(struct WddmLiteDevice *dev);

/*
 * Initialize IH (Interrupt Handler) ring.
 * Sets up the primary interrupt ring buffer so that PSP/SMU
 * completion interrupts have somewhere to land.
 * Must be called BEFORE PSP firmware loading (tinygrad init order).
 * Returns 0 on success, -1 on failure.
 */
int gpu_ih_init(struct WddmLiteDevice *dev);

/*
 * Map system memory pages into the GART.
 *
 * bus_addrs: array of physical/bus addresses (one per 4KB page)
 * num_pages: number of pages to map
 *
 * Returns the GPU virtual address (in the GART aperture) on success,
 * or 0 on failure (no GART slots available).
 * The mapped pages appear contiguous in GPU address space even if
 * the physical pages are scattered.
 */
ULONGLONG gpu_gart_map(struct WddmLiteDevice *dev,
                       const ULONGLONG *bus_addrs, ULONG num_pages);

/*
 * Map a single contiguous DMA buffer into the GART.
 * Convenience wrapper for gpu_gart_map when the physical memory
 * is known to be contiguous.
 *
 * bus_addr: physical/bus address of the buffer start
 * size: buffer size in bytes (rounded up to pages)
 *
 * Returns GPU virtual address or 0 on failure.
 */
ULONGLONG gpu_gart_map_contig(struct WddmLiteDevice *dev,
                              ULONGLONG bus_addr, ULONGLONG size);

/*
 * Unmap pages from GART (revert to dummy page PTEs).
 * gpu_addr: GPU address returned by gpu_gart_map
 * num_pages: number of pages to unmap
 */
void gpu_gart_unmap(struct WddmLiteDevice *dev,
                    ULONGLONG gpu_addr, ULONG num_pages);

/*
 * SMN indirect register access via NBIF RSMU INDEX/DATA pair.
 * This can reach ANY register on the System Management Network,
 * including GC registers that may be inaccessible via direct
 * MMIO BAR when GFXOFF is active.
 *
 * reg is a DWORD offset (same as IP discovery base addresses).
 * Internally converts to byte address for the RSMU_INDEX register.
 */
ULONG gpu_smn_rreg(struct WddmLiteDevice *dev, ULONG reg);
void gpu_smn_wreg(struct WddmLiteDevice *dev, ULONG reg, ULONG val);

/*
 * Flush the Host Data Path (HDP) to ensure CPU writes
 * are visible to the GPU. Should be called before GPU
 * reads from system memory written by CPU.
 */
void gpu_hdp_flush(struct WddmLiteDevice *dev);

/*
 * Send an SMU (System Management Unit) message via MP1 C2PMSG mailbox.
 * msg: PPSMC_MSG_* command ID
 * param: parameter for the message
 * Returns 0 on success, -1 on timeout or failure.
 */
int gpu_smu_send_msg(struct WddmLiteDevice *dev, ULONG msg, ULONG param);

/*
 * Initialize SMU after firmware loading.
 * Checks if SMU is alive, then sends EnableAllSmuFeatures.
 * Must be called after gpu_psp_load_all_fw() and before
 * gpu_disable_gfxoff().
 * Returns 0 on success, -1 if SMU is not alive.
 */
int gpu_smu_enable_features(struct WddmLiteDevice *dev);

/*
 * Perform SMU mode1 reset via debug mailbox.
 * This fully resets the GPU (SOS dies, bootloader restarts).
 * After this, SOS must be reloaded via bootloader commands.
 * Matches tinygrad's __DEBUGSMC_MSG_Mode1Reset (msg=2 on debug mailbox).
 */
int gpu_smu_mode1_reset(struct WddmLiteDevice *dev);

/*
 * Disable GFXOFF power saving.
 * Must be called before accessing any GC registers, as GFXOFF
 * powers down the GC block making all GC registers read as 0.
 * Requires IP discovery to have been run first.
 * Returns 0 on success, -1 on failure.
 */
int gpu_disable_gfxoff(struct WddmLiteDevice *dev);

/*
 * Re-enable GFXOFF power saving.
 */
void gpu_enable_gfxoff(struct WddmLiteDevice *dev);

/*
 * Initialize GFX engine: check RLC status, configure SH_MEM
 * for all VMIDs, set up MEC doorbell range, enable MEC.
 * Requires IP discovery and GFXOFF disabled.
 * Returns 0 on success, -1 on failure.
 */
int gpu_gfx_init(struct WddmLiteDevice *dev);

/*
 * Clean up GFX engine: dequeue any active HQDs, disable MEC.
 */
void gpu_gfx_cleanup(struct WddmLiteDevice *dev);

/*
 * Set up a compute queue by programming HQD registers directly.
 * Uses direct MQD→HQD register programming (tinygrad approach),
 * bypassing MES firmware.
 *
 * queue_idx: queue index (0-7)
 * ring_addr/ring_size: GPU ring buffer
 * rptr_addr: RPTR writeback address (8 bytes)
 * wptr_addr: WPTR poll address (8 bytes)
 * eop_addr/eop_size: end-of-pipe buffer
 * aql: TRUE for AQL queue (used by ROCR/HIP)
 *
 * Returns 0 on success, -1 on failure.
 */
int gpu_setup_compute_queue(struct WddmLiteDevice *dev,
                            ULONG queue_idx,
                            ULONGLONG ring_addr, ULONG ring_size,
                            ULONGLONG rptr_addr, ULONGLONG wptr_addr,
                            ULONGLONG eop_addr, ULONG eop_size,
                            BOOLEAN aql);

/*
 * Deactivate a compute queue's HQD.
 * Must be called before freeing queue DMA buffers so the GPU
 * stops referencing them.
 */
void gpu_deactivate_compute_queue(struct WddmLiteDevice *dev, ULONG queue_idx);

/*
 * Read HQD register diagnostic state for a queue.
 * Returns key register values via output params.
 */
void gpu_read_hqd_diag(struct WddmLiteDevice *dev, ULONG queue_idx,
                        ULONG *out_rptr, ULONG *out_wptr_lo,
                        ULONG *out_status, ULONG *out_active);

#endif /* GPU_INIT_H_INCLUDED */
