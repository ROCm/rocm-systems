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

    BOOLEAN     initialized;
};

/* GFX engine state */
struct GpuGfxState {
    BOOLEAN     rlc_ready;          /* RLC autoload completed */
    BOOLEAN     mec_enabled;        /* MEC (compute) engine running */
    BOOLEAN     sh_mem_configured;  /* SH_MEM_CONFIG set for all VMIDs */
    ULONG       cp_stat;            /* Last CP_STAT readback */
    ULONG       rlc_bootload_status;/* Last RLC bootload status */
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
    ULONG       doorbell_offset; /* Doorbell offset */
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
 * Load SMU firmware via PSP GPCOM ring.
 * Creates the PSP ring in VRAM, submits a LOAD_IP_FW command for
 * the SMU (UCODEType SMC=12), then destroys the ring.
 *
 * fw_dir: directory containing firmware .bin files (e.g. smu_14_0_3.bin).
 * Requires PSP SOS to be alive (call gpu_psp_load_sos first).
 * Returns 0 on success, -1 on failure.
 */
int gpu_psp_load_smu_fw(struct WddmLiteDevice *dev, const char *fw_dir);

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

#endif /* GPU_INIT_H_INCLUDED */
