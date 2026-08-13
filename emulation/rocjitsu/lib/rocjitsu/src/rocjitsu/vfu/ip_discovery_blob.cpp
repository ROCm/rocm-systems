// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/vfu/ip_discovery_blob.h"
#include "rocjitsu/vfu/mmio_registers.h"

#include <algorithm>
#include <cassert>
#include <cstring>

namespace rocjitsu::vfu {

namespace {

// ---------------------------------------------------------------------------
// Constants matching amdgpu_discovery.h / discovery.h
// ---------------------------------------------------------------------------
constexpr uint32_t kDiscoveryTmrSize   = 10 * 1024; // DISCOVERY_TMR_SIZE
constexpr uint32_t kBinarySignature    = 0x28211407;
constexpr uint32_t kDiscoveryTableSig  = 0x53445049; // "IPDS"
constexpr uint32_t kHarvestTableSig    = 0x56524148; // "HARV"
constexpr uint16_t kDiscoveryVersion   = 2;
constexpr int      kTotalTables        = 6;

// Hardware IDs — must match *_HWID in soc15_hw_ip.h exactly.
constexpr uint16_t kHwIdMp1    =   1;  // MP1_HWID
constexpr uint16_t kHwIdThm    =   3;  // THM_HWID
constexpr uint16_t kHwIdSmuio  =   4;  // SMUIO_HWID
constexpr uint16_t kHwIdGc     =  11;  // GC_HWID
constexpr uint16_t kHwIdMmhub  =  34;  // MMHUB_HWID
constexpr uint16_t kHwIdAthub  =  35;  // ATHUB_HWID
constexpr uint16_t kHwIdOsssys =  40;  // OSSSYS_HWID
constexpr uint16_t kHwIdHdp    =  41;  // HDP_HWID
constexpr uint16_t kHwIdSdma0  =  42;  // SDMA0_HWID (instances 0-4 for MI350P)
constexpr uint16_t kHwIdDf     =  46;  // DF_HWID
constexpr uint16_t kHwIdLsdma  =  91;  // LSDMA_HWID
constexpr uint16_t kHwIdNbif   = 108;  // NBIF_HWID
constexpr uint16_t kHwIdUmc    = 150;  // UMC_HWID
constexpr uint16_t kHwIdPcie   =  70;  // PCIE_HWID
constexpr uint16_t kHwIdXgmi   = 200;  // XGMI_HWID
constexpr uint16_t kHwIdMp0    = 255;  // MP0_HWID

// ---------------------------------------------------------------------------
// Low-level serialisation helpers
// ---------------------------------------------------------------------------

struct Writer {
  std::vector<uint8_t> &buf;
  size_t pos = 0;

  void seek(size_t p) { assert(p <= buf.size()); pos = p; }
  size_t tell() const { return pos; }

  void write_u8(uint8_t v)  { assert(pos + 1 <= buf.size()); buf[pos++] = v; }
  void write_u16(uint16_t v) {
    assert(pos + 2 <= buf.size());
    buf[pos] = v & 0xff; buf[pos+1] = v >> 8; pos += 2;
  }
  void write_u32(uint32_t v) {
    assert(pos + 4 <= buf.size());
    buf[pos]   =  v        & 0xff;
    buf[pos+1] = (v >>  8) & 0xff;
    buf[pos+2] = (v >> 16) & 0xff;
    buf[pos+3] = (v >> 24) & 0xff;
    pos += 4;
  }
  void pad(size_t n) { assert(pos + n <= buf.size()); pos += n; }
};

uint16_t checksum_range(const std::vector<uint8_t> &buf, size_t off, size_t len) {
  uint16_t s = 0;
  for (size_t i = 0; i < len; ++i)
    s += buf[off + i];
  return s;
}

// ---------------------------------------------------------------------------
// Emit one 'struct ip' (v2 format, 32-bit base addresses)
//   hw_id u16, num_instance u8, num_base_address u8,
//   major u8, minor u8, revision u8, harvest:4|reserved:4 u8,
//   base_address[num_base] u32[]
// ---------------------------------------------------------------------------
void emit_ip(Writer &w, uint16_t hw_id, uint8_t instance,
             uint8_t major, uint8_t minor, uint8_t revision,
             const std::initializer_list<uint32_t> bases) {
  w.write_u16(hw_id);
  w.write_u8(instance);
  w.write_u8(static_cast<uint8_t>(bases.size()));
  w.write_u8(major);
  w.write_u8(minor);
  w.write_u8(revision);
  w.write_u8(0);  // harvest=0, reserved=0
  for (uint32_t b : bases)
    w.write_u32(b);
}

} // namespace

// ---------------------------------------------------------------------------
// build_gfx944_discovery_blob
//
// IP versions are taken from the GFX9.4.3 (MI300X) aldebaran reference and
// bumped to .4 for GC/MMHUB/SDMA where amdgpu_discovery has explicit
// IP_VERSION(9, 4, 4) cases. The key requirement for the driver is that
// GC_HWIP reports version 9.4.4, which triggers:
//   - aqua_vanjaram_init_soc_config()
//   - gfx_v9_4_3_ip_block
//   - gmc_v9_0_ip_block
//   - vega10_common_ip_block
// ---------------------------------------------------------------------------
std::vector<uint8_t> build_gfx944_discovery_blob() {
  std::vector<uint8_t> blob(kDiscoveryTmrSize, 0);
  Writer w{blob};

  // ----------------------------------------------------------------
  // 1. binary_header  (offset 0, size = 4+2+2+2+2 + 6*8 = 60 bytes)
  //    Layout:
  //      uint32 binary_signature
  //      uint16 version_major
  //      uint16 version_minor
  //      uint16 binary_checksum   <-- filled in last
  //      uint16 binary_size
  //      table_info[6]: { uint16 offset, uint16 checksum, uint16 size, uint16 padding }
  // ----------------------------------------------------------------
  static constexpr size_t kBinaryHdrSize  = 12 + kTotalTables * 8; // 60
  static constexpr size_t kIpDiscoveryOff = kBinaryHdrSize;         // 60

  // Write binary_header fields (leave checksum as 0 for now)
  w.write_u32(kBinarySignature);
  w.write_u16(1); // version_major
  w.write_u16(0); // version_minor
  size_t binary_checksum_pos = w.tell();
  w.write_u16(0); // binary_checksum placeholder
  w.write_u16(0); // binary_size placeholder — filled below

  // table_info[6] — all zeroed first, then IP_DISCOVERY and HARVEST filled in
  size_t table_info_pos = w.tell();
  w.pad(kTotalTables * 8); // 48 bytes

  // ----------------------------------------------------------------
  // 2. ip_discovery_header at offset 60
  //    Layout:
  //      uint32 signature
  //      uint16 version
  //      uint16 size          <-- filled in after we know ip table size
  //      uint32 id
  //      uint16 num_dies
  //      die_info[16]: { uint16 die_id, uint16 die_offset }
  //      uint16 padding[1]
  //    Total: 4+2+2+4+2 + 16*4 + 2 = 80 bytes
  // ----------------------------------------------------------------
  static constexpr size_t kIpHdrInnerSize = 4+2+2+4+2 + 16*4 + 2; // 80
  size_t ip_hdr_pos = w.tell(); // == kIpDiscoveryOff
  w.write_u32(kDiscoveryTableSig);
  w.write_u16(kDiscoveryVersion);
  size_t ip_size_pos = w.tell();
  w.write_u16(0); // size placeholder
  w.write_u32(0x00000001); // id (arbitrary, non-zero)
  w.write_u16(1);           // num_dies = 1

  // die_info[0]: die_id=0 (must match loop index i=0 in amdgpu_discovery_reg_base_init).
  // die_offset is an ABSOLUTE offset from discovery_bin[0], not relative to ip_hdr.
  // die_header immediately follows the ip_discovery_header, so its absolute offset
  // is ip_hdr_pos + kIpHdrInnerSize.
  uint16_t die_hdr_abs_off = static_cast<uint16_t>(ip_hdr_pos + kIpHdrInnerSize);
  w.write_u16(0);                // die_id = 0
  w.write_u16(die_hdr_abs_off);  // die_offset (absolute from blob start)
  // die_info[1..15]: zeroed
  for (int i = 1; i < 16; ++i) { w.write_u16(0); w.write_u16(0); }
  w.write_u16(0); // padding

  // ----------------------------------------------------------------
  // 3. die_header + ip[] array
  //    die_header: uint16 die_id, uint16 num_ips
  // ----------------------------------------------------------------
  size_t die_hdr_pos = w.tell(); // == ip_hdr_pos + kIpHdrInnerSize
  w.write_u16(0); // die_id = 0 (must match die_info[0].die_id and loop index)
  size_t num_ips_pos = w.tell();
  w.write_u16(0); // num_ips placeholder

  // Emit IP blocks.  Each entry: 8 bytes header + 4*num_base bytes.
  // The GC IP version (9, 4, 4) is the key one that triggers the correct
  // driver code paths. Other versions match GFX9.4.3 (aldebaran reference).
  size_t ip_array_start = w.tell();
  int num_ips = 0;

  // All base addresses taken from aldebaran_ip_offset.h INST0_SEGx values.
  // These populate adev->reg_offset[HWIP][0][base_idx] for SOC15_REG_OFFSET.

  // GC — version 9.4.4 → activates gfx_v9_4_3_ip_block
  emit_ip(w, kHwIdGc,    0, 9,  4, 4, {0x2000, 0xa000, 0x2402c00});    ++num_ips;

  // MMHUB — version 9.4.4
  emit_ip(w, kHwIdMmhub, 0, 9,  4, 4, {0x1a000, 0x2408800});           ++num_ips;

  // ATHUB — version 9.4.2
  emit_ip(w, kHwIdAthub, 0, 9,  4, 2, {0xc20, 0x2408c00});             ++num_ips;

  // OSSSYS (IH) — version 4.4.0
  emit_ip(w, kHwIdOsssys,0, 4,  4, 0, {0x10a0, 0x240a000});            ++num_ips;

  // HDP — version 4.4.0
  emit_ip(w, kHwIdHdp,   0, 4,  4, 0, {0xf20, 0x240a400});             ++num_ips;

  // DF — version 3.6.2
  emit_ip(w, kHwIdDf,    0, 3,  6, 2, {0x7000, 0x240b800, 0x7c00000}); ++num_ips;

  // NBIF — version 6.3.1. Segments from aldebaran NBIO_BASE, with seg1 relocated
  // to 0xE00 so RSMU_INDEX/DATA land at BAR5 byte 0x3800/0x3804 (above the
  // threshold where QEMU routes vfio-user callbacks; original 0x14 gives byte 0x50
  // which QEMU maps as direct RAM and bypasses the vfio-user trap path).
  emit_ip(w, kHwIdNbif,  0, 6,  3, 1,
          {0x0, 0xE00, 0xd20, 0x10400, 0x241b000, 0x4040000});          ++num_ips;

  // MP0 — version 13.0.0 (maps to psp_v13_0_ip_block; psp_13_0_0_sos.bin exists
  // in VM firmware; doesn't add ras_v1_0_ip_block which crashes without real hw).
  emit_ip(w, kHwIdMp0,   0, 13, 0, 0,
          {0x16000, 0xdc0000, 0xe00000, 0xe40000, 0x243fc00});           ++num_ips;

  // MP1 (SMU) — version 13.0.0. Same segments as MP0.
  emit_ip(w, kHwIdMp1,   0, 13, 0, 0,
          {0x16000, 0xdc0000, 0xe00000, 0xe40000, 0x243fc00});           ++num_ips;

  // THM — version 13.0.14
  emit_ip(w, kHwIdThm,   0, 13, 0, 14,{0x16600, 0x2400c00});            ++num_ips;

  // SMUIO — version 13.0.6 → smuio_v13_0_6_funcs (13.0.14 is not in the switch).
  emit_ip(w, kHwIdSmuio, 0, 13, 0, 6,
          {0x16800, 0x16a00, 0x2401000, 0x3440000});                     ++num_ips;

  // SDMA — 1 instance (SDMA 4.4.5). Reduced from 5 to avoid exhausting VM
  // invalidation engines: each SDMA ring needs its own engine, and the MMHUB
  // only has 4 available in our configuration.  A single SDMA instance is
  // sufficient for the ring test and basic DMA operations in vfio-user emulation.
  {
    uint32_t base0 = 0x1260;
    uint32_t base1 = 0x12540;
    uint32_t base2 = 0x40a800;
    emit_ip(w, kHwIdSdma0, 0, 4, 4, 5, {base0, base1, base2});
    ++num_ips;
  }

  // XGMI — omitted. GFX9.4.4 sets xgmi.supported via amdgpu_is_multi_aid().
  // Including XGMI IP_VERSION(6,1,0) triggers adev->smuio.funcs->is_host_gpu_xgmi_supported()
  // before smuio funcs are set up, causing a null deref in gmc_v9_0_early_init.

  // PCIE — version 11.0.5. No segment table in aldebaran; use a high address.
  emit_ip(w, kHwIdPcie,  0, 11, 0, 5, {0x11180000});                    ++num_ips;

  // UMC — version 12.0.0. Segments from aldebaran UMC_BASE INST0.
  emit_ip(w, kHwIdUmc,   0, 12, 0, 0, {0x14000, 0x54000, 0x2425800});  ++num_ips;

  // VCN — version 4.0.5 (maps to vcn_v4_0_5_ip_block and jpeg_v4_0_5_ip_block with
  // num_jpeg_inst=1; 4.0.6 gives num_jpeg_inst=2 which triggers a crash in sw_init).
  // UVD_HWID = VCN_HWID = 12; base addresses from MI300X reference.
  emit_ip(w, 12,          0, 4,  0, 5, {0x81000, 0x2430000});               ++num_ips;

  // Fill in num_ips now that we know it
  blob[num_ips_pos]     = static_cast<uint8_t>(num_ips & 0xff);
  blob[num_ips_pos + 1] = static_cast<uint8_t>(num_ips >> 8);

  size_t ip_table_end = w.tell();

  // ip_discovery_header.size = total bytes from ip_hdr_pos to ip_table_end
  uint16_t ip_table_size = static_cast<uint16_t>(ip_table_end - ip_hdr_pos);
  blob[ip_size_pos]     = ip_table_size & 0xff;
  blob[ip_size_pos + 1] = ip_table_size >> 8;

  // ----------------------------------------------------------------
  // 4. Harvest info table (empty — no harvested blocks on MI350P base config)
  //    harvest_info_header: uint32 signature, uint32 version
  //    harvest_table: harvest_info[32] — all zero
  // ----------------------------------------------------------------
  size_t harvest_off = w.tell();
  w.write_u32(kHarvestTableSig);
  w.write_u32(1); // version
  // 32 * (uint16 hw_id, uint8 number_instance, uint8 reserved)
  for (int i = 0; i < 32; ++i) {
    w.write_u16(0); w.write_u8(0); w.write_u8(0);
  }
  size_t harvest_end = w.tell();

  // ----------------------------------------------------------------
  // 5. GC topology table (gc_info_v1_0) — shader engine geometry.
  //    GFX950 (GFX9.4.3): 8 SE, 1 SA/SE, 4 WGP0/SA (→8 CU/SH), 4 RB/SE.
  //    The driver reads this from table_list[GC] in the binary_header.
  //    max_cu_per_sh = 2 * (gc_num_wgp0_per_sa + gc_num_wgp1_per_sa) = 2*4 = 8.
  // ----------------------------------------------------------------
  size_t gc_off = w.tell();
  w.write_u32(0x4347u);       // table_id = GC_TABLE_ID (0x4347 = "GC")
  w.write_u16(1);              // version_major
  w.write_u16(0);              // version_minor
  w.write_u32(96);             // size = sizeof(gc_info_v1_0) = 12 + 21*4 = 96
  // gc_info_v1_0 fields (21 × uint32):
  w.write_u32(8);    // gc_num_se
  w.write_u32(4);    // gc_num_wgp0_per_sa   → max_cu_per_sh = 2*(4+0)=8
  w.write_u32(0);    // gc_num_wgp1_per_sa
  w.write_u32(4);    // gc_num_rb_per_se
  w.write_u32(16);   // gc_num_gl2c
  w.write_u32(256);  // gc_num_gprs
  w.write_u32(256);  // gc_num_max_gs_thds
  w.write_u32(32);   // gc_gs_table_depth
  w.write_u32(256);  // gc_gsprim_buff_depth
  w.write_u32(128);  // gc_parameter_cache_depth
  w.write_u32(1);    // gc_double_offchip_lds_buffer
  w.write_u32(64);   // gc_wave_size
  w.write_u32(8);    // gc_max_waves_per_simd
  w.write_u32(32);   // gc_max_scratch_slots_per_cu
  w.write_u32(64);   // gc_lds_size (KB)
  w.write_u32(4);    // gc_num_sc_per_se
  w.write_u32(1);    // gc_num_sa_per_se
  w.write_u32(2);    // gc_num_packer_per_sc
  w.write_u32(4);    // gc_num_gl2a
  size_t gc_end = w.tell();

  // ----------------------------------------------------------------
  // 6. Fill in binary_header.binary_size and table_info entries
  // ----------------------------------------------------------------
  uint16_t binary_size = static_cast<uint16_t>(gc_end);
  blob[binary_checksum_pos + 2] = binary_size & 0xff;
  blob[binary_checksum_pos + 3] = binary_size >> 8;

  // table_info[IP_DISCOVERY] (index 0): offset=60, checksum, size
  uint16_t ip_size16   = ip_table_size;
  uint16_t ip_checksum = checksum_range(blob, kIpDiscoveryOff, ip_table_size);
  size_t t0 = table_info_pos + 0 * 8;
  blob[t0]   = kIpDiscoveryOff & 0xff;
  blob[t0+1] = kIpDiscoveryOff >> 8;
  blob[t0+2] = ip_checksum & 0xff;
  blob[t0+3] = ip_checksum >> 8;
  blob[t0+4] = ip_size16 & 0xff;
  blob[t0+5] = ip_size16 >> 8;

  // table_info[HARVEST_INFO] (index 2): offset, checksum, size
  uint16_t harv_size     = static_cast<uint16_t>(harvest_end - harvest_off);
  uint16_t harv_off16    = static_cast<uint16_t>(harvest_off);
  uint16_t harv_checksum = checksum_range(blob, harvest_off, harv_size);
  size_t t2 = table_info_pos + 2 * 8;
  blob[t2]   = harv_off16 & 0xff;
  blob[t2+1] = harv_off16 >> 8;
  blob[t2+2] = harv_checksum & 0xff;
  blob[t2+3] = harv_checksum >> 8;
  blob[t2+4] = harv_size & 0xff;
  blob[t2+5] = harv_size >> 8;

  // table_info[GC] (index 1): offset, checksum, size
  uint16_t gc_size16     = static_cast<uint16_t>(gc_end - gc_off);
  uint16_t gc_off16      = static_cast<uint16_t>(gc_off);
  uint16_t gc_checksum   = checksum_range(blob, gc_off, gc_size16);
  size_t t1 = table_info_pos + 1 * 8;
  blob[t1]   = gc_off16 & 0xff;
  blob[t1+1] = gc_off16 >> 8;
  blob[t1+2] = gc_checksum & 0xff;
  blob[t1+3] = gc_checksum >> 8;
  blob[t1+4] = gc_size16 & 0xff;
  blob[t1+5] = gc_size16 >> 8;

  // binary_checksum: sum of all bytes from after the checksum field to binary_size.
  // The driver computes: offset = offsetof(binary_checksum) + sizeof(binary_checksum)
  // i.e. chk_start = 8 + 2 = 10, which includes binary_size itself.
  size_t chk_start = binary_checksum_pos + 2; // skip only binary_checksum, not binary_size
  uint16_t bin_checksum = checksum_range(blob, chk_start, binary_size - chk_start);
  blob[binary_checksum_pos]     = bin_checksum & 0xff;
  blob[binary_checksum_pos + 1] = bin_checksum >> 8;

  return blob;
}

} // namespace rocjitsu::vfu
